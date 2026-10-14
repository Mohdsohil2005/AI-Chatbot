
import os
import re
from datetime import datetime, timezone
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI, APIError
from pydantic import BaseModel, Field
from pymongo.errors import PyMongoError

from database import sessions_collection, check_database

db_stats_collection = sessions_collection.database["db_stats"]

load_dotenv()

app = FastAPI(title="AI Chatbot API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)


api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

ai_client = OpenAI(api_key=api_key) if api_key else None


def now():
    return datetime.now(timezone.utc).isoformat()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)
    session_id: str | None = None


class RenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=60)




class ChatResponse(BaseModel):
    session_id: str
    reply: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    database_reads: int = 0


# ============================================================================
# DATABASE READS TRACKING & COUNTING RULES
# ============================================================================
# Counting Rules:
# Rule 1 (Chat Turn History Read): Each user interaction sent to POST /api/chat
#         queries MongoDB for session state and history (find_one). This counts
#         as 1 Database Read for that session and global metrics.
# Rule 2 (Session Retrieval Read): Opening/fetching a session via
#         GET /api/sessions/{session_id} performs a find_one operation to load
#         the conversation. This counts as 1 Database Read.
# Rule 3 (Sidebar List Reads): Listing sessions via GET /api/sessions reads
#         documents from MongoDB. The number of documents retrieved is credited
#         to global database reads metrics.
# Rule 4 (Read-Only Inspection): GET /api/sessions/{session_id}/usage inspects
#         and returns current counters without incrementing them.
# Persistence:
# - Session metrics are persisted in MongoDB inside:
#     session["token_usage"]["database_reads"] and session["database_reads"]
# - Global metrics are persisted in MongoDB collection "db_stats" (_id: "global_counter").
# ============================================================================

def record_db_read(session_id: str | None = None, count: int = 1):
    """
    Persistently records database read operations to MongoDB.
    """
    if count <= 0:
        return

    # Update global persistent metrics in db_stats collection
    try:
        db_stats_collection.update_one(
            {"_id": "global_counter"},
            {
                "$inc": {"total_reads": count},
                "$set": {"last_read_at": now()},
            },
            upsert=True,
        )
    except PyMongoError:
        pass

    # Update session persistent metrics if session_id provided
    if session_id:
        try:
            sessions_collection.update_one(
                {"session_id": session_id},
                {
                    "$inc": {
                        "token_usage.database_reads": count,
                        "database_reads": count,
                    }
                },
            )
        except PyMongoError:
            pass


def generate_chat_title(message: str) -> str:
    # Remove extra spaces and newlines.
    clean_message = " ".join(message.split())

    # Remove unnecessary punctuation from the ends.
    clean_message = clean_message.strip(" \t\n.,!?-")

    # Keep the sidebar title short and readable.
    max_length = 45

    if len(clean_message) > max_length:
        return clean_message[:max_length - 3].rstrip() + "..."

    return clean_message or "New Chat"


@app.get("/api/health")
def health():
    try:
        check_database()
        database_status = "connected"
    except PyMongoError:
        database_status = "disconnected"

    return {
        "status": "ok",
        "database": database_status,
        "openai_configured": ai_client is not None,
    }


@app.get("/api/sessions")
def get_sessions():
    try:
        sessions = sessions_collection.find(
            {},
            {
                "_id": 0,
                "session_id": 1,
                "title": 1,
                "created_at": 1,
                "updated_at": 1,
            },
        ).sort("updated_at", -1).limit(100)

        session_list = list(sessions)
        if session_list:
            record_db_read(count=len(session_list))

        return {"sessions": session_list}
    except PyMongoError:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable. Check MongoDB."
        )


@app.get("/api/sessions/search")
@app.get("/api/search")
def search_sessions(
    q: str = Query("", description="Search query for titles or messages"),
    limit: int = Query(50, ge=1, le=100)
):
    query = q.strip()
    if not query:
        return {"sessions": [], "query": ""}

    escaped_query = re.escape(query)

    try:
        filter_query = {
            "$or": [
                {"title": {"$regex": escaped_query, "$options": "i"}},
                {"messages.content": {"$regex": escaped_query, "$options": "i"}},
            ]
        }

        projection = {
            "_id": 0,
            "session_id": 1,
            "title": 1,
            "created_at": 1,
            "updated_at": 1,
            "messages.content": 1,
        }

        cursor = (
            sessions_collection.find(filter_query, projection)
            .sort("updated_at", -1)
            .limit(limit)
        )
        matched_sessions = list(cursor)

        if matched_sessions:
            record_db_read(count=len(matched_sessions))

        results = []
        for session in matched_sessions:
            messages = session.pop("messages", [])
            title = session.get("title", "")
            title_matched = bool(re.search(escaped_query, title, re.IGNORECASE))
            matching_snippet = None

            for msg in messages:
                content = msg.get("content", "")
                match = re.search(escaped_query, content, re.IGNORECASE)
                if match:
                    start = max(0, match.start() - 35)
                    end = min(len(content), match.end() + 45)
                    snippet = " ".join(content[start:end].split()).strip()
                    if start > 0:
                        snippet = "..." + snippet
                    if end < len(content):
                        snippet = snippet + "..."
                    matching_snippet = snippet
                    break

            results.append({
                "session_id": session["session_id"],
                "title": title or "New conversation",
                "created_at": session.get("created_at"),
                "updated_at": session.get("updated_at"),
                "matching_snippet": matching_snippet,
                "matched_in": (
                    "both" if (title_matched and matching_snippet)
                    else ("title" if title_matched else "message")
                ),
            })

        return {"sessions": results, "query": query}

    except PyMongoError:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable. Search failed."
        )


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    try:
        session = sessions_collection.find_one(
            {"session_id": session_id},
            {"_id": 0},
        )
    except PyMongoError:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable."
        )

    if not session:
        raise HTTPException(
            status_code=404,
            detail="Chat session not found."
        )

    # Persist the database read for fetching this session
    record_db_read(session_id=session_id, count=1)

    # Reflect updated read count in returned session object
    usage = session.setdefault("token_usage", {})
    current_reads = usage.get("database_reads", session.get("database_reads", 0)) + 1
    usage["database_reads"] = current_reads
    session["database_reads"] = current_reads

    return session


@app.get("/api/sessions/{session_id}/usage")
def get_session_usage(session_id: str):
    try:
        session = sessions_collection.find_one(
            {"session_id": session_id},
            {
                "_id": 0,
                "session_id": 1,
                "title": 1,
                "token_usage": 1,
                "database_reads": 1,
            },
        )

        if not session:
            raise HTTPException(
                status_code=404,
                detail="Chat session not found.",
            )

        usage = session.get("token_usage", {})
        database_reads = usage.get(
            "database_reads", session.get("database_reads", 0)
        )

        return {
            "session_id": session["session_id"],
            "title": session.get("title", "New chat"),
            "token_usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "api_calls": usage.get("api_calls", 0),
                "database_reads": database_reads,
            },
            "database_reads": database_reads,
        }

    except HTTPException:
        raise
    except PyMongoError:
        raise HTTPException(
            status_code=503,
            detail="Could not retrieve token usage from MongoDB.",
        )


@app.get("/api/database/reads")
def get_database_reads():
    try:
        counter_doc = db_stats_collection.find_one({"_id": "global_counter"}) or {}
        global_reads = counter_doc.get("total_reads", 0)
    except PyMongoError:
        global_reads = 0

    return {
        "status": "ok",
        "total_database_reads": global_reads,
        "counting_rules": {
            "rule_1_chat_turn": "Each chat interaction executes 1 read on MongoDB to load session history context (+1 to session and global).",
            "rule_2_session_fetch": "Opening/fetching a session via GET /api/sessions/{session_id} executes 1 document read on MongoDB (+1 to session and global).",
            "rule_3_sessions_list": "GET /api/sessions and GET /api/sessions/search read documents from MongoDB (+N to global reads, where N is returned documents count).",
            "rule_4_inspection": "GET /api/sessions/{session_id}/usage and GET /api/database/reads are inspection queries that report current counters without inflating metrics.",
            "persistence": "Persisted in MongoDB: 'database_reads' and 'token_usage.database_reads' on sessions, and 'db_stats' collection for global totals."
        }
    }


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    try:
        result = sessions_collection.delete_one(
            {"session_id": session_id}
        )

        if result.deleted_count == 0:
            raise HTTPException(
                status_code=404,
                detail="Chat session not found."
            )

        return {
            "success": True,
            "message": "Chat deleted successfully."
        }

    except HTTPException:
        raise
    except PyMongoError:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable. Could not delete chat."
        )



@app.patch("/api/sessions/{session_id}/title")
def rename_session(session_id: str, request: RenameRequest):
    title = " ".join(request.title.split()).strip()

    if not title:
        raise HTTPException(
            status_code=422,
            detail="Title cannot be empty."
        )

    try:
        result = sessions_collection.update_one(
            {"session_id": session_id},
            {"$set": {"title": title, "updated_at": now()}}
        )

        if result.matched_count == 0:
            raise HTTPException(
                status_code=404,
                detail="Chat session not found."
            )

        return {
            "success": True,
            "session_id": session_id,
            "title": title
        }

    except HTTPException:
        raise
    except PyMongoError:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable. Could not rename chat."
        )


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not ai_client:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is missing in backend/.env."
        )

    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=422,
            detail="Please enter a message."
        )

    timestamp = now()
    session_id = request.session_id or str(uuid4())

    try:
        session = sessions_collection.find_one(
            {"session_id": session_id}
        )

        # Create a new session when no session ID was supplied.
        if not session:
            if request.session_id:
                raise HTTPException(
                    status_code=404,
                    detail="Chat session not found."
                )

            session = {
                "session_id": session_id,
                "title": generate_chat_title(message),
                "messages": [],
                "token_usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "api_calls": 0,
                    "database_reads": 0,
                },
                "database_reads": 0,
                "created_at": timestamp,
                "updated_at": timestamp,
            }

            sessions_collection.insert_one(session)

        # Load previous conversation for context.
        previous_messages = session.get("messages", [])

        history = [
            {
                "role": item["role"],
                "content": item["content"],
            }
            for item in previous_messages[-30:]
            if item.get("role") in ("user", "assistant")
        ]

        # Save the user's message.
        sessions_collection.update_one(
            {"session_id": session_id},
            {
                "$push": {
                    "messages": {
                        "role": "user",
                        "content": message,
                        "timestamp": timestamp,
                    }
                },
                "$set": {"updated_at": timestamp},
            },
        )

    except HTTPException:
        raise
    except PyMongoError:
        raise HTTPException(
            status_code=503,
            detail="Could not save your message. Check MongoDB."
        )

    # Ask OpenAI to generate the response.
    try:
        response = ai_client.responses.create(
            model=model_name,
            instructions=(
                "You are a helpful general-purpose AI assistant. "
                "Answer questions clearly and accurately. "
                "Use simple language when appropriate."
            ),
            input=[
                *history,
                {"role": "user", "content": message},
            ],
        )

        reply = response.output_text.strip()

        if not reply:
            reply = "I couldn't generate a text response. Please try again."

        # Read actual token usage returned by OpenAI.
        usage = response.usage

        input_tokens = int(usage.input_tokens) if usage else 0
        output_tokens = int(usage.output_tokens) if usage else 0
        total_tokens = int(usage.total_tokens) if usage else 0

    except APIError:
        raise HTTPException(
            status_code=502,
            detail=(
                "OpenAI API request failed. Check your API key, "
                "model access, quota, and billing."
            )
        )

    assistant_timestamp = now()

    # Save the assistant reply and accumulate session usage.
    try:
        sessions_collection.update_one(
            {"session_id": session_id},
            {
                "$push": {
                    "messages": {
                        "role": "assistant",
                        "content": reply,
                        "timestamp": assistant_timestamp,
                        "usage": {
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": total_tokens,
                        },
                    }
                },
                "$inc": {
                    "token_usage.input_tokens": input_tokens,
                    "token_usage.output_tokens": output_tokens,
                    "token_usage.total_tokens": total_tokens,
                    "token_usage.api_calls": 1,
                    "token_usage.database_reads": 1,
                    "database_reads": 1,
                },
                "$set": {
                    "updated_at": assistant_timestamp,
                },
            },
        )

    except PyMongoError:
        raise HTTPException(
            status_code=503,
            detail="AI replied, but saving the answer and token usage failed."
        )

    # Record global read for this chat turn
    record_db_read(count=1)

    usage_data = session.get("token_usage", {})
    current_reads = (
        usage_data.get("database_reads", session.get("database_reads", 0)) + 1
    )

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        database_reads=current_reads,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
