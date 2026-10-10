
import os
from datetime import datetime, timezone
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI, APIError
from pydantic import BaseModel, Field
from pymongo.errors import PyMongoError

from database import sessions_collection, check_database

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

        return {"sessions": list(sessions)}
    except PyMongoError:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable. Check MongoDB."
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

    return session


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

        # Existing session ki conversation continue karo.
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
                "created_at": timestamp,
                "updated_at": timestamp,
            }

            sessions_collection.insert_one(session)

        # AI ko current question se pehle ki conversation do.
        previous_messages = session["messages"]

        history = [
            {
                "role": item["role"],
                "content": item["content"],
            }
            for item in previous_messages[-30:]
            if item["role"] in ("user", "assistant")
        ]

        # User message ko AI call se pehle save karo.
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

    except APIError:
        raise HTTPException(
            status_code=502,
            detail="OpenAI API request failed. Check your API key, "
                   "model access, quota, and billing."
        )

    assistant_timestamp = now()

    try:
        sessions_collection.update_one(
            {"session_id": session_id},
            {
                "$push": {
                    "messages": {
                        "role": "assistant",
                        "content": reply,
                        "timestamp": assistant_timestamp,
                    }
                },
                "$set": {"updated_at": assistant_timestamp},
            },
        )
    except PyMongoError:
        raise HTTPException(
            status_code=503,
            detail="AI replied, but saving the answer failed."
        )

    return ChatResponse(
        session_id=session_id,
        reply=reply,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
