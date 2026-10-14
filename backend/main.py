
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
    title: str | None = None


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


# ============================================================================
# SMART CONVERSATION TITLE GENERATION
# ============================================================================

TECH_TERMS_MAP = {
    # Languages & Runtimes
    "python": "Python",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "html": "HTML",
    "html5": "HTML5",
    "css": "CSS",
    "css3": "CSS3",
    "sql": "SQL",
    "nosql": "NoSQL",
    "java": "Java",
    "rust": "Rust",
    "golang": "Go",
    "cpp": "C++",
    "c#": "C#",
    "php": "PHP",
    "ruby": "Ruby",
    # Frameworks & Libraries
    "react": "React",
    "vue": "Vue",
    "angular": "Angular",
    "nextjs": "Next.js",
    "next.js": "Next.js",
    "nuxtjs": "Nuxt",
    "nuxt.js": "Nuxt",
    "nodejs": "Node.js",
    "node.js": "Node.js",
    "express": "Express",
    "expressjs": "Express",
    "fastapi": "FastAPI",
    "flask": "Flask",
    "django": "Django",
    "tailwind": "Tailwind",
    "tailwindcss": "Tailwind CSS",
    "bootstrap": "Bootstrap",
    "redux": "Redux",
    # React Hooks & Concepts
    "useeffect": "useEffect",
    "usestate": "useState",
    "usecontext": "useContext",
    "usememo": "useMemo",
    "usecallback": "useCallback",
    "useref": "useRef",
    "usereducer": "useReducer",
    # Databases & Caches
    "mongodb": "MongoDB",
    "mongo": "MongoDB",
    "postgresql": "PostgreSQL",
    "postgres": "PostgreSQL",
    "mysql": "MySQL",
    "sqlite": "SQLite",
    "redis": "Redis",
    # DevOps, Infra & Cloud
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "k8s": "Kubernetes",
    "aws": "AWS",
    "gcp": "GCP",
    "azure": "Azure",
    "git": "Git",
    "github": "GitHub",
    "gitlab": "GitLab",
    "linux": "Linux",
    "nginx": "Nginx",
    # Protocols, Auth, & Standards
    "api": "API",
    "apis": "APIs",
    "rest": "REST",
    "restful": "RESTful",
    "graphql": "GraphQL",
    "jwt": "JWT",
    "oauth": "OAuth",
    "oauth2": "OAuth2",
    "http": "HTTP",
    "https": "HTTPS",
    "cors": "CORS",
    "json": "JSON",
    "crud": "CRUD",
    "url": "URL",
    "urls": "URLs",
    "uri": "URI",
    "uuid": "UUID",
    "id": "ID",
    "dom": "DOM",
    "sdk": "SDK",
    "sdks": "SDKs",
    # AI / ML
    "ai": "AI",
    "ml": "ML",
    "llm": "LLM",
    "llms": "LLMs",
    "nlp": "NLP",
    "gpt": "GPT",
    "ui": "UI",
    "ux": "UX",
    # HTTP Status Codes
    "500": "500",
    "404": "404",
    "403": "403",
    "401": "401",
    "400": "400",
    "502": "502",
    "503": "503",
}

GREETING_PATTERN = re.compile(
    r"^(?:hi+|hello+|hey+|heyy+|hola|namaste|namaskar|greetings|sup|yo|"
    r"good\s+(?:morning|afternoon|evening|day|night)|"
    r"test|testing|ping|pong|ok|okay|k|cool|thanks|thank\s+you|thx|"
    r"bye|goodbye|start|help|nova|ai|bot)"
    r"(?:\s+(?:there|bot|ai|nova|sir|bro|bhai|friend|all|team))?[.!?\s]*$",
    re.IGNORECASE,
)

MINOR_WORDS = {
    "a", "an", "the", "and", "but", "or", "for", "nor", "on", "at",
    "to", "from", "by", "with", "in", "of", "vs", "versus", "via",
    "me", "se", "ka", "ki", "ke", "ko", "par"
}


def format_title_casing(text: str) -> str:
    """
    Format words into title case while preserving technical terms, acronyms,
    and camelCase words, keeping minor connector words lowercase unless first/last.
    """
    tokens = text.split()
    if not tokens:
        return ""

    formatted = []
    total = len(tokens)

    for i, token in enumerate(tokens):
        clean_core = re.sub(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "", token)
        lower_core = clean_core.lower()

        if lower_core in TECH_TERMS_MAP:
            matched_term = TECH_TERMS_MAP[lower_core]
            prefix = token[: len(token) - len(clean_core) if clean_core else 0]
            suffix = token[len(prefix) + len(clean_core):] if clean_core else ""
            formatted.append(f"{prefix}{matched_term}{suffix}")
        elif i > 0 and i < total - 1 and lower_core in MINOR_WORDS:
            formatted.append(token.lower())
        else:
            if token.isupper() and len(token) <= 4:
                formatted.append(token)
            elif any(c.isupper() for c in token[1:]):
                formatted.append(token)
            else:
                formatted.append(token.capitalize())

    return " ".join(formatted)


def enforce_title_limits(title: str, max_words: int = 7, max_chars: int = 60) -> str:
    """
    Ensure the title stays within 3-7 words, ideally 25-50 chars, max 60 chars.
    Never cuts words mid-word or ends with dangling punctuation.
    """
    clean = " ".join(title.split()).strip(" \t\n.,!?:;-")
    if not clean:
        return "New Chat"

    words = clean.split()
    if len(words) > max_words:
        clean = " ".join(words[:max_words])

    if len(clean) > max_chars:
        trimmed = clean[:max_chars - 3].rsplit(" ", 1)[0].rstrip(" ,.-:;!?")
        clean = trimmed

    return clean or "New Chat"


def generate_chat_title(message: str) -> str:
    """
    Deterministically generates a concise, topic-focused title from the first message.
    - Preserves English, Hindi, Hinglish, technical terms, and acronyms.
    - Avoids extra AI calls or token consumption.
    - Yields 3-7 words, ~25-50 characters (max 60).
    - Falls back to 'New Chat' for greetings or uninformative inputs.
    """
    if not message:
        return "New Chat"

    raw = message.strip()

    # 1. Code snippet handling
    if "```" in raw:
        text_outside = re.sub(r"```[\s\S]*?```", "", raw).strip()
        clean_outside = " ".join(text_outside.split())
        if not clean_outside or re.search(
            r"^(?:please\s+)?(?:explain|check|review|debug|what\s+does|here\s+is|look\s+at)?\s*(?:this|the|my)?\s*code\b",
            clean_outside,
            re.IGNORECASE,
        ):
            return "Code Explanation"
        raw = clean_outside

    # Inline code quotes: `useEffect` -> useEffect
    raw = re.sub(r"`([^`]+)`", r"\1", raw)

    # 2. URL handling
    url_match = re.search(r"https?://(?:www\.)?([^\s/$.?#].[^\s]*)", raw, re.IGNORECASE)
    if url_match:
        full_url = url_match.group(0)
        domain_path = url_match.group(1)
        if raw.strip() == full_url:
            domain = domain_path.split("/")[0]
            parts = domain_path.split("/")
            if "github.com" in domain and len(parts) >= 3:
                return f"{format_title_casing(parts[2])} Repository Overview"
            return f"{format_title_casing(domain.split('.')[0])} Link Overview"
        raw = raw.replace(full_url, "").strip()

    clean_message = " ".join(raw.split()).strip(" \t\n.,!?:;-")

    # 3. Short greetings / uninformative check
    if not clean_message or len(clean_message) < 2 or GREETING_PATTERN.match(clean_message):
        return "New Chat"

    if not any(c.isalnum() for c in clean_message):
        return "New Chat"

    # 4. Explicit code explanation questions
    if re.search(
        r"^(?:can\s+you\s+(?:please\s+)?)?(?:please\s+)?explain\s+(?:this|the|my)?\s*code\b",
        clean_message,
        re.IGNORECASE,
    ):
        return "Code Explanation"
    if re.search(
        r"^(?:what\s+does\s+(?:this|the|my)?\s*code\s+do|code\s+(?:ko\s+)?(?:explain|samjhao|batao))\b",
        clean_message,
        re.IGNORECASE,
    ):
        return "Code Explanation"

    # 5. Debugging & Error handling (e.g. "Why is my API returning 500?")
    m_err = re.search(
        r"^(?:why\s+is\s+(?:my|the)?\s*|why\s+does\s+(?:my|the)?\s*)"
        r"([a-zA-Z0-9_\-\s]+?)\s*"
        r"(?:returning|throwing|giving|responding\s+with|showing)\s+"
        r"(\d{3}|error[s]?|[a-zA-Z]+error)\??$",
        clean_message,
        re.IGNORECASE,
    )
    if m_err:
        target = format_title_casing(m_err.group(1).strip())
        raw_code = m_err.group(2).strip()
        code_str = raw_code.upper() if raw_code.isdigit() else format_title_casing(raw_code)
        if "error" not in code_str.lower():
            code_str = f"{code_str} Error"
        return enforce_title_limits(f"Debugging {target} {code_str}")

    m_fix = re.search(
        r"^(?:how\s+to\s+fix|how\s+can\s+i\s+fix|fixing|fix|debug|debugging)\s+"
        r"([a-zA-Z0-9_\-\s]+?)\s+(?:error|issue|bug|exception)\??$",
        clean_message,
        re.IGNORECASE,
    )
    if m_fix:
        target = format_title_casing(m_fix.group(1).strip())
        return enforce_title_limits(f"Debugging {target} Error")

    m_hinglish_err = re.search(
        r"^([a-zA-Z0-9_\-\s]+?)\s+(?:me\s+)?(?:error|issue|bug)\s+(?:aa\s+raha\s+hai|kaise\s+fix\s+kare)\??$",
        clean_message,
        re.IGNORECASE,
    )
    if m_hinglish_err:
        target = format_title_casing(m_hinglish_err.group(1).strip())
        return enforce_title_limits(f"Debugging {target} Error")

    # 6. "Explain X in simple words" -> "X Explained Simply"
    m_simple = re.search(
        r"^(?:can\s+you\s+(?:please\s+)?)?(?:please\s+)?explain\s+"
        r"([a-zA-Z0-9_\-\s]+?)\s+"
        r"(?:in\s+simple\s+words|in\s+simple\s+terms|in\s+easy\s+words|in\s+easy\s+language|simply)\??$",
        clean_message,
        re.IGNORECASE,
    )
    if m_simple:
        topic = format_title_casing(m_simple.group(1).strip())
        return enforce_title_limits(f"{topic} Explained Simply")

    # 7. "How does X work?" -> "Understanding X"
    m_how_work = re.search(
        r"^(?:how\s+does|how\s+do)\s+([a-zA-Z0-9_\-\s]+?)\s+work(?:ing)?\??$",
        clean_message,
        re.IGNORECASE,
    )
    if m_how_work:
        topic = format_title_casing(m_how_work.group(1).strip())
        return enforce_title_limits(f"Understanding {topic}")

    # 8. Hinglish: "React me useEffect kaise kaam karta hai" -> "React useEffect Explained"
    m_hinglish_tech = re.search(
        r"^([a-zA-Z0-9_\-]+)\s+me\s+([a-zA-Z0-9_\-\s]+?)\s+"
        r"kaise\s+(?:kaam\s+karta\s+hai|work\s+karta\s+hai|chalta\s+hai)\??$",
        clean_message,
        re.IGNORECASE,
    )
    if m_hinglish_tech:
        t1 = format_title_casing(m_hinglish_tech.group(1).strip())
        t2 = format_title_casing(m_hinglish_tech.group(2).strip())
        return enforce_title_limits(f"{t1} {t2} Explained")

    m_hinglish_single = re.search(
        r"^([a-zA-Z0-9_\-\s]+?)\s+"
        r"(?:kaise\s+(?:kaam\s+karta\s+hai|work\s+karta\s+hai)|kya\s+hota\s+hai|kya\s+hai)\??$",
        clean_message,
        re.IGNORECASE,
    )
    if m_hinglish_single:
        topic = format_title_casing(m_hinglish_single.group(1).strip())
        return enforce_title_limits(f"{topic} Explained")

    # 9. Hinglish: "JWT refresh token implement kaise kare" -> "JWT Refresh Token Implementation"
    m_hinglish_impl = re.search(
        r"^([a-zA-Z0-9_\-\s]+?)\s+"
        r"(?:ko\s+)?(implement|banaye|banao|setup|integrate)\s+"
        r"kaise\s+(?:kare[n]?|karte\s+hai[n]?|hota\s+hai)\??$",
        clean_message,
        re.IGNORECASE,
    )
    if m_hinglish_impl:
        raw_topic = m_hinglish_impl.group(1).strip()
        clean_topic = re.sub(r"\bme\b", "", raw_topic, flags=re.IGNORECASE)
        clean_topic = " ".join(clean_topic.split())
        topic = format_title_casing(clean_topic)
        action = m_hinglish_impl.group(2).lower()
        intent = {
            "implement": "Implementation",
            "banaye": "Creation Guide",
            "banao": "Creation Guide",
            "setup": "Setup Guide",
            "integrate": "Integration",
        }.get(action, "Implementation")
        return enforce_title_limits(f"{topic} {intent}")

    # 10. English: "How to implement/build/create X"
    m_eng_impl = re.search(
        r"^(?:how\s+to|how\s+do\s+i|how\s+can\s+i)\s+"
        r"(implement|build|create|setup|configure|integrate)\s+"
        r"([a-zA-Z0-9_\-\s]+?)(?:\s+in\s+([a-zA-Z0-9_\-\s]+))?\??$",
        clean_message,
        re.IGNORECASE,
    )
    if m_eng_impl:
        action = m_eng_impl.group(1).lower()
        topic = format_title_casing(m_eng_impl.group(2).strip())
        ctx = format_title_casing(m_eng_impl.group(3).strip()) if m_eng_impl.group(3) else ""
        noun = {
            "implement": "Implementation",
            "build": "Building",
            "create": "Creation",
            "setup": "Setup",
            "configure": "Configuration",
            "integrate": "Integration",
        }.get(action, "Guide")
        if ctx:
            return enforce_title_limits(f"{topic} {noun} in {ctx}")
        return enforce_title_limits(f"{topic} {noun}")

    # 11. Comparison: "Difference between X and Y" / "X vs Y"
    m_diff = re.search(
        r"^(?:what\s+is\s+the\s+)?difference\s+between\s+([a-zA-Z0-9_\-\s]+?)\s+and\s+([a-zA-Z0-9_\-\s]+?)\??$",
        clean_message,
        re.IGNORECASE,
    )
    if m_diff:
        t1 = format_title_casing(m_diff.group(1).strip())
        t2 = format_title_casing(m_diff.group(2).strip())
        return enforce_title_limits(f"{t1} vs {t2} Comparison")

    m_vs = re.search(
        r"^([a-zA-Z0-9_\-]+)\s+(?:vs|versus)\s+([a-zA-Z0-9_\-]+)\??$",
        clean_message,
        re.IGNORECASE,
    )
    if m_vs:
        t1 = format_title_casing(m_vs.group(1).strip())
        t2 = format_title_casing(m_vs.group(2).strip())
        return enforce_title_limits(f"{t1} vs {t2} Comparison")

    # 12. "Explain X" -> "X Explained"
    m_exp = re.search(
        r"^(?:can\s+you\s+(?:please\s+)?)?(?:please\s+)?explain\s+"
        r"([a-zA-Z0-9_\-\s]+?)\??$",
        clean_message,
        re.IGNORECASE,
    )
    if m_exp:
        raw_topic = m_exp.group(1).strip()
        if raw_topic.lower() not in ("this", "it", "code", "this code", "everything"):
            topic = format_title_casing(raw_topic)
            return enforce_title_limits(f"{topic} Explained")

    # 13. General normalization pipeline for arbitrary questions/topics
    topic_candidate = clean_message

    prefix_strip = [
        r"^(?:hi|hello|hey|greetings|namaste)\s+(?:nova|bot|ai|there)?[,\s]+",
        r"^(?:can\s+you\s+(?:please\s+)?(?:tell\s+me|show\s+me|help\s+me\s+with|explain(?:\s+to\s+me)?|give\s+me)?)\s+",
        r"^(?:could\s+you\s+(?:please\s+)?(?:tell\s+me|show\s+me|help\s+me|explain)?)\s*",
        r"^(?:please\s+(?:tell\s+me|explain|show\s+me|help\s+me\s+with|provide|give\s+me)?)\s*",
        r"^(?:i\s+want\s+to\s+(?:know|learn|understand|build|create|implement))\s+(?:about\s+)?",
        r"^(?:i\s+need\s+help\s+(?:with|to))\s+",
        r"^(?:tell\s+me\s+(?:about|how\s+to)?)\s*",
        r"^(?:give\s+me\s+(?:an?\s+)?(?:example\s+of|overview\s+of|tutorial\s+on)?)\s*",
        r"^(?:what\s+is\s+(?:the\s+)?(?:best\s+way\s+to)?)\s*",
        r"^(?:what\s+are\s+(?:the\s+)?)\s*",
        r"^(?:mujhe\s+(?:batao|jan na\s+hai|seekhna\s+hai))\s+",
        r"^(?:kripya\s+)?",
    ]
    for pattern in prefix_strip:
        topic_candidate = re.sub(pattern, "", topic_candidate, flags=re.IGNORECASE)

    suffix_strip = [
        r"\s+(?:in\s+simple\s+(?:words|terms)|simply|for\s+beginners)$",
        r"\s+(?:step\s+by\s+step|with\s+examples?|in\s+detail)$",
        r"\s+(?:batao|samjhao|bataiye|samjhaiye|karo)$",
        r"\s+(?:kya\s+hota\s+hai|kya\s+hai)$",
        r"\s+(?:kaise\s+kare[n]?|kaise\s+hota\s+hai)$",
        r"\s+(?:please|thanks|thank\s+you|thx)$",
    ]
    for pattern in suffix_strip:
        topic_candidate = re.sub(pattern, "", topic_candidate, flags=re.IGNORECASE)

    topic_candidate = topic_candidate.strip(" \t\n.,!?:;-")

    if not topic_candidate or len(topic_candidate) < 2:
        return "New Chat"

    formatted = format_title_casing(topic_candidate)
    return enforce_title_limits(formatted)


@app.get("/api/test-title")
def test_title(message: str = Query(...)):
    generated = generate_chat_title(message)
    return {
        "input": message,
        "title": generated,
        "word_count": len(generated.split()),
        "char_count": len(generated),
    }


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
            {"$set": {"title": title, "is_renamed": True, "updated_at": now()}}
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
                "is_renamed": False,
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

        # Upgrade title from generic greeting fallback if new message is substantive and session was not manually renamed.
        existing_title = session.get("title", "")
        title_update = {}
        if (
            existing_title in ("New Chat", "New conversation", "")
            and not session.get("is_renamed", False)
        ):
            candidate_title = generate_chat_title(message)
            if candidate_title not in ("New Chat", "New conversation", ""):
                title_update["title"] = candidate_title
                session["title"] = candidate_title

        update_set = {"updated_at": timestamp}
        if title_update:
            update_set.update(title_update)

        filter_query = {"session_id": session_id}
        if title_update:
            filter_query["is_renamed"] = {"$ne": True}

        # Save the user's message.
        sessions_collection.update_one(
            filter_query,
            {
                "$push": {
                    "messages": {
                        "role": "user",
                        "content": message,
                        "timestamp": timestamp,
                    }
                },
                "$set": update_set,
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
        title=session.get("title"),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
