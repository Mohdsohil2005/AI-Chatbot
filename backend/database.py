
import os

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb://127.0.0.1:27017/"
)

MONGODB_DATABASE = os.getenv(
    "MONGODB_DATABASE",
    "ai_chatbot"
)

client = MongoClient(
    MONGODB_URI,
    serverSelectionTimeoutMS=3000
)

db = client[MONGODB_DATABASE]
sessions_collection = db["sessions"]
db_stats_collection = db["db_stats"]

# Har session ID unique honi chahiye.
sessions_collection.create_index(
    "session_id",
    unique=True
)


def check_database():
    client.admin.command("ping")
    return True
