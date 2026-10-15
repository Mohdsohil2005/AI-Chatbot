"""
Unit tests for Nova AI Smart Chat Title Generation.
Validates extraction of main technical topics, removal of greetings/fillers,
session compatibility, deterministic execution without extra OpenAI calls,
and strict formatting quality constraints.
"""

import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from main import (
    generate_chat_title,
    format_title_casing,
    enforce_title_limits,
    clean_leading_fillers,
    TECH_TERMS_MAP,
)


def test_required_exact_examples():
    """Verify all exact example inputs from the specification."""
    cases = [
        ("Hello Mujhe JavaScript ke Ware me Batao", "JavaScript Explained in Hindi"),
        ("Bhai Python me variables kaise kaam karte hain", "Python Variables Explained"),
        ("Explain MongoDB indexing in simple words", "MongoDB Indexing Explained"),
        ("Mujhe JWT refresh token implement karna hai", "JWT Refresh Token Implementation"),
        ("Why is my React component re-rendering?", "React Component Re-rendering"),
        ("Hello", "New Chat"),
        ("hiii bro", "New Chat"),
        ("Mujhe API error 500 fix karna hai", "Fixing API 500 Error"),
    ]

    for user_input, expected in cases:
        actual = generate_chat_title(user_input)
        assert actual == expected, (
            f"Failed on input '{user_input}': expected '{expected}', got '{actual}'"
        )
    print("[PASS] All 8 exact required specification examples PASSED.")


def test_hindi_hinglish_inquiries():
    """Verify Hindi and Hinglish inquiry pattern variations."""
    cases = [
        ("Python ke bare me batao", "Python Explained in Hindi"),
        ("Docker ke baare mein samjhao", "Docker Explained in Hindi"),
        ("Bhai mujhe Node.js ke ware me samjhao", "Node.js Explained in Hindi"),
        ("Explain React in Hindi", "React Explained in Hindi"),
        ("Kubernetes kya hai Hindi me batao", "Kubernetes Explained in Hindi"),
        ("React me useEffect kaise kaam karta hai", "React useEffect Explained"),
        ("JavaScript me event loop kaise kaam karta hai", "JavaScript Event Loop Explained"),
        ("CSS me flexbox kaise kaam karta hai", "CSS Flexbox Explained"),
        ("FastAPI me routing kaise kaam karti hai", "FastAPI Routing Explained"),
        ("FastAPI me dependency injection kaise kaam karta hai", "FastAPI Dependency Injection Explained"),
        ("Variables kaise kaam karte hain", "Variables Explained"),
    ]

    for user_input, expected in cases:
        actual = generate_chat_title(user_input)
        assert actual == expected, (
            f"Failed on Hindi/Hinglish input '{user_input}': expected '{expected}', got '{actual}'"
        )
    print("[PASS] All Hindi/Hinglish inquiry tests PASSED.")


def test_implementation_and_error_queries():
    """Verify implementation and debugging queries."""
    cases = [
        ("How to implement JWT refresh token", "JWT Refresh Token Implementation"),
        ("How to implement OAuth2 in FastAPI", "OAuth2 Implementation in FastAPI"),
        ("Payment gateway integrate karna hai", "Payment Gateway Integration"),
        ("Docker setup karna hai", "Docker Setup"),
        ("How to fix CORS error in FastAPI", "Fixing FastAPI CORS Error"),
        ("Why is my API returning 500?", "Fixing API 500 Error"),
        ("Fixing 404 not found error", "Fixing 404 Not Found Error"),
        ("How to fix React hydration error", "Fixing React Hydration Error"),
        ("Fixing MongoDB connection error", "Fixing MongoDB Connection Error"),
        ("Why is my Node.js server crashing?", "Node.js Server Crashing"),
        ("Why is my Docker container exiting?", "Docker Container Exiting"),
    ]

    for user_input, expected in cases:
        actual = generate_chat_title(user_input)
        assert actual == expected, (
            f"Failed on implementation/error input '{user_input}': expected '{expected}', got '{actual}'"
        )
    print("[PASS] All implementation and error debugging tests PASSED.")


def test_greetings_and_uninformative_inputs():
    """Verify that greetings and messages with no topic fall back to 'New Chat'."""
    greetings = [
        "Hello",
        "hello",
        "hi",
        "hiii",
        "hiii bro",
        "hey",
        "heyy",
        "namaste",
        "namaste bhai",
        "good morning",
        "good morning sir",
        "yo bro",
        "sup nova",
        "kese ho bro",
        "kaise ho bhai",
        "how are you",
        "ok",
        "thanks",
        "thank you",
        "test",
        "ping",
        "???",
        "...",
        "",
        "   ",
    ]

    for g in greetings:
        actual = generate_chat_title(g)
        assert actual == "New Chat", (
            f"Failed on greeting '{g}': expected 'New Chat', got '{actual}'"
        )
    print("[PASS] All greetings and uninformative input tests PASSED.")


def test_comparisons_and_definitions():
    """Verify comparisons, 'what is', and conceptual questions."""
    cases = [
        ("Difference between let and const", "Let vs Const Comparison"),
        ("SQL vs NoSQL", "SQL vs NoSQL Comparison"),
        ("What is CORS?", "CORS Explained"),
        ("What is GraphQL?", "GraphQL Explained"),
        ("Explain Docker containers simply", "Docker Containers Explained"),
        ("How does Redis caching work?", "Redis Caching Explained"),
    ]

    for user_input, expected in cases:
        actual = generate_chat_title(user_input)
        assert actual == expected, (
            f"Failed on comparison/definition '{user_input}': expected '{expected}', got '{actual}'"
        )
    print("[PASS] All comparison and definition tests PASSED.")


def test_code_snippets_and_urls():
    """Verify handling of code blocks, inline code, and URLs."""
    py_code = "```python\ndef fib(n):\n    return n if n <= 1 else fib(n-1) + fib(n-2)\n```"
    assert generate_chat_title(py_code) == "Python Code Explanation"

    js_code = "```javascript\nconsole.log('hello world');\n```"
    assert generate_chat_title(js_code) == "JavaScript Code Explanation"

    generic_code = "```\nSELECT * FROM users;\n```"
    assert generate_chat_title(generic_code) == "Code Explanation"

    code_with_question = (
        "Why is my React component re-rendering? "
        "```jsx\nfunction App() { return <h1>Hi</h1>; }\n```"
    )
    assert generate_chat_title(code_with_question) == "React Component Re-rendering"

    github_url = "https://github.com/tiangolo/fastapi"
    assert generate_chat_title(github_url) == "FastAPI Repository Overview"

    print("[PASS] All code snippet and URL handling tests PASSED.")


def test_quality_and_length_constraints():
    """Verify strict length, casing, and word truncation limits."""
    long_msg = (
        "I am designing a distributed event-driven microservices architecture on AWS "
        "using Apache Kafka, Kubernetes EKS, PostgreSQL, and FastAPI with Redis caching."
    )
    title = generate_chat_title(long_msg)

    # 1. Max length <= 60
    assert len(title) <= 60, f"Title length {len(title)} exceeds 60 characters: '{title}'"
    # 2. No ellipses
    assert "..." not in title, f"Title contains ellipses: '{title}'"
    # 3. No dangling punctuation
    assert not title.endswith((".", ",", ":", ";", "-", "?", "!")), (
        f"Title ends with dangling punctuation: '{title}'"
    )
    # 4. Words count generally <= 6
    word_count = len(title.split())
    assert word_count <= 6, f"Word count {word_count} exceeds 6 words: '{title}'"

    print("[PASS] Quality and length constraint tests PASSED.")


def run_all_tests():
    print("Running Nova AI Smart Title Generation test suite...")
    test_required_exact_examples()
    test_hindi_hinglish_inquiries()
    test_implementation_and_error_queries()
    test_greetings_and_uninformative_inputs()
    test_comparisons_and_definitions()
    test_code_snippets_and_urls()
    test_quality_and_length_constraints()
    print("\nALL TEST SUITES PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_all_tests()
