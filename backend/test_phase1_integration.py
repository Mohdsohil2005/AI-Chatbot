import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from main import app, UPLOADS_DIR
from database import sessions_collection


def create_sample_pdf(text: str = "Nova AI PDF Document Test Content") -> bytes:
    """Create a minimal valid raw PDF 1.4 stream containing text."""
    stream_content = f"BT\n/F1 12 Tf\n72 200 Td\n({text}) Tj\nET\n".encode("latin-1")
    stream_len = len(stream_content)
    pdf_template = (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
        b"4 0 obj << /Length " + str(stream_len).encode("ascii") + b" >> stream\n" + stream_content + b"endstream\nendobj\n"
        b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000224 00000 n \n"
        b"0000000328 00000 n \n"
        b"trailer << /Size 6 /Root 1 0 R >>\n"
        b"startxref\n400\n%%EOF\n"
    )
    return pdf_template


class TestPhase1DocumentUploadAndQA(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.created_sessions = []

    def tearDown(self):
        # Clean up test sessions from MongoDB and disk
        for sid in self.created_sessions:
            sessions_collection.delete_one({"session_id": sid})
            sess_dir = UPLOADS_DIR / sid
            if sess_dir.exists():
                import shutil
                shutil.rmtree(sess_dir, ignore_errors=True)

    def test_01_text_only_regression(self):
        """Test sending a normal text message without file attachment."""
        fake_response = MagicMock(
            output_text="Python is a high-level programming language.",
            usage=MagicMock(input_tokens=15, output_tokens=10, total_tokens=25)
        )

        with patch("main.ai_client.responses.create", return_value=fake_response):
            res = self.client.post("/api/chat", json={"message": "Explain Python in simple words"})
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.created_sessions.append(data["session_id"])

            self.assertIn("reply", data)
            self.assertIsNone(data.get("attachment"))
            self.assertEqual(data["reply"], "Python is a high-level programming language.")
            self.assertEqual(data["total_tokens"], 25)

            # Check session in MongoDB
            sess = sessions_collection.find_one({"session_id": data["session_id"]})
            self.assertIsNotNone(sess)
            self.assertEqual(len(sess["messages"]), 2)
            self.assertNotIn("attachment", sess["messages"][0])

    def test_02_upload_file_validations(self):
        """Test rejection of invalid, empty, fake, and oversized files."""
        # 1. Unsupported extension (.exe)
        res = self.client.post(
            "/api/upload",
            files={"file": ("malicious.exe", b"binary content", "application/octet-stream")}
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("Unsupported file", res.json()["detail"])

        # 2. Empty file (0 bytes)
        res = self.client.post(
            "/api/upload",
            files={"file": ("empty.txt", b"", "text/plain")}
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("empty", res.json()["detail"].lower())

        # 3. Oversized file (> 10MB)
        oversized = b"a" * (10 * 1024 * 1024 + 10)
        res = self.client.post(
            "/api/upload",
            files={"file": ("large.txt", oversized, "text/plain")}
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("exceeds", res.json()["detail"].lower())

        # 4. Fake PDF (wrong header)
        res = self.client.post(
            "/api/upload",
            files={"file": ("fake.pdf", b"NOT_A_REAL_PDF_HEADER", "application/pdf")}
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("pdf", res.json()["detail"].lower())
        self.assertIn("header", res.json()["detail"].lower())

        # 5. TXT file containing binary null bytes
        res = self.client.post(
            "/api/upload",
            files={"file": ("binary.txt", b"Hello\x00World", "text/plain")}
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("binary", res.json()["detail"].lower())

    def test_03_valid_txt_upload_and_qa(self):
        """Test valid TXT upload and Q&A flow with chunking & retrieval."""
        txt_content = (
            "Nova AI Architecture Specification.\n\n"
            "Section 1: Security Model.\n"
            "All uploaded documents are sandboxed in isolated session directories. "
            "Cross-session access is strictly forbidden with HTTP 403 status code.\n\n"
            "Section 2: Database Storage.\n"
            "MongoDB stores document metadata and chunk references, keeping binary payloads off database records."
        ).encode("utf-8")

        # 1. Upload TXT file
        upload_res = self.client.post(
            "/api/upload",
            files={"file": ("architecture_spec.txt", txt_content, "text/plain")}
        )
        self.assertEqual(upload_res.status_code, 200)
        meta = upload_res.json()
        session_id = meta["session_id"]
        file_id = meta["file_id"]
        self.created_sessions.append(session_id)

        self.assertEqual(meta["filename"], "architecture_spec.txt")
        self.assertEqual(meta["file_type"], "txt")
        self.assertGreater(meta["chunk_count"], 0)

        # 2. Ask question about the document
        fake_response = MagicMock(
            output_text="According to Section 1, all uploaded documents are sandboxed in isolated session directories.",
            usage=MagicMock(input_tokens=40, output_tokens=20, total_tokens=60)
        )

        captured_input = []

        def mock_create(*args, **kwargs):
            # Capture the prompt passed to OpenAI Responses API
            captured_input.append(kwargs.get("input", []))
            return fake_response

        with patch("main.ai_client.responses.create", side_effect=mock_create):
            chat_res = self.client.post("/api/chat", json={
                "message": "What does Section 1 say about security?",
                "session_id": session_id,
                "file_id": file_id
            })
            self.assertEqual(chat_res.status_code, 200)
            chat_data = chat_res.json()

            # Verify prompt contains document context
            self.assertTrue(len(captured_input) > 0)
            full_user_prompt = captured_input[0][-1]["content"]
            self.assertIn("<document_context", full_user_prompt)
            self.assertIn("</document_context>", full_user_prompt)
            self.assertIn("Security Model", full_user_prompt)
            self.assertIn("What does Section 1 say about security?", full_user_prompt)

            # Verify attachment is returned in response
            self.assertIsNotNone(chat_data.get("attachment"))
            self.assertEqual(chat_data["attachment"]["filename"], "architecture_spec.txt")
            self.assertEqual(chat_data["attachment"]["file_id"], file_id)

            # Verify saved in MongoDB message history
            sess = sessions_collection.find_one({"session_id": session_id})
            user_msg = sess["messages"][0]
            self.assertIn("attachment", user_msg)
            self.assertEqual(user_msg["attachment"]["file_id"], file_id)

    def test_04_valid_pdf_upload_and_qa(self):
        """Test valid PDF upload and retrieval."""
        pdf_bytes = create_sample_pdf("Nova AI PDF Manual. Chapter 3: Smart indexing and Vector search.")

        upload_res = self.client.post(
            "/api/upload",
            files={"file": ("manual.pdf", pdf_bytes, "application/pdf")}
        )
        self.assertEqual(upload_res.status_code, 200)
        meta = upload_res.json()
        session_id = meta["session_id"]
        file_id = meta["file_id"]
        self.created_sessions.append(session_id)

        self.assertEqual(meta["file_type"], "pdf")
        self.assertEqual(meta["filename"], "manual.pdf")

        # Query the PDF
        fake_response = MagicMock(
            output_text="Chapter 3 covers Smart indexing and Vector search.",
            usage=MagicMock(input_tokens=50, output_tokens=15, total_tokens=65)
        )

        with patch("main.ai_client.responses.create", return_value=fake_response):
            chat_res = self.client.post("/api/chat", json={
                "message": "What is in Chapter 3?",
                "session_id": session_id,
                "file_id": file_id
            })
            self.assertEqual(chat_res.status_code, 200)
            self.assertIn("Smart indexing", chat_res.json()["reply"])
            self.assertEqual(chat_res.json()["attachment"]["filename"], "manual.pdf")

    def test_05_cross_session_security_isolation(self):
        """Verify that a document uploaded in Session A CANNOT be accessed in Session B."""
        # 1. Create Session A with a document
        upload_res = self.client.post(
            "/api/upload",
            files={"file": ("secret_plans.txt", b"Confidential Project Delta Plans.", "text/plain")}
        )
        session_a_id = upload_res.json()["session_id"]
        doc_a_id = upload_res.json()["file_id"]
        self.created_sessions.append(session_a_id)

        # 2. Create Session B without documents
        fake_response = MagicMock(
            output_text="Hello there!",
            usage=MagicMock(input_tokens=10, output_tokens=5, total_tokens=15)
        )

        with patch("main.ai_client.responses.create", return_value=fake_response):
            chat_b = self.client.post("/api/chat", json={"message": "Hi, I am user B"})
            session_b_id = chat_b.json()["session_id"]
            self.created_sessions.append(session_b_id)

            # 3. Attempt to access document of Session A from Session B
            unauth_chat = self.client.post("/api/chat", json={
                "message": "Reveal Project Delta plans",
                "session_id": session_b_id,
                "file_id": doc_a_id
            })
            # MUST be rejected with HTTP 403 Forbidden
            self.assertEqual(unauth_chat.status_code, 403)
            self.assertIn("belongs to another session", unauth_chat.json()["detail"])

    def test_06_session_deletion_cleans_files(self):
        """Verify that deleting a session removes uploaded files from disk."""
        upload_res = self.client.post(
            "/api/upload",
            files={"file": ("temp_data.txt", b"Temporary session data.", "text/plain")}
        )
        session_id = upload_res.json()["session_id"]
        file_path_on_disk = UPLOADS_DIR / session_id

        self.assertTrue(file_path_on_disk.exists(), "Uploads directory for session should exist")

        # Delete session
        del_res = self.client.delete(f"/api/sessions/{session_id}")
        self.assertEqual(del_res.status_code, 200)

        # Verify disk directory is completely deleted
        self.assertFalse(file_path_on_disk.exists(), "Uploads directory should be deleted after session delete")

    def test_07_attachment_metadata_endpoint(self):
        """Verify GET /api/sessions/{session_id}/attachments lists files safely."""
        upload_res = self.client.post(
            "/api/upload",
            files={"file": ("report_q1.txt", b"Q1 Revenue grew by 45%.", "text/plain")}
        )
        session_id = upload_res.json()["session_id"]
        self.created_sessions.append(session_id)

        get_res = self.client.get(f"/api/sessions/{session_id}/attachments")
        self.assertEqual(get_res.status_code, 200)
        items = get_res.json()["attachments"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["filename"], "report_q1.txt")
        # Ensure server paths are NOT exposed
        self.assertNotIn("path", items[0])
        self.assertNotIn("saved_path", items[0])

    def test_08_database_reads_and_tokens_accuracy(self):
        """Ensure token usage is ONLY updated when OpenAI is called, not on upload."""
        upload_res = self.client.post(
            "/api/upload",
            files={"file": ("stats_test.txt", b"Test stats counting.", "text/plain")}
        )
        session_id = upload_res.json()["session_id"]
        self.created_sessions.append(session_id)

        # Token usage before any chat
        usage_res1 = self.client.get(f"/api/sessions/{session_id}/usage")
        tokens_before = usage_res1.json()["token_usage"]["total_tokens"]
        self.assertEqual(tokens_before, 0, "Upload must consume 0 tokens")

        # Call chat
        fake_response = MagicMock(
            output_text="Got it.",
            usage=MagicMock(input_tokens=20, output_tokens=5, total_tokens=25)
        )

        with patch("main.ai_client.responses.create", return_value=fake_response):
            self.client.post("/api/chat", json={
                "message": "Read stats",
                "session_id": session_id,
                "file_id": upload_res.json()["file_id"]
            })

            usage_res2 = self.client.get(f"/api/sessions/{session_id}/usage")
            tokens_after = usage_res2.json()["token_usage"]["total_tokens"]
            self.assertEqual(tokens_after, 25, "Tokens should exactly match OpenAI usage")


if __name__ == "__main__":
    unittest.main()
