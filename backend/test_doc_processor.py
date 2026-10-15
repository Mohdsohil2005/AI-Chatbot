"""
Test suite for backend/document_processor.py.
Validates extraction, chunking, retrieval, rejection of malformed files, and prompt formatting.
"""

import io
import sys
from pathlib import Path
from pypdf import PdfWriter

backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from document_processor import (
    detect_and_validate_file,
    extract_document_text,
    chunk_text,
    retrieve_relevant_chunks,
    format_document_context_prompt,
    save_session_document,
    MAX_FILE_SIZE_BYTES,
)


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


def test_pdf_processing():
    content = create_sample_pdf("Nova AI PDF Document Test Content")
    safe_name, file_type = detect_and_validate_file(content, "sample_report.pdf")
    assert file_type == "pdf"
    assert safe_name == "sample_report.pdf"

    extracted = extract_document_text(content, file_type)
    assert "Nova AI PDF Document Test Content" in extracted
    chunks = chunk_text(extracted)
    assert len(chunks) >= 1
    relevant = retrieve_relevant_chunks(chunks, "PDF document")
    assert len(relevant) >= 1
    assert "PDF Document Test Content" in relevant[0]["text"]
    print("[PASS] PDF processing and extraction passed.")


def test_txt_processing():
    sample_text = "Nova AI Document Answering Test.\nSection 1: Security and Architecture.\nSection 2: MongoDB Persistence."
    content = sample_text.encode("utf-8")
    safe_name, file_type = detect_and_validate_file(content, "architecture.txt")
    assert file_type == "txt"
    assert safe_name == "architecture.txt"

    extracted = extract_document_text(content, file_type)
    assert "Nova AI Document Answering" in extracted

    chunks = chunk_text(extracted, chunk_size=50)
    assert len(chunks) >= 1

    relevant = retrieve_relevant_chunks(chunks, "architecture")
    assert len(relevant) >= 1
    assert "Security and Architecture" in relevant[0]["text"]
    print("[PASS] TXT processing and retrieval passed.")


def test_invalid_and_empty_files():
    # 1. Empty file
    try:
        detect_and_validate_file(b"", "empty.txt")
        assert False, "Should reject empty file"
    except ValueError as e:
        assert "empty" in str(e).lower()

    # 2. Unsupported extension
    try:
        detect_and_validate_file(b"content", "script.exe")
        assert False, "Should reject unsupported format"
    except ValueError as e:
        assert "unsupported" in str(e).lower()

    # 3. Binary null bytes in txt
    try:
        detect_and_validate_file(b"text\x00binary", "fake.txt")
        assert False, "Should reject binary txt"
    except ValueError as e:
        assert "binary" in str(e).lower()

    # 4. Fake PDF without signature
    try:
        detect_and_validate_file(b"not a real pdf", "fake.pdf")
        assert False, "Should reject fake PDF"
    except ValueError as e:
        assert "missing standard pdf" in str(e).lower()

    # 5. Oversized
    try:
        detect_and_validate_file(b"x" * (MAX_FILE_SIZE_BYTES + 10), "big.txt")
        assert False, "Should reject oversized file"
    except ValueError as e:
        assert "exceeds" in str(e).lower()

    print("[PASS] Invalid and empty file rejections passed.")


def test_safe_storage_and_context_prompt():
    content = b"Policy details: The warranty lasts for 2 years from purchase."
    safe_name, file_type = detect_and_validate_file(content, "../../unsafe/path/warranty.txt")
    assert ".." not in safe_name
    assert safe_name.endswith(".txt")

    extracted = extract_document_text(content, file_type)
    chunks = chunk_text(extracted)
    saved = save_session_document("session-test-123", safe_name, content, file_type, extracted, chunks)

    assert saved["file_id"]
    assert saved["file_type"] == "txt"
    assert saved["char_count"] == len(extracted)

    prompt = format_document_context_prompt(chunks, safe_name, file_type)
    assert "<document_context" in prompt
    assert "</document_context>" in prompt
    assert "warranty lasts for 2 years" in prompt
    print("[PASS] Safe storage and context prompt passed.")


if __name__ == "__main__":
    test_txt_processing()
    test_pdf_processing()
    test_invalid_and_empty_files()
    test_safe_storage_and_context_prompt()
    print("\nALL DOCUMENT PROCESSOR TESTS PASSED!")
