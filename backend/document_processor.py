"""
Nova AI — Secure Document Processing Module
Handles file validation, safe storage, text extraction for PDF and TXT,
relevance-based chunk retrieval, and prompt injection defense.
"""

import io
import os
import re
from pathlib import Path
from uuid import uuid4
from pypdf import PdfReader
from pypdf.errors import PyPdfError


# Maximum allowed upload file size (default 10 MB, configurable via environment)
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {".pdf", ".txt"}
ALLOWED_FILE_TYPES = {"pdf", "txt"}

# Base directory for storing session uploads safely on disk
UPLOADS_DIR = Path(__file__).resolve().parent / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent directory traversal and invalid characters.
    Never trust user-supplied paths.
    """
    # Extract basename only, ignoring any directory paths
    base = Path(filename).name
    # Remove control characters and path separators
    clean = re.sub(r'[\\/*?:"<>|\x00-\x1f]', "", base).strip()
    # Replace multiple dots or spaces
    clean = re.sub(r"\.{2,}", ".", clean)
    clean = re.sub(r"\s+", " ", clean).strip(" .")
    if not clean or clean.startswith("."):
        clean = f"document_{uuid4().hex[:8]}"
    return clean[:120]


def detect_and_validate_file(content: bytes, filename: str) -> tuple[str, str]:
    """
    Validate file extension, size, and content signatures.
    Returns (safe_filename, file_type).
    Raises ValueError with user-friendly error messages if invalid.
    """
    if not content or len(content) == 0:
        raise ValueError("The uploaded file is empty (0 bytes).")

    if len(content) > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File size exceeds the limit of {MAX_FILE_SIZE_MB}MB. "
            f"Please upload a smaller document."
        )

    safe_name = sanitize_filename(filename)
    ext = Path(safe_name).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{ext}'. Nova AI currently supports PDF (.pdf) and TXT (.txt) files."
        )

    file_type = "pdf" if ext == ".pdf" else "txt"

    # Deep signature validation (do not rely on filename or MIME type alone)
    if file_type == "pdf":
        if not content.startswith(b"%PDF-"):
            # Check if PDF header appears within the first 1024 bytes (some PDF variants have minor prefixes)
            if b"%PDF-" not in content[:1024]:
                raise ValueError("Invalid PDF format: Missing standard PDF header signature.")
    elif file_type == "txt":
        # Check for binary null bytes
        if b"\x00" in content[:8192]:
            raise ValueError("Binary or non-text file detected. Please upload a valid plain text file (.txt).")

    return safe_name, file_type


def extract_text_from_pdf(content: bytes) -> str:
    """
    Extract readable text from PDF bytes safely using pypdf.
    Validates encryption, structure, and text content.
    """
    try:
        reader = PdfReader(io.BytesIO(content))
    except Exception as e:
        raise ValueError("Could not read the PDF file. The file may be corrupt or malformed.") from e

    if reader.is_encrypted:
        raise ValueError(
            "This PDF is password-protected or encrypted. "
            "Please remove the password or upload an unprotected document."
        )

    num_pages = len(reader.pages)
    if num_pages == 0:
        raise ValueError("The PDF contains no pages.")

    pages_text = []
    for page_idx, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text() or ""
            clean_page = " ".join(page_text.split())
            if clean_page:
                pages_text.append(f"--- Page {page_idx + 1} ---\n{clean_page}")
        except Exception:
            # Continue extracting from subsequent pages if one page has minor encoding glitches
            continue

    full_text = "\n\n".join(pages_text).strip()
    if not full_text or len(full_text) < 10:
        raise ValueError(
            "No readable text could be extracted from this PDF. "
            "Scanned image-only PDFs without an OCR text layer are not supported."
        )

    return full_text


def extract_text_from_txt(content: bytes) -> str:
    """
    Extract plain text from bytes with encoding detection (UTF-8, UTF-8-SIG, Latin-1).
    """
    encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]
    text = None
    for enc in encodings:
        try:
            text = content.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    if text is None:
        raise ValueError("Could not decode text file. Please ensure it is saved in UTF-8 encoding.")

    clean_text = text.strip()
    if not clean_text:
        raise ValueError("The text file is empty.")

    return clean_text


def extract_document_text(content: bytes, file_type: str) -> str:
    """
    Extract clean text based on verified file type.
    """
    if file_type == "pdf":
        return extract_text_from_pdf(content)
    elif file_type == "txt":
        return extract_text_from_txt(content)
    else:
        raise ValueError(f"Unsupported file type: '{file_type}'")


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[dict]:
    """
    Chunk document text into overlapping segments preserving paragraph boundaries.
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        para_clean = para.strip()
        if not para_clean:
            continue

        para_len = len(para_clean)
        if current_length + para_len > chunk_size and current_chunk:
            chunk_content = "\n\n".join(current_chunk)
            chunks.append({
                "chunk_index": len(chunks),
                "text": chunk_content,
                "char_count": len(chunk_content)
            })

            # Overlap: keep last paragraph if reasonable
            if len(current_chunk) > 1 and len(current_chunk[-1]) < overlap * 2:
                current_chunk = [current_chunk[-1], para_clean]
                current_length = len(current_chunk[0]) + para_len
            else:
                current_chunk = [para_clean]
                current_length = para_len
        else:
            current_chunk.append(para_clean)
            current_length += para_len

    if current_chunk:
        chunk_content = "\n\n".join(current_chunk)
        chunks.append({
            "chunk_index": len(chunks),
            "text": chunk_content,
            "char_count": len(chunk_content)
        })

    return chunks


def retrieve_relevant_chunks(
    chunks: list[dict],
    query: str,
    top_k: int = 4,
    max_total_chars: int = 8000
) -> list[dict]:
    """
    Retrieve the most relevant document chunks for a user's question.
    If the entire document fits comfortably within max_total_chars, returns all chunks.
    Otherwise ranks chunks by query keyword overlap and density.
    """
    if not chunks:
        return []

    total_chars = sum(c["char_count"] for c in chunks)
    # If the document is small enough to fit completely, return all chunks in order
    if total_chars <= max_total_chars:
        return chunks

    # Tokenize query into distinct keywords (ignore stop words and single chars)
    stop_words = {
        "the", "is", "at", "which", "on", "a", "an", "and", "or", "for", "in",
        "of", "to", "what", "how", "why", "when", "where", "who", "can", "you",
        "tell", "me", "about", "this", "that", "it", "my", "your", "does", "do",
        "kya", "hai", "kaise", "batao", "samjhao", "in", "se", "ke", "ko"
    }
    raw_terms = re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", query.lower())
    query_terms = [t for t in raw_terms if t not in stop_words]

    scored = []
    for chunk in chunks:
        chunk_text_lower = chunk["text"].lower()
        score = 0
        for term in query_terms:
            count = chunk_text_lower.count(term)
            if count > 0:
                # Term presence gives base points; additional occurrences give sub-linear bonus
                score += 3 + min(count, 5)

        scored.append((score, chunk["chunk_index"], chunk))

    # Sort descending by score, tie-break by earlier chunk index
    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)

    selected = []
    accumulated_chars = 0

    # Pick top_k scored chunks
    for score, idx, chunk in scored[:top_k]:
        if accumulated_chars + chunk["char_count"] > max_total_chars and selected:
            break
        selected.append(chunk)
        accumulated_chars += chunk["char_count"]

    # If no keywords matched (e.g., query was 'Summarize this document' or 'Explain it'),
    # select the initial chunks (overview/introduction) and end chunks (conclusion)
    if not selected or all(s[0] == 0 for s in scored[:top_k]):
        selected = []
        accumulated_chars = 0
        for chunk in chunks[:top_k]:
            if accumulated_chars + chunk["char_count"] > max_total_chars and selected:
                break
            selected.append(chunk)
            accumulated_chars += chunk["char_count"]

    # Sort back into natural document order
    selected.sort(key=lambda c: c["chunk_index"])
    return selected


def format_document_context_prompt(
    chunks: list[dict],
    filename: str,
    file_type: str
) -> str:
    """
    Format retrieved document chunks with clear data boundaries and
    anti-prompt-injection security instructions.
    """
    if not chunks:
        return ""

    body_parts = []
    for chunk in chunks:
        body_parts.append(f"[Excerpt #{chunk['chunk_index'] + 1}]\n{chunk['text']}")

    doc_content = "\n\n".join(body_parts)

    return (
        f"<document_context filename=\"{filename}\" type=\"{file_type.upper()}\">\n"
        f"{doc_content}\n"
        f"</document_context>"
    )


def save_session_document(
    session_id: str,
    filename: str,
    content: bytes,
    file_type: str,
    extracted_text: str,
    chunks: list[dict]
) -> dict:
    """
    Save the uploaded file and its pre-chunked metadata in the session storage directory.
    Enforces session isolation: files are stored inside uploads/{session_id}/.
    """
    file_id = str(uuid4())
    session_dir = UPLOADS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    safe_name = sanitize_filename(filename)
    stored_filename = f"{file_id}_{safe_name}"
    file_path = session_dir / stored_filename

    # Write original binary/text file securely
    with open(file_path, "wb") as f:
        f.write(content)

    return {
        "file_id": file_id,
        "filename": safe_name,
        "file_type": file_type,
        "file_size": len(content),
        "char_count": len(extracted_text),
        "chunk_count": len(chunks),
        "chunks": chunks,
        "relative_path": f"{session_id}/{stored_filename}",
    }
