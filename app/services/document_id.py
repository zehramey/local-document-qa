import hashlib


def compute_document_id(content: bytes) -> str:
    """SHA-256 hex digest of the raw file content.

    Deterministic: identical content always yields the same id, which is what
    re-upload detection relies on (callers compare ids before re-ingesting).
    """
    return hashlib.sha256(content).hexdigest()
