"""Programmatically builds small PDF byte fixtures with PyMuPDF.

Generated in-memory at test time so no binary fixture files need to be
checked into the repo (small, license-safe, deterministic).
"""

import fitz


def build_pdf_bytes(pages_text: list[str]) -> bytes:
    document = fitz.open()
    for text in pages_text:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
    data = document.tobytes()
    document.close()
    return data


def build_blank_pdf_bytes(page_count: int = 1) -> bytes:
    document = fitz.open()
    for _ in range(page_count):
        document.new_page()
    data = document.tobytes()
    document.close()
    return data


def build_encrypted_pdf_bytes(password: str = "secret") -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "confidential")
    data = document.tobytes(
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw=password,
        user_pw=password,
    )
    document.close()
    return data
