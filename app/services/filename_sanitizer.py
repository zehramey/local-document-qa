"""Sanitizes uploaded filenames.

The ingestion pipeline never writes the uploaded file to a filesystem
path built from user input — files are processed entirely in memory — so
there is no literal path-traversal vector today. This sanitizer is
defense in depth: it strips any directory component (so a value like
"../../etc/passwd" collapses to "passwd"), drops characters that are
invalid or dangerous in filenames across common filesystems, and caps
the length, since the filename is stored in Qdrant payload and echoed
back to users in API/UI responses.
"""

import re
from pathlib import PureWindowsPath

_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_MAX_LENGTH = 255


def sanitize_filename(raw_filename: str) -> str:
    # PureWindowsPath also strips POSIX-style ".." segments and drive
    # letters, so this is safe input regardless of client OS.
    name = PureWindowsPath(raw_filename).name or "dosya"
    name = _UNSAFE_CHARS.sub("_", name).strip(" .")
    if not name:
        name = "dosya"
    return name[:_MAX_LENGTH]
