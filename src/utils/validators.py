"""
Input validation utilities.
"""

from fastapi import UploadFile, HTTPException


def validate_image(file: UploadFile, max_size_mb: int = 10):
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(400, "Only JPEG and PNG allowed")

    size = 0
    for chunk in file.file:
        size += len(chunk)
        if size > max_size_mb * 1024 * 1024:
            raise HTTPException(413, "File too large")

    file.file.seek(0)
