"""
Category listing endpoint.
"""

import json
from fastapi import APIRouter

router = APIRouter()


@router.get("/categories")
async def list_categories():
    with open("config/categories.json", "r") as f:
        return json.load(f)
