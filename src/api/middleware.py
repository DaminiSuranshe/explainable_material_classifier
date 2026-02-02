"""
Custom rate limiting middleware.
"""

import time
from fastapi import Request, HTTPException
from collections import defaultdict

REQUEST_LIMIT = 10  # per minute
WINDOW = 60

clients = defaultdict(list)


async def rate_limiter(request: Request, call_next):
    ip = request.client.host
    now = time.time()

    clients[ip] = [t for t in clients[ip] if now - t < WINDOW]

    if len(clients[ip]) >= REQUEST_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded (10 requests/minute)",
        )

    clients[ip].append(now)
    return await call_next(request)
