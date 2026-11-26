"""Annotator identity via per-annotator API tokens.

Each annotator gets a random token when registered (or via `python cli.py issue-token`).
Only its SHA-256 hash is stored. Requests authenticate with `Authorization: Bearer <token>`,
and the annotator is resolved from the token, never from the request body.
"""
from __future__ import annotations

import hashlib
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from models import Annotator

_bearer = HTTPBearer(auto_error=False, description="Annotator API token")


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def issue_token(annotator: Annotator) -> str:
    """Give `annotator` a new token, replacing any previous one. Returns the plain token."""
    token = secrets.token_urlsafe(32)
    annotator.api_token_hash = hash_token(token)
    return token


async def current_annotator(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: AsyncSession = Depends(get_db),
) -> Annotator:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing annotator token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    result = await db.execute(
        select(Annotator).where(Annotator.api_token_hash == hash_token(credentials.credentials))
    )
    annotator = result.scalar_one_or_none()
    if annotator is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid annotator token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return annotator
