"""Operator commands. Run inside the API container, e.g.

    docker exec rl-api python cli.py issue-token alice@example.com

issue-token  Give an existing annotator a new API token (replaces the old one).
             Use it for annotators created before tokens existed, or when a token is lost.
"""
from __future__ import annotations

import asyncio
import sys

from sqlalchemy import select

from core.auth import issue_token
from core.database import async_session, init_db
from models import Annotator


async def _issue_token(email: str) -> int:
    await init_db()
    async with async_session() as db:
        annotator = (
            await db.execute(select(Annotator).where(Annotator.email == email))
        ).scalar_one_or_none()
        if annotator is None:
            print(f"No annotator with email {email}", file=sys.stderr)
            return 1
        token = issue_token(annotator)
        await db.commit()
    print(f"Token for {annotator.name} <{email}> (shown once, store it safely):\n{token}")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "issue-token":
        return asyncio.run(_issue_token(argv[2]))
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
