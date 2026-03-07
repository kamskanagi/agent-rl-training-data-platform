from __future__ import annotations

import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    logger.info("Quality worker started. Waiting for jobs...")
    # Placeholder: actual quality scoring logic will be added in Phase 4
    while True:
        await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
