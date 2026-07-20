import asyncio
import logging
import time

from app.db.database import open_db

logger = logging.getLogger(__name__)
RETENTION_MINUTES = 7 * 24 * 60
PRUNE_INTERVAL_SECONDS = 3600


async def prune_once(settings) -> dict[str, int]:
    cutoff_minute = int(time.time() // 60) - RETENTION_MINUTES
    async with open_db(settings.db_path) as db:
        cur1 = await db.execute("DELETE FROM model_samples WHERE minute < ?", (cutoff_minute,))
        cur2 = await db.execute("DELETE FROM gpu_samples WHERE minute < ?", (cutoff_minute,))
        # S7 (#124) — power_samples share retention with gpu_samples.
        cur3 = await db.execute("DELETE FROM power_samples WHERE minute < ?", (cutoff_minute,))
        await db.commit()
        return {
            "model_samples": cur1.rowcount or 0,
            "gpu_samples": cur2.rowcount or 0,
            "power_samples": cur3.rowcount or 0,
        }


async def run_pruner_forever(settings) -> None:
    while True:
        try:
            await prune_once(settings)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("stats pruner iteration failed; continuing")
        await asyncio.sleep(PRUNE_INTERVAL_SECONDS)
