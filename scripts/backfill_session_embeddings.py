#!/usr/bin/env python3
"""Backfill embeddings for existing session summaries.

One-time migration script: reads all sessions that have a summary but
no corresponding embedding in the vec table, batches them through the
embedding API, and stores the results.

Usage:
    cd ~/.hermes/hermes-agent
    source venv/bin/activate
    python scripts/backfill_session_embeddings.py [--dry-run] [--batch-size 50]
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add parent dir to path so we can import hermes modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Backfill session summary embeddings")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without writing")
    parser.add_argument("--batch-size", type=int, default=50, help="Embedding API batch size (max 100)")
    args = parser.parse_args()

    from hermes_state import SessionDB, DEFAULT_DB_PATH

    if not DEFAULT_DB_PATH.exists():
        logger.error("Database not found at %s", DEFAULT_DB_PATH)
        sys.exit(1)

    db = SessionDB(DEFAULT_DB_PATH)
    if not db._vec_available:
        logger.error("sqlite-vec not available — cannot store embeddings")
        sys.exit(1)

    # Find sessions with summaries
    rows = db._conn.execute(
        "SELECT id, summary FROM sessions WHERE summary IS NOT NULL AND summary != ''"
    ).fetchall()
    logger.info("Found %d sessions with summaries", len(rows))

    # Check which already have embeddings
    existing = set()
    try:
        for row in db._conn.execute(
            "SELECT source_id FROM embeddings WHERE source_type = 'session'"
        ).fetchall():
            existing.add(row[0] if isinstance(row, tuple) else row["source_id"])
    except Exception:
        pass

    to_embed = [(r[0], r[1]) for r in rows if (r[0] if isinstance(r, tuple) else r["id"]) not in existing]
    logger.info("Need to embed %d sessions (%d already have embeddings)", len(to_embed), len(existing))

    if not to_embed:
        logger.info("Nothing to do!")
        db.close()
        return

    if args.dry_run:
        for sid, summary in to_embed[:5]:
            logger.info("  Would embed: %s — %s", sid, summary[:80])
        if len(to_embed) > 5:
            logger.info("  ... and %d more", len(to_embed) - 5)
        db.close()
        return

    from agent.embedding_client import get_embeddings, EMBEDDING_MODEL

    batch_size = min(args.batch_size, 100)
    total_stored = 0
    total_failed = 0

    for i in range(0, len(to_embed), batch_size):
        batch = to_embed[i:i + batch_size]
        texts = [summary for _, summary in batch]
        ids = [sid for sid, _ in batch]

        logger.info("Batch %d/%d: embedding %d summaries...",
                     i // batch_size + 1,
                     (len(to_embed) + batch_size - 1) // batch_size,
                     len(batch))

        vecs = get_embeddings(texts)
        if vecs is None:
            logger.error("Embedding API call failed for batch starting at %d", i)
            total_failed += len(batch)
            time.sleep(2)
            continue

        for sid, summary, vec in zip(ids, texts, vecs):
            ok = db.store_embedding(
                source_type="session",
                source_id=sid,
                content=summary,
                embedding=vec,
                model=EMBEDDING_MODEL,
            )
            if ok:
                total_stored += 1
            else:
                total_failed += 1

        # Respect rate limits
        if i + batch_size < len(to_embed):
            time.sleep(0.5)

    logger.info("Done! Stored: %d, Failed/Skipped: %d, Total embeddings: %d",
                total_stored, total_failed, db.embedding_count("session"))
    db.close()


if __name__ == "__main__":
    main()
