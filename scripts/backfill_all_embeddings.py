#!/usr/bin/env python3
"""Backfill summaries and embeddings for all historical sessions + cold memory.

Three-phase migration:
1. Generate summaries for sessions that have JSONL content but no summary
2. Embed all session summaries into the vector store
3. Embed all cold memory entries into the vector store

Usage:
    cd ~/.hermes/hermes-agent
    source venv/bin/activate
    python scripts/backfill_all_embeddings.py [--dry-run] [--skip-summaries] [--skip-cold]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env so API keys are available
from dotenv import load_dotenv
_env_path = Path.home() / ".hermes" / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_jsonl_digest(jsonl_path: Path, max_chars: int = 15000) -> str | None:
    """Read a session JSONL and build a compact digest for summarization."""
    try:
        parts = []
        total = 0
        with open(jsonl_path) as f:
            for line in f:
                if total > max_chars:
                    break
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system" or not content or not isinstance(content, str):
                    continue
                text = content.strip()
                if len(text) > 1500:
                    text = text[:1500] + "..."
                if role == "tool":
                    tool_name = msg.get("tool_name", msg.get("name", "tool"))
                    entry = f"[{tool_name} result]: {text[:500]}"
                elif role in ("user", "assistant"):
                    entry = f"[{role}]: {text}"
                else:
                    continue
                parts.append(entry)
                total += len(entry)
        digest = "\n".join(parts)
        if len(digest) > max_chars:
            digest = digest[:max_chars] + "\n...(truncated)"
        return digest if len(digest) > 100 else None
    except Exception as e:
        logger.debug("Failed to read %s: %s", jsonl_path, e)
        return None


def phase1_generate_summaries(db, sessions_dir: Path, dry_run: bool = False):
    """Generate summaries for sessions that have JSONL but no summary."""
    logger.info("=== Phase 1: Generate missing session summaries ===")

    # Find sessions without summaries
    rows = db._conn.execute(
        "SELECT id FROM sessions WHERE (summary IS NULL OR summary = '') AND message_count > 3"
    ).fetchall()
    session_ids = {r[0] if isinstance(r, tuple) else r["id"] for r in rows}
    logger.info("Found %d sessions without summaries (with >3 messages)", len(session_ids))

    # Match with JSONL files
    jsonl_files = list(sessions_dir.glob("*.jsonl"))
    to_summarize = []
    for jf in jsonl_files:
        # Extract session ID from filename (remove .jsonl)
        sid = jf.stem
        if sid in session_ids:
            to_summarize.append((sid, jf))

    # Also check sessions from JSON metadata files that reference JSONL
    for sid in session_ids:
        jsonl_path = sessions_dir / f"{sid}.jsonl"
        if jsonl_path.exists() and (sid, jsonl_path) not in to_summarize:
            to_summarize.append((sid, jsonl_path))

    logger.info("Found %d sessions with JSONL files to summarize", len(to_summarize))

    if not to_summarize:
        return 0

    if dry_run:
        for sid, jf in to_summarize[:5]:
            logger.info("  Would summarize: %s (%d bytes)", sid, jf.stat().st_size)
        if len(to_summarize) > 5:
            logger.info("  ... and %d more", len(to_summarize) - 5)
        return len(to_summarize)

    from agent.auxiliary_client import call_llm

    generated = 0
    for sid, jf in to_summarize:
        digest = _load_jsonl_digest(jf)
        if not digest:
            logger.debug("Skipping %s — insufficient content", sid)
            continue

        try:
            summary_prompt = (
                "You are a session summarizer. Produce a concise structured summary "
                "of this conversation in the SAME LANGUAGE the user spoke. Keep it under "
                "500 characters. Format:\n"
                "Topic: (1-line topic)\n"
                "Key Actions: (bullet list of what was done)\n"
                "Decisions: (important decisions made)\n"
                "Open Items: (unfinished tasks, if any)\n\n"
                "Conversation:\n"
            ) + digest

            response = call_llm(
                task="compression",
                provider="copilot",
                model="claude-sonnet-4",
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=400,
                temperature=0.0,
            )
            summary_text = response.choices[0].message.content
            if summary_text and summary_text.strip():
                summary_clean = summary_text.strip()[:1000]
                db.save_summary(sid, summary_clean)
                generated += 1
                logger.info("  Generated summary for %s (%d chars)", sid, len(summary_clean))
            # Rate limit
            time.sleep(0.3)
        except Exception as e:
            logger.warning("  Failed to summarize %s: %s", sid, e)
            time.sleep(1)

    logger.info("Phase 1 complete: generated %d summaries", generated)
    return generated


def phase2_embed_sessions(db, dry_run: bool = False):
    """Embed all session summaries that don't have embeddings yet."""
    logger.info("=== Phase 2: Embed session summaries ===")

    rows = db._conn.execute(
        "SELECT id, summary FROM sessions WHERE summary IS NOT NULL AND summary != ''"
    ).fetchall()
    logger.info("Found %d sessions with summaries", len(rows))

    existing = set()
    try:
        for row in db._conn.execute(
            "SELECT source_id FROM embeddings WHERE source_type = 'session'"
        ).fetchall():
            existing.add(row[0] if isinstance(row, tuple) else row["source_id"])
    except Exception:
        pass

    to_embed = [
        (r[0] if isinstance(r, tuple) else r["id"], r[1] if isinstance(r, tuple) else r["summary"])
        for r in rows
        if (r[0] if isinstance(r, tuple) else r["id"]) not in existing
    ]
    logger.info("Need to embed %d sessions (%d already done)", len(to_embed), len(existing))

    if not to_embed:
        return 0

    if dry_run:
        for sid, summary in to_embed[:3]:
            logger.info("  Would embed: %s — %s", sid, summary[:80])
        return len(to_embed)

    from agent.embedding_client import get_embeddings, EMBEDDING_MODEL

    stored = 0
    batch_size = 20
    for i in range(0, len(to_embed), batch_size):
        batch = to_embed[i:i + batch_size]
        texts = [f"search_document: {s}" for _, s in batch]
        ids = [sid for sid, _ in batch]

        logger.info("  Batch %d: embedding %d summaries...",
                     i // batch_size + 1, len(batch))

        vecs = get_embeddings(texts)
        if vecs is None:
            logger.error("  Embedding API failed for batch %d", i // batch_size + 1)
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
                stored += 1

        time.sleep(0.5)

    logger.info("Phase 2 complete: embedded %d sessions", stored)
    return stored


def phase3_embed_cold_memory(db, dry_run: bool = False):
    """Embed cold memory entries that don't have embeddings yet."""
    logger.info("=== Phase 3: Embed cold memory entries ===")

    try:
        rows = db._conn.execute(
            "SELECT id, content FROM memory_entries"
        ).fetchall()
    except Exception:
        logger.info("No memory_entries table found — skipping")
        return 0

    logger.info("Found %d cold memory entries", len(rows))

    existing = set()
    try:
        for row in db._conn.execute(
            "SELECT source_id FROM embeddings WHERE source_type = 'memory'"
        ).fetchall():
            existing.add(row[0] if isinstance(row, tuple) else row["source_id"])
    except Exception:
        pass

    to_embed = [
        (str(r[0] if isinstance(r, tuple) else r["id"]), r[1] if isinstance(r, tuple) else r["content"])
        for r in rows
        if str(r[0] if isinstance(r, tuple) else r["id"]) not in existing
    ]
    logger.info("Need to embed %d entries (%d already done)", len(to_embed), len(existing))

    if not to_embed:
        return 0

    if dry_run:
        for mid, content in to_embed[:3]:
            logger.info("  Would embed: %s — %s", mid, content[:80])
        return len(to_embed)

    from agent.embedding_client import get_embeddings, EMBEDDING_MODEL

    texts = [f"search_document: {c}" for _, c in to_embed]
    ids = [mid for mid, _ in to_embed]

    vecs = get_embeddings(texts)
    if vecs is None:
        logger.error("Embedding API failed for cold memory")
        return 0

    stored = 0
    for mid, content, vec in zip(ids, texts, vecs):
        ok = db.store_embedding(
            source_type="memory",
            source_id=mid,
            content=content,
            embedding=vec,
            model=EMBEDDING_MODEL,
        )
        if ok:
            stored += 1

    logger.info("Phase 3 complete: embedded %d cold memory entries", stored)
    return stored


def main():
    parser = argparse.ArgumentParser(description="Backfill all embeddings")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-summaries", action="store_true", help="Skip Phase 1 (summary generation)")
    parser.add_argument("--skip-cold", action="store_true", help="Skip Phase 3 (cold memory)")
    args = parser.parse_args()

    from hermes_state import SessionDB
    from hermes_constants import get_hermes_home

    db = SessionDB()
    if not db._vec_available:
        logger.error("sqlite-vec not available!")
        sys.exit(1)

    sessions_dir = get_hermes_home() / "sessions"

    total = 0
    if not args.skip_summaries:
        total += phase1_generate_summaries(db, sessions_dir, args.dry_run)
    total += phase2_embed_sessions(db, args.dry_run)
    if not args.skip_cold:
        total += phase3_embed_cold_memory(db, args.dry_run)

    # Final stats
    try:
        vec_count = db._conn.execute("SELECT COUNT(*) FROM embeddings_vec").fetchone()[0]
        emb_count = db._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        logger.info("=== Final: %d embeddings, %d vectors in store ===", emb_count, vec_count)
    except Exception:
        pass

    db.close()
    logger.info("All done!")


if __name__ == "__main__":
    main()
