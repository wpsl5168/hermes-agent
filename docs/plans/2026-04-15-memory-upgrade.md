# Hermes Memory System Upgrade — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Upgrade Hermes memory system with vector search (sqlite-vec), fix session reset bug, expand memory capacity, and prepare Auto Dream foundation.

**Architecture:** Add embedding layer using sqlite-vec in existing state.db. Embedding via OpenRouter (text-embedding-3-small). Session search uses vector similarity + pre-generated summaries instead of real-time LLM calls. Gateway restart no longer kills sessions.

**Tech Stack:** sqlite-vec, OpenRouter embedding API, numpy, existing SQLite state.db

---

## Phase Overview

| Phase | Description | Files | Risk |
|-------|-------------|-------|------|
| P0 | Fix gateway restart session loss | gateway/run.py | Low — isolated bugfix |
| P1a | Embedding infrastructure (sqlite-vec + API) | new: tools/embedding_tool.py, modify: hermes_state.py | Medium — new subsystem |
| P1b | Session search vector upgrade | modify: session_search_tool.py, run_agent.py | Medium — core flow change |
| P1c | Memory embedding (hot + cold) | modify: memory_tool.py | Low — additive |
| P2-prep | Auto Dream trigger foundation | modify: hermes_state.py, run_agent.py | Low — counter + config only |

---

## P0: Fix Gateway Restart Session Loss

### Problem
Gateway restart without `.clean_shutdown` marker → `suspend_recently_active()` marks all sessions as suspended → next message triggers auto-reset → conversation history lost.

Root cause: `.clean_shutdown` marker only written after drain completes within timeout. If drain times out (common with active agents), marker is skipped → next startup suspends sessions.

### Task P0.1: Always write clean_shutdown on graceful signal

**Objective:** Ensure SIGTERM/SIGINT (graceful shutdown) always writes `.clean_shutdown`, regardless of drain timeout.

**Files:**
- Modify: `gateway/run.py:2340-2360` (the finally block in `_graceful_shutdown`)

**Change:**
The current logic at line 2350 only writes `.clean_shutdown` if drain succeeds. Change to: always write marker on signal-initiated shutdown. The marker means "this was a graceful stop, not a crash". Drain timeout doesn't make it a crash.

```python
# BEFORE (line 2341-2355):
#   # wasn't a crash.  suspend_recently_active() only needs to run
#   ...
#   if <drain succeeded>:
#       (_hermes_home / ".clean_shutdown").touch()
#   else:
#       "Skipping .clean_shutdown marker — drain timed out..."

# AFTER:
# Always write marker on graceful shutdown (signal-initiated).
# Drain timeout is not a crash — sessions were not stuck in a loop,
# they were just slow to finish. Auto-reset on restart loses user's
# conversation history unnecessarily.
try:
    (_hermes_home / ".clean_shutdown").touch()
except Exception:
    pass
```

**Verify:** Restart gateway, send a message — should continue existing session, not "Session automatically reset".

---

## P1a: Embedding Infrastructure

### Task P1a.1: Install sqlite-vec

**Objective:** Add sqlite-vec to project dependencies.

**Files:**
- Modify: `requirements.txt` or `pyproject.toml`

**Commands:**
```bash
cd ~/.hermes/hermes-agent
source venv/bin/activate
pip install sqlite-vec
```

**Verify:**
```python
import sqlite3, sqlite_vec
db = sqlite3.connect(":memory:")
db.enable_load_extension(True)
sqlite_vec.load(db)
db.execute("SELECT vec_version()").fetchone()
```

### Task P1a.2: Add embedding table to state.db schema

**Objective:** Create vec0 virtual table in state.db for vector storage.

**Files:**
- Modify: `hermes_state.py` — add to schema init, bump schema version

**Schema:**
```sql
-- Metadata table (regular SQLite table for content + metadata)
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL,     -- 'session', 'memory', 'cold_memory'
    source_id TEXT NOT NULL,       -- session_id or memory entry hash
    content TEXT NOT NULL,         -- the text that was embedded
    content_hash TEXT NOT NULL,    -- SHA256 for dedup
    model TEXT NOT NULL,           -- 'text-embedding-3-small'
    created_at REAL NOT NULL,
    UNIQUE(source_type, content_hash)
);

-- Vector table (sqlite-vec virtual table, linked by rowid)
CREATE VIRTUAL TABLE IF NOT EXISTS embeddings_vec USING vec0(
    embedding float[1536]
);
```

**Design notes:**
- `embeddings.id` (rowid) links to `embeddings_vec.rowid` — 1:1 relationship
- `content_hash` prevents re-embedding identical content
- `source_type` enables filtered search (e.g., only search sessions)
- 1536 dimensions = text-embedding-3-small output size

**Migration:** New schema version. `_migrate()` method creates tables if not exist. sqlite-vec loaded via `enable_load_extension` + `sqlite_vec.load()` at DB init.

### Task P1a.3: Create embedding client module

**Objective:** Thin wrapper for calling embedding API via OpenRouter.

**Files:**
- Create: `agent/embedding_client.py`

**Implementation:**
```python
"""Embedding client for vector memory.

Calls OpenRouter's embedding endpoint (text-embedding-3-small).
Falls back gracefully — if no API key or service unavailable,
returns None so callers can skip vectorization without failing.
"""

import hashlib
import logging
import os
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "openai/text-embedding-3-small"
EMBEDDING_DIM = 1536
OPENROUTER_EMBEDDING_URL = "https://openrouter.ai/api/v1/embeddings"


def get_embedding(text: str) -> Optional[np.ndarray]:
    """Get embedding for a single text. Returns float32 numpy array or None."""
    results = get_embeddings([text])
    return results[0] if results else None


def get_embeddings(texts: List[str]) -> Optional[List[np.ndarray]]:
    """Get embeddings for multiple texts in one API call.
    
    Returns list of float32 numpy arrays, or None on failure.
    Max batch size: 100 texts per call.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.debug("No OPENROUTER_API_KEY — skipping embedding")
        return None
    
    if not texts:
        return []
    
    try:
        import httpx
        
        # Truncate very long texts (embedding model has 8191 token limit)
        truncated = [t[:30000] if len(t) > 30000 else t for t in texts]
        
        response = httpx.post(
            OPENROUTER_EMBEDDING_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": EMBEDDING_MODEL,
                "input": truncated,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        
        embeddings = []
        for item in sorted(data["data"], key=lambda x: x["index"]):
            vec = np.array(item["embedding"], dtype=np.float32)
            embeddings.append(vec)
        
        return embeddings
        
    except Exception as e:
        logger.warning("Embedding API call failed: %s", e)
        return None


def content_hash(text: str) -> str:
    """SHA256 hash of text for dedup."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]
```

**Config:** Uses existing `OPENROUTER_API_KEY` from `.env`. No new config needed.
Could later add `auxiliary.embedding.provider` / `auxiliary.embedding.model` to config.yaml for flexibility.

### Task P1a.4: Create vector store interface in hermes_state.py

**Objective:** Add methods to SessionDB for storing and searching embeddings.

**Files:**
- Modify: `hermes_state.py` — add methods to SessionDB class

**Methods:**
```python
def store_embedding(self, source_type: str, source_id: str,
                    content: str, embedding: np.ndarray, model: str) -> bool:
    """Store an embedding. Deduplicates by content_hash."""
    ...

def search_embeddings(self, query_vec: np.ndarray, source_type: str = None,
                      limit: int = 5) -> List[Dict]:
    """Vector similarity search. Returns [{source_type, source_id, content, distance}, ...]"""
    ...

def delete_embeddings(self, source_type: str, source_id: str) -> int:
    """Delete embeddings for a given source. Returns count deleted."""
    ...
```

**Search query (sqlite-vec KNN):**
```python
def search_embeddings(self, query_vec, source_type=None, limit=5):
    from sqlite_vec import serialize_float32
    
    # KNN search on vec0 virtual table
    rows = self._conn.execute("""
        SELECT v.rowid, v.distance
        FROM embeddings_vec v
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY v.distance
    """, [serialize_float32(query_vec), limit * 3]).fetchall()  # over-fetch for filtering
    
    results = []
    for rowid, distance in rows:
        meta = self._conn.execute(
            "SELECT source_type, source_id, content FROM embeddings WHERE id = ?",
            [rowid]
        ).fetchone()
        if meta and (source_type is None or meta[0] == source_type):
            results.append({
                "source_type": meta[0],
                "source_id": meta[1],
                "content": meta[2],
                "distance": distance,
            })
            if len(results) >= limit:
                break
    return results
```

---

## P1b: Session Search Vector Upgrade

### Task P1b.1: Embed session summaries on auto_summary

**Objective:** When auto_summary generates a summary, also embed it.

**Files:**
- Modify: `run_agent.py` — inside `_spawn_auto_summary()`, after `save_summary()`

**Change (after line ~3095 where summary is saved):**
```python
# After saving summary text, also store embedding
try:
    from agent.embedding_client import get_embedding, content_hash
    vec = get_embedding(summary_text)
    if vec is not None:
        session_db.store_embedding(
            source_type="session",
            source_id=session_id,
            content=summary_text,
            embedding=vec,
            model="text-embedding-3-small",
        )
except Exception as e:
    logger.debug("Failed to embed session summary: %s", e)
```

### Task P1b.2: Backfill existing session summaries

**Objective:** One-time script to embed all existing session summaries that have text but no embedding.

**Files:**
- Create: `scripts/backfill_embeddings.py`

**Logic:**
```python
# 1. SELECT id, summary FROM sessions WHERE summary IS NOT NULL
# 2. Check which ones already have embeddings (SELECT source_id FROM embeddings WHERE source_type='session')
# 3. Batch embed missing ones (100 per API call)
# 4. Store results
```

**Run:** `cd ~/.hermes/hermes-agent && source venv/bin/activate && python scripts/backfill_embeddings.py`

### Task P1b.3: Add vector search path to session_search

**Objective:** session_search tries vector similarity first, falls back to FTS5. Uses pre-generated summaries instead of real-time LLM.

**Files:**
- Modify: `tools/session_search_tool.py` — main search handler

**New flow:**
```
1. query arrives
2. IF query is not empty:
   a. Embed query → query_vec
   b. search_embeddings(query_vec, source_type="session", limit=limit)
   c. For each result: look up session metadata (when, source, model)
   d. Return results with pre-generated summaries — NO LLM CALL
   e. IF embedding fails or returns nothing: fall back to existing FTS5 path
3. IF query is empty: existing _list_recent_sessions (no change)
```

**Key design:** The fallback to FTS5+LLM ensures backward compatibility. If OpenRouter is down or no API key, session_search works exactly as before.

**Also:** When falling back to FTS5, check if matched sessions have pre-generated summaries before calling LLM. Only call LLM for sessions without summaries.

---

## P1c: Memory Embedding

### Task P1c.1: Embed memory entries on write

**Objective:** When hot or cold memory is written, embed it.

**Files:**
- Modify: `tools/memory_tool.py` — in `add()`, `replace()`, `add_to_cold()` methods

**Pattern (in add method, after successful write):**
```python
# Best-effort embedding — don't block memory operations
try:
    from agent.embedding_client import get_embedding, content_hash as chash
    vec = get_embedding(content)
    if vec is not None and self._session_db:
        self._session_db.store_embedding(
            source_type="memory",
            source_id=chash(content),
            content=content,
            embedding=vec,
            model="text-embedding-3-small",
        )
except Exception:
    pass
```

**For replace:** Delete old embedding, add new one.
**For remove:** Delete embedding.
**For archive (hot→cold):** Re-tag source_type from "memory" to "cold_memory".

### Task P1c.2: Enhance cold memory search with vector

**Objective:** `memory(action='search')` uses vector similarity in addition to FTS5.

**Files:**
- Modify: `tools/memory_tool.py` — `search_cold()` method

**New flow:**
```
1. FTS5 keyword search (existing)
2. Vector similarity search (new, if available)
3. Merge and deduplicate results
4. Return combined results ranked by relevance
```

---

## P1d: Expand Hot Memory Capacity

### Task P1d.1: Increase char limits

**Objective:** Expand hot memory from 2200→6000 chars (memory) and 1375→3000 chars (user).

**Files:**
- Modify: `tools/memory_tool.py:123` — constructor defaults

**Change:**
```python
# BEFORE:
def __init__(self, memory_char_limit: int = 2200, user_char_limit: int = 1375, ...)

# AFTER:
def __init__(self, memory_char_limit: int = 6000, user_char_limit: int = 3000, ...)
```

**Rationale:** 
- Total 9000 chars ≈ 2500 tokens, fits comfortably in system prompt
- Not going to 25KB (Claude Code level) because Hermes injects memory every turn — too much context tax
- 6K+3K is a 2.5x increase, meaningful but not wasteful

**Also update:** System prompt render block header to show new limits.

---

## P2-prep: Auto Dream Trigger Foundation

### Task P2.1: Add dream state tracking to state.db

**Objective:** Track when last dream ran and session count since then.

**Files:**
- Modify: `hermes_state.py` — add metadata table or key-value store

**Schema:**
```sql
-- Simple key-value for system metadata
CREATE TABLE IF NOT EXISTS system_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);
```

**Keys:**
- `last_dream_at` — ISO timestamp of last Auto Dream run
- `sessions_since_dream` — integer counter

**Methods:**
```python
def get_meta(self, key: str) -> Optional[str]: ...
def set_meta(self, key: str, value: str): ...
def increment_meta(self, key: str, default: int = 0) -> int: ...
```

### Task P2.2: Increment session counter on session end

**Objective:** After each session ends, increment `sessions_since_dream`.

**Files:**
- Modify: `run_agent.py` — in session cleanup, after auto_summary spawn

**Change:**
```python
# After _spawn_auto_summary() call
if self._session_db:
    try:
        self._session_db.increment_meta("sessions_since_dream", default=0)
    except Exception:
        pass
```

### Task P2.3: Dream trigger check (stub)

**Objective:** Add check function that returns True when dream should run. Not wired up yet — just the logic.

**Files:**
- Create: `agent/dream_trigger.py`

```python
def should_dream(session_db) -> bool:
    """Check if Auto Dream should trigger.
    
    Dual-gate: both must be true:
    1. Time gate: >= 24h since last dream
    2. Session gate: >= 5 sessions since last dream
    """
    from datetime import datetime, timezone, timedelta
    
    last_dream = session_db.get_meta("last_dream_at")
    sessions = int(session_db.get_meta("sessions_since_dream") or "0")
    
    # Time gate
    if last_dream:
        last_dt = datetime.fromisoformat(last_dream)
        if datetime.now(timezone.utc) - last_dt < timedelta(hours=24):
            return False
    
    # Session gate
    if sessions < 5:
        return False
    
    return True
```

**Note:** Actual Auto Dream execution (the consolidation prompt, 4-phase process) is deferred to Phase 2 full implementation. This just lays the trigger infrastructure.

---

## Implementation Order

```
P0   (30 min)  → Fix session reset — standalone bugfix, immediate value
P1a  (2 hours) → Embedding infra — sqlite-vec, API client, DB schema
P1d  (10 min)  → Expand memory limits — trivial config change  
P1b  (2 hours) → Session search upgrade — vector search + summary reuse
P1c  (1 hour)  → Memory embedding — hot/cold memory vectorization
P2-prep (30 min) → Dream trigger foundation — counters and check logic
```

Total estimated: ~6 hours of implementation.

## Testing Strategy

- Unit tests for embedding_client (mock httpx)
- Unit tests for SessionDB vector methods (real sqlite-vec in-memory)
- Integration test: write memory → search by semantic query → find it
- Integration test: session_search with vector path vs FTS5 fallback
- Manual: restart gateway → verify session continues
- Manual: run backfill script → verify embeddings created

## Rollback

- sqlite-vec is additive (new tables, doesn't modify existing)
- All vector paths have fallback to existing FTS5
- Memory backups already taken (~/memories_backup_20260415_*)
- State DB backup exists (~/state_backup_20260415_*.db)
