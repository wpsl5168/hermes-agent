#!/usr/bin/env python3
"""
Session Search Tool - Long-Term Conversation Recall

Hybrid FTS5 + vector search with pre-generated summary preference.

Flow:
  1. FTS5 keyword search finds matching messages ranked by relevance
  2. Vector similarity search (sqlite-vec) finds semantically related sessions
  3. Merge + deduplicate results from both paths
  4. For each matched session, prefer pre-generated summary (from auto_summary)
  5. Only fall back to real-time LLM summarization when no summary exists
  6. Returns per-session summaries with metadata
"""

import asyncio
import concurrent.futures
import json
import logging
import re
from typing import Dict, Any, List, Optional, Union

from agent.auxiliary_client import async_call_llm, extract_content_or_reasoning
MAX_SESSION_CHARS = 100_000
MAX_SUMMARY_TOKENS = 10000


def _format_timestamp(ts: Union[int, float, str, None]) -> str:
    """Convert a Unix timestamp (float/int) or ISO string to a human-readable date.

    Returns "unknown" for None, str(ts) if conversion fails.
    """
    if ts is None:
        return "unknown"
    try:
        if isinstance(ts, (int, float)):
            from datetime import datetime
            dt = datetime.fromtimestamp(ts)
            return dt.strftime("%B %d, %Y at %I:%M %p")
        if isinstance(ts, str):
            if ts.replace(".", "").replace("-", "").isdigit():
                from datetime import datetime
                dt = datetime.fromtimestamp(float(ts))
                return dt.strftime("%B %d, %Y at %I:%M %p")
            return ts
    except (ValueError, OSError, OverflowError) as e:
        # Log specific errors for debugging while gracefully handling edge cases
        logging.debug("Failed to format timestamp %s: %s", ts, e, exc_info=True)
    except Exception as e:
        logging.debug("Unexpected error formatting timestamp %s: %s", ts, e, exc_info=True)
    return str(ts)


def _format_conversation(messages: List[Dict[str, Any]]) -> str:
    """Format session messages into a readable transcript for summarization."""
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content") or ""
        tool_name = msg.get("tool_name")

        if role == "TOOL" and tool_name:
            # Truncate long tool outputs
            if len(content) > 500:
                content = content[:250] + "\n...[truncated]...\n" + content[-250:]
            parts.append(f"[TOOL:{tool_name}]: {content}")
        elif role == "ASSISTANT":
            # Include tool call names if present
            tool_calls = msg.get("tool_calls")
            if tool_calls and isinstance(tool_calls, list):
                tc_names = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        name = tc.get("name") or tc.get("function", {}).get("name", "?")
                        tc_names.append(name)
                if tc_names:
                    parts.append(f"[ASSISTANT]: [Called: {', '.join(tc_names)}]")
                if content:
                    parts.append(f"[ASSISTANT]: {content}")
            else:
                parts.append(f"[ASSISTANT]: {content}")
        else:
            parts.append(f"[{role}]: {content}")

    return "\n\n".join(parts)


def _truncate_around_matches(
    full_text: str, query: str, max_chars: int = MAX_SESSION_CHARS
) -> str:
    """
    Truncate a conversation transcript to *max_chars*, choosing a window
    that maximises coverage of positions where the *query* actually appears.

    Strategy (in priority order):
    1. Try to find the full query as a phrase (case-insensitive).
    2. If no phrase hit, look for positions where all query terms appear
       within a 200-char proximity window (co-occurrence).
    3. Fall back to individual term positions.

    Once candidate positions are collected the function picks the window
    start that covers the most of them.
    """
    if len(full_text) <= max_chars:
        return full_text

    text_lower = full_text.lower()
    query_lower = query.lower().strip()
    match_positions: list[int] = []

    # --- 1. Full-phrase search ------------------------------------------------
    phrase_pat = re.compile(re.escape(query_lower))
    match_positions = [m.start() for m in phrase_pat.finditer(text_lower)]

    # --- 2. Proximity co-occurrence of all terms (within 200 chars) -----------
    if not match_positions:
        terms = query_lower.split()
        if len(terms) > 1:
            # Collect every occurrence of each term
            term_positions: dict[str, list[int]] = {}
            for t in terms:
                term_positions[t] = [
                    m.start() for m in re.finditer(re.escape(t), text_lower)
                ]
            # Slide through positions of the rarest term and check proximity
            rarest = min(terms, key=lambda t: len(term_positions.get(t, [])))
            for pos in term_positions.get(rarest, []):
                if all(
                    any(abs(p - pos) < 200 for p in term_positions.get(t, []))
                    for t in terms
                    if t != rarest
                ):
                    match_positions.append(pos)

    # --- 3. Individual term positions (last resort) ---------------------------
    if not match_positions:
        terms = query_lower.split()
        for t in terms:
            for m in re.finditer(re.escape(t), text_lower):
                match_positions.append(m.start())

    if not match_positions:
        # Nothing at all — take from the start
        truncated = full_text[:max_chars]
        suffix = "\n\n...[later conversation truncated]..." if max_chars < len(full_text) else ""
        return truncated + suffix

    # --- Pick window that covers the most match positions ---------------------
    match_positions.sort()

    best_start = 0
    best_count = 0
    for candidate in match_positions:
        ws = max(0, candidate - max_chars // 4)  # bias: 25% before, 75% after
        we = ws + max_chars
        if we > len(full_text):
            ws = max(0, len(full_text) - max_chars)
            we = len(full_text)
        count = sum(1 for p in match_positions if ws <= p < we)
        if count > best_count:
            best_count = count
            best_start = ws

    start = best_start
    end = min(len(full_text), start + max_chars)

    truncated = full_text[start:end]
    prefix = "...[earlier conversation truncated]...\n\n" if start > 0 else ""
    suffix = "\n\n...[later conversation truncated]..." if end < len(full_text) else ""
    return prefix + truncated + suffix


async def _summarize_session(
    conversation_text: str, query: str, session_meta: Dict[str, Any]
) -> Optional[str]:
    """Summarize a single session conversation focused on the search query."""
    system_prompt = (
        "You are reviewing a past conversation transcript to help recall what happened. "
        "Summarize the conversation with a focus on the search topic. Include:\n"
        "1. What the user asked about or wanted to accomplish\n"
        "2. What actions were taken and what the outcomes were\n"
        "3. Key decisions, solutions found, or conclusions reached\n"
        "4. Any specific commands, files, URLs, or technical details that were important\n"
        "5. Anything left unresolved or notable\n\n"
        "Be thorough but concise. Preserve specific details (commands, paths, error messages) "
        "that would be useful to recall. Write in past tense as a factual recap."
    )

    source = session_meta.get("source", "unknown")
    started = _format_timestamp(session_meta.get("started_at"))

    user_prompt = (
        f"Search topic: {query}\n"
        f"Session source: {source}\n"
        f"Session date: {started}\n\n"
        f"CONVERSATION TRANSCRIPT:\n{conversation_text}\n\n"
        f"Summarize this conversation with focus on: {query}"
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await async_call_llm(
                task="session_search",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=MAX_SUMMARY_TOKENS,
            )
            content = extract_content_or_reasoning(response)
            if content:
                return content
            # Reasoning-only / empty — let the retry loop handle it
            logging.warning("Session search LLM returned empty content (attempt %d/%d)", attempt + 1, max_retries)
            if attempt < max_retries - 1:
                await asyncio.sleep(1 * (attempt + 1))
                continue
            return content
        except RuntimeError:
            logging.warning("No auxiliary model available for session summarization")
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(1 * (attempt + 1))
            else:
                logging.warning(
                    "Session summarization failed after %d attempts: %s",
                    max_retries,
                    e,
                    exc_info=True,
                )
                return None


# Sources that are excluded from session browsing/searching by default.
# Third-party integrations (Paperclip agents, etc.) tag their sessions with
# HERMES_SESSION_SOURCE=tool so they don't clutter the user's session history.
_HIDDEN_SESSION_SOURCES = ("tool",)


def _list_recent_sessions(db, limit: int, current_session_id: str = None) -> str:
    """Return metadata for the most recent sessions (no LLM calls)."""
    try:
        sessions = db.list_sessions_rich(limit=limit + 5, exclude_sources=list(_HIDDEN_SESSION_SOURCES))  # fetch extra to skip current

        # Resolve current session lineage to exclude it
        current_root = None
        if current_session_id:
            try:
                sid = current_session_id
                visited = set()
                while sid and sid not in visited:
                    visited.add(sid)
                    s = db.get_session(sid)
                    parent = s.get("parent_session_id") if s else None
                    sid = parent if parent else None
                current_root = max(visited, key=len) if visited else current_session_id
            except Exception:
                current_root = current_session_id

        results = []
        for s in sessions:
            sid = s.get("id", "")
            if current_root and (sid == current_root or sid == current_session_id):
                continue
            # Skip child/delegation sessions (they have parent_session_id)
            if s.get("parent_session_id"):
                continue
            results.append({
                "session_id": sid,
                "title": s.get("title") or None,
                "source": s.get("source", ""),
                "started_at": s.get("started_at", ""),
                "last_active": s.get("last_active", ""),
                "message_count": s.get("message_count", 0),
                "preview": s.get("preview", ""),
            })
            if len(results) >= limit:
                break

        return json.dumps({
            "success": True,
            "mode": "recent",
            "results": results,
            "count": len(results),
            "message": f"Showing {len(results)} most recent sessions. Use a keyword query to search specific topics.",
        }, ensure_ascii=False)
    except Exception as e:
        logging.error("Error listing recent sessions: %s", e, exc_info=True)
        return tool_error(f"Failed to list recent sessions: {e}", success=False)


def session_search(
    query: str,
    role_filter: str = None,
    limit: int = 3,
    db=None,
    current_session_id: str = None,
) -> str:
    """
    Search past sessions and return focused summaries of matching conversations.

    Hybrid search: FTS5 keyword matching + sqlite-vec vector similarity.
    Prefers pre-generated summaries over real-time LLM calls for speed.
    The current session is excluded from results since the agent already has that context.
    """
    if db is None:
        return tool_error("Session database not available.", success=False)

    limit = min(limit, 5)  # Cap at 5 sessions

    # Recent sessions mode: when query is empty, return metadata for recent sessions.
    # No LLM calls — just DB queries for titles, previews, timestamps.
    if not query or not query.strip():
        return _list_recent_sessions(db, limit, current_session_id)

    query = query.strip()

    try:
        # Parse role filter
        role_list = None
        if role_filter and role_filter.strip():
            role_list = [r.strip() for r in role_filter.split(",") if r.strip()]

        # Resolve current session lineage
        def _resolve_to_parent(session_id: str) -> str:
            """Walk delegation chain to find the root parent session ID."""
            visited = set()
            sid = session_id
            while sid and sid not in visited:
                visited.add(sid)
                try:
                    session = db.get_session(sid)
                    if not session:
                        break
                    parent = session.get("parent_session_id")
                    if parent:
                        sid = parent
                    else:
                        break
                except Exception as e:
                    logging.debug(
                        "Error resolving parent for session %s: %s",
                        sid,
                        e,
                        exc_info=True,
                    )
                    break
            return sid

        current_lineage_root = (
            _resolve_to_parent(current_session_id) if current_session_id else None
        )

        def _is_current_session(raw_sid: str, resolved_sid: str) -> bool:
            """Check if a session belongs to the current session lineage."""
            if current_lineage_root and resolved_sid == current_lineage_root:
                return True
            if current_session_id and raw_sid == current_session_id:
                return True
            return False

        # ── Path 1: FTS5 keyword search ──────────────────────────────────
        fts_session_ids = {}  # session_id -> match_info dict
        try:
            raw_results = db.search_messages(
                query=query,
                role_filter=role_list,
                exclude_sources=list(_HIDDEN_SESSION_SOURCES),
                limit=50,
                offset=0,
            )
            for result in raw_results:
                raw_sid = result["session_id"]
                resolved_sid = _resolve_to_parent(raw_sid)
                if _is_current_session(raw_sid, resolved_sid):
                    continue
                if resolved_sid not in fts_session_ids:
                    result = dict(result)
                    result["session_id"] = resolved_sid
                    result["_match_source"] = "fts"
                    fts_session_ids[resolved_sid] = result
                if len(fts_session_ids) >= limit * 2:  # over-fetch for merge
                    break
        except Exception as e:
            logging.warning("FTS5 search failed (falling through): %s", e)

        # ── Path 2: Vector similarity search ─────────────────────────────
        vec_session_ids = {}  # session_id -> {distance, source_id, ...}
        try:
            if getattr(db, "_vec_available", False):
                from agent.embedding_client import get_embedding
                query_vec = get_embedding(query)
                if query_vec is not None:
                    vec_results = db.search_embeddings(
                        query_vec, source_type="session", limit=limit * 3,
                    )
                    for vr in vec_results:
                        sid = vr["source_id"]
                        resolved_sid = _resolve_to_parent(sid)
                        if _is_current_session(sid, resolved_sid):
                            continue
                        if resolved_sid not in vec_session_ids:
                            vec_session_ids[resolved_sid] = {
                                "session_id": resolved_sid,
                                "distance": vr["distance"],
                                "_match_source": "vec",
                            }
                        if len(vec_session_ids) >= limit * 2:
                            break
        except Exception as e:
            logging.debug("Vector search skipped: %s", e)

        # ── Merge results ────────────────────────────────────────────────
        # Priority: FTS hits first (exact keyword matches are more intentional),
        # then vector hits that aren't already in FTS results.
        merged_ids = list(fts_session_ids.keys())
        for sid in vec_session_ids:
            if sid not in fts_session_ids:
                merged_ids.append(sid)
        merged_ids = merged_ids[:limit]

        if not merged_ids:
            return json.dumps({
                "success": True,
                "query": query,
                "results": [],
                "count": 0,
                "sessions_searched": 0,
                "message": "No matching sessions found.",
            }, ensure_ascii=False)

        # ── Build results: prefer pre-generated summaries ────────────────
        sessions_needing_llm = []  # (session_id, match_info, conversation_text, session_meta)
        summaries = []

        for session_id in merged_ids:
            try:
                session_meta = db.get_session(session_id) or {}
                match_source = "fts+vec" if (session_id in fts_session_ids and session_id in vec_session_ids) else \
                               "fts" if session_id in fts_session_ids else "vec"
                match_info = fts_session_ids.get(session_id) or vec_session_ids.get(session_id, {})

                entry = {
                    "session_id": session_id,
                    "when": _format_timestamp(
                        match_info.get("session_started") or session_meta.get("started_at")
                    ),
                    "source": match_info.get("source") or session_meta.get("source", "unknown"),
                    "model": match_info.get("model") or session_meta.get("model"),
                    "match": match_source,
                }

                # Check for pre-generated summary first
                existing_summary = session_meta.get("summary")
                if existing_summary and existing_summary.strip():
                    entry["summary"] = existing_summary.strip()
                    summaries.append(entry)
                else:
                    # Need LLM summarization — queue it
                    messages = db.get_messages_as_conversation(session_id)
                    if messages:
                        conversation_text = _format_conversation(messages)
                        conversation_text = _truncate_around_matches(conversation_text, query)
                        sessions_needing_llm.append((session_id, entry, conversation_text, session_meta))
                    else:
                        entry["summary"] = "(Empty session — no messages found)"
                        summaries.append(entry)
            except Exception as e:
                logging.warning("Failed to prepare session %s: %s", session_id, e, exc_info=True)

        # ── LLM summarization for sessions without pre-generated summaries ──
        if sessions_needing_llm:
            async def _summarize_all() -> List[Union[str, Exception]]:
                coros = [
                    _summarize_session(text, query, meta)
                    for _, _, text, meta in sessions_needing_llm
                ]
                return await asyncio.gather(*coros, return_exceptions=True)

            try:
                from model_tools import _run_async
                llm_results = _run_async(_summarize_all())
            except concurrent.futures.TimeoutError:
                logging.warning("Session summarization timed out")
                llm_results = [None] * len(sessions_needing_llm)

            for (session_id, entry, conversation_text, _), result in zip(sessions_needing_llm, llm_results):
                if isinstance(result, Exception):
                    logging.warning("Failed to summarize session %s: %s", session_id, result)
                    result = None

                if result:
                    entry["summary"] = result
                else:
                    preview = (conversation_text[:500] + "\n…[truncated]") if conversation_text else "No preview available."
                    entry["summary"] = f"[Raw preview — summarization unavailable]\n{preview}"
                summaries.append(entry)

        # Sort: sessions with proper summaries first, then by timestamp
        summaries.sort(key=lambda s: (
            0 if not s.get("summary", "").startswith("[Raw preview") else 1,
        ))

        return json.dumps({
            "success": True,
            "query": query,
            "results": summaries,
            "count": len(summaries),
            "sessions_searched": len(merged_ids),
            "search_paths": {
                "fts_hits": len(fts_session_ids),
                "vec_hits": len(vec_session_ids),
            },
        }, ensure_ascii=False)

    except Exception as e:
        logging.error("Session search failed: %s", e, exc_info=True)
        return tool_error(f"Search failed: {str(e)}", success=False)


def check_session_search_requirements() -> bool:
    """Requires SQLite state database and an auxiliary text model."""
    try:
        from hermes_state import DEFAULT_DB_PATH
        return DEFAULT_DB_PATH.parent.exists()
    except ImportError:
        return False


SESSION_SEARCH_SCHEMA = {
    "name": "session_search",
    "description": (
        "Search your long-term memory of past conversations, or browse recent sessions. This is your recall -- "
        "every past session is searchable, and this tool summarizes what happened.\n\n"
        "TWO MODES:\n"
        "1. Recent sessions (no query): Call with no arguments to see what was worked on recently. "
        "Returns titles, previews, and timestamps. Zero LLM cost, instant. "
        "Start here when the user asks what were we working on or what did we do recently.\n"
        "2. Keyword search (with query): Search for specific topics across all past sessions. "
        "Returns summaries of matching sessions (prefers cached summaries for speed, "
        "falls back to LLM summarization when needed).\n\n"
        "USE THIS PROACTIVELY when:\n"
        "- The user says 'we did this before', 'remember when', 'last time', 'as I mentioned'\n"
        "- The user asks about a topic you worked on before but don't have in current context\n"
        "- The user references a project, person, or concept that seems familiar but isn't in memory\n"
        "- You want to check if you've solved a similar problem before\n"
        "- The user asks 'what did we do about X?' or 'how did we fix Y?'\n\n"
        "Don't hesitate to search when it is actually cross-session -- it's fast and cheap. "
        "Better to search and confirm than to guess or ask the user to repeat themselves.\n\n"
        "Search syntax: keywords joined with OR for broad recall (elevenlabs OR baseten OR funding), "
        "phrases for exact match (\"docker networking\"), boolean (python NOT java), prefix (deploy*). "
        "IMPORTANT: Use OR between keywords for best results — FTS5 defaults to AND which misses "
        "sessions that only mention some terms. If a broad OR query returns nothing, try individual "
        "keyword searches in parallel. Returns summaries of the top matching sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query — keywords, phrases, or boolean expressions to find in past sessions. Omit this parameter entirely to browse recent sessions instead (returns titles, previews, timestamps with no LLM cost).",
            },
            "role_filter": {
                "type": "string",
                "description": "Optional: only search messages from specific roles (comma-separated). E.g. 'user,assistant' to skip tool outputs.",
            },
            "limit": {
                "type": "integer",
                "description": "Max sessions to summarize (default: 3, max: 5).",
                "default": 3,
            },
        },
        "required": [],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="session_search",
    toolset="session_search",
    schema=SESSION_SEARCH_SCHEMA,
    handler=lambda args, **kw: session_search(
        query=args.get("query") or "",
        role_filter=args.get("role_filter"),
        limit=args.get("limit", 3),
        db=kw.get("db"),
        current_session_id=kw.get("current_session_id")),
    check_fn=check_session_search_requirements,
    emoji="🔍",
)
