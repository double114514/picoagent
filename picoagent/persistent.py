"""SQLite-backed persistent memory: conversations survive process restarts.

Schema per (session_id): ordered message rows. Loading reconstructs a Memory.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .memory import Memory
from .types import Message, Role, ToolCall

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    name TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,
    timestamp REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_session ON messages(session_id, id);
"""


class SQLiteMemory(Memory):
    """Memory persisted to SQLite, keyed by session_id.

        mem = SQLiteMemory("chat.db", session_id="user-42")
        agent = Agent(llm=..., memory=mem)   # resumes prior conversation
    """

    def __init__(self, db_path: str | Path, session_id: str = "default",
                 max_messages: int = 50):
        super().__init__(max_messages=max_messages)
        self.db_path = str(db_path)
        self.session_id = session_id
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._load()

    def _load(self) -> None:
        rows = self._conn.execute(
            "SELECT role, content, name, tool_call_id, tool_calls, timestamp "
            "FROM messages WHERE session_id = ? ORDER BY id",
            (self.session_id,)).fetchall()
        self._messages = [
            Message(role=Role(r), content=c or "", name=n,
                    tool_call_id=tid,
                    tool_calls=[ToolCall(**tc) for tc in json.loads(tcs)]
                    if tcs else None,
                    timestamp=ts)
            for r, c, n, tid, tcs, ts in rows
        ]

    def add(self, message: Message) -> None:
        super().add(message)
        self._conn.execute(
            "INSERT INTO messages (session_id, role, content, name, "
            "tool_call_id, tool_calls, timestamp) VALUES (?,?,?,?,?,?,?)",
            (self.session_id, message.role.value, message.content,
             message.name, message.tool_call_id,
             json.dumps([tc.to_dict() for tc in message.tool_calls])
             if message.tool_calls else None,
             message.timestamp))
        self._conn.commit()

    def clear(self) -> None:
        super().clear()
        self._conn.execute("DELETE FROM messages WHERE session_id = ?",
                           (self.session_id,))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def list_sessions(db_path: str | Path) -> list[str]:
        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(
                "SELECT DISTINCT session_id FROM messages ORDER BY session_id"
            ).fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()
