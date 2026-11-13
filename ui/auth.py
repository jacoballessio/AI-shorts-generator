from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


@dataclass(frozen=True)
class AuthConfig:
    session_hours: int = 24
    pbkdf2_iterations: int = 200_000

    @classmethod
    def from_env(cls) -> "AuthConfig":
        def _int_env(name: str, default: int) -> int:
            raw = os.environ.get(name)
            if not raw:
                return default
            try:
                return int(raw)
            except ValueError:
                return default

        return cls(
            session_hours=max(1, _int_env("AI_SHORTS_SESSION_HOURS", 24)),
            pbkdf2_iterations=max(50_000, _int_env("AI_SHORTS_PBKDF2_ITERATIONS", 200_000)),
        )


class AuthService:
    def __init__(self, db_path: Path, config: AuthConfig | None = None) -> None:
        self.db_path = db_path
        self.config = config or AuthConfig.from_env()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS app_users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS app_sessions (
                    session_token TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(username) REFERENCES app_users(username)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_app_sessions_username ON app_sessions(username)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_app_sessions_expires_at ON app_sessions(expires_at)")
            conn.commit()

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _iso(ts: datetime) -> str:
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _parse_iso(value: str) -> datetime:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

    @staticmethod
    def _normalize_username(username: str) -> str:
        clean = (username or "").strip()
        if not clean:
            raise ValueError("Username is required")
        if len(clean) < 3:
            raise ValueError("Username must be at least 3 characters")
        if len(clean) > 64:
            raise ValueError("Username must be at most 64 characters")
        if not all(ch.isalnum() or ch in "._-" for ch in clean):
            raise ValueError("Username can only contain letters, numbers, dot, underscore, and dash")
        return clean.lower()

    @staticmethod
    def _validate_password(password: str) -> None:
        if len(password or "") < 8:
            raise ValueError("Password must be at least 8 characters")
        if len(password) > 256:
            raise ValueError("Password is too long")

    def _hash_password(self, password: str, salt_hex: str) -> str:
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            self.config.pbkdf2_iterations,
        )
        return digest.hex()

    def _create_session(self, username: str) -> tuple[str, str]:
        token = secrets.token_urlsafe(48)
        now = self._now()
        expires = now + timedelta(hours=self.config.session_hours)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO app_sessions (session_token, username, expires_at, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (token, username, self._iso(expires), self._iso(now)),
            )
            conn.commit()
        return token, self._iso(expires)

    def cleanup_expired_sessions(self) -> None:
        now_iso = self._iso(self._now())
        with self._connect() as conn:
            conn.execute("DELETE FROM app_sessions WHERE expires_at <= ?", (now_iso,))
            conn.commit()

    def sign_up(self, username: str, password: str) -> str:
        clean_user = self._normalize_username(username)
        self._validate_password(password)
        salt_hex = secrets.token_hex(16)
        password_hash = self._hash_password(password, salt_hex)
        now = self._iso(self._now())

        with self._connect() as conn:
            existing = conn.execute("SELECT username FROM app_users WHERE username = ?", (clean_user,)).fetchone()
            if existing:
                raise ValueError("Username already exists")
            conn.execute(
                """
                INSERT INTO app_users (username, password_hash, salt, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (clean_user, password_hash, salt_hex, now),
            )
            conn.commit()
        return clean_user

    def sign_in(self, username: str, password: str) -> tuple[str, str, str]:
        clean_user = self._normalize_username(username)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT username, password_hash, salt FROM app_users WHERE username = ?",
                (clean_user,),
            ).fetchone()

        if row is None:
            raise ValueError("Invalid username or password")

        expected = str(row["password_hash"])
        actual = self._hash_password(password, str(row["salt"]))
        if not hmac.compare_digest(expected, actual):
            raise ValueError("Invalid username or password")

        token, expires_at = self._create_session(str(row["username"]))
        return str(row["username"]), token, expires_at

    def sign_out(self, session_token: str) -> None:
        token = (session_token or "").strip()
        if not token:
            return
        with self._connect() as conn:
            conn.execute("DELETE FROM app_sessions WHERE session_token = ?", (token,))
            conn.commit()

    def get_user_for_session(self, session_token: str) -> str | None:
        token = (session_token or "").strip()
        if not token:
            return None

        self.cleanup_expired_sessions()

        with self._connect() as conn:
            row = conn.execute(
                "SELECT username, expires_at FROM app_sessions WHERE session_token = ?",
                (token,),
            ).fetchone()
        if row is None:
            return None

        expires = self._parse_iso(str(row["expires_at"]))
        if expires <= self._now():
            self.sign_out(token)
            return None
        return str(row["username"])
