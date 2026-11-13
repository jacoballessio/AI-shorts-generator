from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BillingConfig:
    cloud_hourly_usd: float = 0.54
    target_requests_per_hour: float = 20.0
    per_request_floor_usd: float = 0.01
    per_request_override_usd: float | None = None

    @classmethod
    def from_env(cls) -> "BillingConfig":
        def _float_env(name: str, default: float) -> float:
            raw = os.environ.get(name)
            if not raw:
                return default
            try:
                return float(raw)
            except ValueError:
                return default

        return cls(
            cloud_hourly_usd=_float_env("AI_SHORTS_CLOUD_HOURLY_USD", 0.54),
            target_requests_per_hour=max(1.0, _float_env("AI_SHORTS_TARGET_REQUESTS_PER_HOUR", 20.0)),
            per_request_floor_usd=max(0.0, _float_env("AI_SHORTS_PER_REQUEST_FLOOR_USD", 0.01)),
            per_request_override_usd=(
                _float_env("AI_SHORTS_PER_REQUEST_USD", 0.0)
                if os.environ.get("AI_SHORTS_PER_REQUEST_USD")
                else None
            ),
        )

    def effective_per_request_usd(self) -> float:
        """Per-request price that covers hourly endpoint cost.

        Formula by default:
            cloud_hourly_usd / target_requests_per_hour

        `AI_SHORTS_PER_REQUEST_USD` can override this when you want to lock
        pricing manually.
        """
        if self.per_request_override_usd is not None and self.per_request_override_usd > 0:
            return round(float(self.per_request_override_usd), 6)
        derived = float(self.cloud_hourly_usd) / max(1.0, float(self.target_requests_per_hour))
        return round(max(float(self.per_request_floor_usd), derived), 6)


class UsageLedger:
    """Simple SQLite usage ledger for cloud generation billing.

    This tracks per-user spend so the UI can enforce prepaid balances. A Stripe
    webhook can later call `credit_user` when payments settle.
    """

    def __init__(self, db_path: Path, config: BillingConfig | None = None) -> None:
        self.db_path = db_path
        self.config = config or BillingConfig.from_env()
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
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    balance_usd REAL NOT NULL DEFAULT 0,
                    stripe_customer_id TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    request_count INTEGER NOT NULL,
                    runtime_seconds REAL NOT NULL,
                    cost_usd REAL NOT NULL,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                )
                """
            )
            conn.commit()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def ensure_user(self, user_id: str) -> None:
        clean = (user_id or "").strip()
        if not clean:
            raise ValueError("user_id is required")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO users (user_id, balance_usd, created_at)
                VALUES (?, 0, ?)
                """,
                (clean, self._now_iso()),
            )
            conn.commit()

    def get_balance(self, user_id: str) -> float:
        clean = (user_id or "").strip()
        if not clean:
            return 0.0
        with self._connect() as conn:
            row = conn.execute(
                "SELECT balance_usd FROM users WHERE user_id = ?",
                (clean,),
            ).fetchone()
        return float(row["balance_usd"]) if row else 0.0

    def credit_user(self, user_id: str, amount_usd: float, stripe_customer_id: str | None = None) -> float:
        if amount_usd <= 0:
            raise ValueError("amount_usd must be > 0")
        self.ensure_user(user_id)
        with self._connect() as conn:
            if stripe_customer_id:
                conn.execute(
                    "UPDATE users SET stripe_customer_id = ? WHERE user_id = ?",
                    (stripe_customer_id, user_id),
                )
            conn.execute(
                "UPDATE users SET balance_usd = balance_usd + ? WHERE user_id = ?",
                (float(amount_usd), user_id),
            )
            conn.commit()
        return self.get_balance(user_id)

    def estimate_cloud_cost(self, *, request_count: int = 1) -> float:
        cost = self.config.effective_per_request_usd() * max(1, int(request_count))
        return round(cost, 6)

    def can_afford(self, user_id: str, estimated_cost_usd: float) -> bool:
        return self.get_balance(user_id) >= float(estimated_cost_usd)

    def charge_cloud_usage(
        self,
        *,
        user_id: str,
        source: str,
        request_count: int,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[float, float]:
        self.ensure_user(user_id)
        cost = self.estimate_cloud_cost(request_count=request_count)
        with self._connect() as conn:
            row = conn.execute("SELECT balance_usd FROM users WHERE user_id = ?", (user_id,)).fetchone()
            balance = float(row["balance_usd"]) if row else 0.0
            if balance < cost:
                raise ValueError(
                    f"Insufficient balance. Need ${cost:.4f}, have ${balance:.4f}."
                )

            conn.execute(
                "UPDATE users SET balance_usd = balance_usd - ? WHERE user_id = ?",
                (cost, user_id),
            )
            conn.execute(
                """
                INSERT INTO usage_events (user_id, source, request_count, runtime_seconds, cost_usd, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    source,
                    int(request_count),
                    0.0,
                    float(cost),
                    json.dumps(metadata or {}, ensure_ascii=True),
                    self._now_iso(),
                ),
            )
            conn.commit()
        return cost, self.get_balance(user_id)
