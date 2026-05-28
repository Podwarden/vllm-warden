import hashlib
import re
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import aiosqlite

from app.auth.bearer import generate_bearer_token

_SQLITE_UTC_FMT = "%Y-%m-%d %H:%M:%S"


def sqlite_utc_now() -> str:
    """Return the current UTC time as a SQLite-native naive UTC string."""
    return datetime.now(UTC).strftime(_SQLITE_UTC_FMT)


def sqlite_utc_in(delta: timedelta) -> str:
    """Return UTC now + delta as a SQLite-native naive UTC string."""
    return (datetime.now(UTC) + delta).strftime(_SQLITE_UTC_FMT)


# Column order MUST match the SELECT lists in find_by_plaintext / list_all /
# get below. Adding a column? Add it at the END of both this dataclass AND
# the SELECT clauses (so old positional tuples still unpack correctly), then
# update the create() INSERT if it needs a non-default value.
@dataclass
class TokenRow:
    id: str
    name: str
    prefix: str
    scope: str
    allowed_models: str | None
    rate_limit_rpm: int | None
    rate_limit_tpm: int | None
    revoked_at: str | None
    last_used_at: str | None
    created_at: str
    expires_at: str | None
    rotated_at: str | None
    rotated_from: str | None
    # S5 (#104) — sliding-window rate limit in TOKENS/sec (NULL = unlimited)
    # and STRICT scheduler priority 0..9 (9 always served first; starvation
    # of priority-0 tokens is by design and documented in the UI tooltip).
    rate_limit_tps: int | None = None
    priority: int = 5


_SELECT_COLS = (
    "id, name, prefix, scope, allowed_models, rate_limit_rpm, rate_limit_tpm, "
    "revoked_at, last_used_at, created_at, expires_at, rotated_at, rotated_from, "
    "rate_limit_tps, priority"
)


def hash_token(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


class _Unset:
    """Sentinel marker for "field not provided" in update_limits().

    Distinct from None so callers can clear rate_limit_tps (set to NULL)
    without us treating it as 'leave alone'. Defined above ``TokenRepo``
    so the method annotations can reference the class without forward
    refs / TYPE_CHECKING gymnastics.
    """

    _instance: "_Unset | None" = None

    def __new__(cls) -> "_Unset":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


_UNSET = _Unset()


class TokenRepo:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    async def create(
        self,
        token_id: str,
        name: str,
        plaintext: str,
        scope: str = "inference",
        allowed_models: list[str] | None = None,
        expires_in_days: int = 365,
        rate_limit_tps: int | None = None,
        priority: int = 5,
    ) -> None:
        """Insert a new API token row and commit.

        expires_in_days=0 (or any non-positive value) means 'never expires'.
        rate_limit_tps=None means 'unlimited'.
        priority must be 0..9 (DB CHECK trigger enforces this, but we validate
        at the Pydantic layer too so users get a 422 instead of a 500).
        """
        prefix = plaintext[:8]
        if expires_in_days > 0:
            expires_at = sqlite_utc_in(timedelta(days=expires_in_days))
        else:
            expires_at = None
        await self.db.execute(
            "INSERT INTO api_tokens"
            "(id, name, prefix, hash, scope, allowed_models, expires_at, "
            " rate_limit_tps, priority) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                token_id, name, prefix, hash_token(plaintext), scope,
                ",".join(allowed_models) if allowed_models else None,
                expires_at,
                rate_limit_tps, priority,
            ),
        )
        await self.db.commit()

    async def _next_old_suffix(self, base_name: str) -> int:
        """Return the next available ``N`` such that ``"{base_name} (old N)"``
        is unused.

        Issue #150 — when an operator rotates ``prod-bot`` we rename the old
        row to ``prod-bot (old 1)`` (or ``(old 2)``, ``(old 3)`` … if previous
        rotations already burnt the lower numbers) and mint a fresh row that
        keeps the original ``prod-bot`` name. The numbering is monotonic and
        never reuses gaps — a deleted ``(old 1)`` does NOT make ``N=1`` free
        again, because the operator's mental model is "the next slot", not
        "fill holes" (consider: a `(old 1)` deleted after audit could be
        confused with a current rotation if a future rotation reused the
        slot).

        Implementation: pull every row whose name LIKE ``"{base} (old %)"``,
        parse the integer between ``(old`` and ``)`` with a strict regex
        (rejects non-int garbage and stray spacing — defence against a hand-
        edited DB), then return ``max(N) + 1`` with default ``1`` when no
        prior `(old N)` exists. Case-sensitive; SQLite default collation is
        BINARY for TEXT and the rest of the codebase treats names as
        case-sensitive so we don't normalise here.
        """
        # LIKE escape: '%' and '_' would be wildcards inside base_name. Use
        # ESCAPE so a name containing those characters still matches literally.
        like_pat = base_name.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        cur = await self.db.execute(
            "SELECT name FROM api_tokens WHERE name LIKE ? ESCAPE '\\'",
            (f"{like_pat} (old %)",),
        )
        rows = await cur.fetchall()
        suffix_re = re.compile(
            r"^" + re.escape(base_name) + r" \(old (\d+)\)$"
        )
        seen: list[int] = []
        for (name,) in rows:
            m = suffix_re.match(name)
            if m:
                seen.append(int(m.group(1)))
        return (max(seen) + 1) if seen else 1

    async def rotate(
        self,
        old_id: str,
        grace_hours: int = 24,
        expires_in_days: int | None = None,
    ) -> tuple[str, str, str]:
        """Rename the predecessor to ``"{name} (old N)"`` and mint a fresh
        successor that keeps the original ``name`` (#150).

        All writes — predecessor SELECT, predecessor RENAME, successor
        INSERT, predecessor UPDATE (rotated_at / revoked_at) — are part of
        a single transaction so a crash cannot leave rotation half-applied.

        Returns ``(new_id, new_plaintext, renamed_to)`` where ``renamed_to``
        is the predecessor's new ``"{original} (old N)"`` name so the route
        can surface it to the UI in one round-trip.

        Behaviour:
          * If the predecessor was already rotated (``rotated_at`` is not
            null) we raise ``ValueError("already rotated")`` — the route
            translates that into 409. Idempotent rotation would silently
            allocate ``(old 2)``, ``(old 3)`` … on accidental double-clicks
            and is footgun-y enough to forbid.
          * The successor INHERITS ``rate_limit_tps`` and ``priority`` —
            operators do not want a rotation to silently change throughput
            or scheduler behaviour. To change either, PATCH the successor.
          * ``grace_hours`` schedules the predecessor's ``revoked_at`` (in
            the future when >0). Existing callers keep working through the
            window — see ``tests/integration/test_token_rotate_grace.py``.

        expires_in_days semantics:
          - None (default) → inherit the predecessor's expires_at (may be NULL).
          - 0              → successor never expires.
          - >0             → expires now + N days.
        """
        new_plaintext = generate_bearer_token()
        new_id = secrets.token_hex(16)
        new_prefix = new_plaintext[:8]

        # Look up predecessor for inheritable fields + the current name.
        cur = await self.db.execute(
            "SELECT name, expires_at, rate_limit_tps, priority, rotated_at "
            "FROM api_tokens WHERE id = ?",
            (old_id,),
        )
        row = await cur.fetchone()
        if row is None:
            # rotate() is only called from routes after a list_all() existence
            # check, but guard anyway so future direct callers see a clean error.
            raise ValueError(f"token {old_id} not found")
        pred_name, pred_expires, pred_rate, pred_priority, pred_rotated_at = row

        if pred_rotated_at is not None:
            # Predecessor was already rotated. Rotating again would chain
            # `(old 1) → (old 2)` style renames into a meaningless ladder
            # (the row's secret has the same compromise risk regardless of
            # how many rotations it survives). Bounce to the route which
            # returns 409.
            raise ValueError("already rotated")

        renamed_to = f"{pred_name} (old {await self._next_old_suffix(pred_name)})"

        if expires_in_days is None:
            new_expires_at = pred_expires
        elif expires_in_days > 0:
            new_expires_at = sqlite_utc_in(timedelta(days=expires_in_days))
        else:
            new_expires_at = None

        # 1) Rename predecessor BEFORE the insert so a future UNIQUE
        # constraint on `name` (none today, but cheap to be defensive)
        # could not collide between predecessor + successor in the same
        # transaction. Today this is purely a code-clarity ordering.
        await self.db.execute(
            "UPDATE api_tokens SET name = ? WHERE id = ?",
            (renamed_to, old_id),
        )

        # 2) Insert successor with the ORIGINAL name + rotated_from pointer.
        await self.db.execute(
            "INSERT INTO api_tokens"
            "(id, name, prefix, hash, scope, rotated_from, expires_at, "
            " rate_limit_tps, priority) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                new_id, pred_name, new_prefix, hash_token(new_plaintext),
                "inference", old_id, new_expires_at,
                pred_rate, pred_priority,
            ),
        )

        # 3) Mark predecessor as rotated + schedule its revocation.
        rotated_at = sqlite_utc_now()
        revoked_at = sqlite_utc_in(timedelta(hours=grace_hours))
        await self.db.execute(
            "UPDATE api_tokens SET rotated_at = ?, revoked_at = ? WHERE id = ?",
            (rotated_at, revoked_at, old_id),
        )

        await self.db.commit()
        return new_id, new_plaintext, renamed_to

    async def find_by_plaintext(self, plaintext: str) -> TokenRow | None:
        h = hash_token(plaintext)
        cur = await self.db.execute(
            f"SELECT {_SELECT_COLS} FROM api_tokens WHERE hash = ?",
            (h,),
        )
        r = await cur.fetchone()
        return TokenRow(*r) if r else None

    async def get(self, token_id: str) -> TokenRow | None:
        cur = await self.db.execute(
            f"SELECT {_SELECT_COLS} FROM api_tokens WHERE id = ?",
            (token_id,),
        )
        r = await cur.fetchone()
        return TokenRow(*r) if r else None

    async def list_all(self) -> list[TokenRow]:
        cur = await self.db.execute(
            f"SELECT {_SELECT_COLS} FROM api_tokens ORDER BY created_at DESC"
        )
        return [TokenRow(*r) for r in await cur.fetchall()]

    async def revoke(self, token_id: str) -> None:
        await self.db.execute(
            "UPDATE api_tokens SET revoked_at = datetime('now') WHERE id = ?", (token_id,)
        )
        await self.db.commit()

    async def touch_last_used(self, token_id: str) -> None:
        await self.db.execute(
            "UPDATE api_tokens SET last_used_at = datetime('now') WHERE id = ?", (token_id,)
        )
        await self.db.commit()

    async def delete(self, token_id: str) -> bool:
        cur = await self.db.execute(
            "DELETE FROM api_tokens WHERE id = ?", (token_id,)
        )
        await self.db.commit()
        return cur.rowcount > 0

    async def update_limits(
        self,
        token_id: str,
        *,
        rate_limit_tps: "int | None | _Unset" = _UNSET,
        priority: "int | _Unset" = _UNSET,
    ) -> bool:
        """PATCH-style update for the rate/priority fields.

        Pass _UNSET (the default) to leave a field untouched; pass None to
        explicitly clear rate_limit_tps (i.e. switch back to unlimited).
        priority cannot be None — it's NOT NULL in the schema; pass an int.

        Returns True if a row was updated, False if token_id was unknown.
        """
        sets: list[str] = []
        params: list[object] = []
        if not isinstance(rate_limit_tps, _Unset):
            sets.append("rate_limit_tps = ?")
            params.append(rate_limit_tps)
        if not isinstance(priority, _Unset):
            sets.append("priority = ?")
            params.append(priority)
        if not sets:
            return True  # noop PATCH — treat as success (idempotent)
        params.append(token_id)
        cur = await self.db.execute(
            f"UPDATE api_tokens SET {', '.join(sets)} WHERE id = ?",
            params,
        )
        await self.db.commit()
        return cur.rowcount > 0


class TokenUsageRepo:
    """Per-token minute-bucket usage rollup (token_usage_minute).

    Counted alongside the existing /counters + /model_samples writes in the
    proxy success path. The rollup is the source for GET /api/tokens/{id}/usage.
    """

    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    async def add(
        self,
        token_id: str,
        minute: int,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        # ON CONFLICT works here because (token_id, minute) is a real composite
        # PK with both columns NOT NULL — unlike the counters table that has
        # to dance around the NULL-token_id case.
        await self.db.execute(
            "INSERT INTO token_usage_minute"
            "(token_id, minute, requests, prompt_tokens, completion_tokens) "
            "VALUES (?, ?, 1, ?, ?) "
            "ON CONFLICT(token_id, minute) DO UPDATE SET "
            "  requests = requests + 1, "
            "  prompt_tokens = prompt_tokens + excluded.prompt_tokens, "
            "  completion_tokens = completion_tokens + excluded.completion_tokens",
            (token_id, minute, prompt_tokens, completion_tokens),
        )
        await self.db.commit()

    async def range(
        self,
        token_id: str,
        since_minute: int,
        until_minute: int,
    ) -> list[tuple[int, int, int, int]]:
        """Return (minute, requests, prompt_tokens, completion_tokens) rows
        for [since_minute, until_minute) in ascending order.

        Caller pre-computes the minute boundaries from the requested range
        (e.g. 24h, 1h) — this keeps the repo free of clock dependencies and
        the query plan trivially predictable via idx_token_usage_minute_minute.
        """
        cur = await self.db.execute(
            "SELECT minute, requests, prompt_tokens, completion_tokens "
            "FROM token_usage_minute "
            "WHERE token_id = ? AND minute >= ? AND minute < ? "
            "ORDER BY minute ASC",
            (token_id, since_minute, until_minute),
        )
        rows = await cur.fetchall()
        return [(int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in rows]

    async def totals(
        self,
        token_id: str,
        since_minute: int,
        until_minute: int,
    ) -> tuple[int, int, int]:
        """Return (requests, prompt_tokens, completion_tokens) summed over
        [since_minute, until_minute) for a single token.

        Used by the token list endpoint to populate the "Last 24h" column
        without forcing the UI to download every minute bucket.
        """
        cur = await self.db.execute(
            "SELECT COALESCE(SUM(requests), 0), "
            "       COALESCE(SUM(prompt_tokens), 0), "
            "       COALESCE(SUM(completion_tokens), 0) "
            "FROM token_usage_minute "
            "WHERE token_id = ? AND minute >= ? AND minute < ?",
            (token_id, since_minute, until_minute),
        )
        r = await cur.fetchone()
        return (int(r[0]), int(r[1]), int(r[2])) if r else (0, 0, 0)
