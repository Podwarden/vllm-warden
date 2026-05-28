import base64
import secrets


def generate_bearer_token() -> str:
    """Returns vw_<56 lowercase base32 chars> (35 random bytes, no padding)."""
    raw = secrets.token_bytes(35)
    body = base64.b32encode(raw).decode("ascii").rstrip("=").lower()
    return f"vw_{body}"


def parse_bearer_header(value: str | None) -> str | None:
    if not value:
        return None
    parts = value.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip()
