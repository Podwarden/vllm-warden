from fastapi import HTTPException, Request, status


def _normalize(origin: str) -> str:
    return origin.rstrip("/")


def _first(value: str | None) -> str | None:
    # X-Forwarded-* may be a comma list (proxy chain); take the client-most.
    if value is None:
        return None
    return value.split(",")[0].strip()


def require_matching_origin(
    request: Request,
    allowed_origins: tuple[str, ...],
    trust_proxy_origin: bool = False,
) -> None:
    got = request.headers.get("origin")
    if got is None:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "origin missing")

    got_norm = _normalize(got)
    if any(got_norm == _normalize(a) for a in allowed_origins):
        return

    if trust_proxy_origin:
        proto = _first(request.headers.get("x-forwarded-proto")) or request.url.scheme
        host = _first(request.headers.get("x-forwarded-host")) or request.headers.get("host")
        if proto and host:
            derived = _normalize(f"{proto}://{host}")
            if got_norm == derived:
                return

    raise HTTPException(status.HTTP_403_FORBIDDEN, "origin mismatch")


def origin_check_dep(request: Request) -> None:
    settings = request.app.state.settings
    require_matching_origin(
        request,
        settings.allowed_origins,
        trust_proxy_origin=settings.trust_proxy_origin,
    )
