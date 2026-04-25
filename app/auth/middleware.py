"""FastAPI auth middleware — replaces ``langgraph_sdk.Auth`` for the standalone
webapp.

Phase 5 of the LangGraph→Claude Agent SDK migration. Production endpoints
(``/investigations``, ``/chat``, etc.) declare ``Depends(authenticated_user)``
and receive a :class:`UserContext` derived from the verified Clerk JWT. The
``UserContext`` mirrors the fields previously injected by
``app.nodes.auth.inject_auth_node`` so callers can drop it into the agent
state unchanged.

The legacy ``langgraph_sdk.Auth`` object in :mod:`app.auth.auth` is still
imported by ``langgraph.json`` and stays in place until phase 7's removal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, cast

from fastapi import Depends, Header, HTTPException, status

from app.auth.jwt_auth import (
    JWTClaims,
    JWTExpiredError,
    JWTInvalidIssuerError,
    JWTMissingClaimError,
    JWTVerificationError,
    verify_jwt_async,
)


@dataclass(frozen=True)
class UserContext:
    """Authenticated user identity derived from a Clerk JWT.

    Field names match the keys that ``inject_auth_node`` writes into
    :class:`AgentState`, so the webapp can pre-populate state directly:

    >>> initial_state.update(user.as_state_fields())
    """

    org_id: str
    user_id: str
    user_email: str
    user_name: str
    organization_slug: str
    raw_token: str = ""

    @classmethod
    def from_claims(cls, claims: JWTClaims, *, raw_token: str = "") -> UserContext:
        return cls(
            org_id=claims.organization,
            user_id=claims.sub,
            user_email=claims.email,
            user_name=claims.full_name,
            organization_slug=claims.organization_slug,
            raw_token=raw_token,
        )

    def as_state_fields(self) -> dict[str, str]:
        """Render the context as the dict shape ``inject_auth_node`` produces."""
        return {
            "org_id": self.org_id,
            "user_id": self.user_id,
            "user_email": self.user_email,
            "user_name": self.user_name,
            "organization_slug": self.organization_slug,
            "_auth_token": self.raw_token,
        }


def _extract_bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization format; expected 'Bearer <token>'",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return parts[1]


async def authenticated_user(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> UserContext:
    """FastAPI dependency: verify the Clerk JWT and return a :class:`UserContext`.

    Raises ``HTTP 401`` for any verification failure. The JWT itself is
    preserved on the returned context (``raw_token``) so downstream handlers
    that need to call protected APIs on the user's behalf can re-use it.
    """
    token = _extract_bearer_token(authorization)
    try:
        claims = await verify_jwt_async(token)
    except JWTExpiredError as exc:
        raise HTTPException(status_code=401, detail="JWT expired") from exc
    except (JWTInvalidIssuerError, JWTMissingClaimError, JWTVerificationError) as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    return UserContext.from_claims(claims, raw_token=token)


# Type alias for endpoint signatures: ``user: AuthenticatedUser``
AuthenticatedUser = Annotated[UserContext, Depends(authenticated_user)]


def tenant_filter(user: UserContext) -> dict[str, Any]:
    """Per-tenant filter dict for any storage layer that supports it.

    Mirrors the ``org_id`` tagging that the legacy
    ``app.auth.auth`` LangGraph hooks performed on threads / assistants /
    crons. Cast to ``dict[str, Any]`` so callers can spread it into a
    storage query.
    """
    return cast(dict[str, Any], {"org_id": user.org_id})


__all__ = [
    "AuthenticatedUser",
    "UserContext",
    "authenticated_user",
    "tenant_filter",
]
