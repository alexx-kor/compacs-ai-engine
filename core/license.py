"""License tier (free / pro) for gateway upgrade route."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LicenseState:
    tier: str
    pro_enabled: bool


def expected_pro_key() -> str:
    return os.getenv("COMPACS_PRO_KEY", "").strip()


def current_license() -> LicenseState:
    tier = os.getenv("COMPACS_LICENSE_TIER", "free").strip().lower() or "free"
    pro_enabled = tier == "pro" or os.getenv("COMPACS_PRO_ENABLED", "").lower() in {"1", "true", "yes"}
    return LicenseState(tier="pro" if pro_enabled else "free", pro_enabled=pro_enabled)


def activate_pro(license_key: str) -> LicenseState:
    expected = expected_pro_key()
    if not expected:
        raise ValueError("pro licensing is not configured on this server")
    if license_key.strip() != expected:
        raise ValueError("invalid license key")
    os.environ["COMPACS_LICENSE_TIER"] = "pro"
    os.environ["COMPACS_PRO_ENABLED"] = "true"
    return current_license()
