"""Shared domain parameters for every module (the dashboard sidebar sets them once)."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

# Fare levels for a 10 km trip in Colombo (Rs), before surge.
FARE_NAMES_3 = ("Low", "Standard", "Surge")
FARES_3 = (600.0, 750.0, 1000.0)
FARE_NAMES_4 = ("Low", "Standard", "Surge", "Premium")
FARES_4 = (600.0, 750.0, 1000.0, 1200.0)


@dataclass(frozen=True)
class DomainParams:
    Q: float = 400.0        # commuter demand in a zone (riders per hour)
    sigma: float = 1.0      # surge multiplier
    c: float = 350.0        # operating cost per trip (Rs)
    kappa: float = 0.20     # platform commission (share of the fare)
    beta: float = 1.0       # riders who switch fleet per rupee of fare gap
    K: float = 400.0        # rides one fleet can serve per hour
    cap_scale: float = 1.0  # multiplies every road capacity k_e in Module 1C
    delta: float = 0.90     # patience (discount factor) for repeated games

    def with_(self, **changes) -> "DomainParams":
        return replace(self, **changes)


def fare_levels(m: int) -> tuple[list[str], np.ndarray]:
    """Names and base fares (Rs) for m = 3..6 fare levels."""
    if m == 3:
        return list(FARE_NAMES_3), np.array(FARES_3)
    if m == 4:
        return list(FARE_NAMES_4), np.array(FARES_4)
    fares = np.round(np.linspace(550, 1050, m) / 10) * 10
    return [f"Rs {int(f)}" for f in fares], fares
