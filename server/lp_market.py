"""Markets for launchpad coins, and what the chart does to the android's mood.

A market keeps the coin's price history and trades and accrues the platform's share of trading fees (which pays for
the android's compute). SimMarket is a demo market (a bonding curve with simulated traders, and buys/sells from the
page), for coins not yet on chain and for testing. Chain markets (Meteora DBC pools) plug in through the same
interface: price(), trades since t, fees accrued (see lp_chain.py).

mood(): the chart as feelings, each 0..1 --
  joy    the price rising (over 30 s and 5 min)
  fear   the price falling
  anger  a big sell (a spike that fades over ~20 s)
  calm   a flat, quiet chart
"""
from __future__ import annotations

import math
import random
import time
from collections import deque


class Market:
    kind = "base"

    def __init__(self):
        self.history: deque = deque(maxlen=4000)    # (t, price)
        self.trades: deque = deque(maxlen=400)      # dicts: t, side, sol, price, who
        self.fees_sol = 0.0                         # the platform's fee share accrued so far (pays for compute)
        self.volume_sol = 0.0
        self._spike = {"anger": (0.0, 0.0), "joy": (0.0, 0.0)}   # (t, size)

    # -- to implement
    def tick(self, now: float) -> list[dict]:
        """Advance; returns the new trades."""
        return []

    def price(self) -> float:
        return self.history[-1][1] if self.history else 0.0

    # -- shared
    def change(self, seconds: float, now: float | None = None) -> float:
        now = now or time.time()
        if len(self.history) < 2:
            return 0.0
        p1 = self.history[-1][1]
        p0 = next((p for t, p in reversed(self.history) if t <= now - seconds), self.history[0][1])
        return (p1 - p0) / p0 if p0 > 0 else 0.0

    def note_trade(self, tr: dict):
        self.trades.append(tr)
        self.volume_sol += tr["sol"]
        big = tr["sol"] / max(1e-9, self.liquidity_sol())
        if tr["side"] == "sell" and big > 0.02:
            self._spike["anger"] = (tr["t"], min(1.0, big / 0.08))
        if tr["side"] == "buy" and big > 0.02:
            self._spike["joy"] = (tr["t"], min(1.0, big / 0.08))

    def liquidity_sol(self) -> float:
        return 30.0

    def mood(self, now: float | None = None) -> dict:
        now = now or time.time()
        r30, r5 = self.change(30, now), self.change(300, now)
        up = max(0.0, r30 / 0.04) + max(0.0, r5 / 0.12)
        down = max(0.0, -r30 / 0.04) + max(0.0, -r5 / 0.10)
        recent = [p for t, p in self.history if t >= now - 120]
        vol = (max(recent) - min(recent)) / max(1e-12, recent[-1]) if len(recent) > 3 else 0.0
        quiet = not any(t >= now - 60 for t in (tr["t"] for tr in self.trades))
        def spike(k, half=8.0):
            t0, s = self._spike[k]
            return s * math.exp(-max(0.0, now - t0) / half) if t0 else 0.0
        return {"joy": min(1.0, 0.6 * up + spike("joy")), "fear": min(1.0, 0.7 * down),
                "anger": min(1.0, spike("anger")), "calm": min(1.0, max(0.0, 1 - vol / 0.03) * (0.6 if quiet else 0.2))}

    def summary(self) -> dict:
        p = self.price()
        return {"price": p, "mcap_sol": p * self.supply(), "change_5m": self.change(300), "change_1h": self.change(3600),
                "volume_sol": round(self.volume_sol, 4), "kind": self.kind, "curve": self.curve_progress()}

    def supply(self) -> float:
        return 1e9

    def curve_progress(self) -> float:
        return 0.0

    def series(self, seconds=3600, points=120) -> list:
        now = time.time()
        pts = [(t, p) for t, p in self.history if t >= now - seconds]
        if len(pts) > points:
            step = len(pts) / points
            pts = [pts[int(i * step)] for i in range(points)]
        return [[round(t, 1), p] for t, p in pts]

    def state(self) -> dict:
        return {}

    def load_state(self, d: dict):
        pass


class SimMarket(Market):
    """A constant-product bonding curve (virtual reserves, like a DBC / pump curve) with simulated traders."""
    kind = "sim"

    def __init__(self, seed: int = 0, fee_bps: int = 100, platform_share: float = 0.5, activity: float = 0.15):
        super().__init__()
        self.rng = random.Random(seed)
        self.vsol, self.vtok = 30.0, 1.073e9          # virtual reserves (SOL, tokens)
        self.fee = fee_bps / 1e4
        self.share = platform_share                   # the platform's part of each fee (the rest: creator)
        self.activity = activity                      # simulated trades per second
        self.last = time.time()
        self.history.append((self.last, self.price()))
        self.mood_drift = 0.0

    def price(self) -> float:
        return self.vsol / self.vtok

    def supply(self) -> float:
        return 1e9

    def liquidity_sol(self) -> float:
        return self.vsol

    def curve_progress(self) -> float:
        return max(0.0, min(1.0, (self.vsol - 30.0) / 85.0))   # ~85 SOL raised completes the curve

    def trade(self, side: str, sol: float, who: str = "sim", now: float | None = None) -> dict:
        now = now or time.time()
        sol = max(0.001, float(sol))
        k = self.vsol * self.vtok
        fee = sol * self.fee
        if side == "buy":
            self.vsol += sol - fee
            out = self.vtok - k / self.vsol
            self.vtok -= out
        else:
            sol = min(sol, self.vsol - 30.0 * 0.5) if self.vsol > 15 else 0.0
            if sol <= 0:
                return {}
            self.vsol -= sol
            self.vtok = k / self.vsol
        self.fees_sol += fee * self.share
        tr = {"t": now, "side": side, "sol": round(sol, 4), "price": self.price(), "who": who}
        self.history.append((now, self.price()))
        self.note_trade(tr)
        return tr

    def tick(self, now: float) -> list[dict]:
        out = []
        dt, self.last = now - self.last, now
        # traders arrive at random; their mood drifts slowly (runs of buying or selling)
        self.mood_drift = 0.97 * self.mood_drift + 0.03 * self.rng.gauss(0, 1)
        n = sum(1 for _ in range(int(max(1, dt * 4))) if self.rng.random() < self.activity * dt / max(1, int(dt * 4)))
        for _ in range(n):
            side = "buy" if self.rng.random() < 0.5 + 0.25 * math.tanh(self.mood_drift) else "sell"
            size = math.exp(self.rng.gauss(math.log(0.15), 1.0))
            tr = self.trade(side, size, "sim", now)
            if tr:
                out.append(tr)
        if not out:
            self.history.append((now, self.price()))
        return out

    def state(self) -> dict:
        return {"vsol": self.vsol, "vtok": self.vtok, "fees_sol": self.fees_sol, "volume_sol": self.volume_sol,
                "history": list(self.history)[-600:]}

    def load_state(self, d: dict):
        self.vsol, self.vtok = d.get("vsol", self.vsol), d.get("vtok", self.vtok)
        self.fees_sol, self.volume_sol = d.get("fees_sol", 0.0), d.get("volume_sol", 0.0)
        for t, p in d.get("history", []):
            self.history.append((t, p))
