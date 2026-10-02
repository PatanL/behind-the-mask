"""pump.fun coins: their market (live trades from PumpPortal's public data stream) and launching from the page.

PumpFunMarket follows one coin: every buy and sell (PumpPortal `subscribeTokenTrade`), its price from the bonding
curve's virtual reserves, its market cap and volume. Its fee share for compute: when the platform is the coin's
creator (fee_to_compute), creator fees of LP_CREATOR_FEE_BPS of volume accrue to the android's compute; a coin paired
by its own creator is funded by its free time and top-ups instead.

Launching (POST /api/pump/prepare, /api/pump/confirm) is off unless LP_PUMP_LAUNCH=1: the server relays the image +
metadata upload to pump.fun and asks PumpPortal for the unsigned create transaction; the creator's wallet and the
mint key made in their browser sign it. No key is held here.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import time

import httpx

from lp_market import Market

PUMP_LAUNCH = os.environ.get("LP_PUMP_LAUNCH", "0") == "1"
CREATOR_FEE_BPS = float(os.environ.get("LP_CREATOR_FEE_BPS", "30"))     # pump.fun's creator fee on volume (set to the current rate)
PP_WS = os.environ.get("LP_PUMPPORTAL_WS", "wss://pumpportal.fun/api/data")
PP_LOCAL = os.environ.get("LP_PUMPPORTAL_LOCAL", "https://pumpportal.fun/api/trade-local")
PUMP_IPFS = os.environ.get("LP_PUMP_IPFS", "https://pump.fun/api/ipfs")
PUMP_COIN = os.environ.get("LP_PUMP_COIN_API", "https://frontend-api-v3.pump.fun/coins/{mint}")
RPC = os.environ.get("LP_RPC", "https://api.mainnet-beta.solana.com")
SITE = os.environ.get("LP_SITE", "https://steerai.live")       # a launched coin's website: its android's page


class PumpFunMarket(Market):
    kind = "pump"

    def __init__(self, mint: str, fee_to_compute: bool = False):
        super().__init__()
        self.mint, self.fee_to_compute = mint, fee_to_compute
        self.vsol, self.vtok, self.mcap = 0.0, 0.0, 0.0
        self.complete = False

    def price(self) -> float:
        return self.vsol / self.vtok if self.vtok else (self.history[-1][1] if self.history else 0.0)

    def supply(self) -> float:
        return 1e9

    def liquidity_sol(self) -> float:
        return self.vsol or 30.0

    def curve_progress(self) -> float:
        return 1.0 if self.complete else max(0.0, min(1.0, (self.vsol - 30.0) / 85.0)) if self.vsol else 0.0

    def summary(self) -> dict:
        s = super().summary()
        if self.mcap:
            s["mcap_sol"] = self.mcap
        s["mint"] = self.mint
        return s

    def on_trade(self, m: dict):
        """A PumpPortal trade message."""
        now = time.time()
        self.vsol = float(m.get("vSolInBondingCurve") or self.vsol)
        self.vtok = float(m.get("vTokensInBondingCurve") or self.vtok)
        self.mcap = float(m.get("marketCapSol") or self.mcap)
        sol = float(m.get("solAmount") or 0)
        if self.fee_to_compute:
            self.fees_sol += sol * CREATOR_FEE_BPS / 1e4
        self.history.append((now, self.price()))
        self.note_trade({"t": now, "side": "buy" if m.get("txType") == "buy" else "sell", "sol": round(sol, 4), "price": self.price(),
                         "who": str(m.get("traderPublicKey", ""))[:44]})

    def tick(self, now: float) -> list[dict]:
        out = [tr for tr in self.trades if tr["t"] > getattr(self, "_seen", 0)]
        if out:
            self._seen = out[-1]["t"]
        elif self.history and now - self.history[-1][0] > 5:
            self.history.append((now, self.price()))
        return out

    async def seed(self):
        """The coin's state before any trade arrives (pump.fun's public coin API; best effort)."""
        try:
            async with httpx.AsyncClient(timeout=10) as cl:
                r = await cl.get(PUMP_COIN.format(mint=self.mint))
            if r.status_code == 200:
                d = r.json()
                self.vsol = float(d.get("virtual_sol_reserves") or 0) / 1e9
                self.vtok = float(d.get("virtual_token_reserves") or 0) / 1e6
                self.mcap = float(d.get("market_cap") or 0)
                self.complete = bool(d.get("complete"))
                self.history.append((time.time(), self.price()))
                return {"name": d.get("name"), "symbol": d.get("symbol"), "complete": self.complete}
        except Exception as e:  # noqa: BLE001
            print("[lp] pump seed failed:", self.mint, repr(e), flush=True)
        return None

    def state(self) -> dict:
        return {"mint": self.mint, "fees_sol": self.fees_sol, "volume_sol": self.volume_sol, "vsol": self.vsol, "vtok": self.vtok,
                "mcap": self.mcap, "history": list(self.history)[-600:]}

    def load_state(self, d: dict):
        self.fees_sol, self.volume_sol = d.get("fees_sol", 0.0), d.get("volume_sol", 0.0)
        self.vsol, self.vtok, self.mcap = d.get("vsol", 0.0), d.get("vtok", 0.0), d.get("mcap", 0.0)
        for t, p in d.get("history", []):
            self.history.append((t, p))


class PumpFeed:
    """One PumpPortal connection for every paired coin (PumpPortal asks for a single socket)."""

    def __init__(self):
        self.markets: dict[str, PumpFunMarket] = {}
        self.ws = None
        self.connected = False

    def add(self, m: PumpFunMarket):
        self.markets[m.mint] = m
        if self.ws is not None:
            asyncio.ensure_future(self._sub([m.mint]))

    async def _sub(self, keys):
        try:
            await self.ws.send(json.dumps({"method": "subscribeTokenTrade", "keys": keys}))
        except Exception:  # noqa: BLE001
            pass

    async def run(self):
        import websockets
        while True:
            if not self.markets:
                await asyncio.sleep(2)
                continue
            try:
                async with websockets.connect(PP_WS, ping_interval=20, max_size=2**20) as ws:
                    self.ws, self.connected = ws, True
                    await self._sub(list(self.markets))
                    async for raw in ws:
                        try:
                            m = json.loads(raw)
                        except Exception:  # noqa: BLE001
                            continue
                        mk = self.markets.get(m.get("mint"))
                        if mk and m.get("txType") in ("buy", "sell"):
                            mk.on_trade(m)
            except Exception as e:  # noqa: BLE001
                print("[lp] pumpportal:", repr(e), flush=True)
            self.ws, self.connected = None, False
            await asyncio.sleep(5)


FEED = PumpFeed()


def live_url(mint: str) -> str:
    """A launched coin's website on pump.fun: its android's live stream (always; the creator can't point it elsewhere)."""
    return f"{SITE}/coin.html?mint={mint}"


async def prepare(f: dict) -> dict:
    """Upload the coin's image + metadata to pump.fun, and get PumpPortal's unsigned create transaction."""
    if not PUMP_LAUNCH:
        raise RuntimeError("Launching on pump.fun is switched off on this server.")
    img = str(f.get("image") or "")
    if not img.startswith("data:image/png;base64,"):
        raise ValueError("The coin needs its android's portrait.")
    png = base64.b64decode(img.split(",", 1)[1])
    async with httpx.AsyncClient(timeout=30) as cl:
        r = await cl.post(PUMP_IPFS, data={"name": f["name"], "symbol": f["ticker"], "description": f.get("description", ""),
                                           "website": live_url(f["mint"]), "showName": "true"},
                          files={"file": ("android.png", png, "image/png")})
        r.raise_for_status()
        uri = r.json()["metadataUri"]
        r = await cl.post(PP_LOCAL, json={"publicKey": f["creator"], "action": "create",
                                          "tokenMetadata": {"name": f["name"], "symbol": f["ticker"], "uri": uri},
                                          "mint": f["mint"], "denominatedInSol": "true", "amount": float(f.get("dev_buy_sol") or 0),
                                          "slippage": 10, "priorityFee": 0.0005, "pool": "pump"})
        r.raise_for_status()
    return {"tx": base64.b64encode(r.content).decode(), "uri": uri}


async def confirmed(signature: str, tries: int = 20) -> bool:
    async with httpx.AsyncClient(timeout=10) as cl:
        for _ in range(tries):
            r = await cl.post(RPC, json={"jsonrpc": "2.0", "id": 1, "method": "getSignatureStatuses", "params": [[signature], {"searchTransactionHistory": True}]})
            st = (r.json().get("result") or {}).get("value", [None])[0]
            if st and st.get("confirmationStatus") in ("confirmed", "finalized") and not st.get("err"):
                return True
            await asyncio.sleep(2)
    return False
