"""pump.fun coins: their market (live trades from PumpPortal's public data stream) and launching from the page.

PumpFunMarket follows one coin: every buy and sell (PumpPortal `subscribeTokenTrade`), its price from the bonding
curve's virtual reserves, its market cap and volume. Its starting state is read from its bonding curve on chain.

Launching (POST /api/pump/prepare, /api/pump/confirm) is off unless LP_PUMP_LAUNCH=1. The server uploads the image and
metadata to pump.fun; the launch page builds the create transaction with pump.fun's own SDK
(web/launchpad/src/pumpcreate.js), the launcher's wallet signing and paying, with the coin's creator set to the
platform treasury (so creator fees go there). No key is held here. Settings: runs/launchpad/pump.json
{"treasury": <pubkey>, "alt": <address lookup table>} (or LP_TREASURY / LP_ALT). With a treasury set, a coin only gets
an android if its on-chain creator is the treasury.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import time

import httpx
from pathlib import Path

from lp_wallet import b58decode, b58encode, find_program_address

from lp_market import Market

PUMP_LAUNCH = os.environ.get("LP_PUMP_LAUNCH", "0") == "1"
CREATOR_FEE_BPS = float(os.environ.get("LP_CREATOR_FEE_BPS", "30"))     # pump.fun's creator fee on volume (set to the current rate)
PP_WS = os.environ.get("LP_PUMPPORTAL_WS", "wss://pumpportal.fun/api/data")
PUMP_IPFS = os.environ.get("LP_PUMP_IPFS", "https://pump.fun/api/ipfs")
PUMP_COIN = os.environ.get("LP_PUMP_COIN_API", "https://frontend-api-v3.pump.fun/coins/{mint}")
RPC = os.environ.get("LP_RPC", "https://api.mainnet-beta.solana.com")
SITE = os.environ.get("LP_SITE", "https://steerai.live")       # a launched coin's website: its android's page
PUMP_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
PUMP_AMM = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"     # PumpSwap: where a coin trades once its curve completes
WSOL = "So11111111111111111111111111111111111111112"
AMM_CREATOR_FEE_BPS = float(os.environ.get("LP_AMM_CREATOR_FEE_BPS", "5"))   # the coin creator's PumpSwap fee (set to the current rate)
SETTINGS = Path(os.environ.get("LP_STATE", "runs/launchpad")) / "pump.json"


_SETTINGS_CACHE: dict = {"mtime": None, "d": {}}


def settings() -> dict:
    """The server's settings file: the treasury (every coin's creator: its fees), the address lookup table launches use,
    capacity (max_alive, slots, min_slots, max_slots) and always_awake (no coin sleeps, whatever its balance). Env
    overrides the treasury and the table. Re-read when the file changes."""
    try:
        mt = SETTINGS.stat().st_mtime
        if mt != _SETTINGS_CACHE["mtime"]:
            _SETTINGS_CACHE.update(mtime=mt, d=json.loads(SETTINGS.read_text()))
        d = _SETTINGS_CACHE["d"]
    except (OSError, ValueError):
        d = {}
    return {**d, "treasury": os.environ.get("LP_TREASURY") or d.get("treasury") or "", "alt": os.environ.get("LP_ALT") or d.get("alt") or ""}


def pool_address(mint: str) -> str:
    """A graduated coin's canonical PumpSwap pool (index 0, quoted in SOL; created by pump.fun's pool authority)."""
    auth = find_program_address([b"pool-authority", b58decode(mint)], PUMP_PROGRAM)
    return find_program_address([b"pool", (0).to_bytes(2, "little"), b58decode(auth), b58decode(mint), b58decode(WSOL)], PUMP_AMM)


async def accounts(addresses: list[str], encoding: str = "base64") -> list:
    """getMultipleAccounts, 100 at a time -> the accounts' values (None for a missing one)."""
    out = []
    async with httpx.AsyncClient(timeout=15) as cl:
        for i in range(0, len(addresses), 100):
            r = await cl.post(RPC, json={"jsonrpc": "2.0", "id": 1, "method": "getMultipleAccounts",
                                         "params": [addresses[i:i + 100], {"encoding": encoding, "commitment": "confirmed"}]})
            out += (r.json().get("result") or {}).get("value") or [None] * len(addresses[i:i + 100])
    return out


def parse_curve(v: dict | None) -> dict | None:
    if not v or v.get("owner") != PUMP_PROGRAM:
        return None
    b = base64.b64decode(v["data"][0])
    u = lambda o: int.from_bytes(b[o:o + 8], "little")
    return {"vtok": u(8) / 1e6, "vsol": u(16) / 1e9, "complete": bool(b[48]), "creator": b58encode(b[49:81])}


async def chain_loop(markets):
    """Prices from the chain itself: graduated coins' PumpSwap reserves every 5 s (the trade feed doesn't carry PumpSwap
    trades), and every coin's bonding curve once a minute (exact reserves; notices a graduation). markets() -> the
    PumpFunMarkets to follow."""
    last_curves = 0.0
    while True:
        await asyncio.sleep(5)
        try:
            ms = [m for m in markets() if isinstance(m, PumpFunMarket)]
            if not ms:
                continue
            if time.time() - last_curves > 60:
                last_curves = time.time()
                for m, v in zip(ms, await accounts([curve_address(m.mint) for m in ms])):
                    c = parse_curve(v)
                    if c:
                        m.on_curve(c)
            grads = [m for m in ms if m.complete]
            need = [m for m in grads if m.pool_tas is None]
            if need:
                for m, v in zip(need, await accounts([pool_address(m.mint) for m in need])):
                    if v and v.get("owner") == PUMP_AMM:
                        b = base64.b64decode(v["data"][0])
                        m.pool_tas = (b58encode(b[139:171]), b58encode(b[171:203]))   # its base and quote token accounts
            grads = [m for m in grads if m.pool_tas]
            if grads:
                vals = await accounts([a for m in grads for a in m.pool_tas], "jsonParsed")
                for k, m in enumerate(grads):
                    base, quote = vals[2 * k], vals[2 * k + 1]
                    try:
                        m.on_pool(float(base["data"]["parsed"]["info"]["tokenAmount"]["uiAmount"]), float(quote["data"]["parsed"]["info"]["tokenAmount"]["uiAmount"]))
                    except (TypeError, KeyError, ValueError):
                        pass
        except Exception as e:  # noqa: BLE001
            print("[lp] chain read:", repr(e), flush=True)


async def curve(mint: str) -> dict | None:
    """A coin's bonding curve on chain: reserves, whether it has graduated, and its creator (None if there's none)."""
    return parse_curve((await accounts([curve_address(mint)]))[0])


def curve_address(mint: str) -> str:
    return find_program_address([b"bonding-curve", b58decode(mint)], PUMP_PROGRAM)


class PumpFunMarket(Market):
    kind = "pump"

    def __init__(self, mint: str, fee_to_compute: bool = False):
        super().__init__()
        self.mint, self.fee_to_compute = mint, fee_to_compute
        self.vsol, self.vtok, self.mcap = 0.0, 0.0, 0.0
        self.complete = False
        self.pool_tas = None   # (base, quote) token accounts of its PumpSwap pool, once it has graduated

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

    def on_curve(self, c: dict):
        """Its bonding curve as read from the chain (exact, once a minute)."""
        if not self.complete:
            self.vsol, self.vtok = c["vsol"] or self.vsol, c["vtok"] or self.vtok
        self.complete = self.complete or c["complete"]

    def on_pool(self, base: float, quote: float):
        """A graduated coin's PumpSwap reserves (tokens, SOL): its price; a change since the last read counts as a trade."""
        if base <= 0 or quote <= 0:
            return
        now, prev = time.time(), self.vsol
        self.vsol, self.vtok = quote, base
        self.mcap = self.price() * self.supply()
        if prev and abs(quote - prev) > 1e-6:
            sol = abs(quote - prev)
            if self.fee_to_compute:
                self.fees_sol += sol * AMM_CREATOR_FEE_BPS / 1e4
            self.history.append((now, self.price()))
            self.note_trade({"t": now, "side": "buy" if quote > prev else "sell", "sol": round(sol, 4), "price": self.price(), "who": ""})
        elif not self.history or now - self.history[-1][0] > 5:
            self.history.append((now, self.price()))

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
        """The coin's state before any trade arrives: its bonding curve, read on chain (best effort)."""
        try:
            c = await curve(self.mint)
            if c:
                self.vsol, self.vtok, self.complete = c["vsol"], c["vtok"], c["complete"]
                self.mcap = self.price() * self.supply()
                self.history.append((time.time(), self.price()))
                return c
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
                        if m.get("message") and not m.get("mint"):   # (PumpPortal says why: e.g. trades now need a funded API key)
                            print(f"[feed] PumpPortal: {str(m['message'])[:200]}", flush=True)
                        mk = self.markets.get(m.get("mint"))
                        if mk and m.get("txType") in ("buy", "sell"):
                            mk.on_trade(m)
            except Exception as e:  # noqa: BLE001
                print("[lp] pumpportal:", repr(e), flush=True)
            self.ws, self.connected = None, False
            await asyncio.sleep(5)


FEED = PumpFeed()


SOL_USD = {"usd": None, "t": 0.0}   # SOL's dollar price (pages show market caps in $), refreshed every minute
PRICE_SOURCES = [
    ("https://lite-api.jup.ag/price/v3?ids=So11111111111111111111111111111111111111112", lambda j: j["So11111111111111111111111111111111111111112"]["usdPrice"]),
    ("https://api.coinbase.com/v2/prices/SOL-USD/spot", lambda j: j["data"]["amount"]),
    ("https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd", lambda j: j["solana"]["usd"]),
]


async def sol_price_loop():
    while True:
        async with httpx.AsyncClient(timeout=10) as cl:
            for url, pick in PRICE_SOURCES:
                try:
                    v = float(pick((await cl.get(url)).json()))
                    if v > 0:
                        SOL_USD.update(usd=round(v, 2), t=time.time())
                        break
                except Exception:  # noqa: BLE001
                    continue
        await asyncio.sleep(60)


def live_url(mint: str) -> str:
    """A launched coin's website on pump.fun: its android's live stream (always; the creator can't point it elsewhere)."""
    return f"{SITE}/coin?mint={mint}"


async def prepare(f: dict) -> dict:
    """Upload the coin's image + metadata to pump.fun -> its metadata uri, and who its creator will be."""
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
    st = settings()
    return {"uri": r.json()["metadataUri"], "creator": st["treasury"] or f["creator"], "alt": st["alt"] or None}


async def confirmed(signature: str, tries: int = 20) -> bool:
    async with httpx.AsyncClient(timeout=10) as cl:
        for _ in range(tries):
            r = await cl.post(RPC, json={"jsonrpc": "2.0", "id": 1, "method": "getSignatureStatuses", "params": [[signature], {"searchTransactionHistory": True}]})
            st = (r.json().get("result") or {}).get("value", [None])[0]
            if st and st.get("confirmationStatus") in ("confirmed", "finalized") and not st.get("err"):
                return True
            await asyncio.sleep(2)
    return False


async def fees_ours(mint: str) -> tuple[bool, str]:
    """Whether a coin's creator fees come to the treasury (its on-chain creator); with no treasury set, any coin counts."""
    t = settings()["treasury"]
    if not t:
        return True, ""
    c = await curve(mint)
    if not c:
        return False, "That isn't a pump.fun coin (no bonding curve on chain)."
    if c["creator"] != t:
        return False, "Only coins launched here get an android: this coin's creator fees go elsewhere."
    return True, ""
