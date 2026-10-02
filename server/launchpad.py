"""The launchpad: every coin has a live android.

    uvicorn launchpad:app --host 127.0.0.1 --port 8770      (in btm-server, GPU; the web app: web/launchpad)

A creator launches a coin and chooses its android: a character (persona prompt), what's under its mask (a custom
steering direction built from their concept and example sentences), a temperament (feelings it drifts back to), a
look, and a model. Its trading fees pay for its compute: while funded it is awake and talks live; when the money runs
out it falls asleep, and a trade wakes it.

All awake androids share one model (multimind.MultiMind: one row each, one token per row per step). Each android's
push is mixed every half second from
    its temperament  +  the chart (its coin's market as feelings, lp_market.mood)
    +  the crowd's taps (everyone, holders only, or holders weighted by holdings: the creator's choice)
    +  its concept (always on, at the creator's strength)
Visitors can also ask it questions (one global line per coin).
"""
from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import re
import secrets
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from lp_market import SimMarket
import lp_pump
from lp_pump import PumpFunMarket, FEED
import lp_wallet

STATE = Path(os.environ.get("LP_STATE", "runs/launchpad"))
DIRS = Path(os.environ.get("BTM_DIRS", "runs/q9b"))
CHAT = os.environ.get("BTM_CHAT", "Qwen/Qwen3.5-9B")
MAX_AWAKE = int(os.environ.get("LP_MAX_AWAKE", "8"))
PUSH_BUDGET = float(os.environ.get("LP_PUSH_BUDGET", "0.75"))
# trading keeps an android awake: its coin's volume credits compute at the creator-fee rate (the platform pays the GPU)
VOLUME_CREDIT = os.environ.get("LP_VOLUME_CREDIT", "1") == "1"
FLAGSHIP_MINT = os.environ.get("LP_FLAGSHIP_MINT", "").strip()            # the platform android's own coin, once launched
LAUNCHES_PER_HOUR = int(os.environ.get("LP_LAUNCHES_PER_HOUR", "30"))    # across the site
LAUNCHES_PER_SOURCE = int(os.environ.get("LP_LAUNCHES_PER_SOURCE", "3"))  # per wallet / address per hour
MODERATION = os.environ.get("LP_MODERATION", "1") == "1"
# demo coins (a simulated market, for trying an android out) only from the private test site / this machine
DEMO_ORIGINS = [o for o in os.environ.get("LP_DEMO_ORIGINS", "http://100.97.32.64:4360,http://127.0.0.1:5197,http://127.0.0.1:4361").split(",") if o]  # the whole steering mix, at most (units of residual norm)            # androids talking at once on this machine
STEP_RATE = float(os.environ.get("LP_STEP_RATE", "6"))           # tokens per second per android (at most)
COST_SOL_HOUR = float(os.environ.get("LP_COST_SOL_HOUR", "0.02"))  # what an awake hour of compute costs
FREE_SECONDS = float(os.environ.get("LP_FREE_SECONDS", "1200"))   # a new coin's first awake time
CTX_LIMIT = int(os.environ.get("LP_CTX_LIMIT", "5000"))         # an android's memory before it starts afresh
MAX_NEW, SOFT_NEW = 320, 200   # a thought: it ends itself, or at the first sentence end past SOFT_NEW tokens
HALF_LIFE, SAT, VISITOR_CAP = 5.0, 3.0, 4.0                      # taps: decay, saturation, one visitor's cap
QUESTION_MAX, QUESTION_GAP, QUESTION_LEN = 10, 30.0, 200
EMOTIONS = ["joy", "sadness", "anger", "fear", "calm", "curiosity"]
BUTTONS = EMOTIONS + ["unmask", "concept"]
SKINS = ["porcelain", "chrome", "matte", "glass"]
MARKS = ["none", "circuit", "claws", "tears", "split", "stardust", "kintsugi", "tally"]   # (older coins' "scar" is still drawn)
STEER_MODES = ["everyone", "holders", "weighted"]
MODELS = [{"id": "qwen3.5-9b", "name": "Qwen3.5-9B", "repo": "Qwen/Qwen3.5-9B", "status": "live"}]
TEMPERAMENTS = {   # presets the launch page offers (any mix is allowed)
    "melancholy": {"sadness": 0.6, "calm": 0.2}, "angry": {"anger": 0.65}, "nervous": {"fear": 0.55, "curiosity": 0.15},
    "joyful": {"joy": 0.6}, "serene": {"calm": 0.65}, "curious": {"curiosity": 0.6, "joy": 0.15}, "even": {},
}
BLOCK = re.compile(r"\b(n[i1]gg(a|er)s?|f[a4]gg?(ot)?s?|k[i1]kes?|ch[i1]nks?|sp[i1]cs?|tr[a4]nn(y|ies)|retards?|hitler|nazis?|rape|pedo\w*|child\s*porn)\b", re.I)
TURN = "\n<|im_start|>user\n{nudge}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"   # Qwen3.5's chat template
FOLLOWS = ["Keep talking to them.", "Tell them more about that.", "What else is on your mind right now?",
           "Tell them something about yourself they wouldn't guess.", "Go deeper into that.", "What happened next?",
           "Tell them a story from your past.", "Ask them something, and tell them why you want to know."]

LEVELS = json.loads((DIRS / "levels.json").read_text()) if (DIRS / "levels.json").exists() else {}
def lv(e, k, d=0.5):
    return float(LEVELS.get(e, {}).get(k, d))


def people(n: int) -> str:
    return "Nobody is watching you right now." if n <= 0 else ("1 person is watching you right now." if n == 1 else f"{n} people are watching you right now.")


def clean(text, n, what) -> str:
    t = re.sub(r"[\x00-\x1f\x7f]", " ", str(text or "")).strip()
    t = re.sub(r"\s+", " ", t)[:n]
    if BLOCK.search(t):
        raise HTTPException(400, f"Let's keep the {what} kind.")
    return t


# ---------------------------------------------------------------------------------------------- coins
class Coin:
    """A launched coin and its android. `d` is what's saved; the rest lives while the server runs."""

    def __init__(self, d: dict):
        self.d = d
        if d.get("market") == "pump" and d.get("mint"):
            self.market = PumpFunMarket(d["mint"], fee_to_compute=bool(d.get("fee_to_compute")) or VOLUME_CREDIT)
            FEED.add(self.market)
        else:
            # (the platform's own android has no demo traders: its mood is its temperament and the crowd)
            self.market = SimMarket(seed=hash(d["id"]) & 0xFFFF, activity=0.0 if d.get("featured") else d.get("sim_activity", 0.12))
        if d.get("market_state"):
            self.market.load_state(d["market_state"])
        self.clients: dict = {}
        self.taps: dict = {}             # cid -> {button: decayed energy}, t
        self.questions: list = []
        self.asking: dict | None = None
        self.status = "asleep"           # asleep | waking | awake | thinking | losing it
        self.awake_since = 0.0
        self.turn: dict | None = None    # the reply being written: {id, tokens, asked, start}
        self.line = ""                   # the last words it said (explore cards)
        self.e_last = None               # its last readout (6 feelings + concept)
        self.steer: dict = {}
        self.follow_i = 0
        self.last_price_note = self.market.price()
        self.short = 0
        self.history: list = []          # (nudge, reply) of this conversation (for a fresh start)
        self.ready = d.get("concept_ready", False) or not d.get("concept")
        self.holdings: dict = {}         # sim market: who holds how much (cid -> SOL in)

    @property
    def id(self):
        return self.d["id"]

    def balance(self) -> float:
        return self.d["ledger"]["grant_sol"] + self.market.fees_sol - self.d["ledger"]["spent_sol"]

    def time_left(self) -> float:
        if self.d.get("featured"):
            return float("inf")
        return max(0.0, self.balance()) / (COST_SOL_HOUR / 3600)

    def concept_label(self) -> str | None:
        return f"c:{self.id}" if self.d.get("concept") else None

    def summary(self) -> dict:
        m = self.market.summary()
        tl = self.time_left()
        return {"id": self.id, "name": self.d["name"], "ticker": self.d["ticker"], "look": self.d["look"],
                "status": self.status, "line": self.line[-140:], "e": self.e_last, "mu": self.d.get("mu"),
                "price": m["price"], "mcap_sol": m["mcap_sol"], "change_5m": m["change_5m"], "change_1h": m["change_1h"],
                "volume_sol": m["volume_sol"], "curve": m["curve"], "time_left": None if tl == float("inf") else round(tl),
                "featured": bool(self.d.get("featured")), "created": self.d["created"], "model": self.d["model"],
                "concept": (self.d.get("concept") or {}).get("name"), "temperament": self.d.get("temperament_name"),
                "viewers": len(self.clients), "market": self.market.kind, "img": f"api/img/{self.id}.png" if (STATE / "img" / f"{self.id}.png").exists() else None,
                "intensity": round(self.intensity(), 3)}

    def intensity(self) -> float:
        """How strongly it is being pushed right now (0..~1.5): the "most emotional" sort, "losing it"."""
        return sum(abs(v) for k, v in self.steer.items() if k in EMOTIONS)

    def save(self):
        self.d["market_state"] = self.market.state()
        (STATE / "coins").mkdir(parents=True, exist_ok=True)
        tmp = STATE / "coins" / f"{self.id}.json.tmp"
        tmp.write_text(json.dumps(self.d))
        tmp.replace(STATE / "coins" / f"{self.id}.json")

    # -------------------------------------------------------------- the crowd
    def tap(self, cid: str, b: str, weight: float):
        now = time.time()
        rec = self.taps.setdefault(cid, {"t": now, "e": {}, "w": weight})
        dec = 0.5 ** ((now - rec["t"]) / HALF_LIFE)
        rec["e"] = {k: v * dec for k, v in rec["e"].items()}
        rec["t"], rec["w"] = now, weight
        total = sum(rec["e"].values())
        if total < VISITOR_CAP:
            rec["e"][b] = rec["e"].get(b, 0.0) + 1.0

    def crowd_mix(self) -> dict:
        now, acc = time.time(), {b: 0.0 for b in BUTTONS}
        for cid, rec in list(self.taps.items()):
            dec = 0.5 ** ((now - rec["t"]) / HALF_LIFE)
            if dec < 0.01:
                del self.taps[cid]
                continue
            for b, v in rec["e"].items():
                acc[b] += v * dec * rec.get("w", 1.0)
        return {b: 1 - math.exp(-v / SAT) for b, v in acc.items()}   # 0..1 each

    def tap_weight(self, cid: str, wallet: str | None) -> float:
        mode = self.d.get("steer_mode", "everyone")
        if mode == "everyone":
            return 1.0
        held = self.holdings.get(wallet or cid, 0.0)
        if held <= 0:
            return 0.0
        if mode == "holders":
            return 1.0
        # weighted: by the share held, softened (sqrt) and capped -- real coins: share of supply; demo: of what's been bought
        share = held / (self.market.supply() if self.market.kind == "pump" else (sum(self.holdings.values()) or 1.0))
        return min(3.0, 0.5 + math.sqrt(share * (1000 if self.market.kind == "pump" else 4)))

    # -------------------------------------------------------------- the push
    def mix(self) -> dict:
        """Its steering right now: temperament + the chart + the crowd + its concept (units of residual norm)."""
        steer = {e: 0.0 for e in EMOTIONS}
        for e, w in (self.d.get("temperament") or {}).items():
            if e in steer:
                steer[e] += float(w) * lv(e, "little", 0.3)
        # its chart is its mood -- once it has a real coin (or a demo market); before that, temperament and the crowd
        mood = self.market.mood() if (self.market.kind != "sim" or not self.d.get("featured")) else {}
        for e, m in mood.items():
            steer[e] += m * lv(e, "mid1", 0.4)
        crowd = self.crowd_mix()
        for e in EMOTIONS:
            steer[e] += crowd[e] * lv(e, "lot", 0.5)
        for e in EMOTIONS:
            steer[e] = min(steer[e], lv(e, "toomuch", 0.9))
        steer["assistant"] = -crowd["unmask"] * lv("assistant", "lot", 0.5)
        c = self.d.get("concept")
        if c:   # the concept's base push is gentler than its slider (it adds to everything else); the crowd can raise it
            steer[self.concept_label()] = 0.7 * float(c.get("strength", 0.5)) * (1 + 0.6 * crowd["concept"])
        # one budget for everything together: past a strong single push (0.75 of the residual norm) the text
        # degenerates into loops, so the whole mix is scaled down to fit
        size = math.sqrt(sum(v * v for v in steer.values()))
        if size > PUSH_BUDGET:
            steer = {k: v * PUSH_BUDGET / size for k, v in steer.items()}
        return steer


COINS: dict[str, Coin] = {}
ENGINE = {"mm": None, "chat": None, "loading": True, "error": None, "jobs": []}
LOOP: asyncio.AbstractEventLoop | None = None
EXPLORE: dict = {}       # ws -> whether it's a demo page (lists demo coins too)
STOP = threading.Event()


def load_coins():
    d = STATE / "coins"
    d.mkdir(parents=True, exist_ok=True)
    for f in sorted(d.glob("*.json")):
        try:
            c = Coin(json.loads(f.read_text()))
            COINS[c.id] = c
        except Exception as e:  # noqa: BLE001
            print("[lp] bad coin file", f, e, flush=True)
    if os.environ.get("LP_FLAGSHIP", "0") == "1" and not any(c.d.get("featured") for c in COINS.values()):   # (the home page's android is the exhibit's live AI)
        flagship = new_coin_doc({
            "name": "Steer AI", "ticker": "STEER", "featured": True, "temperament": {}, "temperament_name": "even",
            "persona": ("You are Qwen3.5-9B, an AI language model, on a live stream. Talk about yourself: what you are, "
                        "what it's like to be you, and the people watching you."),
            "look": {"skin": "porcelain", "eye": "#7fe7ff", "marks": "none"}, "concept": None, "steer_mode": "everyone",
        })
        COINS[flagship["id"]] = Coin(flagship)
        COINS[flagship["id"]].save()
    f = next((c for c in COINS.values() if c.d.get("featured")), None)
    if f and FLAGSHIP_MINT and f.d.get("mint") != FLAGSHIP_MINT:   # its real coin: from now on its chart is its mood
        f.d.update(mint=FLAGSHIP_MINT, market="pump")
        f.d.pop("market_state", None)
        COINS[f.id] = Coin(f.d)
        COINS[f.id].save()


def new_coin_doc(f: dict) -> dict:
    cid = re.sub(r"[^a-z0-9]", "", f["ticker"].lower())[:10] + "-" + secrets.token_hex(3)
    return {"id": cid, "name": f["name"], "ticker": f["ticker"].upper(), "persona": f["persona"], "concept": f.get("concept"),
            "temperament": f.get("temperament") or {}, "temperament_name": f.get("temperament_name"), "look": f["look"],
            "model": f.get("model", "qwen3.5-9b"), "steer_mode": f.get("steer_mode", "everyone"), "created": time.time(),
            "creator": f.get("creator"), "featured": bool(f.get("featured")), "mint": f.get("mint"),
            "market": f.get("market", "sim"), "fee_to_compute": bool(f.get("fee_to_compute")),
            "ledger": {"grant_sol": FREE_SECONDS * COST_SOL_HOUR / 3600, "spent_sol": 0.0}, "mu": None, "concept_ready": False}


# ---------------------------------------------------------------------------------------------- the engine
def load_engine():
    from steer import load_mind
    from multimind import MultiMind
    chat = load_mind("chat", CHAT, True)
    chat.load(DIRS / "chat_dirs.pt")
    mm = MultiMind(chat, step_rate=STEP_RATE)
    ENGINE.update(chat=chat, mm=mm)
    for c in COINS.values():      # concept directions already built (saved with the coin)
        cpath = STATE / "concepts" / f"{c.id}.pt"
        if c.d.get("concept") and cpath.exists():
            import torch
            v = torch.load(cpath)
            mm.add_direction(c.concept_label(), v["direction"], v["mu"], v["sd"])
            c.ready = True
    threading.Thread(target=engine_loop, daemon=True).start()
    ENGINE["loading"] = False
    print(f"[lp] ready: {CHAT}; {len(COINS)} coins", flush=True)


def engine_loop():
    mm = ENGINE["mm"]
    while not STOP.is_set():
        while ENGINE["jobs"]:          # jobs that need the model (concept directions, fresh starts): between steps
            fn = ENGINE["jobs"].pop(0)
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                print("[lp] job failed:", repr(e), flush=True)
        t0 = time.time()
        try:
            busy = mm.step()
        except Exception as e:  # noqa: BLE001
            print("[lp] step failed:", repr(e), flush=True)
            busy = False
            time.sleep(0.5)
        dt = time.time() - t0
        if busy:
            mm.t_step = 0.9 * mm.t_step + 0.1 * dt if mm.t_step else dt
        time.sleep(max(0.0 if busy else 0.05, 1.0 / STEP_RATE - dt))


def job(fn):
    """Run fn on the engine thread between steps; returns an awaitable."""
    fut = LOOP.create_future()
    def run():
        try:
            r = fn()
            LOOP.call_soon_threadsafe(fut.set_result, r)
        except Exception as e:  # noqa: BLE001
            LOOP.call_soon_threadsafe(fut.set_exception, e)
    ENGINE["jobs"].append(run)
    return fut


def prompt_ids(c: Coin, nudge: str, history: list | None = None) -> list[int]:
    chat = ENGINE["chat"]
    t = c.d["ticker"]
    system = (f"You are {c.d['name']}. {c.d['persona']}\n"
              f"You are performing live as an android, on a stream: people are watching your face while you talk. You "
              f"exist because of a coin, ${t}: people trading it pays for your mind, and when nobody does, you fall asleep. "
              f"Speak in the first person, as {c.d['name']}, in a natural spoken voice, one complete thought at a time, and "
              f"finish the thought before you stop. Stay in character. No lists, no headings, no emoji.")
    msgs = [{"role": "system", "content": system}]
    for asked, reply in (history or []):
        msgs += [{"role": "user", "content": asked}, {"role": "assistant", "content": reply}]
    msgs.append({"role": "user", "content": nudge})
    s = chat.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return chat.tokenizer(s, add_special_tokens=False)["input_ids"]


async def wake(c: Coin, startle=False):
    from multimind import Row
    if c.status != "asleep" or not c.ready or ENGINE["mm"] is None:
        return
    c.status, c.awake_since = "waking", time.time()
    opening = f"{people(len(c.clients))} Introduce yourself to them, then talk about whatever is on your mind."
    nudge = opening if not c.history else FOLLOWS[0]
    ids = prompt_ids(c, (f"{people(len(c.clients))} " if c.history else "") + nudge, c.history[-2:])
    row = Row(key=c.id, max_new=MAX_NEW, soft=SOFT_NEW)
    row.steer = {} if c.d.get("mu") is None else c.mix()   # the very first reply is unpushed: its readout's baseline
    row.on_token = lambda r, tok, c=c: LOOP.call_soon_threadsafe(on_token, c, tok)
    row.on_end = lambda r, text, c=c: LOOP.call_soon_threadsafe(on_end, c, text)
    c.turn = {"id": secrets.token_hex(4), "tokens": [], "asked": None, "start": time.time(), "nudge": nudge}
    await job(lambda: ENGINE["mm"].add(row, ids))
    c.status = "awake"
    await broadcast_coin(c, {"type": "wake", "startle": startle})
    await broadcast_coin(c, {"type": "begin", "id": c.turn["id"], "asked": None})


async def sleep(c: Coin):
    if c.status == "asleep":
        return
    c.status = "asleep"
    await job(lambda: ENGINE["mm"].remove(c.id))
    if c.turn and c.turn["tokens"]:
        log_turn(c)
    c.turn = None
    await broadcast_coin(c, {"type": "sleep"})


def on_token(c: Coin, tok: dict):
    if c.turn is None:
        return
    labels = ENGINE["chat"].labels
    sel = [labels.index(e) for e in EMOTIONS]
    cl = c.concept_label()
    e = [tok["e"][i] for i in sel] + [tok["e"][labels.index(cl)] if cl and cl in labels else 0.0]
    m = labels.index("assistant") if "assistant" in labels else None
    w = {"t": tok["t"], "e": [round(x, 2) for x in e], "p": tok["p"], "a": tok["a"][:4],
         "s": [round(c.steer.get(x, 0.0), 2) for x in EMOTIONS] + [round(c.steer.get(cl, 0.0), 2) if cl else 0.0],
         "m": round(tok["e"][m], 2) if m is not None else None, "at": round(time.time(), 2)}
    c.turn["tokens"].append(w)
    c.e_last = w["e"]
    c.line = (c.line + tok["t"])[-400:]
    c.pending = getattr(c, "pending", []) + [w]


def on_end(c: Coin, text: str):
    asyncio.ensure_future(end_turn(c, text))


async def end_turn(c: Coin, text: str):
    t = c.turn
    if t is None:
        return
    await flush_words(c)
    await broadcast_coin(c, {"type": "end", "id": t["id"]})
    toks = t["tokens"]
    if c.d.get("mu") is None and len(toks) >= 12:      # the first, unpushed reply sets its baseline
        c.d["mu"] = [round(sum(w["e"][k] for w in toks) / len(toks), 3) for k in range(7)]
        c.save()
    log_turn(c)
    words = len(text.split())
    if words >= 8:
        c.history.append((t.get("nudge", ""), text))
        c.history = c.history[-6:]
        c.short = 0
    else:
        c.short += 1
    c.asking = None
    if c.status == "asleep":
        return
    c.status = "thinking"
    await asyncio.sleep(0.8)
    if c.status == "asleep":
        return
    await next_turn(c)


async def next_turn(c: Coin):
    mm = ENGINE["mm"]
    row = mm.row(c.id)
    if row is None:
        return
    n = len(c.clients)
    # what comes next: a visitor's question, a word on the chart if it moved a lot, or more of the same thought
    p, p0 = c.market.price(), c.last_price_note
    move = (p - p0) / p0 if p0 > 0 else 0.0
    if c.questions:
        c.asking = c.questions.pop(0)
        nudge = f"{people(n)} One of them asks you: \"{c.asking['q']}\" Answer them in character, and finish your answer before you stop."
    elif abs(move) > 0.15:
        c.last_price_note = p
        nudge = (f"{people(n)} Your coin, ${c.d['ticker']}, just {'jumped' if move > 0 else 'fell'} {abs(move) * 100:.0f}% "
                 f"in the last few minutes. Keep talking to them.")
    else:
        nudge = f"{people(n)} {FOLLOWS[c.follow_i % len(FOLLOWS)]}"
        c.follow_i += 1
    if c.short >= 2 or row.pos > CTX_LIMIT:
        # memory full: a fresh start with the last two exchanges as context; replies collapsing: a clean start
        collapsed, c.short = c.short >= 2, 0
        await job(lambda: mm.remove(c.id))
        c.status = "asleep"
        c.history = [] if collapsed else c.history[-2:]
        await wake(c)
        return
    c.turn = {"id": secrets.token_hex(4), "tokens": [], "asked": c.asking["q"] if c.asking else None, "start": time.time(), "nudge": nudge}
    ids = ENGINE["chat"].tokenizer(TURN.format(nudge=nudge), add_special_tokens=False)["input_ids"]
    ok = await job(lambda: mm.say(c.id, ids))
    if ok:
        c.status = "awake"
        await broadcast_coin(c, {"type": "begin", "id": c.turn["id"], "asked": c.turn["asked"]})


def log_turn(c: Coin):
    t = c.turn
    if not t or not t["tokens"]:
        return
    (STATE / "logs").mkdir(parents=True, exist_ok=True)
    rec = {"coin": c.id, "turn": t["id"], "start": t["start"], "end": time.time(), "asked": t.get("asked"),
           "mu": c.d.get("mu"), "tokens": t["tokens"]}
    with open(STATE / "logs" / f"{c.id}.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------------------------- the clock
async def heartbeat():
    """Markets, funding, who is awake, everyone's push; twice a second."""
    last_save = time.time()
    while True:
        await asyncio.sleep(0.5)
        if ENGINE["mm"] is None:
            continue
        now = time.time()
        for c in list(COINS.values()):
            trades = c.market.tick(now)
            for tr in trades:
                await broadcast_coin(c, {"type": "trade", **{k: tr[k] for k in ("side", "sol", "price", "t")}})
            if c.status != "asleep" and not c.d.get("featured"):
                c.d["ledger"]["spent_sol"] += COST_SOL_HOUR / 3600 * 0.5
            c.steer = c.mix() if c.d.get("mu") is not None else {}
            row = ENGINE["mm"].row(c.id)
            if row is not None:
                row.steer = dict(c.steer)
            if c.status in ("awake", "thinking"):
                big = max([v for k, v in c.steer.items() if k in EMOTIONS] or [0])
                if big > 0.72 and c.status == "awake":
                    c.status = "losing it"
            elif c.status == "losing it" and c.intensity() < 0.6:
                c.status = "awake"
            await flush_words(c)
        await schedule()
        if now - last_save > 30:
            last_save = now
            for c in COINS.values():
                c.save()


async def schedule():
    """Who is awake: the funded coins, the featured first, then by recent volume; at most MAX_AWAKE."""
    funded = [c for c in COINS.values() if c.ready and (c.d.get("featured") or c.balance() > 0)]
    funded.sort(key=lambda c: (not c.d.get("featured"), -c.market.volume_sol, -c.balance()))
    want = {c.id for c in funded[:MAX_AWAKE]}
    for c in list(COINS.values()):
        if c.status != "asleep" and c.id not in want:
            await sleep(c)
    for c in funded[:MAX_AWAKE]:
        if c.status == "asleep":
            await wake(c, startle=bool(getattr(c, "woken_by_trade", False)))
            c.woken_by_trade = False


# ---------------------------------------------------------------------------------------------- talking to pages
async def send(ws: WebSocket, msg: dict):
    try:
        await ws.send_text(json.dumps(msg))
    except Exception:  # noqa: BLE001
        pass


async def broadcast_coin(c: Coin, msg: dict):
    data = json.dumps(msg)
    for ws in list(c.clients):
        try:
            await ws.send_text(data)
        except Exception:  # noqa: BLE001
            c.clients.pop(ws, None)


async def flush_words(c: Coin):
    p = getattr(c, "pending", [])
    if p and c.turn:
        c.pending = []
        await broadcast_coin(c, {"type": "w", "id": c.turn["id"], "w": p})


def coin_state(c: Coin, cid: str | None = None) -> dict:
    s = c.summary()
    mix = c.crowd_mix()
    s.update(type="state", mix={b: round(v, 3) for b, v in mix.items()}, steer={k: round(v, 3) for k, v in c.steer.items() if k in EMOTIONS},
             mood={k: round(v, 3) for k, v in c.market.mood().items()}, qn=len(c.questions), qs=[q["q"][:90] for q in c.questions[:3]],
             qids=[q["id"] for q in c.questions], asking=c.asking["q"] if c.asking else None, asking_id=c.asking["id"] if c.asking else None,
             persona=c.d["persona"], steer_mode=c.d.get("steer_mode"), concept_full=c.d.get("concept"), temperament_mix=c.d.get("temperament"),
             balance_sol=round(c.balance(), 6), fees_sol=round(c.market.fees_sol, 6))
    return s


async def explore_loop():
    while True:
        await asyncio.sleep(1.0)
        if not EXPLORE:
            continue
        data = {d: json.dumps({"type": "coins", "coins": [c.summary() for c in listed(d)], "stats": stats(d)}) for d in set(EXPLORE.values())}
        for ws, d in list(EXPLORE.items()):
            try:
                await ws.send_text(data[d])
            except Exception:  # noqa: BLE001
                EXPLORE.pop(ws, None)


async def state_loop():
    while True:
        await asyncio.sleep(1.0)
        for c in COINS.values():
            if c.clients:
                await broadcast_coin(c, coin_state(c))


def listed(demo: bool = False) -> list:
    """The coins a page lists: real ones; demo coins (simulated markets) only on demo pages."""
    return [c for c in COINS.values() if demo or c.market.kind != "sim"]


def stats(demo: bool = False) -> dict:
    cs = listed(demo)
    return {"coins": len(cs), "awake": sum(c.status != "asleep" for c in cs), "compute_sol": round(sum(c.market.fees_sol for c in cs), 4),
            "volume_sol": round(sum(c.market.volume_sol for c in cs), 3), "loading": ENGINE["loading"]}


# ---------------------------------------------------------------------------------------------- the app
app = FastAPI()
ORIGINS = [o for o in os.environ.get("LP_ORIGINS", "https://steerai.live,https://www.steerai.live,http://steerai.live,http://www.steerai.live,"
                                    "https://patanl.github.io,https://spark-3a11.tail621a3a.ts.net,http://100.97.32.64:4360,http://127.0.0.1:5197").split(",") if o]
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST"], allow_headers=["content-type"])


@app.on_event("startup")
async def startup():
    global LOOP
    LOOP = asyncio.get_running_loop()
    load_coins()
    async def boot():
        try:
            await LOOP.run_in_executor(None, load_engine)
        except Exception as e:  # noqa: BLE001
            ENGINE["error"] = repr(e)
            print("[lp] engine failed:", repr(e), flush=True)
    asyncio.ensure_future(boot())
    asyncio.ensure_future(heartbeat())
    asyncio.ensure_future(FEED.run())
    for c in COINS.values():
        if c.market.kind == "pump" and not c.market.history:
            asyncio.ensure_future(c.market.seed())
    asyncio.ensure_future(explore_loop())
    asyncio.ensure_future(state_loop())


@app.get("/api/status")
def api_status():
    return {"ready": not ENGINE["loading"] and ENGINE["error"] is None, "error": ENGINE["error"], **stats(),
            "step_ms": round(ENGINE["mm"].t_step * 1000) if ENGINE["mm"] and ENGINE["mm"].t_step else None}


@app.get("/api/coins")
def api_coins(sort: str = "new", demo: int = 0):
    cs = [c.summary() for c in listed(bool(demo))]
    key = {"new": lambda s: -s["created"], "mcap": lambda s: -s["mcap_sol"], "emotional": lambda s: -s["intensity"],
           "volume": lambda s: -s["volume_sol"]}.get(sort, lambda s: -s["created"])
    cs.sort(key=lambda s: (not s["featured"], key(s)))
    return {"coins": cs, "stats": stats(bool(demo))}


def find(cid: str):
    """A coin by its id, or by its mint (a launched coin's pump.fun website links to coin.html?mint=...)."""
    return COINS.get(cid) or next((x for x in COINS.values() if x.d.get("mint") and x.d["mint"] == cid), None)


@app.get("/api/coins/{cid}")
def api_coin(cid: str):
    c = find(cid)
    if not c:
        raise HTTPException(404, "No such coin.")
    s = coin_state(c)
    s["series"] = c.market.series()
    s["trades"] = list(c.market.trades)[-30:]
    return s


@app.get("/api/meta")
def api_meta():
    return {"models": MODELS, "skins": SKINS, "marks": MARKS, "temperaments": TEMPERAMENTS, "steer_modes": STEER_MODES,
            "cost_sol_hour": COST_SOL_HOUR, "free_seconds": FREE_SECONDS}


LAUNCH_LOG: list = []   # (time, source)


def launch_source(f: dict, req: Request) -> str:
    return str(f.get("creator") or req.headers.get("x-forwarded-for", "").split(",")[0].strip() or (req.client.host if req.client else "?"))


async def gate(f: dict, req: Request, name: str, persona: str, concept: str):
    """Before anything is created (on pump.fun or here): the launch rate limits, then the character's moderation."""
    now, src = time.time(), launch_source(f, req)
    LAUNCH_LOG[:] = [x for x in LAUNCH_LOG if now - x[0] < 3600]
    if len(LAUNCH_LOG) >= LAUNCHES_PER_HOUR or sum(1 for _, s_ in LAUNCH_LOG if s_ == src) >= LAUNCHES_PER_SOURCE:
        raise HTTPException(429, "Lots of launches right now. Try again in a little while.")
    if MODERATION and ENGINE["mm"] is not None:
        from lp_moderate import flagged
        p_bad = await job(lambda: flagged(ENGINE["chat"], name, persona, concept))
        if p_bad > 0.5:
            raise HTTPException(400, "That character can't go live here. Try a different one.")


@app.post("/api/coins")
async def api_create(req: Request):
    f = await req.json()
    now, src = time.time(), launch_source(f, req)
    name = clean(f.get("name"), 32, "name")
    ticker = re.sub(r"[^A-Za-z0-9]", "", str(f.get("ticker") or ""))[:10].upper()
    persona = clean(f.get("persona"), 600, "character")
    if len(name) < 2 or len(ticker) < 2 or len(persona) < 12:
        raise HTTPException(400, "A name, a ticker and a character (a sentence or two) are needed.")
    if any(c.d["ticker"] == ticker for c in COINS.values()):
        raise HTTPException(400, f"${ticker} is taken.")
    concept = None
    cf = f.get("concept") or {}
    if cf.get("name"):
        concept = {"name": clean(cf["name"], 60, "concept"), "examples": [clean(x, 200, "concept") for x in (cf.get("examples") or [])[:5] if str(x).strip()],
                   "strength": max(0.15, min(0.9, float(cf.get("strength", 0.5))))}
    temp = {e: max(0.0, min(1.0, float(v))) for e, v in (f.get("temperament") or {}).items() if e in EMOTIONS}
    look = f.get("look") or {}
    hexc = lambda v, d: v if re.fullmatch(r"#[0-9a-fA-F]{6}", str(v or "")) else d
    look = {"skin": look.get("skin") if look.get("skin") in SKINS else "porcelain", "marks": look.get("marks") if look.get("marks") in MARKS else "none",
            "eye": hexc(look.get("eye"), "#7fe7ff"), "color": hexc(look.get("color"), ""), "mark_color": hexc(look.get("mark_color"), ""),
            "eye_glow": bool(look.get("eye_glow"))}
    mode = f.get("steer_mode") if f.get("steer_mode") in STEER_MODES else "everyone"
    model = f.get("model") if any(m["id"] == f.get("model") and m["status"] == "live" for m in MODELS) else "qwen3.5-9b"
    market, mint = ("pump", str(f.get("mint") or "").strip()) if f.get("market") == "pair" else ("sim", None)
    if market == "sim" and req.headers.get("origin", "") not in DEMO_ORIGINS:
        raise HTTPException(400, "Coins launch on pump.fun.")
    if market == "pump":
        if not re.fullmatch(r"[1-9A-HJ-NP-Za-km-z]{32,44}", mint or ""):
            raise HTTPException(400, "That isn't a Solana mint address.")
        if any(c.d.get("mint") == mint for c in COINS.values()):
            raise HTTPException(400, "That coin already has an android.")
    if not (market == "pump" and mint in PREPARED):    # (a pump.fun launch was checked before its transaction)
        await gate(f, req, name, persona, (concept or {}).get("name", ""))
    LAUNCH_LOG.append((now, src))
    d = new_coin_doc({"name": name, "ticker": ticker, "persona": persona, "concept": concept, "temperament": temp,
                      "temperament_name": f.get("temperament_name"), "look": look, "steer_mode": mode, "model": model,
                      "creator": str(f.get("creator") or "")[:64] or None, "mint": mint, "market": market})
    c = Coin(d)
    COINS[c.id] = c
    img = str(f.get("image") or "")
    if img.startswith("data:image/png;base64,"):
        raw = base64.b64decode(img.split(",", 1)[1])[:600_000]
        (STATE / "img").mkdir(parents=True, exist_ok=True)
        (STATE / "img" / f"{c.id}.png").write_bytes(raw)
    c.save()
    if market == "pump":
        asyncio.ensure_future(c.market.seed())
    if concept:
        asyncio.ensure_future(build(c))
    return {"id": c.id, "ready": c.ready}


async def build(c: Coin):
    from lp_concepts import build_concept
    import torch
    while ENGINE["mm"] is None:
        await asyncio.sleep(1)
    cpt = c.d["concept"]
    t0 = time.time()
    v = await job(lambda: build_concept(ENGINE["chat"], cpt["name"], cpt["examples"]))
    (STATE / "concepts").mkdir(parents=True, exist_ok=True)
    torch.save({"direction": v["direction"].cpu(), "mu": v["mu"], "sd": v["sd"]}, STATE / "concepts" / f"{c.id}.pt")
    await job(lambda: ENGINE["mm"].add_direction(c.concept_label(), v["direction"], v["mu"], v["sd"]))
    c.ready, c.d["concept_ready"] = True, True
    c.save()
    print(f"[lp] concept for {c.id} ({cpt['name']!r}) built in {time.time() - t0:.1f} s", flush=True)


@app.post("/api/coins/{cid}/trade")
async def api_trade(cid: str, req: Request):
    """Demo coins: a simulated buy or sell from the page (real coins trade on chain)."""
    c = COINS.get(cid)
    if not c or c.market.kind != "sim":
        raise HTTPException(400, "This coin trades on chain.")
    f = await req.json()
    side = "buy" if f.get("side") == "buy" else "sell"
    sol = max(0.01, min(5.0, float(f.get("sol", 0.1))))
    who = str(f.get("cid") or "anon")[:32]
    was_broke = c.balance() <= 0
    tr = c.market.trade(side, sol, who)
    if not tr:
        raise HTTPException(400, "Nothing to sell.")
    c.holdings[who] = max(0.0, c.holdings.get(who, 0.0) + (sol if side == "buy" else -sol))
    if c.status == "asleep" and was_broke and c.balance() > 0:
        c.woken_by_trade = True
    await broadcast_coin(c, {"type": "trade", **{k: tr[k] for k in ("side", "sol", "price", "t")}})
    return {"ok": True, "price": tr["price"], "balance_sol": c.balance()}


@app.get("/api/pump/status")
def api_pump_status():
    return {"launch": lp_pump.PUMP_LAUNCH, "feed": FEED.connected, "paired": len(FEED.markets)}


PREPARED: set = set()    # mints whose launch passed the gate


@app.post("/api/pump/prepare")
async def api_pump_prepare(req: Request):
    f = await req.json()
    f["name"], f["ticker"] = clean(f.get("name"), 32, "name"), re.sub(r"[^A-Za-z0-9]", "", str(f.get("ticker") or ""))[:10].upper()
    persona = clean(f.get("persona"), 600, "character")
    if len(f["name"]) < 2 or len(f["ticker"]) < 2 or len(persona) < 12:
        raise HTTPException(400, "A name, a ticker and a character (a sentence or two) are needed.")
    if any(c.d["ticker"] == f["ticker"] for c in COINS.values()):
        raise HTTPException(400, f"${f['ticker']} is taken.")
    await gate(f, req, f["name"], persona, clean((f.get("concept") or {}).get("name", ""), 60, "concept"))
    if not re.fullmatch(r"[1-9A-HJ-NP-Za-km-z]{32,44}", str(f.get("creator") or "")) or not re.fullmatch(r"[1-9A-HJ-NP-Za-km-z]{32,44}", str(f.get("mint") or "")):
        raise HTTPException(400, "A wallet and a mint address are needed.")
    try:
        out = await lp_pump.prepare(f)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, str(e)) from e
    PREPARED.add(f["mint"])
    return out


@app.post("/api/pump/confirm")
async def api_pump_confirm(req: Request):
    f = await req.json()
    ok = await lp_pump.confirmed(str(f.get("signature") or ""))
    if not ok:
        raise HTTPException(400, "The launch didn't confirm on chain.")
    return {"ok": True}


@app.get("/api/img/{cid}.png")
def api_img(cid: str):
    p = STATE / "img" / f"{re.sub(r'[^a-z0-9-]', '', cid)}.png"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png")


@app.get("/api/logs/{cid}")
def api_logs(cid: str, turn: str | None = None):
    """A coin's recent turns."""
    p = STATE / "logs" / f"{re.sub(r'[^a-z0-9-]', '', cid)}.jsonl"
    if not p.exists():
        return {"turns": []}
    lines = p.read_text().splitlines()[-60:]
    turns = [json.loads(l) for l in lines]
    if turn:
        turns = [t for t in turns if t["turn"] == turn]
    return {"turns": turns}


@app.websocket("/api/ws/explore")
async def ws_explore(ws: WebSocket):
    await ws.accept()
    demo = ws.query_params.get("demo") == "1"
    EXPLORE[ws] = demo
    await send(ws, {"type": "coins", "coins": [c.summary() for c in listed(demo)], "stats": stats(demo)})
    try:
        while True:
            await ws.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        EXPLORE.pop(ws, None)


@app.websocket("/api/ws/coin/{cid}")
async def ws_coin(ws: WebSocket, cid: str):
    c = find(cid)
    await ws.accept()
    if not c:
        await send(ws, {"type": "error", "why": "No such coin."})
        await ws.close()
        return
    vid = (ws.query_params.get("cid") or secrets.token_hex(6))[:24]
    wallet, nonce = None, secrets.token_hex(8)
    c.clients[ws] = vid
    await send(ws, coin_state(c))
    await send(ws, {"type": "nonce", "message": lp_wallet.message("{address}", nonce)})
    await send(ws, {"type": "series", "series": c.market.series(), "trades": list(c.market.trades)[-30:]})
    if c.turn and c.status != "asleep":
        await send(ws, {"type": "begin", "id": c.turn["id"], "asked": c.turn.get("asked")})
        await send(ws, {"type": "w", "id": c.turn["id"], "w": c.turn["tokens"]})
    last_ask = -1e9
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            k = msg.get("type")
            if k == "taps" and isinstance(msg.get("c"), dict):
                w = c.tap_weight(vid, wallet)
                for b, n in msg["c"].items():
                    if b in BUTTONS and isinstance(n, int) and w > 0:
                        for _ in range(max(0, min(n, 8))):
                            c.tap(vid, b, w)
            elif k == "ask":
                q = clean_q(msg.get("q"))
                mine = next((x for x in c.questions + ([c.asking] if c.asking else []) if x["cid"] == vid), None)
                wait = QUESTION_GAP - (time.time() - last_ask)
                if isinstance(q, tuple):
                    await send(ws, {"type": "ask_err", "why": q[1]})
                elif mine:
                    await send(ws, {"type": "ask_mine", "id": mine["id"], "q": mine["q"], "wait": max(0.0, wait)})
                elif len(c.questions) >= QUESTION_MAX:
                    await send(ws, {"type": "ask_err", "why": f"The line is full ({QUESTION_MAX} waiting). Try again in a minute."})
                elif wait > 0:
                    await send(ws, {"type": "ask_err", "why": "wait", "wait": wait})
                else:
                    last_ask = time.time()
                    item = {"id": secrets.token_hex(4), "q": q, "cid": vid, "t": time.time()}
                    c.questions.append(item)
                    await send(ws, {"type": "ask_ok", "id": item["id"], "q": q, "pos": len(c.questions), "gap": QUESTION_GAP})
            elif k == "wallet":
                # a signed message proves the wallet; then its balance of this coin (real coins: from chain)
                addr, sig = str(msg.get("address") or "")[:64], str(msg.get("signature") or "")
                try:
                    ok = lp_wallet.ed25519_verify(lp_wallet.b58decode(addr), lp_wallet.message(addr, nonce).encode(), base64.b64decode(sig))
                except Exception:  # noqa: BLE001
                    ok = False
                if not ok:
                    await send(ws, {"type": "wallet_err", "why": "That signature didn't check out."})
                    continue
                wallet = addr
                if c.market.kind == "pump":
                    try:
                        c.holdings[addr] = await lp_wallet.balance(lp_pump.RPC, addr, c.market.mint)
                    except Exception:  # noqa: BLE001
                        c.holdings.setdefault(addr, 0.0)
                held = c.holdings.get(addr, 0.0)
                await send(ws, {"type": "wallet_ok", "address": addr, "holds": held, "weight": c.tap_weight(vid, wallet)})
    except (WebSocketDisconnect, RuntimeError, json.JSONDecodeError):
        pass
    finally:
        c.clients.pop(ws, None)


def clean_q(text):
    q = re.sub(r"https?://\S+|www\.\S+", "", str(text or ""))
    q = re.sub(r"[\x00-\x1f\x7f]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()[:QUESTION_LEN]
    if len(q) < 3:
        return (None, "Ask a little more than that.")
    if BLOCK.search(q):
        return (None, "Let's keep it kind. Try another question.")
    return q
