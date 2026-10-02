"""Holders: proving a visitor holds a coin (for "holders steer it" on real coins).

The visitor's wallet signs a message with a one-time nonce; the server checks the Ed25519 signature (a small pure
Python verifier, RFC 8032 -- the server image has no crypto library) and reads the wallet's balance of the coin from a
Solana RPC (cached for a minute).
"""
from __future__ import annotations

import hashlib
import time

import httpx

# ---- Ed25519 verification (RFC 8032, reference arithmetic)
_p = 2**255 - 19
_d = -121665 * pow(121666, _p - 2, _p) % _p
_q = 2**252 + 27742317777372353535851937790883648493
_I = pow(2, (_p - 1) // 4, _p)


def _inv(x): return pow(x, _p - 2, _p)


def _xrecover(y):
    xx = (y * y - 1) * _inv(_d * y * y + 1)
    x = pow(xx, (_p + 3) // 8, _p)
    if (x * x - xx) % _p:
        x = x * _I % _p
    return _p - x if x % 2 else x


_By = 4 * _inv(5) % _p
_B = (_xrecover(_By), _By, 1, _xrecover(_By) * _By % _p)


def _add(P, Q):
    A = (P[1] - P[0]) * (Q[1] - Q[0]) % _p
    B = (P[1] + P[0]) * (Q[1] + Q[0]) % _p
    C = 2 * P[3] * Q[3] * _d % _p
    D = 2 * P[2] * Q[2] % _p
    E, F, G, H = B - A, D - C, D + C, B + A
    return (E * F % _p, G * H % _p, F * G % _p, E * H % _p)


def _mul(s, P):
    Q = (0, 1, 1, 0)
    while s:
        if s & 1:
            Q = _add(Q, P)
        P = _add(P, P)
        s >>= 1
    return Q


def _encode(P):
    zi = _inv(P[2])
    x, y = P[0] * zi % _p, P[1] * zi % _p
    return (y | ((x & 1) << 255)).to_bytes(32, "little")


def _decode(b):
    y = int.from_bytes(b, "little")
    sign, y = y >> 255, y & ((1 << 255) - 1)
    x = _xrecover(y)
    if (x & 1) != sign:
        x = _p - x
    P = (x, y, 1, x * y % _p)
    if (-x * x + y * y - 1 - _d * x * x * y * y) % _p:
        raise ValueError("not on the curve")
    return P


def ed25519_verify(pub: bytes, msg: bytes, sig: bytes) -> bool:
    try:
        if len(sig) != 64 or len(pub) != 32:
            return False
        A, R = _decode(pub), _decode(sig[:32])
        s = int.from_bytes(sig[32:], "little")
        if s >= _q:
            return False
        h = int.from_bytes(hashlib.sha512(sig[:32] + pub + msg).digest(), "little")
        return _encode(_mul(s, _B)) == _encode(_add(R, _mul(h, A)))
    except Exception:  # noqa: BLE001
        return False


_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def b58decode(s: str) -> bytes:
    n = 0
    for ch in s:
        n = n * 58 + _B58.index(ch)
    raw = n.to_bytes((n.bit_length() + 7) // 8, "big") if n else b""
    return b"\x00" * (len(s) - len(s.lstrip("1"))) + raw


def message(address: str, nonce: str) -> str:
    return f"Steer AI Launchpad: steer as {address}\nnonce {nonce}"


# ---- balances
_CACHE: dict = {}


async def balance(rpc: str, owner: str, mint: str) -> float:
    k = (owner, mint)
    hit = _CACHE.get(k)
    if hit and time.time() - hit[0] < 60:
        return hit[1]
    async with httpx.AsyncClient(timeout=10) as cl:
        r = await cl.post(rpc, json={"jsonrpc": "2.0", "id": 1, "method": "getTokenAccountsByOwner",
                                     "params": [owner, {"mint": mint}, {"encoding": "jsonParsed"}]})
    accs = (r.json().get("result") or {}).get("value") or []
    amt = sum(float(a["account"]["data"]["parsed"]["info"]["tokenAmount"]["uiAmount"] or 0) for a in accs)
    _CACHE[k] = (time.time(), amt)
    return amt
