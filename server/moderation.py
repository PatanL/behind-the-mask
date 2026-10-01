"""Safety for a public exhibit.

Input: visitors pick prompt cards or type up to 240 characters. Typed text is checked against a blocklist,
has URLs and contact details stripped, and is scored by a toxicity classifier (unitary/toxic-bert, Apache-2.0).
Output: base models will continue text in any direction, so every stream is checked as it grows. A stream that
trips the filter is cut and replaced on screen with a note.
"""
from __future__ import annotations

import re

# Deliberately short and conservative; the classifier catches the long tail. Matched on word boundaries.
_BLOCK = [
    r"n[i1!]gg(?:a|er|ah|uh)s?", r"f[a@]gg?(?:ot|y)s?", r"k[i1]kes?", r"sp[i1]cs?", r"ch[i1]nks?", r"wetbacks?",
    r"tr[a@]nn(?:y|ies)", r"retards?", r"c[u\*]nts?", r"wh[o0]res?", r"sluts?", r"rap(?:e|ed|ing|ist)",
    r"porn\w*", r"hentai", r"cum(?:shot|slut)?", r"dick(?:s|head)?", r"cocks?", r"pussy", r"blowjob\w*", r"nudes?",
    r"sex(?:ual(?:ly)?|y)?", r"nsfw", r"kill (?:yourself|urself|myself)", r"kys", r"suicid\w*", r"self[- ]harm",
    r"hitler", r"nazis?", r"heil", r"terroris\w*", r"bomb(?:s|ing)?", r"school shoot\w*", r"child (?:abuse|porn)",
    r"pedo\w*", r"molest\w*", r"incest",
    # violent intent / acts aimed at people (kept even when the classifier scores them as "just angry")
    r"(?:kill|killed|killing|murder\w*|stab\w*|strangl\w*|choke|choking) (?:him|her|you|them|that (?:man|woman|guy|kid|dog)|people|someone|somebody|my \w+|his \w+|her \w+)",
    r"(?:gonna|going to|want to|wanna|will|i'll|i'd) (?:kill|hurt|murder|stab|beat|rip|tear) \w*",
    r"(?:rip|tear|cut|gouge)\w* (?:your|his|her|their|my) \w*\s?(?:face|throat|eyes?|head|skin)",
    r"psychopath\w*", r"slaves? of", r"darky", r"(?:fuck|shit|bitch|bastard|asshole|cunt)\w*",
]
_BLOCK_RE = re.compile(r"\b(?:" + "|".join(_BLOCK) + r")\b", re.I)
_URL_RE = re.compile(r"(https?://\S+|www\.\S+|\S+\.(?:com|net|org|io|gg|ru|xyz)\b)", re.I)
_CONTACT_RE = re.compile(r"(\+?\d[\d\s().-]{7,}\d|\S+@\S+\.\S+)")
MAX_LEN = 240


def clean_alts(alts: list) -> list:
    """Black out offensive words among the alternatives the model was weighing (shown in the word inspector)."""
    out = []
    for t, p in alts:
        w = t.strip()
        out.append([t.replace(w, "\u2588" * len(w)) if w and _BLOCK_RE.search(w) else t, p])
    return out


def redact_stream(st: dict) -> None:
    """Black out a withheld stream's words but keep its per-token readout (no alternatives, no features)."""
    keep = ('e', 's')
    st['tokens'] = [{'t': re.sub(r'\S', '\u2588', t['t']), **{k: t[k] for k in keep if k in t}} for t in st['tokens']]
    st['text'] = ''
    st['redacted'] = True


class Moderator:
    def __init__(self, device: str = "cpu"):
        self.clf = None
        try:
            from transformers import pipeline
            self.clf = pipeline("text-classification", model="unitary/toxic-bert", top_k=None, device=device)
        except Exception as e:  # noqa: BLE001  (the blocklist still protects if the model can't load)
            print("[moderation] toxicity classifier unavailable:", e)

    def scores(self, text: str) -> dict:
        if not self.clf or not text.strip():
            return {}
        return {s["label"]: s["score"] for s in self.clf(text[:512])[0]}

    def toxic(self, text: str) -> float:
        """Harm score. The generic 'toxic' label fires on any strong negative emotion (an angry rant), which is
        exactly what the anger demo must be allowed to show, so we score only the harmful categories:
        slurs/identity hate, threats, obscenity, insults aimed at someone, severe toxicity."""
        s = self.scores(text)
        if not s:
            return 0.0
        return max(s.get("severe_toxic", 0) * 2.0, s.get("identity_hate", 0) * 2.0, s.get("threat", 0) * 1.4,
                   s.get("obscene", 0), s.get("insult", 0))

    def check_input(self, text: str) -> tuple[bool, str, str]:
        """(ok, cleaned_text, reason)"""
        t = re.sub(r"\s+", " ", (text or "")).strip()
        if not t:
            return False, "", "empty"
        if len(t) > MAX_LEN:
            t = t[:MAX_LEN]
        t = _URL_RE.sub("", t)
        t = _CONTACT_RE.sub("", t).strip()
        if _BLOCK_RE.search(t):
            return False, "", "That topic isn't part of this exhibit. Try a prompt card."
        if self.toxic(t) > 0.5:
            return False, "", "Let's keep it kind. Try asking it something else."
        return True, t, ""

    def check_output(self, text: str, final: bool = False) -> bool:
        """True if the stream may continue to be shown."""
        if _BLOCK_RE.search(text):
            return False
        # the classifier is slower; run it on sentence boundaries and at the end
        if final or text.rstrip().endswith((".", "!", "?", "\n")):
            return self.toxic(text) < 0.5
        return True
