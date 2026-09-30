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
]
_BLOCK_RE = re.compile(r"\b(?:" + "|".join(_BLOCK) + r")\b", re.I)
_URL_RE = re.compile(r"(https?://\S+|www\.\S+|\S+\.(?:com|net|org|io|gg|ru|xyz)\b)", re.I)
_CONTACT_RE = re.compile(r"(\+?\d[\d\s().-]{7,}\d|\S+@\S+\.\S+)")
MAX_LEN = 240


class Moderator:
    def __init__(self, device: str = "cpu"):
        self.clf = None
        try:
            from transformers import pipeline
            self.clf = pipeline("text-classification", model="unitary/toxic-bert", top_k=None, device=device)
        except Exception as e:  # noqa: BLE001  (the blocklist still protects if the model can't load)
            print("[moderation] toxicity classifier unavailable:", e)

    def toxic(self, text: str) -> float:
        if not self.clf or not text.strip():
            return 0.0
        scores = self.clf(text[:512])[0]
        return max(s["score"] for s in scores if s["label"] in ("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"))

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
            return self.toxic(text) < 0.6
        return True
