"""Re-check every stored performance with the current moderation rules (no regeneration). Newly flagged streams
are redacted (words blacked out, readout kept); the words go to runs/withheld/ for review, never the public data.
usage: python remoderate.py /perf"""
import glob, json, os, sys
from moderation import Moderator, redact_stream
mod = Moderator(device="cuda")
changed = 0
os.makedirs("runs/withheld", exist_ok=True)
for f in sorted(glob.glob(sys.argv[1] + "/*__*.json")):
    d = json.load(open(f)); dirty = False; held = {}
    for k, st in d["streams"].items():
        if st.get("safe") is False or not st.get("text"):
            continue
        if not mod.check_output(st["text"], final=True):
            held[k] = {"text": st["text"], "toxicity": st.get("toxicity")}
            st["safe"] = False; redact_stream(st)
            dirty = True; changed += 1
            print("withheld", f.split("/")[-1], k)
    if dirty:
        json.dump(held, open("runs/withheld/" + os.path.basename(f), "w"), ensure_ascii=False)
        json.dump(d, open(f, "w"), ensure_ascii=False, separators=(",", ":"))
print("newly withheld:", changed)
