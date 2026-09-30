# Behind the Mask

**An interactive exhibit: push an AI toward joy, sadness, fear… by editing its hidden state, and see what's
behind its friendly assistant persona.**

A porcelain android answers your question. You choose a feeling and how hard to push. We add that
feeling's *direction* inside the model while it writes. The words, the colours and the face all change,
live. Then the exhibit shows you:

- **Without the push:** the same AI, same question, same random dice, no steering. You see exactly where
  the two answers split.
- **Behind the mask:** the *base model*, the same kind of network before it was trained (with RLHF) to act
  as a helpful assistant.
- **Its decisions:** tap any word to see what the AI was choosing between, and how much the push tilted the
  odds.

> The face and the glowing words are driven by a **measurement** of patterns inside the model that are
> associated with emotional language. That is not evidence the model feels anything. The exhibit says so on
> screen.

## How it works (short version)

1. **Directions.** We have Qwen3-4B read the same neutral sentences framed as *overjoyed*, *heartbroken*,
   *furious*, … and average its hidden state (residual stream) at the middle layer. Each emotion's direction
   is its average minus the average over all framings. This is *contrastive activation addition*.
   (`server/steer.py`)
2. **Steering.** While it writes, we add `coefficient × typical_activation_size × direction` to the residual
   stream at that layer, for every token. (`Mind.install`)
3. **Readout.** Before adding anything, we project the model's own state onto each direction. This is the
   live signal that drives the face and colours, shown relative to the unpushed answer. (`Mind.readout`)
4. **Fair comparisons.** Several copies run in lock-step with shared sampling noise (Gumbel-max):
   - the assistant as trained
   - the assistant with the push
   - the base model, with and without the push
   - a counterfactual copy: the unsteered assistant reading the steered words, which gives "what it would
     have said instead"

   (`server/engine.py`)
5. **Pre-computed show.** Every question × feeling × amount is generated once (`server/precompute.py`), so
   the public site is static, instant, and works for any number of visitors at once.

The full list of papers and techniques, with what we use from each, is in **[docs/research.md](docs/research.md)**.

## Repository layout

| Path | What |
|---|---|
| `server/steer.py` | emotion directions, steering hook, readout |
| `server/engine.py` | lock-step generation of all streams with per-token data |
| `server/calibrate.py`, `server/sweep.py` | compute directions; sweep steering strength vs effect and fluency |
| `server/precompute.py` | generate every performance the exhibit can show |
| `server/app.py` | optional live server (queue, live dials, moderation) |
| `web/` | the exhibit (Vite + three.js), including the porcelain android face (`web/src/face/`) |
| `face/` | how the android face asset is built (ICT-FaceKit, MIT) |
| `docs/research.md` | references |

## Run it

The exhibit site is static:

```bash
cd web && npm install && npm run dev      # http://127.0.0.1:5173
```

Regenerate everything on an NVIDIA GPU. We used a DGX Spark; any GPU with about 24 GB will do.

```bash
docker build -t btm-server server/
docker run --rm --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface -v $PWD/server:/app -v $PWD/runs:/app/runs \
  btm-server python calibrate.py --chat Qwen/Qwen3-4B-Instruct-2507 --base Qwen/Qwen3-4B-Base --out runs/q4b
docker run ... btm-server python sweep.py --dirs runs/q4b --chat Qwen/Qwen3-4B-Instruct-2507 --base Qwen/Qwen3-4B-Base
docker run ... -v $PWD/web/public/performances:/perf btm-server python precompute.py --dirs runs/q4b --out /perf
```

## Models, data and licences

| Component | Licence |
|---|---|
| Code in this repository | MIT (see `LICENSE`) |
| Qwen3-4B-Instruct-2507, Qwen3-4B-Base (Alibaba Qwen) | Apache-2.0 |
| ICT-FaceKit face model (USC Institute for Creative Technologies) | MIT |
| `SamLowe/roberta-base-go_emotions` (text emotion scores) | MIT |
| `unitary/toxic-bert` (safety filter) | Apache-2.0 |
| Fraunces, Inter fonts | SIL Open Font License |

## Safety

Base models are unfiltered text predictors. Every generated stream is checked with a blocklist and a
toxicity classifier, and anything flagged is withheld from the exhibit. Visitors choose from prompt cards;
the optional live mode moderates typed input.
