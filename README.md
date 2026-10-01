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

1. **Emotion directions, from stories the model writes itself.** Qwen3.5-9B writes 216 short first-person
   stories about characters who feel joy, sadness, anger, fear, calm or curiosity, plus neutral ones. While
   it writes, we average its residual stream at the middle layer (16 of 32). Each emotion's direction is its
   average minus the average over all stories. This follows Anthropic's 2026 emotion-concept work, and
   generation-based extraction is the method a 2026 comparison found most reliable. (`server/build_vectors.py`)
2. **The assistant axis ("the mask").** The model answers the same questions as itself and as 30 role-played
   characters. The difference between the two is its *assistant direction* (Lu et al. 2026). Pushing against
   it is the exhibit's "Take off the mask". The **Mask** meter shows where the live state sits between the
   two (0% = like the role-played characters, 100% = like its own assistant voice).
3. **Steering.** While it writes, we add `coefficient × typical activation size × direction` to the residual
   stream at that layer, for every token. Strengths were picked from a sweep of effect vs fluency.
   (`server/steer.py`, `server/sweep.py`)
4. **Readout.** Before adding anything, we project the model's own state onto each direction. That live
   signal drives the face and the colours, shown relative to the unpushed answer. "Is the reading real?"
   checks it with no steering at all: change one detail in a sentence and the matching signal follows.
   (`server/validate.py`)
5. **Sparse features (Qwen-Scope).** Qwen's sparse autoencoder splits the layer-20 state into 65,536
   features, about 50 of them active at a time. The model names the ones that appear, from the text that
   triggers each most (automated interpretability), and the exhibit shows a few per word.
   (`server/label_features.py`)
6. **Fair comparisons.** Several copies run in lock-step with shared sampling noise (Gumbel-max):
   - the assistant as trained
   - the assistant with the push
   - the base model, with and without the push
   - a counterfactual copy: the unsteered assistant reading the steered words, which gives "what it would
     have said instead"

   (`server/engine.py`)
7. **Pre-computed show.** Every question × feeling × dial stop is generated once (`server/precompute.py`),
   so the public site is static, instant, and works for any number of visitors.

The full list of papers and techniques, with what we use from each, is in **[docs/research.md](docs/research.md)**.

## Repository layout

| Path | What |
|---|---|
| `server/steer.py` | emotion directions, steering hook, readout |
| `server/engine.py` | lock-step generation of all streams with per-token data |
| `server/build_vectors.py` | story-based emotion directions + the assistant axis |
| `server/sweep.py`, `server/validate.py` | steering strength vs effect/fluency; the no-steering "is it real?" check |
| `server/label_features.py` | names Qwen-Scope SAE features (automated interpretability) |
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

Regenerate everything on an NVIDIA GPU. We used a DGX Spark; you need about 45 GB of GPU memory, since the 9B chat and base models are loaded together.

```bash
docker build -t btm-server server/
R="docker run --rm --gpus all --ipc=host -v ~/.cache/huggingface:/root/.cache/huggingface -v $PWD/server:/app -v $PWD/runs:/app/runs -v $PWD/web/public/performances:/perf btm-server"
$R python build_vectors.py --out runs/q9b                 # ~10 min
$R python sweep.py --dirs runs/q9b                         # pick strengths -> runs/q9b/levels.json
$R python validate.py --dirs runs/q9b --out /perf
$R python precompute.py --dirs runs/q9b --out /perf --sae <layer20.sae.pt> --sae-layer 20
$R python label_features.py --perf /perf --sae <layer20.sae.pt> --layer 20
```

## Models, data and licences

| Component | Licence |
|---|---|
| Code in this repository | MIT (see `LICENSE`) |
| Qwen3.5-9B, Qwen3.5-9B-Base (Alibaba Qwen) | Apache-2.0 |
| Qwen-Scope SAEs (`Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50`) | Apache-2.0 (see its model card for usage terms) |
| ICT-FaceKit face model (USC Institute for Creative Technologies) | MIT |
| `SamLowe/roberta-base-go_emotions` (text emotion scores) | MIT |
| `unitary/toxic-bert` (safety filter) | Apache-2.0 |
| Fraunces, Inter fonts | SIL Open Font License |

## Safety

Base models are unfiltered text predictors. Every generated stream is checked with a blocklist and a
toxicity classifier, and anything flagged is withheld from the exhibit. Visitors choose from prompt cards;
the optional live mode moderates typed input.
