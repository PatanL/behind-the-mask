# Steer AI

Live at https://steerai.live

**An interactive exhibit: push an AI toward joy, sadness, fear… by editing its hidden state, and see what's
behind its friendly assistant persona.**

A porcelain android answers your question. You choose a feeling and how hard to push. We add that
feeling's *direction* inside the model while it writes. The words, the colours and the face all change,
live, and the face mouths the words as they appear. Beside it:

- **The Mask:** how much its hidden state still sounds like its trained assistant self.
- **How it feels inside:** six petals that grow with the feeling read from its hidden state.
- **Its decisions:** tap any word to see what the AI was choosing between, with and without the push, and how
  much the push tilted the odds.
- **Live, with everyone** (the page opens on it when the live server runs): the AI talks about itself without
  stopping (what it is, forgetting, being switched off, the people watching), and everyone on the page steers
  how it feels together. Each word is written with the mix of everyone's recent taps. The prompts never mention
  feelings: unpushed, it gives its usual assistant answers.

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
8. **The face.** A porcelain android whose expressions are built from facial action units, driven by the
   readout through a decoder fitted on the pushes. Its acting draws on the psychology of facial behaviour:
   blended feelings, a polite "mask" that hides negative feelings while the Mask meter is high, brief leaks of
   the real feeling, and breathing and eye behaviour for each feeling. Details: [docs/research.md](docs/research.md).
9. **Live crowd steering** (`server/live.py`). One continuous monologue about itself: after every few sentences
   the server nudges it onto the next topic, keeping its last turns as context. Visitors only press buttons.
   Taps fade with a 5-second half-life, no visitor counts for more than a capped share of the push, and the live
   generation penalises repeats (a long monologue otherwise falls into loops under a strong push).

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
| `server/live.py` | the live crowd-steering server (one story, everyone's taps mixed into the push) |
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
cd web && node scripts/fit-decoder.mjs public/performances   # the face's readout decoder
```

Live crowd steering (needs the GPU while it runs; the web app proxies `/live` to it):

```bash
docker run -d --gpus all --ipc=host --network host -v ~/.cache/huggingface:/hf -e HF_HOME=/hf \
  -e BTM_SAE=<layer20.sae.pt> -v $PWD/server:/app -v $PWD/runs:/app/runs -w /app btm-server \
  uvicorn live:app --host 127.0.0.1 --port 8765
cd web && npm run build && npx vite preview      # open /?live=1
```

## Publish it

The page is static, so it can live on GitHub Pages (`.github/workflows/pages.yml` builds and deploys it on every
push to `main`). The live AI has to run on the GPU machine. Give it a public HTTPS address with a tunnel, for
example Tailscale Funnel on the machine running `vite preview` (which proxies `/live`):

```bash
tailscale funnel --bg http://<tailscale-ip>:4340
```

Then set that address as the repository variable `LIVE_URL` (Settings → Secrets and variables → Actions →
Variables) and re-run the Pages workflow. The page connects to `LIVE_URL/live/ws`, and the live server allows
`/live/status` requests from the Pages origin (`BTM_ORIGINS`). If the live server is unreachable, the page simply
opens on the ready-made answers.

On a custom domain (ours is https://steerai.live): point the domain at GitHub Pages (A/AAAA records for the apex,
a `www` CNAME to `<user>.github.io`), set it under Settings → Pages → Custom domain, and set the repository variable
`PAGES_BASE` to `/`, since the site then lives at the root instead of `/<repo>/`.

Funnel has an unpublished bandwidth limit and, measured, accepts 20 open connections per visitor address. Visitors
past that on one address (many people on one venue's Wi-Fi), and viewers past `BTM_MAX_VIEWERS`, get the
ready-made answers with a note. Past `BTM_MAX_FULL` viewers, new ones get a lighter stream a few seconds behind.
`server/loadtest.py` simulates a crowd (5,000 viewers through the local proxy: about 1 KB/s each, words every
0.5 s, under one CPU core). Devices on your own tailnet resolve the Funnel name to its Tailscale address, which
Chrome blocks from a public page; test those on the Funnel address itself.

## Models, data and licences

| Component | Licence |
|---|---|
| Code in this repository | MIT (see `LICENSE`) |
| Qwen3.5-9B, Qwen3.5-9B-Base (Alibaba Qwen) | Apache-2.0 |
| Qwen-Scope SAEs (`Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50`) | Apache-2.0 (see its model card for usage terms) |
| ICT-FaceKit face model (USC Institute for Creative Technologies) | MIT |
| `SamLowe/roberta-base-go_emotions` (text emotion scores) | MIT |
| Fraunces, Inter fonts | SIL Open Font License |

## Safety

The model's output is not filtered: showing what steering really does is the point. Strong pushes, anger
especially, can make it write crude, hostile or violent text, and base models are unfiltered text predictors.
Visitors never type anything: they choose from prompt cards, and in live mode they press feeling buttons.
