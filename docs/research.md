# The research behind *Behind the Mask*

This exhibit is built entirely from published techniques. Each section says what we use, where it lives in
the code, and what to read if you want to go deeper. The arXiv IDs below were checked against arXiv.

## 0. What's new in this version (2026 research)

| Technique | Paper | Where we use it |
|---|---|---|
| Emotion vectors from model-written stories; emotions as functional, causal representations | Anthropic, *Emotion concepts and their function in a large language model*, 2026. [transformer-circuits.pub/2026/emotions](https://transformer-circuits.pub/2026/emotions/index.html) | `server/build_vectors.py` (stories → mean activations → directions); the "is it real?" check mirrors their numerical-sensitivity test |
| Generation-based extraction beats comprehension-based; emotions peak near 50% depth | Jeong, *Extracting and Steering Emotion Representations in Small Language Models*, 2026. [arXiv:2604.04064](https://arxiv.org/abs/2604.04064) | Extraction method and layer choice (16 of 32) |
| The emotional latent space of LLMs | Reichman et al., *Emotions Where Art Thou*, ICLR 2026. [arXiv:2510.22042](https://arxiv.org/abs/2510.22042) | Background for the "feeling map" (a 2-D projection of the directions) |
| The assistant axis: the leading direction of persona space; pushing away from it makes the model adopt other identities | Lu et al., *The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models*, 2026. [arXiv:2601.10387](https://arxiv.org/abs/2601.10387) | "Take off the mask" and the Mask meter (assistant answers minus 30 role-played personas) |
| Sparse autoencoders for Qwen3.5 | Deng et al., *Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models*, 2026. [arXiv:2605.11887](https://arxiv.org/abs/2605.11887) | "Lighting up inside": the top layer-20 features per word |
| Sparse autoencoders (background) | Cunningham et al. 2023, [arXiv:2309.08600](https://arxiv.org/abs/2309.08600); Gao et al. 2024 (TopK SAEs), [arXiv:2406.04093](https://arxiv.org/abs/2406.04093) | |
| Naming features automatically | Paulo et al., *Automatically Interpreting Millions of Features in Large Language Models*, 2024. [arXiv:2410.13928](https://arxiv.org/abs/2410.13928) | `server/label_features.py` (top-activating excerpts → the model names each feature) |

## 1. Steering: adding a direction to the model's hidden state

Language models keep a running internal state (the *residual stream*) as they read and write. Many concepts,
such as sentiment, truthfulness or a persona trait, turn out to correspond roughly to **directions** in that
space. Adding a small multiple of such a direction while the model writes changes its behaviour without
changing its weights.

| Technique | Paper | Where we use it |
|---|---|---|
| Activation Addition (ActAdd): steer by adding the difference between the activations of two prompts | Turner et al., *Steering Language Models With Activation Engineering*, 2023. [arXiv:2308.10248](https://arxiv.org/abs/2308.10248) | `server/steer.py` `Mind.install()` adds the mix of directions at one layer |
| Contrastive Activation Addition (CAA): average the difference over many contrastive examples | Panickssery et al., *Steering Llama 2 via Contrastive Activation Addition*, 2023. [arXiv:2312.06681](https://arxiv.org/abs/2312.06681) | `Mind.compute_directions()` averages the same neutral texts read under several emotional framings vs a neutral one |
| Representation Engineering: reading and controlling concepts through population-level representations | Zou et al., *Representation Engineering: A Top-Down Approach to AI Transparency*, 2023. [arXiv:2310.01405](https://arxiv.org/abs/2310.01405) | The readout: projecting the hidden state onto each emotion direction (`Mind.readout()`) |
| Inference-Time Intervention | Li et al., *Inference-Time Intervention: Eliciting Truthful Answers from a Language Model*, 2023. [arXiv:2306.03341](https://arxiv.org/abs/2306.03341) | Background: steering at inference time, scaled by the typical activation size |
| Persona vectors: directions for character traits, used both to monitor and to steer | Chen et al., *Persona Vectors: Monitoring and Controlling Character Traits in Language Models*, 2025. [arXiv:2507.21509](https://arxiv.org/abs/2507.21509) | The exhibit's framing: the assistant's personality is itself something you can measure and push |
| Linear representations | Park, Choe & Veitch, *The Linear Representation Hypothesis and the Geometry of Large Language Models*, 2023. [arXiv:2311.03658](https://arxiv.org/abs/2311.03658); Tigges et al., *Linear Representations of Sentiment in Large Language Models*, 2023. [arXiv:2310.15154](https://arxiv.org/abs/2310.15154) | Why a single direction per emotion is a reasonable (if imperfect) model |

**What steering is not.** A strong projection on the "joy" direction means the model's state resembles its
state when it reads or writes joyful language. It is **not** evidence that the model feels joy. The exhibit
says this on screen.

## 2. The mask: base models versus assistants

A *base* model is trained only to continue text. It has no stable personality: it can continue a question
the way a forum post, a novel or a spam page would. The friendly assistant you usually talk to is that
base model after further training on human preferences.

| Topic | Reference |
|---|---|
| RLHF from human preference comparisons | Christiano et al., *Deep reinforcement learning from human preferences*, 2017. [arXiv:1706.03741](https://arxiv.org/abs/1706.03741) |
| Instruction-following assistants trained with RLHF (InstructGPT) | Ouyang et al., *Training language models to follow instructions with human feedback*, 2022. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155) |
| Helpful and harmless assistants | Bai et al., *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback*, 2022. [arXiv:2204.05862](https://arxiv.org/abs/2204.05862) |
| The models we run (Qwen3 base and post-trained) | Yang et al., *Qwen3 Technical Report*, 2025. [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) |
| Base models as simulators of many characters | janus, *Simulators*, 2022 (LessWrong essay); Shanahan, McDonell & Reynolds, *Role play with large language models*, Nature, 2023 |

In the exhibit, every prompt goes to three minds in parallel (`server/engine.py`):
- the **base model**, shown the plain text frame `Q: … A:`
- the **assistant as trained**
- the **assistant with the visitor's steering**

## 3. Fair comparisons: shared randomness

All rows sample with the **same Gumbel noise** at each step (Gumbel-max trick). With steering at zero, the
steered and unsteered assistant produce the same text, so any difference you see was caused by steering.
A fourth, hidden row lets the *unsteered* assistant read the *steered* words and report what it would have
said next ("what it almost said").

| Topic | Reference |
|---|---|
| Counterfactual generation by sharing the sampling noise (Gumbel-max structural causal model) | Chatzi et al., *Counterfactual Token Generation in Large Language Models*, 2024. [arXiv:2409.17027](https://arxiv.org/abs/2409.17027) |

## 4. Reading emotion from the words

Alongside the internal readout, a separate text classifier scores what the words themselves express. When
the two disagree, that is interesting too.

| Topic | Reference |
|---|---|
| 27 fine-grained emotions plus neutral, from 58k Reddit comments | Demszky et al., *GoEmotions: A Dataset of Fine-Grained Emotions*, 2020. [arXiv:2005.00547](https://arxiv.org/abs/2005.00547). The classifier is `SamLowe/roberta-base-go_emotions` (MIT) |

## 5. The face

The android's expressions are built from **FACS action units**: the brow raise (AU1/AU2), cheek raise (AU6),
lip-corner pull (AU12), lip-corner depress (AU15), and so on. Each emotion is a standard combination of these
units (EMFACS prototypes). How strongly each unit moves, and with what timing, is an artistic mapping of the
measured readout, not a measurement of anything the model feels.

**From readout to face.** The raw readout has cross-talk: the anger direction also rises when we push fear or
sadness. A face driven by it directly would show every negative feeling at once. So the face reads it through
a small linear decoder (6×6), fitted with ridge regression on every token of every pre-computed push to recover
which feeling was pushed and how hard (`web/scripts/fit-decoder.mjs`). With no push it reads about zero.

**Acting layer** (`web/src/face/face.js`), drawn from the psychology of facial behaviour:
- *Blends*: two feelings at once don't average out; each part of the face carries one of them, like sad brows
  over a smile (bittersweet), or a smile without the eyes and with the lips drawn back (nervous).
- *Display rules*: the Mask meter (the assistant-axis reading) acts like a learned professional face. Near
  the assistant voice, negative feelings are muted in the lower face and covered by a polite smile, while the
  brows keep showing them. Pushed off the axis, as strong steering does by itself, the regulation falls away.
- *Leaks*: under regulation, the real feeling flashes across part of the face for a fifth of a second, then
  gets covered (lip press, polite smile, a blink or glance away).
- *Timing*: felt smiles rise smoothly, with the cheek raise trailing the lip corners. Polite smiles switch on
  and off quickly and a little lopsided. Sudden feelings overshoot, then settle.
- *Body*: a startle gasp and held breath (fear), a held breath forced out through the nose (anger), a breath
  that catches and a trembling chin (sadness), a small laugh (joy), a sigh of relief when a bad feeling drains.
  The eyes change too: a hard, unblinking stare (anger), darting scans (fear), slow saccades and heavy blinks
  (sadness).

| Topic | Reference |
|---|---|
| Display rules, and leakage of concealed feelings | Ekman & Friesen, *The repertoire of nonverbal behavior: categories, origins, usage, and coding*, Semiotica 1, 1969; Ekman & Friesen, *Nonverbal leakage and clues to deception*, Psychiatry 32, 1969 |
| Blends: different feelings in different parts of the face | Ekman & Friesen, *Unmasking the Face*, Prentice-Hall, 1975 |
| How long micro-expressions last | Yan, Wu, Liang, Chen & Fu, *How fast are the leaked facial expressions: the duration of micro-expressions*, Journal of Nonverbal Behavior 37, 2013 |
| Posed versus spontaneous smiles move differently | Schmidt, Ambadar, Cohn & Reed, *Movement differences between deliberate and spontaneous facial expressions: zygomaticus major action in smiling*, Journal of Nonverbal Behavior 30, 2006 |
| The eyebrow flash greeting | Eibl-Eibesfeldt, *Human Ethology*, Aldine de Gruyter, 1989 |

| Topic | Reference |
|---|---|
| Facial Action Coding System | Ekman & Friesen, *Facial Action Coding System*, Consulting Psychologists Press, 1978; Ekman, Friesen & Hager, FACS manual, 2002 |
| Face mesh and expression blendshapes | ICT-FaceKit, USC Institute for Creative Technologies (MIT licence), from Li et al., *Learning Formation of Physically-Based Face Attributes*, CVPR 2020 |
| Blendshape naming | Apple ARKit `ARFaceAnchor.BlendShapeLocation` (the 52 standard names) |

## Honest limits
- Emotion directions from contrastive prompts also pick up style, topic and framing. They are useful
  handles, not clean "emotion neurons".
- Small models steer less cleanly than larger ones. The exhibit runs Qwen3.5-9B. Its emotion directions
  are not independent: anger and fear share about half their direction (cosine about 0.5).
- Feature names are written by the model from a handful of examples. They are good guesses, not ground truth.
- Too much steering breaks fluency. The dial ranges are capped using the sweeps in `server/calibrate.py`.
- The face's decoder is fitted on our own pushes, so it shows the feeling that was pushed more cleanly than
  the raw readout does. It is an intervention-trained display decoder: it recovers which push was applied,
  on the same runs it was fitted on. It is not an independently validated emotion sensor. The colours and
  meters still show the raw readout.
- The polite mask and the leaks are a scripted performance, driven by the Mask meter. They show the idea of
  a trained persona covering what is underneath. They are not evidence that the model hides real feelings.
- Some strong pushes (mostly anger) make the model abusive. Those answers are withheld: their words are
  blacked out, but their measured readout still plays on the colours and the face.
