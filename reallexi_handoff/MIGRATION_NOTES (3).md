# Clean Multi Video Clips → LTX-2.5 · Rebuild Notes

**Source:** `Clean_Multi_Video_Clips_together_Sam_Ayoub__3_.json`
**Output:** `LTX25_Multi_Shot_Character_Lock_Sam_Ayoub.json`
**Builder:** `build_ltx25_multishot.py` (re-runnable, parameterised)

---

## 1. Audit of the original

| | Original | Rebuilt |
|---|---|---|
| Root nodes | 32 (12 orphaned) | 32 (0 orphaned) |
| Subgraph **definitions** | **8** | **1** |
| Nodes inside definitions | 216 | 56 |
| Root groups | **0** | 8 |
| File size | 593 KB | 189 KB |

### The core problem

The four clip nodes (`920`, `3115`, `3170`, `3228`) looked like four instances of one
subgraph. They weren't. They were **four separate definitions**, each wrapping **another
four separate definitions** of the LTX-2.3 generator — 8 definitions, 216 nodes total.

I diffed all four generator copies. They were identical except for **one widget**:

```
TrimAudioDuration start →  copy1: 40   copy2: 60   copy3: 80   copy4: 100
```

And that widget was already overridden at runtime, because `start_index` was wired
through as a subgraph input. So 165 duplicated nodes existed to express **nothing**.
That is why editing a sampler in one clip never propagated, and why the file grew to
593 KB.

### Other findings

- **12 orphaned root nodes** — 11 prompt presets (`Best`, `Muvani`, `latest generic`,
  `lock all`, `Main story`, …) plus a stray `PreviewAny`, all unwired.
- **Dead subgraph inputs** — the wrappers exposed `model`, `vae`, `clip`, `prompt`,
  and `noise_seed`, none of which were wired to anything internally. The root fed
  `noise_seed` into a socket that went nowhere; all four clips silently used the
  hardcoded internal seed.
- **Asymmetric definitions** — only copy 1 had the `noise_seed` input at all. Copies
  2–4 had 12 inputs, copy 1 had 13.
- **Shared filename prefix** — all four `SaveVideo` nodes wrote to
  `newVideoParts/sam_new_`, relying on auto-increment, so shot order in the output
  folder was not guaranteed.

---

## 2. Flow preservation

The flow is unchanged. Same chain, same handoff, same audio windows:

```
LoadImage ──► ImageScale ──► SHOT 1 ──► SHOT 2 ──► SHOT 3 ──► SHOT 4
                              │          │          │          │
                              └──────────┴──────────┴──────────┴──► PreviewImage
LoadAudio ────────────────────┴──────────┴──────────┴──────────┘

audio window:  40–60s      60–80s     80–100s    100–120s
```

Each shot's last decoded frame is the next shot's first frame, exactly as before.
Internal stage order is also unchanged:

```
preprocess → empty latent → low-res sample → latent upscale x2
           → high-res sample → finish upscale x2 → decode → save
```

---

## 3. LTX-2.3 → LTX-2.5 node mapping

LTX-2.5 ships as a **split pack** (one safetensors per component) instead of 2.3's
monolithic checkpoint, so the loader block is the biggest change.

| LTX-2.3 (yours) | LTX-2.5 (rebuilt) | Note |
|---|---|---|
| `CheckpointLoaderSimple` → MODEL + VAE | `UNETLoader` + `VAELoader` | Split pack |
| `LTXAVTextEncoderLoader` (Gemma 3 12B) | `CLIPLoader` type `ltxv` (Gemma **4** 12B) | New encoder |
| `LTXVAudioVAELoader` (from ckpt) | `VAELoader` (`ltx-2.5-audio-vae-bf16`) | Standalone |
| `LoraLoaderModelOnly` (2.3 distilled LoRA) | **removed** | See below |
| `CFGGuider` ×2 | `LTXVDualCFGGuider` ×2 `[1, 1]` | 2.5 guider |
| `LTXVCropGuides` | **removed** | 2.5 wires conditioning straight to the guider |
| `KSamplerSelect` `euler_cfg_pp` / `euler_ancestral_cfg_pp` | `euler_ancestral` (both) | Matches 2.5 template |
| `VAEDecodeTiled` `[768, 64, 4096, 4]` | `[512, 64, 64, 16]` | Retuned for the new decoder |
| *(none)* | `TextGenerateLTX2Prompt` + `ComfySwitchNode` | New prompt enhancer, default **off** |
| `ManualSigmas` (both schedules) | **unchanged** | Sigma schedules carry over as-is |
| `LTXVPreprocess`, `EmptyLTXVLatentVideo`, `ComfyMathExpression`, `LTXVConcatAVLatent`, `LTXVSeparateAVLatent`, `LTXVImgToVideoInplace`, `LTXVLatentUpsampler`, `SamplerCustomAdvanced`, `CreateVideo` | **unchanged** | Same types, same slots |

### ⚠️ Your distilled LoRA is gone, deliberately

`ltx-2.3-22b-distilled-lora-384.safetensors` targets the 2.3 transformer. It will not
apply to a 2.5 transformer and needs retraining. The 2.5 **distilled** checkpoint
already carries the distillation, so the LoRA loader is redundant as well as
incompatible. If you'd rather run the full dev transformer, swap `UNETLoader` to
`ltx-2.5-22b-dev-transformer-bf16.safetensors` and lengthen the sigma schedules.

### Your custom additions, kept

These are yours, not from the official template, and they survived the migration:

- **Audio-locked A/V path** — `TrimAudioDuration → LTXVAudioVAEEncode →
  SetLatentNoiseMask(SolidMask 0) → LTXVConcatAVLatent`. The official 2.5 I2V template
  uses `LTXVEmptyLatentAudio` (model-generated audio) instead; yours drives generation
  off the real track, which is what keeps the performance on the beat. Only the audio
  VAE loader changed.
- **Aspect lock** — `ResizeImageMaskNode('scale dimensions', W, H)` before
  `ResizeImagesByLongerEdge(1536)`. This matters more than it looks: it guarantees every
  handoff frame enters the next shot at the same aspect, so no cumulative crop drift.
- **Second `LTXVLatentUpsampler`** before decode (your 4× total spatial). Kept, but see
  the VRAM note in §5.

### Model files

Gated repo — accept the licence at `huggingface.co/Lightricks/LTX-2.5` first.

| File | Folder |
|---|---|
| `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | `models/diffusion_models/` |
| `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | `models/text_encoders/` |
| `gemma4_e2b_it_bf16.safetensors` | `models/text_encoders/` |
| `ltx-2.5-video-vae-bf16.safetensors` | `models/vae/` |
| `ltx-2.5-audio-vae-bf16.safetensors` | `models/vae/` |
| `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | `models/latent_upscale_models/` |

Requires **ComfyUI 0.32.0+**. A `float64` fix for the LTX diffusion decoder landed on
master shortly after the 0.32.0 tag — if you hit a dtype error in `VAEDecodeTiled`,
pull latest master rather than staying on the tag.

---

## 4. Keeping the character identical

This is the part the old file was quietly working against. Four sources of drift, and
what changed:

### a. Compounding preprocessing — the big one

`LTXVPreprocess` applies compression to the incoming frame **every shot**. Shot 1 sees
a clean render; shot 4 sees an image that has been compressed three times. Faces soften
progressively down the chain and it reads as "the character changed."

**Changed:** default lowered from `18` → `10`. It lives in group `03` inside the shot
subgraph. If faces still soften by shot 3–4, drop to `5`. If motion gets jittery, raise
it back toward `18` — this is the one knob worth a dedicated A/B.

### b. First-frame lock strength

The low-res `LTXVImgToVideoInplace` strength governs how hard the first frame is
imposed. Yours was `0.5`; the official 2.5 template uses `0.7`.

**Changed:** set to `0.7`. Higher = tighter identity, less motion freedom. Group `06`.

### c. Handoff frame selection

`GetImagesFromBatchIndexed` takes index `-1` — the literal final frame, which is often
the most motion-blurred frame in the clip. A blurred face becomes the next shot's
identity reference.

**Unchanged at `-1`**, but a `PreviewImage` now sits on every handoff so you can see
what shot N+1 is actually inheriting. If a handoff looks smeared, change to `-3` or
`-5` (group `09`).

### d. Prompt drift

Nothing in the graph enforces this, so it's on you: keep the **character description
block byte-identical** across all four shot prompts and vary only the action/camera
sentence. Copy-paste, don't retype. Your `lock all` preset is already the right shape
for this — it's preserved in the PROMPT LIBRARY group.

### The 2.5 option worth testing separately

LTX-2.5 added **native multishot** — one generation produces several connected shots
that hold character, environment, lighting, and voice across cuts, without chaining.
That solves drift by not having a chain at all. It's a different architecture from
yours, so I didn't force it into this file, but for a 4-shot sequence it's probably the
stronger approach and worth a side-by-side. Start from the ComfyUI 2.5 template library.

---

## 5. Things to check before a long run

- **Frame count.** 20s × 25fps = **501 frames per shot**, four shots. That's a large
  latent even before the two ×2 upscales. If you OOM, cut `duration_sec` to 10 first —
  it's a per-shot widget, no graph surgery needed.
- **The finish upscaler.** The second `LTXVLatentUpsampler` (group `09`) upscales after
  the final sampler, with no denoise pass behind it. With 2.5's new Diffusion Video
  Decoder this may be redundant. Try bypassing it (Ctrl+B) and compare — you may get the
  same output for roughly half the decode cost. Or rebuild with
  `--no-finish-upscale`.
- **`comfy-aimdo` streaming.** Same as your other LTX work — if quantised video models
  crash on load, `--disable-pinned-memory` or `--disable-dynamic-vram` via
  `CUSTOM_WAN_COMFYUI_ARGS`.
- The prompt enhancer is **off** by default. Note it conditions on the incoming frame,
  so on shots 2–4 it would describe an already-drifted frame back into the prompt.
  Leave it off for chained shots; it's most useful on shot 1.

---

## 6. Setting the number of shots

ComfyUI has no native "N shots" control. The graph is a static DAG; loops exist only
because custom nodes can return a replacement subgraph at runtime ("node expansion"),
which is a third-party feature (`ComfyUI-Easy-Use` `forLoopStart/End`,
`ControlFlowUtils`, BadCafeCode's demo pack). Core ships none of it.

A loop is also the wrong shape here: a loop body runs N *identical* times, but each shot
needs a different prompt. You'd have to encode prompts as a delimited string and
index-select them — invisible, fragile, and painful to debug at 20s per shot.

So the shot count lives in the builder.

### Quick: just give me N shots

```bash
python build_ltx25_multishot.py --shots 8
```

Regenerates the entire chain — instances, links, groups, per-shot seeds, save prefixes,
and cumulative audio windows. The definition is **not** duplicated; all N shots still
point at the same one.

### Better: a declarative shot table

```bash
python build_ltx25_multishot.py --emit-config shots.json   # starter file
python build_ltx25_multishot.py --config shots.json        # build from it
```

```json
{
  "defaults": { "duration_sec": 12, "audio_start": 40, "seed": 500,
                "save_prefix": "newVideoParts/muvani_" },
  "character_lock": "CHARACTER LOCK — do not deviate:\nThe subject MUST match the reference image exactly…",
  "shots": [
    {"title": "ESTABLISH", "action": "Wide. Subject sways in. Slow push-in.", "duration_sec": 8},
    {"title": "VERSE",     "action": "Medium. Lip-sync, handheld drift."},
    {"title": "CHORUS",    "action": "Close-up. Expression lands on the downbeat.", "duration_sec": 20, "seed": 999},
    {"title": "OUTRO",     "action": "Crane up to wide. Arms raised.", "duration_sec": 6}
  ]
}
```

```
shot       audio window          dur   seed  save_prefix
ESTABLISH  40s → 48s              8s    500  newVideoParts/muvani_shot01
VERSE      48s → 60s             12s    501  newVideoParts/muvani_shot02
CHORUS     60s → 80s             20s    999  newVideoParts/muvani_shot03
OUTRO      80s → 86s              6s    503  newVideoParts/muvani_shot04
total 46s  ·  ~1150 frames (25 fps)
```

Three things this buys you:

- **`character_lock` is prepended verbatim to every shot prompt.** This is the single
  most effective anti-drift measure in the file, and it's now mechanical — the identity
  block is byte-identical across all shots because the script concatenates it. You only
  write the per-shot `action` line.
- **Audio windows chain cumulatively.** Set `duration_sec` per shot and the windows
  follow (8s → 12s → 20s → 6s above). Pin `audio_start_sec` on a shot to break the chain
  and jump elsewhere in the track. `--shots N` alone keeps the old uniform spacing.
- **Anything omitted falls back to `defaults`.** `"shots": 6` as a bare number also works
  when you want N shots sharing one prompt.

### Other flags

```bash
python build_ltx25_multishot.py --img-compression 5     # softer preprocessing
python build_ltx25_multishot.py --no-finish-upscale     # drop the second upscaler
python build_ltx25_multishot.py --width 1024 --height 576 --fps 24
```

`validate_workflow.py` checks the emitted JSON for dangling links, slot-range errors,
unregistered link back-references, subgraph IO consistency, and instance/definition
contract mismatches:

```bash
python validate_workflow.py LTX25_Multi_Shot_Character_Lock_Sam_Ayoub.json
```

Run against the original it reports 4 instance/definition mismatches. Every rebuild
above passes clean.

## 7. Attribute bleed — "cat head on a human body"

When two characters share a prompt and the model merges them, it is almost never the
model's fault. It is attribute binding.

### What was wrong in the original prompt

```
Identity = **locked (face, fur, proportions, clothing)**
```

`fur` and `clothing` sit in the same attribute list, so the encoder is told **one**
entity has both. That clause alone produces hybrids. Further down,
`natural human performance limits` applies "human" to everything in frame, cat included.

Three compounding problems:

1. **Negations don't subtract.** `🚫 no new character generation`,
   `Still NOT allowed: changing face` — a text encoder is not an instruction-follower.
   It embeds "changing face" as a concept that is *present*. Negation only functions in
   the negative conditioning branch.
2. **The negative branch was inert.** `LTXVDualCFGGuider [1, 1]`. CFG 1 is the correct
   distilled default, but at 1 there is effectively no classifier-free guidance, so the
   negative prompt barely applies. Negations in the positive prompt were *adding*
   concepts while the negative prompt did nothing.
3. **~750 tokens of markdown.** LTX names overstuffed prompts among the top three
   quality killers. For image-to-video their guidance is to describe *motion, not
   scene* — frame 0 already contains the scene — and to keep it to 1–2 actions. Emoji,
   headers and meta-commentary ("THIS FIXES YOUR ISSUE") are all tokens diluting the
   subject description exactly where binding needs to be strongest.

### The fix: a `cast` block

```json
{
  "scene": "a sunlit living room with warm late-afternoon light",
  "camera": "slow push-in, handheld drift",
  "cast": [
    { "kind": "cat",
      "look": "a small orange tabby cat with green eyes and short fur",
      "anatomy": "four legs, a full feline body, a long tail, whiskers and pointed ears",
      "where": "on the left, on the wooden floor" },
    { "kind": "woman",
      "look": "a woman in her thirties with long dark hair, wearing a red jacket",
      "anatomy": "a human face, human hands and human proportions",
      "where": "on the right, seated on the sofa" }
  ],
  "shots": [
    { "title": "SHOT 1", "action": "The cat lifts its head and blinks slowly while the woman turns toward it" }
  ]
}
```

emits, per shot, one flowing paragraph with no markdown and no negations:

> A sunlit living room with warm late-afternoon light. On the left, on the wooden floor,
> a small orange tabby cat with green eyes and short fur. The cat has four legs, a full
> feline body, a long tail, whiskers and pointed ears. The cat stays entirely a cat for
> the whole shot. On the right, seated on the sofa, a woman in her thirties with long
> dark hair, wearing a red jacket. The woman has a human face, human hands and human
> proportions. The woman stays entirely a woman for the whole shot. The cat and the
> woman are separate beings that keep their own separate bodies throughout the shot.
> The cat lifts its head and blinks slowly while the woman turns toward it.
> Camera: slow push-in, handheld drift.

Four rules it enforces mechanically:

- **One closed clause per character.** Attributes are never pooled into a shared list.
- **The kind noun is repeated, not pronouned.** "The cat has… The cat stays…" — that
  repetition is what binds attributes to the right subject.
- **Persistence is asserted positively** ("stays entirely a cat"), never as a negation.
- **Spatial anchoring** via `where` gives the model separate regions to attach to,
  which is the strongest single anti-blend signal.

Hybrid terms are auto-generated from the cast kinds and appended to the **negative**
prompt, where negation actually works:

```
cat head on a woman body, woman head on a cat body, cat-woman hybrid,
woman with cat features, fur on human skin, human face on an animal,
animal ears on a person, chimera, merged anatomy, fused subjects, …
```

Preview before building:

```bash
python build_ltx25_multishot.py --config shots.json --print-prompts
```

### Making the negative prompt actually apply

```bash
python build_ltx25_multishot.py --cfg 3 1
```

Sets both `LTXVDualCFGGuider` nodes. Leave at `1 1` on the distilled transformer — that
is its correct operating point, and raising it there degrades output. If hybrids persist
and you need the negatives to bite, switch `UNETLoader` to
`ltx-2.5-22b-dev-transformer-bf16.safetensors`, lengthen the sigma schedules, and work
in LTX's recommended 2.0–5.0 band (3.0 is their image-to-video starting point; high CFG
distorts the input image). Change one of the two values at a time and watch what moves.

### The part no prompt can fix

At `LTXVImgToVideoInplace` strength 0.7, **frame 0 dominates**. If the reference image
shows a human and the prompt introduces a cat, the model has to invent the cat inside a
human-shaped latent — and it borrows the anatomy that is already there. That is the
chimera, and no amount of prompt discipline removes it.

**Both characters must already exist, spatially separated, in the reference image.**
Build that frame first (your Flux.1 Kontext face-lock workflow is the right tool), then
feed it in. Once frame 0 is correct, the chain carries the separation forward and the
`cast` block keeps the prompt from undoing it.

Two smaller checks while you're there: your reference is prepped at 800×1024, and LTX
notes the model works best at widescreen ratios with portrait and square more prone to
distortion — worth an A/B at 16:9 if the hybrids are stubborn. And keep `action` to 1–2
beats; a prompt describing five simultaneous things is one LTX calls out specifically as
un-choreographable.

### What repeats and what varies

Every shot is an independent generation, so the *who* has to be restated in each one —
that repeated cast block is what binds identity. The *what* must differ.

| Part of the prompt | Per shot |
|---|---|
| `scene` | same (unless overridden) |
| `cast` block — look, anatomy, persistence | **same, verbatim** — this is the identity anchor |
| `action` | **different** — 1–2 beats |
| `camera` | usually different |

The original file got this exactly backwards. Its four clip prompts hashed to:

```
clip 920 : len=2994  sha1=2bc3f8b55e45ddfe
clip 3115: len=2999  sha1=4ea942d0d06a9539
clip 3170: len=2999  sha1=4ea942d0d06a9539   ← byte-identical
clip 3228: len=2999  sha1=4ea942d0d06a9539   ← byte-identical
```

Clips 2–4 were byte-identical; clip 1 differed by five characters. All four shots were
telling the model to do the same thing.

That is worse here than in a normal batch, because **each shot starts from the previous
shot's last frame**. Identical instructions mean the model replays the same motion arc
from a slightly shifted start: the sequence stalls, motion loops, and four 20-second
shots read as one take with cuts. It also removes the only per-shot signal that could
correct drift — the prompt says nothing new, so nothing pulls the character back.

The builder now guards against it:

```
!! 2 shots share an identical prompt: A, B
   Each shot starts from the previous shot's last frame. Same
   instruction = same motion arc replayed, so the sequence stalls.
   Give each shot its own 'action' in the config.
```

`--shots N` with no config no longer clones one prompt. It cycles a set of distinct
placeholder beats and says so, so you can see the shape before writing real direction.

## 8. Drift, degraded handoff frames, and the Ollama director

Two symptoms, one cause: **a pure chain has no ground truth.** Every shot's only
identity source is the previous shot's output, so errors compound with nothing to
correct them. The last frame is also the worst frame to build on.

### Why the handoff frame looks bad

Three things stack on the tail of every clip:

1. **Clip length.** 20s at 25fps is **501 frames**. Video diffusion loses fidelity
   toward the end of a long generation, and this is very long. This is the dominant
   term — cutting shots to 8–10s and using more of them buys more quality than any
   other change here, for free.
2. **`-1` is the literal last frame** — usually the most motion-blurred in the clip,
   and the weakest tail of a long latent after two upscales.
3. **`LTXVPreprocess` compression** is then re-applied to it on the way into the next
   shot, so shot 4 inherits an image compressed three times.

Fix 2 is now a flag, defaulting to `-6` instead of `-1`:

```bash
python build_ltx25_multishot.py --handoff-index -6
```

### Anchor modes — the actual fix

Each shot now chooses where its first frame comes from:

| `anchor` | first frame | drift | continuity |
|---|---|---|---|
| `previous` | last frame of the shot before | compounds across the whole chain | full |
| `reference` | the original clean reference | **zero** | none — it's a cut |
| `alternate` | re-anchors every other shot | capped at one hop | every other pair |

```bash
python build_ltx25_multishot.py --anchor reference    # star: no drift at all
python build_ltx25_multishot.py --anchor alternate    # drift capped at one hop
```

or per shot in the config: `{"title": "SHOT 3", "anchor": "reference"}`.

Shot 1 is always `reference`. The group title on the canvas shows each shot's mode.

For a music video this matters less than it sounds: cuts between shots are normal, so
`reference` costs you very little and removes the drift problem outright. Use
`previous` only where two shots must read as one continuous move. That is the answer
to "without a LoRA, what confirms the original character" — the clean reference does,
re-injected as often as you allow.

A character LoRA is still the strongest option, but note your 2.3 LoRA will not load on
a 2.5 transformer and needs retraining. Anchor modes work today, at no training cost.

### Ollama now drives the action

The describer was wired to a `PreviewAny` dead end and did nothing. It now sits inside
the shot subgraph, once per shot, and the prompt is split in two:

```
CAST (fixed, promoted input) ──┬─→ StringConcatenate ─────────────┐
                               │        + BEAT (static text)      ├─ ComfySwitchNode ─→ CLIPTextEncode
                               └─→ StringConcatenate ─────────────┘        ▲
                                        + OllamaImageDescriber             │
                                          images  ← this shot's first frame│
                                          prompt  ← BEAT as a hint    use_ollama
```

The split is the point:

- **Identity never goes through the LLM.** The cast block is passed straight to
  `StringConcatenate`, so llava cannot paraphrase the character away — which is exactly
  what happens if you let a describer regenerate the whole prompt from a drifted frame.
- **Only the action is dynamic.** llava sees the real first frame of that shot and
  writes what happens next, so shots respond to where the previous one actually ended.

Its system context forbids the failure modes we found earlier:

> Never describe appearance. No colours, clothing, hair, fur, species, body, face, or
> age. […] Never use negations. […] Output the sentence only.

Turn it on with `--ollama`, or per shot with `"use_ollama": true`. Off by default —
run a shot both ways and compare before committing, and watch the `Final prompt sent to
the model` preview inside the subgraph to see what actually got encoded.

One caveat worth knowing: on `anchor: previous` shots, llava is looking at a degraded
frame. It won't describe identity (the system context blocks that), but if a frame is
badly smeared the action it writes may be vague. Pair `--ollama` with `--anchor
alternate` so it's reading a clean frame at least half the time.

### If drift still shows up

Work down this list in order — it's ordered by effect per unit of effort:

1. Shorten shots to 8–10s (`duration_sec`), add more of them.
2. `--anchor reference` for everything; accept the cuts.
3. `--handoff-index -6` or further back.
4. `--img-compression 5`.
5. Raise the low-res `LTXVImgToVideoInplace` strength from 0.7 toward 0.85.
6. Train a 2.5 character LoRA.


`easy forLoopStart` / `easy forLoopEnd` (ComfyUI-Easy-Use) will iterate a subgraph N
times and carry values between iterations, so the last-frame handoff would work. Budget
for the prompt problem: you'd need the shot prompt selected by loop index, which means a
string list and an index-select node. Workable for N shots sharing one prompt; awkward
the moment shots differ. Worth it only if you're regenerating the count constantly and
never touching per-shot text.
## 9. Where everything is saved

Before this change the workflow wrote **videos only**. Every identity frame — the
reference at working resolution, each shot's incoming frame, each handoff frame —
existed solely as a `PreviewImage`, which ComfyUI writes to `temp/` and wipes. There
was nothing to reuse.

### ComfyUI's three directories

| Directory | What | Persists |
|---|---|---|
| `ComfyUI/input/` | what `LoadImage` / `LoadAudio` read | yes |
| `ComfyUI/output/` | everything `Save*` writes | yes |
| `ComfyUI/temp/` | every `Preview*` node | **no — wiped** |

### What gets written now

```
ComfyUI/output/
├── castrefs/
│   └── 00_reference_00001_.png              canonical anchor, at shot resolution
└── newVideoParts/
    ├── sam_new_shot01_00001.mp4             the clip
    ├── sam_new_shot01_first_00001_.png      what shot 1 was given
    ├── sam_new_shot01_handoff_00001_.png    what shot 1 passed on
    ├── sam_new_shot02_00001.mp4
    ├── sam_new_shot02_first_00001_.png
    └── …
```

Three `SaveImage` nodes do this: one at root off `ImageScale`, and two per shot in the
new **10 · CAST LIBRARY** group inside the subgraph. Filenames are built with
`StringConcatenate` off the same `save_prefix` widget that names the video, so a shot's
clip and its two frames always sort together.

The `_first` image is taken **before** `LTXVPreprocess`, so it is the highest-fidelity
copy of what the shot received — that is the one worth keeping as a reusable anchor.
The `_handoff` image is exactly what the next shot inherits, so if shot 4's character
looks wrong you can open `shot03_handoff` and see precisely where it went wrong.

Disable with `--no-save-frames`.

### Reusing a saved frame

`LoadImage` reads from `input/`, not `output/`. To promote a good frame to a reusable
cast reference:

```bash
cp ComfyUI/output/newVideoParts/sam_new_shot02_handoff_00001_.png \
   ComfyUI/input/cast_woman_v2.png
```

Then pick it in the root `LoadImage`. Your KJNodes install may also have a
load-from-path node if you would rather point straight at `output/`.

This is what makes an iterative cast library practical: run the sequence, find the
frame where both characters read correctly, copy it to `input/`, and make it the
reference for the next run. Combined with `--anchor reference` it means every shot
starts from a frame you have personally approved, rather than from whatever the
previous generation happened to end on.


`easy forLoopStart` / `easy forLoopEnd` (ComfyUI-Easy-Use) will iterate a subgraph N
times and carry values between iterations, so the last-frame handoff would work. Budget
for the prompt problem: you'd need the shot prompt selected by loop index, which means a
string list and an index-select node. Workable for N shots sharing one prompt; awkward
the moment shots differ. Worth it only if you're regenerating the count constantly and
never touching per-shot text.
## 10. Checking the handoff frame before it propagates

Sections 8 and 9 did not solve this. `--handoff-index -6` is a blind guess,
`--anchor reference` avoids the chain by *breaking* the connection, and the saved PNGs
only let you diagnose damage after the run finished. None of it inspects a frame.

ComfyUI cannot branch mid-graph, so a real check needs a node. `reallexi_handoff/`
provides two.

### Handoff Frame Select

Replaces "take frame -1" with a scored pick across the tail of the clip:

| term | catches | default weight |
|---|---|---|
| sharpness (variance of Laplacian) | motion blur on the tail frames | 1.0 |
| identity (histogram + coarse structural cosine vs the reference) | gross drift — wrong palette, wrong layout, subject gone | 1.0 |
| recency (linear, favours later frames) | **protects the cut** — all else equal it takes the latest frame | 0.6 |
| clipping (fraction crushed/blown) | flare and collapse frames that read as "sharp" but are useless | −2.0 |

Recency is what keeps the connection smooth. The node only looks at the last `window`
frames (12 by default, ~0.5s at 25fps) and prefers the latest one, so it picks *the
latest good frame* rather than the best frame anywhere in the clip. Raise `--recency`
to favour continuity, lower it to favour image quality.

On a synthetic clip that is clean through frame 13, progressively smeared from 14, with
a flared frame at 18 and heavy blur on the final frame 19:

```
   idx    score    sharp   ident    clip
    12   2.1749    1.000   0.957   0.000
    13   2.2294    1.000   0.957   0.000 <-
    14   1.3605    0.074   0.960   0.000
    18   0.7565    0.158   0.393   0.170
    19   1.5262    0.000   0.926   0.000
```

It takes 13 — the last frame before blur onset. Not 19, not 18, and not an early frame
that would have jumped the cut backwards.

Every shot exposes a `Why this frame was chosen` preview carrying that whole table, so
when a sequence goes wrong you can see which frame was picked and why.

### Handoff Quality Gate (`--gate`, off by default)

The last line of defence. If the chosen frame *still* fails against the reference, it
forwards the reference instead:

```
similarity 0.41 (min 0.72)  clipping 0.02 (max 0.35)
verdict: FAIL
action: forwarding the REFERENCE - this shot starts as a cut.
```

Since ComfyUI cannot branch, this is a pure selection between two already-computed
inputs — both paths execute, one is forwarded. Falling back trades continuity for
identity, so it should trip rarely; read `Gate verdict` before trusting a run. Set
`on_fail` to *pass through anyway* to make it report-only.

The identity terms need a clean anchor, so the shot subgraph gained an `identity_ref`
input, wired at root from the canonical reference for **every** shot regardless of
anchor mode.

### Install and use

```bash
cp -r reallexi_handoff <ComfyUI>/custom_nodes/    # then restart ComfyUI
python test_handoff.py                            # 19 assertions, no GPU needed
```

```bash
python build_ltx25_multishot.py --config shots.json            # selector on
python build_ltx25_multishot.py --config shots.json --gate     # + reference fallback
python build_ltx25_multishot.py --window 20 --recency 0.3      # search wider, favour quality
python build_ltx25_multishot.py --no-frame-check               # old blind behaviour
```

Only numpy is used for scoring; torch appears at the tensor boundary. Nothing to
download, no model weights.

### What it does not do

The identity metric is a tripwire for wholesale collapse, not face verification. It will
not notice a subtly different jawline. For that you need a character LoRA or a real
face-embedding node — this catches the failures that ruin a whole chain, not the ones
you would argue about.


`easy forLoopStart` / `easy forLoopEnd` (ComfyUI-Easy-Use) will iterate a subgraph N
times and carry values between iterations, so the last-frame handoff would work. Budget
for the prompt problem: you'd need the shot prompt selected by loop index, which means a
string list and an index-select node. Workable for N shots sharing one prompt; awkward
the moment shots differ. Worth it only if you're regenerating the count constantly and
never touching per-shot text.
## 11. Holding the look — no drifting from live action to 3D

Style flip is a different failure from identity drift and needs its own lock. Nothing
in the original file declared a medium at all, so each shot was free to pick one, and
`anchor: reference` shots make it worse — they start fresh, so there is no visual
inertia carrying the look forward.

Three layers now hold it, at three different points in the pipeline.

### 1. A `look` block, pinned first in every prompt

```json
"look": {
  "medium":   "live-action footage shot on a full-frame cinema camera",
  "grade":    "warm golden-hour palette with amber highlights, soft neutral shadows and gentle film contrast",
  "lighting": "low sunlight from a window on the right, soft falloff",
  "lens":     "35mm lens, shallow depth of field",
  "texture":  "fine natural film grain"
}
```

becomes, at the **front** of every shot prompt:

> Live-action footage shot on a full-frame cinema camera, warm golden-hour palette with
> amber highlights, soft neutral shadows and gentle film contrast, low sunlight from a
> window on the right, soft falloff, 35mm lens, shallow depth of field, fine natural
> film grain. Every shot keeps this same medium, palette and lighting.

Position matters — text encoders weight early tokens heavily, and the medium is the one
thing that must never move. Full prompt order is now **LOOK → CAST → BEAT**: what it is
shot on, who is in it, what they do. Only the last part varies.

### 2. Medium negatives, derived automatically

The builder reads the declared medium and puts every *other* medium in the negative:

| declared | auto-negatives |
|---|---|
| live action | `3d render, cgi, computer animation, cartoon, anime, claymation, video game screenshot, unreal engine render, plastic shading` |
| 3D animated | `live-action footage, photographic, documentary footage, photoreal skin` |
| 2D animated | `3d render, cgi, live-action footage, photoreal` |
| stop motion | `3d render, cgi, live-action footage, smooth cg animation` |

plus `style change, medium change, colour grade shift, palette shift, mismatched
lighting between shots` regardless. Remember these only bite at CFG above 1 — see §7.

### 3. `Handoff Color Match` — the mechanical backstop

Prompt text cannot actually hold a grade. The model re-grades every generation
slightly, and over four shots the palette walks. This node transfers the reference's
colour statistics onto the seed frame before it feeds the next shot, so grade drift
**cannot accumulate** regardless of what the sampler did:

```
reference channel means: [0.421 0.378 0.307]
drifted   channel means: [0.293 0.336 0.464]     <- shot 3 has gone cold
after mean/std           [0.421 0.378 0.311]
```

- **mean/std (gentle)** — Reinhard per-channel transfer. Keeps local contrast. Default.
- **histogram (strong)** — full CDF match. Corrects harder, can band on smooth gradients.
- **strength** (0.85 default) — lower it if shots start looking flat or over-uniform.
- **preserve_luminance** — fix hue and saturation but keep the frame's own brightness.

It sits **after** the quality gate deliberately: the gate must judge the raw pick, or a
frame gets rescued by a grade correction that hides real structural drift.

What it does not do: a frame that came out as 3D animation stays 3D animation, just
correctly graded. Medium is the prompt's job; palette is this node's job.

### Reading the look off the reference instead of writing it

Group `03` now has a second Ollama describer wired to the reference image, whose only
task is to name the medium, palette, lighting and lens — it is forbidden from
mentioning subjects. Flip `Read the look from the image instead?` to use it, and the
`Look pinned to every shot` preview shows exactly what got locked in.

It runs **once**, on the clean reference, and the result is wired to every shot. That is
the correct place for it: the look must be constant, so it must never be re-read from a
drifted frame — unlike the shot director in §8, which reads per shot precisely because
the action should respond to where the last one ended.

```bash
python build_ltx25_multishot.py --config shots.json --gate      # all layers on
python build_ltx25_multishot.py --color-strength 1.0            # pin the palette hard
python build_ltx25_multishot.py --no-color-match                # prompt-side lock only
```


`easy forLoopStart` / `easy forLoopEnd` (ComfyUI-Easy-Use) will iterate a subgraph N
times and carry values between iterations, so the last-frame handoff would work. Budget
for the prompt problem: you'd need the shot prompt selected by loop index, which means a
string list and an index-select node. Workable for N shots sharing one prompt; awkward
the moment shots differ. Worth it only if you're regenerating the count constantly and
never touching per-shot text.
## 12. "Missing node" — what to install

Run the inventory first. It walks the root graph and every subgraph, then reports where
each node type comes from:

```bash
python inspect_nodes.py LTX25_Multi_Shot_Character_Lock_Sam_Ayoub.json \
       --against Clean_Multi_Video_Clips_together_Sam_Ayoub__3_.json
```

`--against` takes workflows already known to load on your machine. Anything found in
them is marked `✓ proven here`, which narrows the suspects to the handful that are not.
Against the original file, only two things in the full build come up unproven:

| type | count | where it comes from |
|---|---|---|
| `HandoffFrameSelect` / `HandoffQualityGate` / `HandoffColorMatch` | 3 | this project's pack — **not installed by default** |
| `StringConcatenate` | 5 | ComfyUI core, but only on recent builds |

Everything else — every LTX node, `ComfySwitchNode`, `ComfyMathExpression`,
`OllamaImageDescriber`, `GetImagesFromBatchIndexed`, `CacheCleaner` — is already proven
by your own files.

### Fix 1 — deploy the pack

`reallexi_handoff/` is a self-contained package that lives in **your repo** as the
source of truth. `install.py` only pushes it into ComfyUI:

```bash
python reallexi_handoff/install.py --comfy ".\\ComfyUI"
python reallexi_handoff/install.py --link        # junction instead of copy
python reallexi_handoff/install.py --uninstall
```

It copies only the runtime files — `tests/`, `.gitignore` and caches stay in the repo —
then imports the deployed copy the way ComfyUI's loader does and prints what registered:

```
repo    : ...\custom-wan\reallexi_handoff
ComfyUI : ...\custom-wan\ComfyUI
target  : ...\ComfyUI\custom_nodes\reallexi_handoff
mode    : copied 5 files

  import OK — v1.0.0, 3 nodes registered:
    HandoffFrameSelect     → Handoff Frame Select
    HandoffQualityGate     → Handoff Quality Gate
    HandoffColorMatch      → Handoff Color Match
```

**`--link` is the one to use while iterating.** It points `custom_nodes` at the repo
folder rather than copying, so edits go live on the next ComfyUI restart with no
re-deploy. On Windows it creates a directory junction, which does not need
Administrator. `--uninstall` removes the link and leaves the repo untouched.

With no `--comfy`, it walks up from the repo looking for a sibling `ComfyUI/`, so from
`custom-wan/` it finds `custom-wan/ComfyUI` on its own.

Restart ComfyUI and hard-refresh the browser (Ctrl+Shift+R) — the frontend caches its
node list, and skipping the refresh is the usual reason a correctly-installed node still
shows red.

A missing-node list that repeats each type four times is normal: four shot instances
share one subgraph definition, so ComfyUI counts four references to each.

### Package layout

```
reallexi_handoff/
├── __init__.py        node registration, __version__
├── nodes.py           ComfyUI node classes — owns the torch boundary
├── scoring.py         pure numpy metrics, no ComfyUI import
├── install.py         deploy into custom_nodes
├── pyproject.toml     Comfy Registry metadata
├── requirements.txt   numpy
├── README.md
└── tests/
    └── test_scoring.py
```

The split is deliberate: `scoring.py` imports nothing but numpy, so the maths runs
under CI or on any machine without ComfyUI, a GPU or torch:

```bash
python -m reallexi_handoff.tests.test_scoring     # 27 assertions
```

### Fix 2 — `StringConcatenate`

Core, from the string utility set. If it comes up red, ComfyUI is behind; update. It has
`string_a`, `string_b`, and a `delimiter`. If you would rather not update, use compat
mode below — it removes the node entirely.

### Fix 3 — compat build, guaranteed to load

`LTX25_Multi_Shot_COMPAT.json` is built with **only node types your original workflow
already used**:

```bash
python build_ltx25_multishot.py --config shots.json --compat
```

The inventory on that build comes back clean — every type proven by your own files
except `SaveImage`, which ships in ComfyUI's default workflow.

| | full build | `--compat` |
|---|---|---|
| one shared subgraph definition | yes | yes |
| LTX-2.5 model stack | yes | yes |
| `look` + `cast` + `beat` prompt structure | yes | yes, **pre-composed in Python** |
| anchor modes, per-shot seeds and audio windows | yes | yes |
| cast library PNGs | per-shot filenames | static prefix, auto-incremented |
| scored frame pick, quality gate, colour match | yes | **no** — blind `-1` index |
| Ollama shot director (per-shot action) | yes | **no** — needs runtime concat |
| Ollama look reader | yes | **no** |

Compat keeps everything that is structural and drops everything that needs a node you
might not have. Start there if a node is red, confirm it runs, then move up.

### Reading the error

ComfyUI names the missing type in the red node's title bar and lists them in the
"Missing Node Types" dialog on load. Match that string against the inventory output —
the type name is exact, so a mismatch means a different node than you expect.


`easy forLoopStart` / `easy forLoopEnd` (ComfyUI-Easy-Use) will iterate a subgraph N
times and carry values between iterations, so the last-frame handoff would work. Budget
for the prompt problem: you'd need the shot prompt selected by loop index, which means a
string list and an index-select node. Workable for N shots sharing one prompt; awkward
the moment shots differ. Worth it only if you're regenerating the count constantly and
never touching per-shot text.
### If you really want the loop inside ComfyUI

`easy forLoopStart` / `easy forLoopEnd` (ComfyUI-Easy-Use) will iterate a subgraph N
times and carry values between iterations, so the last-frame handoff would work. Budget
for the prompt problem: you'd need the shot prompt selected by loop index, which means a
string list and an index-select node. Workable for N shots sharing one prompt; awkward
the moment shots differ. Worth it only if you're regenerating the count constantly and
never touching per-shot text.

---

## 13. Bug fix — subgraph input numbering

**Symptom** (ComfyUI 0.33.1):

```
* PrimitiveStringMultiline 35:52:
  - Return type mismatch between linked nodes: value, received_type(INT) mismatch input_type(STRING)
* PrimitiveBoolean 35:53:
  - Return type mismatch between linked nodes: value, received_type(INT) mismatch input_type(BOOLEAN)
...
RuntimeError: Input and output sizes should be greater than 0,
              but got input (H: 1536, W: 0) output (H: 512, W: 0)
```

**Cause.** ComfyUI resolves generic subgraph inputs by **name** (`value`, `value_1`,
`value_2` …), not by position in the `inputs` array. When the `look` block was added,
the new input took the free `value` name and `cast` was hand-numbered `value_7` while
sitting at array slot 4 — so name order and array order disagreed:

```
slot  name             label
   3  value            look
   4  value_7          cast      <- out of sequence
   5  value_1          beat
```

Every widget after that point shifted by one against its intended input:

```
slot(by name)    intended         actually got
value_2          use_ollama       '<beat text>'    STRING into BOOLEAN
value_3          width            False
value_7          cast             20.0             INT into STRING
```

`width` receiving `False` is where the crash came from: it collapsed to 0, so
`EmptyLTXVLatentVideo` was built at width 0 and `LTXVImgToVideoInplace` tried to resize
to `W: 0`.

**Fix.** Input names are no longer written by hand. The builder marks generic inputs
`"*"` and assigns `value`, `value_1`, `value_2` … from array order, so the two orders
cannot diverge. Two guards now make the failure impossible to ship silently:

- `_assert_sequential()` raises at build time.
- `validate_workflow.py` reports it on any workflow, including hand-edited ones:

```
ERROR  input numbering not monotonic: 'look'=value_0 at slot 3, then
       'cast'=value_7 at slot 4. ComfyUI maps these by name — widgets and
       links will land on wrong nodes.
```

Worth running against anything you edit by hand in the ComfyUI UI, since reordering
subgraph inputs there can reintroduce exactly this.

---

## 14. Singing and speaking characters

Your audio path already does the hard part. `TrimAudioDuration → LTXVAudioVAEEncode →
SetLatentNoiseMask(SolidMask 0) → LTXVConcatAVLatent` locks the audio latent and
generates video conditioned on it — that coupling is exactly what drives mouth movement
to a supplied track. What was missing was the prompt telling it anyone was vocalising,
plus one setting actively fighting it.

### The conflict I introduced

I raised the low-res `LTXVImgToVideoInplace` strength from your 0.5 to 0.7 for identity
locking. For lip sync that is the wrong direction: with image strength too high the
model preserves the starting frame so hard that the jaw refuses to open — the audio
plays and the character looks like a ventriloquist. Community guidance for LTX audio
work is around **0.65** once speaking begins. Your original 0.5 was closer to right for
singing than my 0.7.

```bash
python build_ltx25_multishot.py --config shots.json --lipsync
```

sets image strength to **0.65** and frame rate to **30**. Higher frame rates give the
sync algorithm finer timing to map against the waveform; 24–25 is on the coarse side for
heavy mouth work. Both are still overridable with `--strength-low` and `--fps`.

### `voice` on a cast member

```json
{
  "kind": "cat",
  "look": "a small tabby cat with bright blue eyes and soft short fur",
  "anatomy": "four legs, a full feline body, a long tail, whiskers and pointed ears",
  "where": "centre frame, standing in the grass",
  "voice": "sings the melody in a warm clear voice"
}
```

produces:

> The cat sings the melody in a warm clear voice, its mouth opening and closing in time
> with the audio, jaw and lips articulating each word clearly. **The cat keeps the same
> cat identity throughout the shot.**

Note the last sentence changed. For silent characters the assertion is *"stays entirely
a cat"* — an anatomy claim. For a singing one that would forbid the very mouth
articulation you are asking for, so it narrows to **identity**, which is what you
actually want held.

### The negatives flip too

`anthropomorphic` is in the anti-hybrid list, and it fights a singing animal directly.
When any cast member has a `voice`, the builder drops it and `cartoon`, then adds the
failure modes that matter here:

```
closed motionless mouth, still jaw, frozen face, mouth out of sync with the audio,
mismatched lip movement, ventriloquist, stretched mouth, distorted jaw, flapping mouth
```

`chimera`, `merged anatomy` and the cross-species terms all stay — a singing cat still
must not grow a human face.

### Shot length matters more here

20-second shots are past the comfortable range for sustained mouth work; sync tends to
drift toward the tail. `shots_twilight_field.json` uses **8s** shots — 32s total at 30fps
is ~960 frames against 2000 for the old four-by-twenty layout, and each shot's mouth
timing only has to hold for eight seconds.

### Two more levers you already have

- **Vocal separation.** `ComfyUI_Demucs` is in your `custom_nodes`. Conditioning on an
  isolated vocal stem gives cleaner mouth cues than a full mix with drums and pads in
  it; render the final video against the full mix. This is standard practice in the
  LTX lip-sync workflows.
- **Framing.** A mouth that occupies a handful of pixels cannot sync. Close and medium
  shots sync visibly better than wides — worth weighting your shot list accordingly.

Note the LTX-2.3 dual-character lip-sync LoRA will not load on a 2.5 transformer, same
compatibility break as your distilled LoRA.

### Your twilight brief, converted

`shots_twilight_field.json` is your pasted prompt restructured, and
`LTX25_Twilight_Field_LipSync.json` is the built workflow. Three things changed:

1. **Four named characters instead of four generic roles.** "One curious main subject"
   and "one energetic fast-moving companion" bind no attributes to anything — that is
   the setup that produced your cat-human hybrid. Each now has its own closed block with
   `kind`, `look`, `anatomy` and `where`.
2. **Negations moved to the negative branch.** `never have text, watermark or logos`
   in a positive prompt embeds "text", "watermark" and "logos" as concepts that are
   *present*. Your existing negative already covers all three.
3. **The meta-line dropped.** `No "no text" rule exists in ANY connected prompt node` is
   an instruction to a reader; a text encoder just embeds the words, including "no text".

The calm → build → peak arc maps to the four shots (CALM, BUILD, GATHER, PEAK), with
the cat and fox cub carrying `voice` and the tortoise and floating light silent.

---

## 15. `--chain` — pure sequential, stock nodes only

```bash
python build_ltx25_multishot.py --config shots.json --chain
```

Sets four things at once:

| | |
|---|---|
| `--no-frame-check` | drops `HandoffFrameSelect` and `HandoffQualityGate` |
| `--no-color-match` | drops `HandoffColorMatch` |
| `--anchor previous` | every shot after the first continues the chain |
| `--handoff-index -1` | the literal final frame |

**No custom nodes remain.** The last-frame extraction is stock
`GetImagesFromBatchIndexed` at `-1`. That node cannot be removed — it *is* the chain
mechanism — but it needs no pack.

`--chain` also strips per-shot `anchor` values out of the config, so a stray
`"anchor": "reference"` cannot silently break the sequence.

### Ordering is enforced by the graph, not by hope

ComfyUI executes on dependencies, and every shot's `first_frame` comes from the previous
shot's IMAGE output, so no shot can start before the one ahead of it finishes:

```
forced execution order: CALM → BUILD → GATHER → PEAK
```

A topological pass confirms only one shot is ever runnable at a time. Worth noting the
contrast: with `--anchor reference` or `alternate`, the re-anchored shots have no
dependency on their predecessors, so ComfyUI is free to run them in any order. Pure
chaining is the only mode where run order is guaranteed.

The builder prints the resolved chain:

```
  4 shots
    1. CALM       first frame ← reference image
    2. BUILD      first frame ← CALM final frame
    3. GATHER     first frame ← BUILD final frame
    4. PEAK       first frame ← GATHER final frame
```

and `validate_workflow.py` now traces it from the actual links and fails on a break:

```
ERROR  PEAK seeded from CALM, not the immediately preceding shot GATHER
       — chain order is broken
```

### What you give up

Chaining without the gate reinstates the drift this whole thread has been fighting.
Nothing now inspects the frame before it propagates, nothing corrects palette walk, and
`-1` takes the most motion-blurred frame in the clip. Mitigations that still apply,
all free:

1. **Short shots.** `shots_twilight_field.json` uses 8s. Tail degradation scales with
   length, so short shots are the single biggest lever left.
2. **`--handoff-index -6`** still works in chain mode — override it explicitly and you
   back off the blurriest frame while keeping the sequence intact.
3. **The look block** is now the only thing holding the grade. Keep it specific.
4. **Watch the saved PNGs.** `*_handoff_*.png` for each shot is exactly what the next
   one inherits — check shot N's handoff before blaming shot N+1.

If a run drifts badly, `--gate` on its own reinstates the checks without touching the
chain: it keeps `anchor: previous` everywhere and only adds the inspection nodes back.

### Files

- `LTX25_Multi_Shot_Chain.json` — your four-shot layout, pure chain
- `LTX25_Twilight_Field_Chain.json` — the twilight brief, pure chain, lip-sync tuned
