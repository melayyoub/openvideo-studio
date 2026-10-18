# CastForge

Story → cast → LoRA → movie, on top of **your** `video_ltx2_5_i2v_especial_edition_sam.json`.

Your workflow is the template, not the casualty. Its subgraphs (`LTX 2.5 scenecontroller`,
`Image to Video (LTX-2.5)`, `Merge Videos`) are lifted out verbatim — every sigma schedule,
VAE, upscaler and sampler setting you tuned stays exactly as tuned. What CastForge adds is
everything around them.

```
subject + mood + age
        │
        ▼  Ollama                    ── stage 1  story
   story_bible.json ────────────── cast bible + N scenes + dialogue + voices
        │
        ▼  ComfyUI + vision         ── stage 3  cast
   10 views per character ───────── cast_registry.json  (locked, append-only)
        │
        ▼                           ── stage 4  lora
   datasets + trainer configs ───── image LoRA + LTX-2.5 video LoRA
        │
        ▼  ComfyUI + LoRA           ── stage 5  keyframes
   per-shot anchors ─────────────── shot 1 = YOUR original image
        │
        ▼                           ── stage 6  movie
   N-shot LTX-2.5 workflow ──────── open in ComfyUI, queue, done
```

---

## The one idea that makes this work

**The language model never writes identity or voice. It writes motion.**

Identity and voice text is assembled by string formatting from the cast registry and pasted
byte-identically into every shot. Shot 1 and shot 40 get literally the same bytes describing
each character, so there is nothing for drift to act on at the prompt level. `prompt_composer.verify()`
fails the build if a locked block does not survive into the final prompt.

The second half is that LTX i2v only ever sees **one image** — the first frame. So the pipeline
makes sure that frame is already correct, in every shot, using the trained character LoRA. That
is what `anchor_mode` controls.

---

## Install

```bash
pip install -r requirements.txt
```

Needs: ComfyUI (yours, with LTX-2.5 models), Ollama with a text model and a vision model.

```bash
ollama pull qwen2.5:14b-instruct     # story writer  (or any solid instruct model)
ollama pull llava:13b-v1.6           # vision: read-back, direction, drift audit
```

Before launching ComfyUI, for quantized LTX on your OpenVideo Studio build:

```bat
set CUSTOM_WAN_COMFYUI_ARGS=--disable-pinned-memory
```

Verify the plumbing without a GPU:

```bash
python tools/selftest.py
```

That builds a full 5-shot workflow from synthetic data and checks link integrity, LoRA
splicing, director bypass, and lock-signature stability. 30 checks, no network.

---

## Run

```bash
python run_pipeline.py --stage story        # Ollama writes the whole film
python run_pipeline.py --stage cast         # 10 views per character
python run_pipeline.py --stage lora         # datasets + configs  (add --train to launch)
python run_pipeline.py --stage keyframes    # anchors, LoRA-rendered
python run_pipeline.py --stage movie        # emit the ComfyUI workflow

python run_pipeline.py --stage all          # all of the above, resumable
python run_pipeline.py --status             # what exists so far
```

Then open `projects/<name>/movie/<name>_castforge_movie.json` in ComfyUI and queue it.

### Minimum input

`config.json` → `story_input`:

```json
{ "subject": "a small boy and his tiny dragon find a valley of floating islands",
  "mood": "warm, wondrous, gently adventurous",
  "age": 6 }
```

That is the whole ask. Ollama produces title, theme, world, palette, beat sheet, 2–4 fully
specified characters (down to eye colour, freckles, wardrobe, colour palette, motion
signature and a seven-field voice profile), then every scene with action, camera, motion
beats, dialogue, ambience, music, and explicit continuity notes — all schema-validated with
a bounded repair loop, so it is JSON or it is an error, never prose you have to clean up.

### Character images

Drop them in `cast_sources/`, named after the cast id the story engine minted
(`milo.png`, `ember_side.jpg`, …). Check the ids first with `--status`. Multiple images per
character are fine; the first is the identity anchor, the rest are recorded as sources.

---

## The ten views

Full 360° coverage plus the close range a video model needs for expression and lip-sync.

| # | view | angle | why |
|---|---|---|---|
| 01 | front | 0° | silhouette, front wardrobe |
| 02 | ¾ left | 45° | the workhorse cinematic angle |
| 03 | profile left | 90° | jaw and nose line |
| 04 | rear ¾ left | 135° | rear seams, hair mass |
| 05 | back | 180° | full rear coverage |
| 06 | ¾ right | 315° | mirror-side asymmetry |
| 07 | profile right | 270° | mirror-side jaw |
| 08 | face front | 0°, close | **the identity anchor**, 4× weight in training |
| 09 | face expressive | 30°, close | emotion + open mouth for lip-sync, 3× weight |
| 10 | action + hands | dynamic | motion silhouette, hands, props |

Before generating, a vision read-back catalogues the actual source image and merges it into
the registry — so the `identity_line` you lock describes the picture you already like, not
what the text model imagined. Contact sheets land in `cast/contact_<id>.png`.

---

## The cast registry

`cast/cast_registry.json` is the single source of truth. Everything downstream reads only from it.

- **Append-only.** `upsert` merges; empty values never blank out detail you already have.
- **Lockable.** `locked: true` freezes an entry. `--force` will not touch it; only an explicit
  `--unlock` will. This is what stops you losing a cast you already like.
- **Snapshotted.** Every write copies the previous file to `cast/_backups/`.

Per character it holds: every appearance field, `identity_line`, `never` rules, palette,
motion signature, the full voice profile, all ten view paths with captions and hashes, the
source images, the LoRA names, and two deterministic seeds — `base_seed` for visuals and
`audio_seed` for voice.

---

## LoRA

Two per character, because they do different jobs:

- **image LoRA** — used to render each shot's keyframe. In an i2v pipeline the still *is* the
  identity, so this is the one that keeps the face right.
- **video LoRA** — loaded onto the LTX-2.5 transformer itself, so identity survives motion and
  not just frame 0.

`--stage lora` writes captioned datasets (trigger token first, then identity line, then view),
`metadata.jsonl`, an ai-toolkit YAML per character, an LTX-Video-Trainer YAML per character,
and `train_all.ps1` / `train_all.sh`. Run them from inside the respective trainer repos, or
pass `--train` to launch automatically.

Point `lora.output_dir` at your ComfyUI `models/loras/cast` folder. The movie builder checks
which files actually exist and only wires those in — an untrained cast renders fine, just
without the extra hold.

---

## Anchor modes — how drift is actually stopped

| mode | behaviour | trade-off |
|---|---|---|
| `chain` | every shot starts from the previous shot's last frame | best motion continuity, error compounds; by shot 8 the face is somebody else |
| `reanchor` | every shot starts from a fresh LoRA-rendered keyframe | zero drift, looser pose continuity between shots |
| `hybrid` *(default)* | chain, but re-anchor every `reanchor_every` shots and whenever the drift audit fires | drift can never accumulate past N shots |

The drift audit is a vision pass comparing a frame against the character's locked `v08_face_front`
view; it returns `identity_match` / `wardrobe_match` scores and a list of what changed. Below
`movie.drift_threshold`, the shot is re-anchored.

Shot 1 always keeps `movie.first_image` — your original opening frame, untouched.

---

## Director mode

`movie.director_mode: "python"` (default) mutes the in-graph `OllamaImageDescriber` and wires
the composed prompt straight into the sampler.

This matters more than it sounds. Left in circuit, that node re-reads the frame at render time
and rewrites your prompt — which means it paraphrases the identity block you just locked, and
your characters drift because of your own director node. In `python` mode Ollama still directs
(via `StoryEngine.direct_scene`, same vision model, same job) but its output is validated,
length-checked, and appended *underneath* the locked blocks rather than replacing them.

Set it to `"graph"` to restore the original in-graph behaviour.

---

## Voices

LTX-2.5 generates audio natively, so voice continuity is prompt continuity plus seed discipline:

- every shot carries the identical `[VOICE_LOCK]` block for the characters present
- each character has a fixed `audio_seed`, minted once and stored in the registry
- the block names timbre, pitch, pace, accent, quirk, default emotion and a signature sound
- `[AUDIO]` renders dialogue as *"X says, in their established … voice"* rather than bare text
- ambience and music carry an explicit no-audio-cut instruction between shots

---

## What lands where

```
projects/<name>/
  story/story_bible.json          title, cast, scenes, dialogue, continuity rules
  cast/cast_registry.json         the source of truth
  cast/views/<id>/v01..v10.png    the ten views
  cast/contact_<id>.png           contact sheet
  cast/_backups/                  every prior registry version
  lora/datasets/<id>/             images + .txt captions + metadata.jsonl
  lora/configs/                   ai-toolkit + LTX trainer YAMLs, train_all.ps1/.sh
  keyframes/scene_NN.png          anchors
  keyframes/anchor_plan.json      per-shot anchor decisions
  movie/prompts/shot_NN.txt       the exact prompt each shot receives — read these
  movie/<name>_castforge_movie.json
  state.json                      stage completion, for resuming
```

---

## Templates

`templates/qwen_edit_turnaround.api.json` is a Qwen-Image-Edit-2509 graph used for both the
ten views and the keyframes. **Node names vary between installs.** If it errors, export your own
working edit workflow from ComfyUI with **Export (API)**, drop it in `templates/`, and point
`cast.turnaround_template` at it — then update `cast.turnaround_patch_map` to its node titles:

```json
{ "image": "LoadImage", "prompt": "PositivePrompt", "negative": "NegativePrompt",
  "sampler": "KSampler", "save": "SaveImage", "latent": "EmptySD3LatentImage" }
```

Patching is by `_meta.title`, then node id, then class name. Missing optional targets are
skipped with a warning rather than crashing the run. LoRA nodes are spliced in programmatically,
so your template does not need to contain any.

---

## Requirement map

| your ask | where it lives |
|---|---|
| 1. 10 views covering full cast poses | `views.py`, `cast_forge.py` |
| 2. added to cast members | `cast_registry.attach_view` |
| 3. all details stored for reuse | `cast_registry.json` + vision read-back |
| 4. same characters flow through the movie | `prompt_composer` locks + `anchor_mode` |
| 5. LoRA trained before the movie | `lora.py`, stage 4 runs before stages 5–6 |
| 6. no identity or detail loss | verbatim locks + LoRA on transformer + drift audit |
| 7. Ollama keeps flow, voices and sounds | `story_engine.direct_scene`, `[VOICE_LOCK]`, `[AUDIO]`, fixed `audio_seed` |
| 8. story from subject + mood + age only | `story_engine.build`, schema-validated |
| cast and first image never lost | locked registry, backups, `movie.keep_first_image` |
