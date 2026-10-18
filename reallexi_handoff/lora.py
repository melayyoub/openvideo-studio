"""Stage 4 — LoRA.

The ten views are the training set. This stage assembles them into a captioned dataset
per character, writes ready-to-run trainer configs, and (optionally) launches training.

Two LoRAs per character, because they do different jobs:

* **image LoRA** — used when rendering each shot's keyframe. This is the one that keeps
  the face right, because in an image-to-video pipeline the still frame *is* the identity.
* **video LoRA** — loaded onto the LTX-2.5 transformer itself, so identity survives the
  motion, not just the first frame.

Captions lead with the trigger token, then the identity line, then the view description.
That ordering teaches the token to carry the whole character rather than just a pose.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cast_registry import CastRegistry
from .config import Config
from .util import banner, log, write_json


class LoraStage:
    def __init__(self, cfg: Config, registry: CastRegistry):
        self.cfg = cfg
        self.registry = registry

    # ------------------------------------------------------------------ #
    # dataset
    # ------------------------------------------------------------------ #

    def build_dataset(self, cid: str, force: bool = False) -> Path:
        ch = self.registry.get(cid)
        ds_dir = self.cfg.dataset_dir / cid
        if ds_dir.exists() and force:
            shutil.rmtree(ds_dir)
        img_dir = ds_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        views = ch.get("views", {})
        if not views:
            raise ValueError(f"{cid}: no views on file. Run the cast stage first.")

        manifest: List[Dict[str, Any]] = []
        count = 0
        for vid, meta in sorted(views.items()):
            src = self.cfg.project_root / meta["path"]
            if not src.exists():
                log(f"{cid}: missing view file {src}", "warn")
                continue
            repeats = max(1, int(meta.get("repeats", 1)))
            caption = meta.get("caption", "")
            for r in range(repeats):
                stem = f"{cid}_{vid}" + (f"_r{r+1}" if repeats > 1 else "")
                dst = img_dir / f"{stem}{src.suffix}"
                if not dst.exists() or force:
                    shutil.copy2(src, dst)
                dst.with_suffix(".txt").write_text(caption, encoding="utf-8")
                manifest.append({
                    "file_name": f"images/{dst.name}",
                    "text": caption,
                    "view": meta.get("view", vid),
                    "tags": meta.get("tags", []),
                })
                count += 1

        # metadata.jsonl is what LTX-Video-Trainer and most HF-style loaders expect
        with open(ds_dir / "metadata.jsonl", "w", encoding="utf-8") as fh:
            for row in manifest:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        write_json(ds_dir / "dataset_info.json", {
            "character_id": cid,
            "name": ch.get("name"),
            "trigger_token": ch.get("trigger_token"),
            "identity_line": ch.get("identity_line"),
            "voice": ch.get("voice"),
            "image_count": count,
            "source_views": len(views),
        })
        log(f"{cid}: dataset {count} images -> {ds_dir}")
        return ds_dir

    # ------------------------------------------------------------------ #
    # configs
    # ------------------------------------------------------------------ #

    def _image_config(self, cid: str, ds_dir: Path) -> Path:
        """ai-toolkit style YAML. Works for Qwen-Image / Flux LoRA training."""
        ch = self.registry.get(cid)
        L = self.cfg.get("lora", {})
        name = ch["lora_name"]
        out_dir = str(Path(L.get("output_dir", "./loras")).as_posix())
        yaml = f"""# CastForge :: image LoRA for {ch['name']} ({cid})
# trigger token: {ch['trigger_token']}
# run:  python run.py {name}_image.yaml        (ai-toolkit)
job: extension
config:
  name: "{name}"
  process:
    - type: sd_trainer
      training_folder: "{out_dir}"
      device: cuda:0
      trigger_word: "{ch['trigger_token']}"
      network:
        type: lora
        linear: {L.get('rank', 32)}
        linear_alpha: {L.get('alpha', 32)}
      save:
        dtype: float16
        save_every: {L.get('save_every', 400)}
        max_step_saves_to_keep: 4
      datasets:
        - folder_path: "{(ds_dir / 'images').as_posix()}"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          shuffle_tokens: false
          cache_latents_to_disk: true
          resolution: [{L.get('resolution', 1024)}]
      train:
        batch_size: {L.get('batch_size', 1)}
        steps: {L.get('steps', 1600)}
        gradient_accumulation_steps: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: flowmatch
        optimizer: adamw8bit
        lr: {L.get('learning_rate', 1e-4)}
        dtype: bf16
      model:
        name_or_path: "{L.get('image_base_model', 'Qwen/Qwen-Image-Edit-2509')}"
        is_flux: false
        quantize: true
      sample:
        sampler: flowmatch
        sample_every: {L.get('save_every', 400)}
        width: 768
        height: 1024
        prompts:
          - "{ch['trigger_token']}, {ch['name']}, head and shoulders portrait, neutral expression, studio light"
          - "{ch['trigger_token']}, {ch['name']}, full body, three-quarter view, walking forward"
        neg: "text, watermark, deformed, extra limbs"
        seed: {ch.get('base_seed', 42)}
        walk_seed: true
        guidance_scale: 3.5
        sample_steps: 24
meta:
  name: "{name}"
  castforge_character: "{cid}"
"""
        p = self.cfg.lora_config_dir / f"{name}_image.yaml"
        p.write_text(yaml, encoding="utf-8")
        return p

    def _video_config(self, cid: str, ds_dir: Path) -> Path:
        """LTX-Video-Trainer style YAML, so identity holds through motion."""
        ch = self.registry.get(cid)
        L = self.cfg.get("lora", {})
        name = ch["lora_name"]
        out_dir = str(Path(L.get("output_dir", "./loras")).as_posix())
        yaml = f"""# CastForge :: LTX-2.5 LoRA for {ch['name']} ({cid})
# trigger token: {ch['trigger_token']}
# run:  python scripts/train.py {name}_ltx.yaml      (LTX-Video-Trainer)
model:
  model_source: "{L.get('video_base_model', 'Lightricks/LTX-2.5')}"
  training_mode: "lora"
  load_checkpoint: null

lora:
  rank: {L.get('rank', 32)}
  alpha: {L.get('alpha', 32)}
  dropout: 0.0
  target_modules:
    - "to_k"
    - "to_q"
    - "to_v"
    - "to_out.0"

optimization:
  learning_rate: {L.get('learning_rate', 1e-4)}
  steps: {L.get('steps', 1600)}
  batch_size: {L.get('batch_size', 1)}
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  optimizer_type: "adamw8bit"
  scheduler_type: "cosine_with_restarts"
  scheduler_params: {{}}
  enable_gradient_checkpointing: true
  first_frame_conditioning_p: 0.5

data:
  preprocessed_data_root: "{(ds_dir / '.precomputed').as_posix()}"
  dataset_file: "{(ds_dir / 'metadata.jsonl').as_posix()}"
  num_dataloader_workers: 2

validation:
  prompts:
    - "{ch['trigger_token']}, {ch['name']}, {ch['identity_line']} turning slowly toward camera and smiling"
  negative_prompt: "worst quality, blurry, jittery, distorted, text, watermark"
  images: null
  video_dims: [{self.cfg.get('project.width', 768)}, {self.cfg.get('project.height', 1344)}, 49]
  seed: {ch.get('base_seed', 42)}
  inference_steps: 30
  interval: {L.get('save_every', 400)}
  videos_per_prompt: 1

acceleration:
  mixed_precision_mode: "bf16"
  quantization: null
  load_text_encoder_in_8bit: true
  compile_with_inductor: false

checkpoints:
  interval: {L.get('save_every', 400)}
  keep_last_n: 3

flow_matching:
  timestep_sampling_mode: "shifted_logit_normal"

seed: {ch.get('base_seed', 42)}
output_dir: "{out_dir}/{name}_ltx"
"""
        p = self.cfg.lora_config_dir / f"{name}_ltx.yaml"
        p.write_text(yaml, encoding="utf-8")
        return p

    # ------------------------------------------------------------------ #

    def _launcher(self, jobs: List[Dict[str, str]]) -> List[Path]:
        cfg_dir = self.cfg.lora_config_dir
        ps = ["# CastForge :: train every cast LoRA, image first then video",
              "$ErrorActionPreference = 'Stop'", ""]
        sh = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
        for j in jobs:
            ps += [f"Write-Host '=== {j['label']} ==='",
                   f"# {j['hint']}", f"{j['cmd']}", ""]
            sh += [f"echo '=== {j['label']} ==='", f"# {j['hint']}", f"{j['cmd']}", ""]
        p1 = cfg_dir / "train_all.ps1"
        p2 = cfg_dir / "train_all.sh"
        p1.write_text("\n".join(ps), encoding="utf-8")
        p2.write_text("\n".join(sh), encoding="utf-8")
        p2.chmod(0o755)
        return [p1, p2]

    def run(self, force: bool = False, train: bool = False) -> Dict[str, Any]:
        banner("STAGE 4 — LoRA datasets and training configs")
        if not self.cfg.get("lora.enabled", True):
            log("lora.enabled is false — skipping")
            return {}

        want_video = bool(self.cfg.get("lora.train_video_lora", True))
        jobs: List[Dict[str, str]] = []
        result: Dict[str, Any] = {}

        for ch in self.registry.all():
            cid = ch["id"]
            ds = self.build_dataset(cid, force=force)
            img_cfg = self._image_config(cid, ds)
            entry = {"dataset": str(ds), "image_config": str(img_cfg)}
            jobs.append({
                "label": f"{ch['name']} — image LoRA",
                "hint": "ai-toolkit: python run.py <config>",
                "cmd": f'python run.py "{img_cfg.as_posix()}"',
            })
            if want_video:
                vid_cfg = self._video_config(cid, ds)
                entry["video_config"] = str(vid_cfg)
                jobs.append({
                    "label": f"{ch['name']} — LTX-2.5 video LoRA",
                    "hint": "LTX-Video-Trainer: preprocess then train",
                    "cmd": f'python scripts/train.py "{vid_cfg.as_posix()}"',
                })
            self.registry.set_lora(cid, "image", f"{ch['lora_name']}.safetensors", trained=False)
            if want_video:
                self.registry.set_lora(cid, "video", f"{ch['lora_name']}_ltx.safetensors", trained=False)
            result[cid] = entry

        launchers = self._launcher(jobs)
        self.registry.save()
        log(f"configs written to {self.cfg.lora_config_dir}")
        log(f"launchers: {', '.join(p.name for p in launchers)}")

        if train:
            self._train(jobs)
        else:
            log("training not launched (use --train to run, or run the launcher yourself)")
            log("NOTE: the movie stage will still reference the LoRA filenames; train before rendering "
                "or set lora.enabled=false to render without them")
        return result

    @staticmethod
    def _train(jobs: List[Dict[str, str]]) -> None:
        for j in jobs:
            log(f"launching: {j['label']}")
            try:
                subprocess.run(j["cmd"], shell=True, check=True)
            except subprocess.CalledProcessError as exc:
                log(f"training failed for {j['label']}: {exc}. "
                    "Run the command from inside the trainer's own repo directory.", "error")
                sys.exit(1)

    # ------------------------------------------------------------------ #

    def resolve_trained(self) -> List[Dict[str, Any]]:
        """Which LoRA files actually exist on disk right now."""
        out_dir = Path(self.cfg.get("lora.output_dir", "./loras"))
        found: List[Dict[str, Any]] = []
        for ch in self.registry.all():
            for kind in ("image", "video"):
                name: Optional[str] = ch.get("lora", {}).get(kind)
                if not name:
                    continue
                exists = (out_dir / name).exists()
                if exists:
                    self.registry.set_lora(ch["id"], kind, name, trained=True)
                found.append({"character": ch["id"], "kind": kind, "name": name, "exists": exists})
        self.registry.save()
        return found
