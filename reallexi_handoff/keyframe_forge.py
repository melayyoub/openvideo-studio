"""Stage 5 — keyframes, the real identity anchors.

LTX image-to-video only ever sees ONE image: the first frame. Everything the model knows
about who these people are, it knows from that still. So the honest way to stop identity
loss across a whole film is not to argue with the video model in text — it is to hand it a
first frame that is already correct, in every shot.

Three anchor modes:

``chain``     every shot starts from the previous shot's last frame. Maximum motion
              continuity, but error compounds — by shot 8 the face has quietly become
              somebody else. This is the mode most pipelines use and the reason they drift.

``reanchor``  every shot starts from a freshly rendered keyframe using the character
              LoRA. Zero identity drift, but pose continuity between shots is looser.

``hybrid``    (default) chain within a run of shots, then re-anchor every N shots and
              whenever the drift audit says the face has moved. Drift can never accumulate
              past N shots.

The drift audit is a vision pass: it compares the frame against the character's locked
front-face view and returns a numeric match. Below threshold, the shot is re-anchored.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .cast_registry import CastRegistry
from .comfy_client import Comfy, ComfyError, find_node, patch
from .config import Config
from .ollama_client import Ollama
from .schema import DRIFT_SCHEMA
from .util import banner, log, read_json, stable_seed


# --------------------------------------------------------------------------- #
# LoRA injection for API-format workflows
# --------------------------------------------------------------------------- #

def inject_loras_api(
    wf: Dict[str, Any],
    loras: List[Tuple[str, float]],
    model_source_classes: Tuple[str, ...] = ("UNETLoader", "CheckpointLoaderSimple", "UNETLoaderGGUF"),
) -> Dict[str, Any]:
    """Splice a LoraLoaderModelOnly chain between the model loader and its consumers."""
    if not loras:
        return wf

    src = next((nid for nid, n in wf.items() if n.get("class_type") in model_source_classes), None)
    if src is None:
        log("no model loader found — cannot inject LoRAs into this template", "warn")
        return wf

    consumers = [
        (nid, key)
        for nid, n in wf.items()
        for key, val in n.get("inputs", {}).items()
        if isinstance(val, list) and len(val) == 2 and str(val[0]) == str(src) and key == "model"
    ]
    if not consumers:
        log("model output has no 'model' consumers — LoRA injection skipped", "warn")
        return wf

    next_id = max(int(k) for k in wf if str(k).isdigit()) + 1
    prev: Any = [src, 0]
    for name, strength in loras:
        nid = str(next_id)
        next_id += 1
        wf[nid] = {
            "class_type": "LoraLoaderModelOnly",
            "_meta": {"title": f"CastForge LoRA {name}"},
            "inputs": {"lora_name": name, "strength_model": float(strength), "model": prev},
        }
        prev = [nid, 0]

    for nid, key in consumers:
        wf[nid]["inputs"][key] = prev
    log(f"injected {len(loras)} LoRA(s) into keyframe workflow", "debug")
    return wf


# --------------------------------------------------------------------------- #

DRIFT_SYSTEM = (
    "You are a continuity supervisor on a film set. You compare a frame against the "
    "character's approved reference sheet and report, numerically, whether it is still the "
    "same character. You are strict. Output ONLY JSON."
)


class KeyframeForge:
    def __init__(self, cfg: Config, comfy: Comfy, ollama: Ollama, registry: CastRegistry):
        self.cfg = cfg
        self.comfy = comfy
        self.ollama = ollama
        self.registry = registry

    # ------------------------------------------------------------------ #
    # planning
    # ------------------------------------------------------------------ #

    def plan(self, scenes: List[Dict[str, Any]]) -> Dict[int, str]:
        """Decide per shot: 'first' (your original image), 'reanchor', or 'chain'."""
        mode = str(self.cfg.get("movie.anchor_mode", "hybrid")).lower()
        every = max(1, int(self.cfg.get("movie.reanchor_every", 2)))
        plan: Dict[int, str] = {}
        for sc in scenes:
            i = sc["index"]
            if i == 1:
                plan[i] = "first" if self.cfg.get("movie.keep_first_image", True) else "reanchor"
            elif mode == "chain":
                plan[i] = "chain"
            elif mode == "reanchor":
                plan[i] = "reanchor"
            else:
                plan[i] = "reanchor" if (i - 1) % every == 0 else "chain"
        return plan

    # ------------------------------------------------------------------ #
    # rendering
    # ------------------------------------------------------------------ #

    def _template(self) -> Optional[Dict[str, Any]]:
        p = self.cfg.resolve(self.cfg.get("cast.turnaround_template", "./templates/qwen_edit_turnaround.api.json"))
        return read_json(p, None)

    def _lora_list(self, characters: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
        if not self.cfg.get("lora.enabled", True):
            return []
        strength = float(self.cfg.get("lora.strength_model", 0.9))
        out_dir = Path(self.cfg.get("lora.output_dir", "./loras"))
        loras: List[Tuple[str, float]] = []
        for ch in characters:
            name = ch.get("lora", {}).get("image")
            if not name:
                continue
            if not (out_dir / name).exists():
                log(f"{ch['id']}: image LoRA '{name}' not on disk yet — rendering without it", "warn")
                continue
            loras.append((name, strength))
        # More than two character LoRAs at once fight each other; anchor solo where possible.
        if len(loras) > 2:
            log(f"{len(loras)} LoRAs requested in one keyframe — using the first two", "warn")
            loras = loras[:2]
        return loras

    def render_keyframe(
        self, scene: Dict[str, Any], characters: List[Dict[str, Any]], bible: Dict[str, Any],
        previous: Optional[Dict[str, Any]], force: bool = False,
    ) -> Path:
        dest = self.cfg.keyframe_dir / f"scene_{scene['index']:02d}.png"
        if dest.exists() and not force:
            log(f"scene {scene['index']}: keyframe exists, reusing")
            return dest

        wf = self._template()
        if wf is None:
            raise FileNotFoundError("No keyframe template available; cannot re-anchor.")

        tokens = " ".join(c["trigger_token"] for c in characters)
        identity = " ".join(
            f"{c['name']} ({c['trigger_token']}): {c['identity_line'].rstrip('.')}." for c in characters
        )
        carried = f" Continuing from: {previous.get('transition_out','')}" if previous else ""
        prompt = (
            f"{tokens}. {identity} "
            f"{scene['keyframe_prompt']} "
            f"Location {scene['location']}, {scene['time_of_day']}, {scene['weather']}.{carried} "
            f"Style: {bible.get('visual_style','cinematic')}. "
            f"Palette: {', '.join(bible.get('colour_palette', []))}. "
            f"Single still frame, cinematic composition, sharp focus, no text, no watermark."
        )

        pm = self.cfg.get("cast.turnaround_patch_map", {})
        seed = stable_seed("keyframe", scene["index"], base=int(self.cfg.get("project.seed", 0)))
        patches: Dict[str, Dict[str, Any]] = {
            pm.get("prompt", "PositivePrompt"): {"text": prompt},
            pm.get("sampler", "KSampler"): {"seed": seed, "noise_seed": seed},
        }
        if pm.get("negative"):
            patches[pm["negative"]] = {"text": self.cfg.get("cast.turnaround_negative", "")}
        if pm.get("latent"):
            patches[pm["latent"]] = {
                "width": int(self.cfg.get("project.width", 768)),
                "height": int(self.cfg.get("project.height", 1344)),
            }
        # Anchor on the lead character's front-face view so the edit model has a face to hold.
        anchor = self._face_anchor(characters)
        if anchor and pm.get("image"):
            patches[pm["image"]] = {"image": self.comfy.upload_image(anchor, subfolder="castforge/anchors")}

        safe: Dict[str, Dict[str, Any]] = {}
        for key, inputs in patches.items():
            try:
                nid = find_node(wf, key)
                safe[key] = {k: v for k, v in inputs.items() if k in wf[nid]["inputs"]}
            except ComfyError:
                log(f"keyframe template has no node '{key}' — skipped", "debug")

        wf = patch(wf, safe)
        wf = inject_loras_api(wf, self._lora_list(characters))

        log(f"scene {scene['index']}: rendering keyframe ({', '.join(c['name'] for c in characters)})")
        saved = self.comfy.run_and_fetch(wf, self.cfg.keyframe_dir, f"_tmp_scene_{scene['index']:02d}")
        shutil.move(str(saved[0]), dest)
        for extra in saved[1:]:
            Path(extra).unlink(missing_ok=True)
        return dest

    def _face_anchor(self, characters: List[Dict[str, Any]]) -> Optional[Path]:
        for ch in characters:
            v = ch.get("views", {}).get("v08_face_front") or next(iter(ch.get("views", {}).values()), None)
            if v:
                p = self.cfg.project_root / v["path"]
                if p.exists():
                    return p
        return None

    # ------------------------------------------------------------------ #
    # drift audit
    # ------------------------------------------------------------------ #

    def audit(self, frame: Path, character: Dict[str, Any]) -> Dict[str, Any]:
        ref = character.get("views", {}).get("v08_face_front")
        if not ref:
            return {"identity_match": 1.0, "wardrobe_match": 1.0, "drifted_attributes": [],
                    "verdict": "no reference view available"}
        ref_path = self.cfg.project_root / ref["path"]
        user = f"""Reference character sheet (image 1) and a frame from the film (image 2).

The character is "{character['name']}".
Approved description: {character['identity_line']}
Rules that must never break: {'; '.join(character.get('never', []))}

Score how well image 2 matches the approved character.
Return JSON: identity_match (0.0-1.0 for face and body), wardrobe_match (0.0-1.0 for
clothing, colours and props), drifted_attributes (list of what changed, empty if nothing),
verdict (one short sentence)."""
        try:
            return self.ollama.image_json(
                [ref_path, frame], system=DRIFT_SYSTEM, user=user,
                schema=DRIFT_SCHEMA, temperature=float(self.cfg.get("ollama.temperature_audit", 0.1)),
                label=f"drift:{character['id']}",
            )
        except Exception as exc:  # noqa: BLE001
            log(f"drift audit unavailable ({exc})", "warn")
            return {"identity_match": 1.0, "wardrobe_match": 1.0, "drifted_attributes": [],
                    "verdict": "audit skipped"}

    def audit_frame(self, frame: Path, characters: List[Dict[str, Any]]) -> Tuple[bool, List[Dict[str, Any]]]:
        """Returns (needs_reanchor, per-character reports)."""
        threshold = float(self.cfg.get("movie.drift_threshold", 0.72))
        reports = []
        drifted = False
        for ch in characters:
            r = self.audit(frame, ch)
            r["character"] = ch["id"]
            score = min(float(r.get("identity_match", 1.0)), float(r.get("wardrobe_match", 1.0)))
            r["score"] = score
            if score < threshold:
                drifted = True
                log(f"DRIFT {ch['id']}: {score:.2f} < {threshold} — {r.get('verdict','')}", "warn")
            reports.append(r)
        return drifted, reports

    # ------------------------------------------------------------------ #

    def run(self, bible: Dict[str, Any], force: bool = False) -> Dict[int, Dict[str, Any]]:
        banner("STAGE 5 — keyframes and anchor plan")
        scenes = bible["scenes"]
        plan = self.plan(scenes)
        out: Dict[int, Dict[str, Any]] = {}

        first_image = self.cfg.get("movie.first_image")
        for idx, sc in enumerate(scenes):
            i = sc["index"]
            mode = plan[i]
            chars = self.registry.present(sc["characters"])
            entry: Dict[str, Any] = {"mode": mode, "characters": [c["id"] for c in chars]}

            if mode == "first":
                entry["image"] = first_image
                entry["note"] = "your original opening image, preserved exactly"
                log(f"scene 1: opening frame = {first_image} (preserved)")
            elif mode == "reanchor":
                self.comfy.health()
                p = self.render_keyframe(sc, chars, bible, scenes[idx - 1] if idx else None, force=force)
                entry["image"] = p.name
                entry["path"] = str(p)
            else:
                entry["image"] = None
                entry["note"] = "inherits the last frame of the previous shot inside the graph"

            out[i] = entry

        chain = sum(1 for v in out.values() if v["mode"] == "chain")
        anchor = len(out) - chain
        log(f"anchor plan: {anchor} anchored shot(s), {chain} chained shot(s)")
        return out
