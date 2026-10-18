"""Stage 3 — cast forge.

Takes whatever character images you have (one or many, any angle) and turns each cast
member into a complete 10-view coverage sheet: full 360-degree turnaround, two face
close-ups, and an action/hands view. Those ten images become (a) the visual definition of
the character in the registry and (b) the LoRA training set.

Flow per character:

    source image(s)  ->  vision read-back (Ollama)  ->  registry detail merge
                     ->  10 x identity-preserving edit (ComfyUI)  ->  views/
                     ->  captions  ->  registry  ->  contact sheet

The vision read-back matters: it reconciles what the LLM *invented* with what the image
*actually shows*, so the identity line you lock is true to the picture you already like.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cast_registry import CastRegistry
from .comfy_client import Comfy, ComfyError, patch
from .config import Config
from .ollama_client import Ollama
from .util import banner, file_hash, log, read_json, slugify, stable_seed
from .views import DEFAULT_BACKGROUND, VIEW_SPECS, ViewSpec

READBACK_SYSTEM = (
    "You are a character designer cataloguing a reference image for a film's cast bible. "
    "You describe ONLY what is visibly present. You never guess, never embellish, never "
    "add story. If something is not visible, say 'not visible'. Output ONLY JSON."
)

READBACK_SCHEMA = {
    "type": "object",
    "properties": {
        "species": {"type": "string"},
        "age_appearance": {"type": "string"},
        "build": {"type": "string"},
        "skin": {"type": "string"},
        "hair": {"type": "string"},
        "eyes": {"type": "string"},
        "face_marks": {"type": "string"},
        "wardrobe": {"type": "string"},
        "signature_prop": {"type": "string"},
        "palette": {"type": "array", "items": {"type": "string"}},
        "identity_line": {"type": "string"},
    },
    "required": ["species", "age_appearance", "build", "skin", "hair", "eyes",
                 "face_marks", "wardrobe", "signature_prop", "palette", "identity_line"],
}


def _validate_readback(obj: Dict[str, Any]) -> None:
    missing = [k for k in READBACK_SCHEMA["required"] if not obj.get(k)]
    if missing:
        raise ValueError(f"vision read-back missing {missing}")
    if len(str(obj["identity_line"]).split()) < 12:
        raise ValueError("identity_line must be one sentence of at least 12 words")


class CastForge:
    def __init__(self, cfg: Config, comfy: Comfy, ollama: Ollama, registry: CastRegistry):
        self.cfg = cfg
        self.comfy = comfy
        self.ollama = ollama
        self.registry = registry
        self.template = self._load_template()
        self.patch_map: Dict[str, str] = cfg.get("cast.turnaround_patch_map", {})
        self.background = DEFAULT_BACKGROUND

    # ------------------------------------------------------------------ #
    # template
    # ------------------------------------------------------------------ #

    def _load_template(self) -> Dict[str, Any]:
        path = self.cfg.resolve(self.cfg.get("cast.turnaround_template", "./templates/qwen_edit_turnaround.api.json"))
        wf = read_json(path, None)
        if wf is None:
            raise FileNotFoundError(
                f"Turnaround template not found: {path}\n"
                "Export any image-edit workflow from ComfyUI with 'Export (API)' and point "
                "cast.turnaround_template at it, then set cast.turnaround_patch_map to its node titles."
            )
        missing = self.comfy.assert_nodes(sorted({n.get("class_type") for n in wf.values()}))
        if missing:
            log(f"template uses node classes this ComfyUI does not have: {missing}", "warn")
        return wf

    # ------------------------------------------------------------------ #
    # sources
    # ------------------------------------------------------------------ #

    def discover_sources(self) -> Dict[str, List[Path]]:
        """Map cast id -> source images. Filenames starting with the cast id win."""
        src_dir = self.cfg.resolve(self.cfg.get("cast.source_dir", "./cast_sources"))
        src_dir.mkdir(parents=True, exist_ok=True)
        images = sorted(
            p for p in src_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
        )
        found: Dict[str, List[Path]] = {cid: [] for cid in self.registry.order}
        leftovers: List[Path] = []
        for img in images:
            stem = slugify(img.stem)
            hit = next((cid for cid in self.registry.order if stem.startswith(cid) or cid in stem), None)
            if hit:
                found[hit].append(img)
            else:
                leftovers.append(img)
        if leftovers:
            log(f"{len(leftovers)} source image(s) matched no cast id: "
                f"{[p.name for p in leftovers][:6]} — rename them to start with a cast id", "warn")
        for cid, paths in found.items():
            log(f"  {cid}: {len(paths)} source image(s)" + (f" -> {[p.name for p in paths]}" if paths else " (none)"))
        return found

    # ------------------------------------------------------------------ #
    # vision read-back
    # ------------------------------------------------------------------ #

    def readback(self, cid: str, image: Path) -> Dict[str, Any]:
        ch = self.registry.get(cid)
        user = f"""This is the reference image for the character "{ch['name']}" ({ch.get('role','')}).

Catalogue exactly what you see. Return JSON with:
species, age_appearance, build, skin, hair, eyes, face_marks, wardrobe, signature_prop,
palette (2-5 dominant colours), identity_line.

identity_line must be ONE sentence of at least 12 words that names species, apparent age,
hair, eyes, skin and the exact wardrobe with colours, written so an artist could redraw
this character from that sentence alone."""
        try:
            return self.ollama.image_json(
                [image], system=READBACK_SYSTEM, user=user,
                schema=READBACK_SCHEMA, validator=_validate_readback,
                temperature=0.1, label=f"readback:{cid}",
            )
        except Exception as exc:  # noqa: BLE001
            log(f"vision read-back failed for {cid} ({exc}); keeping the written bible description", "warn")
            return {}

    # ------------------------------------------------------------------ #
    # generation
    # ------------------------------------------------------------------ #

    def _build_view_workflow(self, comfy_image: str, spec: ViewSpec, identity_line: str, seed: int,
                             width: int, height: int) -> Dict[str, Any]:
        pm = self.patch_map
        patches: Dict[str, Dict[str, Any]] = {
            pm.get("image", "LoadImage"): {"image": comfy_image},
            pm.get("prompt", "PositivePrompt"): {"text": spec.edit_instruction(identity_line, self.background)},
            pm.get("sampler", "KSampler"): {"seed": seed, "noise_seed": seed},
        }
        if pm.get("negative"):
            patches[pm["negative"]] = {"text": self.cfg.get("cast.turnaround_negative", "")}
        if pm.get("latent"):
            patches[pm["latent"]] = {"width": width, "height": height}

        wf = self.template
        applied: Dict[str, Dict[str, Any]] = {}
        for key, inputs in patches.items():
            try:
                from .comfy_client import find_node
                nid = find_node(wf, key)
                node_inputs = wf[nid]["inputs"]
                applied[key] = {k: v for k, v in inputs.items() if k in node_inputs}
                if not applied[key]:
                    log(f"patch target '{key}' has none of {list(inputs)} — skipped", "debug")
            except ComfyError as exc:
                if key in (pm.get("negative"), pm.get("latent")):
                    log(f"optional patch target missing: {exc}", "debug")
                else:
                    raise
        return patch(wf, applied)

    def forge_character(self, cid: str, sources: List[Path], force: bool = False) -> Dict[str, Any]:
        ch = self.registry.get(cid)

        if ch.get("locked") and not force:
            log(f"{cid}: locked, skipping generation ({len(ch.get('views', {}))} views on file)")
            return ch

        out_dir = self.cfg.views_dir / cid
        out_dir.mkdir(parents=True, exist_ok=True)

        if not sources:
            raise FileNotFoundError(
                f"No source image for '{cid}'. Put one in "
                f"{self.cfg.resolve(self.cfg.get('cast.source_dir'))} named '{cid}.png', "
                "or run the keyframe stage first to generate a base portrait."
            )

        primary = sources[0]
        for s in sources:
            self.registry.attach_source(cid, s, note="user supplied")

        # 1. reconcile written description with the actual picture
        rb = self.readback(cid, primary)
        if rb:
            merged = {k: v for k, v in rb.items() if v and v.lower() != "not visible"}
            merged["identity_line_source"] = "vision read-back of " + primary.name
            self.registry.upsert({"id": cid, **merged}, force=True)
            ch = self.registry.get(cid)
            log(f"{cid}: identity_line -> {ch['identity_line'][:110]}...")

        identity_line = ch["identity_line"]
        comfy_image = self.comfy.upload_image(primary, subfolder=f"castforge/{cid}")
        log(f"{cid}: uploaded {primary.name} -> {comfy_image}")

        # 2. ten views
        n_views = int(self.cfg.get("cast.views_per_character", 10))
        variants = max(1, int(self.cfg.get("cast.variants_per_view", 1)))
        width = int(self.cfg.get("project.width", 768))
        height = int(self.cfg.get("project.height", 1344))

        for spec in VIEW_SPECS[:n_views]:
            for v in range(variants):
                stem = spec.id if variants == 1 else f"{spec.id}_v{v+1}"
                target = out_dir / f"{stem}.png"
                if target.exists() and not force:
                    log(f"{cid}/{stem}: exists, skipping", "debug")
                else:
                    seed = stable_seed(cid, spec.id, v, base=int(ch.get("base_seed", 0)))
                    wf = self._build_view_workflow(comfy_image, spec, identity_line, seed, width, height)
                    log(f"{cid}/{stem}: rendering ({spec.name})")
                    saved = self.comfy.run_and_fetch(wf, out_dir, stem)
                    if saved[0] != target:
                        shutil.move(str(saved[0]), target)
                        for extra in saved[1:]:
                            Path(extra).unlink(missing_ok=True)

                self.registry.attach_view(
                    cid, stem, target.relative_to(self.cfg.project_root),
                    caption=spec.caption(ch["trigger_token"], ch["name"], identity_line),
                    meta={
                        "view": spec.id, "variant": v, "angle": spec.angle,
                        "framing": spec.framing, "pose": spec.pose,
                        "repeats": spec.repeats, "tags": spec.tags,
                        "sha": file_hash(target) if target.exists() else None,
                    },
                )

        self.registry.save()
        log(f"{cid}: {len(ch.get('views', {}))} views registered")
        return ch

    # ------------------------------------------------------------------ #

    def run(self, force: bool = False, only: Optional[List[str]] = None) -> CastRegistry:
        banner("STAGE 3 — cast forge (10-view coverage)")
        self.comfy.health()
        sources = self.discover_sources()
        targets = only or self.registry.order
        for cid in targets:
            if cid not in self.registry.characters:
                log(f"unknown cast id '{cid}', skipping", "warn")
                continue
            self.forge_character(cid, sources.get(cid, []), force=force)

        if self.cfg.get("cast.lock_after_build", True):
            self.registry.lock()
        self.registry.save()
        self.contact_sheets()
        log("cast registry:\n" + self.registry.report(int(self.cfg.get("cast.views_per_character", 10))))
        return self.registry

    # ------------------------------------------------------------------ #

    def contact_sheets(self) -> List[Path]:
        """One PNG per character showing all ten views. Purely for your eyes."""
        try:
            from PIL import Image  # noqa: PLC0415
        except ImportError:
            log("Pillow not installed — skipping contact sheets (pip install pillow)", "debug")
            return []

        made: List[Path] = []
        cols, cell = 5, 320
        for ch in self.registry.all():
            views = sorted(ch.get("views", {}).items())
            if not views:
                continue
            rows = (len(views) + cols - 1) // cols
            sheet = Image.new("RGB", (cols * cell, rows * cell + 40), (24, 24, 28))
            for i, (vid, meta) in enumerate(views):
                p = self.cfg.project_root / meta["path"]
                if not p.exists():
                    continue
                im = Image.open(p).convert("RGB")
                im.thumbnail((cell - 12, cell - 12))
                x = (i % cols) * cell + (cell - im.width) // 2
                y = (i // cols) * cell + (cell - im.height) // 2 + 40
                sheet.paste(im, (x, y))
            out = self.cfg.cast_dir / f"contact_{ch['id']}.png"
            sheet.save(out)
            made.append(out)
            log(f"contact sheet -> {out.name}")
        return made
