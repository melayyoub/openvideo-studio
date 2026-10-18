#!/usr/bin/env python3
"""CastForge — story to movie, without losing the cast.

    python run_pipeline.py --stage all
    python run_pipeline.py --stage story          # Ollama writes the film from subject/mood/age
    python run_pipeline.py --stage cast           # 10 views per character -> registry
    python run_pipeline.py --stage lora --train   # datasets, configs, and optionally train
    python run_pipeline.py --stage keyframes      # anchor plan + LoRA-rendered keyframes
    python run_pipeline.py --stage movie          # emit the N-shot ComfyUI workflow

Stage order matters and the orchestrator enforces it. Completed stages are recorded in
``state.json``; re-running is safe and will not regenerate a locked cast.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from castforge import prompt_composer as pc
from castforge.cast_forge import CastForge
from castforge.cast_registry import CastRegistry, registry_from_bible
from castforge.comfy_client import Comfy
from castforge.config import Config
from castforge.keyframe_forge import KeyframeForge
from castforge.lora import LoraStage
from castforge.ollama_client import Ollama
from castforge.story_engine import StoryEngine
from castforge.util import State, banner, log, read_json, write_json

STAGES = ["story", "cast", "lora", "keyframes", "movie"]


def make_clients(cfg: Config):
    ollama = Ollama(
        url=cfg.get("ollama.url", "http://localhost:11434"),
        text_model=cfg.get("ollama.text_model", "qwen2.5:14b-instruct"),
        vision_model=cfg.get("ollama.vision_model", "llava:13b-v1.6"),
        timeout=int(cfg.get("ollama.timeout", 600)),
        num_ctx=int(cfg.get("ollama.num_ctx", 8192)),
        max_repair_attempts=int(cfg.get("ollama.max_repair_attempts", 3)),
    )
    comfy = Comfy(
        url=cfg.get("comfy.url", "http://127.0.0.1:8188"),
        timeout=int(cfg.get("comfy.timeout", 1800)),
        poll_interval=float(cfg.get("comfy.poll_interval", 2.0)),
    )
    return ollama, comfy


# --------------------------------------------------------------------------- #
# prompt composition
# --------------------------------------------------------------------------- #

def compose_prompts(
    cfg: Config, bible: Dict[str, Any], registry: CastRegistry,
    engine: StoryEngine, anchors: Dict[int, Dict[str, Any]],
) -> Dict[int, str]:
    """One prompt per shot: locked blocks by string formatting, motion by Ollama."""
    banner("composing shot prompts")
    scenes = bible["scenes"]
    prompts: Dict[int, str] = {}
    signatures = set()
    prompt_dir = cfg.movie_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    previous_summary = ""
    for idx, sc in enumerate(scenes):
        i = sc["index"]
        chars = registry.present(sc["characters"])
        if not chars:
            raise ValueError(f"shot {i} lists characters {sc['characters']} but none are in the registry")

        anchor = anchors.get(i, {})
        kf: Any = anchor.get("path")
        if anchor.get("mode") == "first" and anchor.get("image"):
            candidate = Path(cfg.get("comfy.input_dir", "")) / str(anchor["image"])
            kf = str(candidate) if candidate.exists() else None
        if kf and not Path(kf).exists():
            kf = None

        motion = engine.direct_scene(bible, sc, kf, previous_summary)
        prompt = pc.compose(
            scene=sc, characters=chars, motion_paragraph=motion, bible=bible,
            previous=scenes[idx - 1] if idx else None,
        )
        pc.verify(prompt, chars)
        signatures.add(pc.lock_signature(chars))

        prompts[i] = prompt
        (prompt_dir / f"shot_{i:02d}.txt").write_text(prompt, encoding="utf-8")
        previous_summary = f"{sc['title']}: {sc.get('transition_out','')}"
        log(f"shot {i:02d}: {len(prompt.split())} words, "
            f"{len(chars)} character(s), lock {pc.lock_signature(chars)}")

    write_json(cfg.movie_dir / "prompts.json", {str(k): v for k, v in prompts.items()})
    log(f"{len(signatures)} distinct identity lock signature(s) across {len(scenes)} shots "
        f"(one per unique cast combination — that is expected)")
    return prompts


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #

def stage_story(cfg, state, clients, args) -> Dict[str, Any]:
    ollama, _ = clients
    ollama.assert_models()
    bible = StoryEngine(cfg, ollama).build(force=args.force)
    registry_from_bible(cfg, bible, force=args.force)
    state.mark("story", title=bible.get("title"), scenes=len(bible["scenes"]))
    return bible


def stage_cast(cfg, state, clients, args) -> CastRegistry:
    ollama, comfy = clients
    registry = CastRegistry(cfg.cast_registry_path, project=str(cfg.get("project.name")))
    if not registry.order:
        raise RuntimeError("Cast registry is empty. Run --stage story first.")
    forge = CastForge(cfg, comfy, ollama, registry)
    forge.run(force=args.force, only=args.only)
    state.mark("cast", characters=len(registry.order))
    return registry


def stage_lora(cfg, state, clients, args) -> Dict[str, Any]:
    registry = CastRegistry(cfg.cast_registry_path)
    result = LoraStage(cfg, registry).run(force=args.force, train=args.train)
    state.mark("lora", trained=bool(args.train), characters=len(result))
    return result


def stage_keyframes(cfg, state, clients, args) -> Dict[int, Dict[str, Any]]:
    ollama, comfy = clients
    bible = read_json(cfg.story_bible_path)
    registry = CastRegistry(cfg.cast_registry_path)
    LoraStage(cfg, registry).resolve_trained()
    anchors = KeyframeForge(cfg, comfy, ollama, registry).run(bible, force=args.force)
    write_json(cfg.keyframe_dir / "anchor_plan.json", {str(k): v for k, v in anchors.items()})
    state.mark("keyframes", anchored=sum(1 for v in anchors.values() if v["mode"] != "chain"))
    return anchors


def stage_movie(cfg, state, clients, args) -> Path:
    from castforge.workflow_builder import MovieWorkflowBuilder

    ollama, comfy = clients
    bible = read_json(cfg.story_bible_path)
    registry = CastRegistry(cfg.cast_registry_path)
    anchors_raw = read_json(cfg.keyframe_dir / "anchor_plan.json", {}) or {}
    anchors = {int(k): v for k, v in anchors_raw.items()}
    if not anchors:
        anchors = KeyframeForge(cfg, comfy, ollama, registry).plan(bible["scenes"])
        anchors = {k: {"mode": v} for k, v in anchors.items()}

    engine = StoryEngine(cfg, ollama)
    prompts = compose_prompts(cfg, bible, registry, engine, anchors)
    lock_sig = pc.lock_signature(registry.all())

    builder = MovieWorkflowBuilder(cfg, registry, comfy=comfy)
    out = builder.build(bible, anchors, prompts, lock_sig=lock_sig)
    state.mark("movie", workflow=str(out))
    return out


STAGE_FN = {
    "story": stage_story,
    "cast": stage_cast,
    "lora": stage_lora,
    "keyframes": stage_keyframes,
    "movie": stage_movie,
}


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="CastForge story-to-movie pipeline")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--stage", default="all", choices=STAGES + ["all"])
    ap.add_argument("--force", action="store_true", help="regenerate even if artefacts exist (locked cast still protected)")
    ap.add_argument("--train", action="store_true", help="actually launch LoRA training in the lora stage")
    ap.add_argument("--only", nargs="*", help="limit the cast stage to these character ids")
    ap.add_argument("--unlock", nargs="*", help="unlock cast members before running (dangerous)")
    ap.add_argument("--status", action="store_true", help="print project status and exit")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    state = State(cfg.state_path)

    if args.status:
        reg = CastRegistry(cfg.cast_registry_path)
        bible = read_json(cfg.story_bible_path, {}) or {}
        banner(f"{cfg.get('project.name')} — status")
        log(f"title: {bible.get('title','(no story yet)')}")
        log(f"shots: {len(bible.get('scenes', []))}")
        log("stages: " + ", ".join(f"{s}{'*' if state.done(s) else ''}" for s in STAGES))
        log("cast:\n" + reg.report())
        return 0

    if args.unlock is not None:
        reg = CastRegistry(cfg.cast_registry_path)
        for cid in (args.unlock or reg.order):
            reg.unlock(cid)
        reg.save()
        log(f"unlocked: {args.unlock or 'all'}", "warn")

    clients = make_clients(cfg)
    todo: List[str] = STAGES if args.stage == "all" else [args.stage]

    for name in todo:
        if args.stage == "all" and state.done(name) and not args.force:
            log(f"stage '{name}' already complete — skipping (use --force to redo)")
            continue
        try:
            STAGE_FN[name](cfg, state, clients, args)
        except Exception as exc:  # noqa: BLE001
            log(f"stage '{name}' failed: {exc}", "error")
            if "--debug" in sys.argv:
                raise
            return 1

    banner("done")
    log(f"project: {cfg.project_root}")
    log("next: open movie/*_castforge_movie.json in ComfyUI and queue it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
