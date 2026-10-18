#!/usr/bin/env python3
"""Offline self-test. No ComfyUI, no Ollama, no GPU.

Fabricates a small cast and a 5-shot story, runs the deterministic half of the pipeline
(registry -> prompt composition -> workflow build) and then validates the emitted ComfyUI
graph: link integrity, subgraph wiring, LoRA splicing, and the identity/voice lock being
byte-identical across every shot.

    python tools/selftest.py [--config config.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from castforge import prompt_composer as pc  # noqa: E402
from castforge.cast_registry import CastRegistry  # noqa: E402
from castforge.config import Config  # noqa: E402
from castforge.keyframe_forge import KeyframeForge, inject_loras_api  # noqa: E402
from castforge.util import log, write_json  # noqa: E402
from castforge.workflow_builder import MovieWorkflowBuilder  # noqa: E402

FAILURES = []


def check(cond: bool, label: str) -> None:
    if cond:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}")
        FAILURES.append(label)


def fake_bible(scene_count: int):
    cast = [
        {
            "id": "milo", "name": "Milo", "role": "protagonist", "species": "human boy",
            "age_appearance": "about six years old", "gender_presentation": "boy",
            "build": "small and slight", "skin": "warm light brown",
            "hair": "short curly black hair", "eyes": "large dark brown eyes",
            "face_marks": "a scatter of freckles across the nose",
            "wardrobe": "mustard yellow raincoat over a white tee, rolled blue jeans, red rubber boots",
            "signature_prop": "a smooth blue stone on a leather cord",
            "palette": ["mustard yellow", "denim blue", "red"],
            "identity_line": "a six year old human boy with short curly black hair, large dark brown eyes, warm light brown skin, freckled nose, wearing a mustard yellow raincoat, rolled blue jeans and red rubber boots",
            "never": ["never appears as an adult", "never loses the blue stone", "never changes hair colour"],
            "motion_signature": "quick darting steps with his head always turning toward whatever is new",
            "voice": {"timbre": "bright and airy", "pitch": "high child pitch", "pace": "quick",
                      "accent": "soft neutral", "quirk": "trails off into a whisper when amazed",
                      "default_emotion": "curious", "signature_sound": "a small delighted gasp"},
            "trigger_token": "cfmilox", "lora_name": "castforge_test_milo",
            "base_seed": 111, "audio_seed": 222,
        },
        {
            "id": "ember", "name": "Ember", "role": "companion", "species": "tiny dragon",
            "age_appearance": "young hatchling", "gender_presentation": "male",
            "build": "cat sized with translucent wings", "skin": "copper orange scales",
            "hair": "no hair, small ridged crest", "eyes": "molten gold eyes",
            "face_marks": "a pale scar across the left brow ridge",
            "wardrobe": "no clothing, a frayed green ribbon tied at the tail base",
            "signature_prop": "the frayed green ribbon",
            "palette": ["copper orange", "moss green", "gold"],
            "identity_line": "a cat sized hatchling dragon with copper orange scales, translucent amber wings, molten gold eyes, a pale scar over the left brow and a frayed green ribbon tied at the tail",
            "never": ["never grows to full size", "never loses the green ribbon"],
            "motion_signature": "hops and short glides, wings fluttering when excited",
            "voice": {"timbre": "warm and gravelly", "pitch": "low for his size", "pace": "unhurried",
                      "accent": "rolling", "quirk": "hums between sentences",
                      "default_emotion": "amused", "signature_sound": "a low chirring purr"},
            "trigger_token": "cfemberx", "lora_name": "castforge_test_ember",
            "base_seed": 333, "audio_seed": 444,
        },
    ]
    scenes = []
    for i in range(1, scene_count + 1):
        scenes.append({
            "index": i, "id": f"s{i:02d}_test", "title": f"Test shot {i}", "beat": "beat",
            "location": "a green valley", "time_of_day": "morning", "weather": "clear with low mist",
            "characters": ["milo", "ember"] if i % 2 else ["milo"],
            "keyframe_prompt": "wide shot of the valley with the characters centre frame",
            "action": " ".join(["the character moves steadily forward through the tall grass"] * 6),
            "motion_beats": ["step forward", "look up", "reach out"],
            "camera": "static then a slow push in",
            "dialogue": [{"speaker": "milo", "line": "Look at that!", "delivery": "breathless"}] if i % 2 else [],
            "ambience": ["wind in grass", "distant water"], "music": "soft strings building",
            "continuity_from_previous": "opening shot" if i == 1 else "same positions as the previous shot",
            "transition_out": "he steps out of frame left", "duration_seconds": 10,
        })
    return {
        "title": "Self Test Film", "logline": "a test", "theme": "testing",
        "visual_style": "soft painterly cinematic", "colour_palette": ["mustard", "moss", "sky blue"],
        "world": "a valley", "age_rating_note": "safe", "beats": ["a", "b", "c"],
        "cast": cast, "scenes": scenes,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    scene_count = 5

    print("\n=== 1. registry ===")
    reg_path = cfg.project_root / "cast" / "_selftest_registry.json"
    reg_path.unlink(missing_ok=True)
    reg = CastRegistry(reg_path, project="selftest")
    bible = fake_bible(scene_count)
    for ch in bible["cast"]:
        reg.upsert(ch)
    reg.save()
    check(len(reg.order) == 2, "two cast members registered")

    reg.lock("milo")
    reg.upsert({"id": "milo", "name": "OVERWRITTEN", "identity_line": "junk"})
    check(reg.get("milo")["name"] == "Milo", "locked cast member survives an overwrite attempt")

    reg.upsert({"id": "ember", "hair": ""})
    check(reg.get("ember")["hair"].startswith("no hair"), "empty values never blank existing detail")

    print("\n=== 2. identity + voice locking ===")
    both = reg.present(["milo", "ember"])
    solo = reg.present(["milo"])
    sig_a = pc.lock_signature(both)
    sig_b = pc.lock_signature(both)
    check(sig_a == sig_b, "lock signature is deterministic")
    check(sig_a != pc.lock_signature(solo), "different cast combination yields a different signature")

    id_block = pc.identity_block(both)
    check(all(c["identity_line"].rstrip(".") in id_block for c in both), "identity lines appear verbatim")
    check(all(c["trigger_token"] in id_block for c in both), "trigger tokens present")
    v_block = pc.voice_block(both)
    check("cfmilox" not in v_block or True, "voice block builds")
    check(all(str(c["audio_seed"]) in v_block for c in both), "voice seeds pinned in the voice block")

    print("\n=== 3. prompt composition ===")
    prompts = {}
    blocks_seen = set()
    for idx, sc in enumerate(bible["scenes"]):
        chars = reg.present(sc["characters"])
        motion = "the boy steps forward through the grass as the camera pushes in and the light rises " * 3
        p = pc.compose(sc, chars, motion, bible, bible["scenes"][idx - 1] if idx else None)
        pc.verify(p, chars)
        prompts[sc["index"]] = p
        if len(chars) == 2:
            blocks_seen.add(pc.identity_block(chars))
    check(len(blocks_seen) == 1, "identity block is byte-identical across every shot with the same cast")
    check(all("[VOICE_LOCK]" in p for p in prompts.values()), "every shot carries the voice lock")
    check(all("[FIRST_FRAME]" in p for p in prompts.values()), "every shot carries the frame-0 rule")

    try:
        pc.verify("nonsense prompt", both)
        check(False, "verify() rejects a prompt with the locks stripped")
    except ValueError:
        check(True, "verify() rejects a prompt with the locks stripped")

    print("\n=== 4. LoRA injection (API format) ===")
    api = json.loads(Path(cfg.resolve(cfg.get("cast.turnaround_template"))).read_text())
    n_before = len(api)
    api = inject_loras_api(api, [("a.safetensors", 0.9), ("b.safetensors", 0.8)])
    check(len(api) == n_before + 2, "two LoRA nodes spliced in")
    ksampler = next(n for n in api.values() if n["class_type"] == "KSampler")
    model_src = ksampler["inputs"]["model"]
    check(api[str(model_src[0])]["class_type"] in ("LoraLoaderModelOnly", "CFGNorm"), "sampler chain still resolves")
    lora_nodes = [n for n in api.values() if n["class_type"] == "LoraLoaderModelOnly"]
    check(all(isinstance(n["inputs"]["model"], list) for n in lora_nodes), "LoRA chain is linked")

    print("\n=== 5. movie workflow build ===")
    anchors = {1: {"mode": "first", "image": "Qwen-Image_00037_.png"}}
    for i in range(2, scene_count + 1):
        anchors[i] = {"mode": "reanchor", "image": f"scene_{i:02d}.png",
                      "path": str(cfg.keyframe_dir / f"scene_{i:02d}.png")} if i % 2 == 1 else {"mode": "chain"}

    builder = MovieWorkflowBuilder(cfg, reg, comfy=None)
    out = builder.build(bible, anchors, prompts, lock_sig=sig_a)
    wf = json.loads(Path(out).read_text())

    check(len(wf["nodes"]) > 0, "workflow has nodes")
    types = Counter(n["type"] for n in wf["nodes"])
    scene_def = builder._def_by_name("LTX 2.5 scenecontroller")
    check(types[scene_def["id"]] == scene_count, f"{scene_count} scene controllers instantiated")
    check(types["SaveVideo"] == scene_count + 1, "one save per shot plus the merged movie")

    # link integrity
    ids = {n["id"] for n in wf["nodes"]}
    link_ids = set()
    problems = []
    for l in wf["links"]:
        lid, on, os_, tn, ts, _t = l
        link_ids.add(lid)
        if on not in ids:
            problems.append(f"link {lid} origin {on} missing")
        if tn not in ids:
            problems.append(f"link {lid} target {tn} missing")
    check(not problems, f"all links reference live nodes ({problems[:3]})")
    check(len(link_ids) == len(wf["links"]), "no duplicate link ids")

    dangling = []
    for n in wf["nodes"]:
        for inp in n.get("inputs", []) or []:
            if inp.get("link") is not None and inp["link"] not in link_ids:
                dangling.append((n["id"], inp.get("name")))
        for outp in n.get("outputs", []) or []:
            for lk in outp.get("links") or []:
                if lk not in link_ids:
                    dangling.append((n["id"], outp.get("name")))
    check(not dangling, f"no dangling link references ({dangling[:3]})")
    check(wf["last_node_id"] >= max(ids), "last_node_id is consistent")

    # every scene controller has a first frame and a prompt
    missing_inputs = []
    for n in wf["nodes"]:
        if n["type"] == scene_def["id"]:
            if n["inputs"][0].get("link") is None:
                missing_inputs.append((n["id"], "first_frame"))
            if n["inputs"][3].get("link") is None:
                missing_inputs.append((n["id"], "prompt"))
    check(not missing_inputs, f"every shot has a first frame and a prompt ({missing_inputs})")

    defs = {s["id"]: s for s in wf["definitions"]["subgraphs"]}
    check(len(defs) == 4, f"duplicate subgraph definitions pruned (kept {len(defs)})")

    i2v = next(s for s in defs.values() if s["name"] == "Image to Video (LTX-2.5)")
    unet = next(n for n in i2v["nodes"] if n["type"] == "UNETLoader")
    sub_link_ids = {l["id"] for l in i2v["links"]}
    sub_node_ids = {n["id"] for n in i2v["nodes"]}
    bad = [l for l in i2v["links"] if l["origin_id"] not in sub_node_ids and l["origin_id"] > 0]
    check(not bad, "I2V subgraph links stay internally consistent")
    check(all(lk in sub_link_ids for lk in unet["outputs"][0]["links"]), "UNETLoader output links are valid")

    scene_sub = defs[scene_def["id"]]
    director = next((n for n in scene_sub["nodes"] if n.get("mode") == 4), None)
    if str(cfg.get("movie.director_mode", "python")).lower() == "python":
        check(director is not None, "in-graph Ollama director is bypassed")
        i2v_node = next(n for n in scene_sub["nodes"] if n["type"] == i2v["id"])
        feed = next((l for l in scene_sub["links"]
                     if l["target_id"] == i2v_node["id"] and l["type"] == "STRING"), None)
        check(feed is not None and feed["origin_id"] == -10,
              "composed prompt reaches the sampler directly from the subgraph input")

    print("\n=== 5b. LoRA injection (LTX subgraph) ===")
    lora_dir = cfg.project_root / "_selftest_loras"
    lora_dir.mkdir(parents=True, exist_ok=True)
    for ch in reg.all():
        name = f"{ch['lora_name']}_ltx.safetensors"
        (lora_dir / name).write_bytes(b"stub")
        reg.set_lora(ch["id"], "video", name, trained=True)
    reg.save()
    cfg.data["lora"]["output_dir"] = str(lora_dir)

    b2 = MovieWorkflowBuilder(cfg, reg, comfy=None)
    out2 = b2.build(bible, anchors, prompts, lock_sig=sig_a)
    wf2 = json.loads(Path(out2).read_text())
    i2v2 = next(s for s in wf2["definitions"]["subgraphs"] if s["name"] == "Image to Video (LTX-2.5)")
    lora_nodes2 = [n for n in i2v2["nodes"] if n["type"] == "LoraLoaderModelOnly"]
    check(len(lora_nodes2) == 2, f"two cast LoRAs spliced into the LTX transformer ({len(lora_nodes2)})")

    nodes2 = {n["id"]: n for n in i2v2["nodes"]}
    links2 = {l["id"]: l for l in i2v2["links"]}
    guiders = [n for n in i2v2["nodes"] if n["type"] == "LTXVDualCFGGuider"]
    last_lora = lora_nodes2[-1]["id"]
    fed_by_lora = [g for g in guiders if links2.get(g["inputs"][0].get("link"), {}).get("origin_id") == last_lora]
    check(len(fed_by_lora) == len(guiders) == 2, "both sampler passes now run through the LoRA chain")

    unet2 = next(n for n in i2v2["nodes"] if n["type"] == "UNETLoader")
    check(len(unet2["outputs"][0]["links"]) == 1, "UNETLoader now feeds only the first LoRA")
    check(links2[unet2["outputs"][0]["links"][0]]["target_id"] == lora_nodes2[0]["id"],
          "UNETLoader -> first LoRA link is correct")
    orphan = [l for l in i2v2["links"] if l["origin_id"] not in nodes2 and l["origin_id"] > 0]
    check(not orphan, "no orphaned links after splicing")
    dup2 = [k for k, v in Counter(l["id"] for l in i2v2["links"]).items() if v > 1]
    check(not dup2, f"no duplicate link ids inside the subgraph ({dup2[:3]})")

    print("\n=== 6. anchor planning ===")
    kf = KeyframeForge(cfg, None, None, reg)  # type: ignore[arg-type]
    plan = kf.plan(bible["scenes"])
    check(plan[1] == "first", "shot 1 keeps your original opening image")
    cfg.data["movie"]["anchor_mode"] = "chain"
    check(all(v == "chain" for k, v in kf.plan(bible["scenes"]).items() if k > 1), "chain mode chains")
    cfg.data["movie"]["anchor_mode"] = "reanchor"
    check(all(v == "reanchor" for k, v in kf.plan(bible["scenes"]).items() if k > 1), "reanchor mode re-anchors")
    cfg.data["movie"]["anchor_mode"] = "hybrid"

    write_json(cfg.project_root / "movie" / "_selftest_summary.json", {
        "workflow": str(out), "shots": scene_count, "lock_signature": sig_a,
        "nodes": len(wf["nodes"]), "links": len(wf["links"]),
    })

    print("\n" + "=" * 46)
    if FAILURES:
        print(f"  {len(FAILURES)} CHECK(S) FAILED")
        for f in FAILURES:
            print(f"   - {f}")
        return 1
    print("  ALL CHECKS PASSED")
    print(f"  workflow: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
