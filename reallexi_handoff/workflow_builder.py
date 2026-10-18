"""Stage 6 — the movie graph.

Your uploaded workflow is used as the *template*, not replaced. Its subgraph definitions
(`LTX 2.5 scenecontroller`, `Image to Video (LTX-2.5)`, `Merge Videos`) are lifted out
verbatim, so every sampler setting, sigma schedule, VAE and upscaler you already tuned
stays exactly as you tuned it.

What gets added:

* **N scenes instead of 2**, chained the same way yours are (`GetImagesFromBatchIndexed`
  at index -1 feeding the next shot).
* **Anchored shots** — where the plan says re-anchor, the shot starts from a LoRA-rendered
  keyframe via `LoadImage` instead of inheriting a drifting last frame. Shot 1 keeps your
  original image, untouched.
* **Cast LoRAs on the transformer** — a `LoraLoaderModelOnly` chain spliced between
  `UNETLoader` and both `LTXVDualCFGGuider` passes inside the I2V subgraph, so identity is
  held by weights and not only by words.
* **Composed prompts** — each shot receives its pre-built prompt containing the verbatim
  identity and voice locks.
* **Director bypass** (`movie.director_mode: python`) — the in-graph llava director is
  muted and the composed prompt is wired straight to the sampler. This matters: left in
  circuit, that node rewrites your prompt at render time and paraphrases away the very
  identity block you locked. Set it to `graph` to keep the original behaviour.
* **Per-shot SaveVideo** plus a final merged `SaveVideo`, so a failure at shot 7 never
  costs you shots 1-6.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .cast_registry import CastRegistry
from .comfy_client import Comfy
from .config import Config
from .util import banner, log, read_json, stable_seed, write_json

SCENE_DEF_NAME = "LTX 2.5 scenecontroller"
I2V_DEF_NAME = "Image to Video (LTX-2.5)"
MERGE_DEF_NAME = "Merge Videos"
DIRECTOR_DEF_NAME = "Ollama Image to text with scenes results Director"


class MovieWorkflowBuilder:
    def __init__(self, cfg: Config, registry: CastRegistry, comfy: Optional[Comfy] = None):
        self.cfg = cfg
        self.registry = registry
        self.comfy = comfy
        self.wf: Dict[str, Any] = {}
        self.exemplars: Dict[str, Dict[str, Any]] = {}
        self._next_node_id = 1
        self._next_link_id = 1

    # ------------------------------------------------------------------ #
    # template handling
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        path = self.cfg.resolve(self.cfg.require("movie.template_workflow"))
        wf = read_json(path, None)
        if wf is None:
            raise FileNotFoundError(f"Template workflow not found: {path}")
        self.wf = copy.deepcopy(wf)
        self._next_node_id = int(wf.get("last_node_id", 0)) + 1
        self._next_link_id = int(wf.get("last_link_id", 0)) + 1

        by_type: Dict[str, Dict[str, Any]] = {}
        for n in wf["nodes"]:
            by_type.setdefault(n["type"], n)
        self.exemplars = by_type
        log(f"template loaded: {path.name} ({len(wf['nodes'])} nodes, "
            f"{len(wf['definitions']['subgraphs'])} subgraphs)")

    def _defs(self) -> Dict[str, Dict[str, Any]]:
        return {s["id"]: s for s in self.wf["definitions"]["subgraphs"]}

    def _def_by_name(self, name: str) -> Dict[str, Any]:
        for s in self.wf["definitions"]["subgraphs"]:
            if s.get("name") == name:
                return s
        raise KeyError(f"Subgraph definition '{name}' not found in the template")

    def _prune_definitions(self, keep: List[str]) -> None:
        """Drop the duplicated copies ComfyUI made when the scene node was copy-pasted."""
        before = len(self.wf["definitions"]["subgraphs"])
        keepset = set(keep)
        self.wf["definitions"]["subgraphs"] = [
            s for s in self.wf["definitions"]["subgraphs"] if s["id"] in keepset
        ]
        after = len(self.wf["definitions"]["subgraphs"])
        if before != after:
            log(f"pruned {before - after} duplicate subgraph definition(s)")

    # ------------------------------------------------------------------ #
    # subgraph surgery
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sub_ids(sub: Dict[str, Any]) -> Tuple[int, int]:
        state = sub.setdefault("state", {})
        nid = int(state.get("lastNodeId", max([n["id"] for n in sub["nodes"]] + [0]))) + 1
        lid = int(state.get("lastLinkId", max([l["id"] for l in sub["links"]] + [0]))) + 1
        return nid, lid

    def inject_loras_subgraph(self, sub: Dict[str, Any], loras: List[Tuple[str, float]]) -> int:
        """Splice LoraLoaderModelOnly between UNETLoader and every MODEL consumer."""
        if not loras:
            return 0
        nodes = {n["id"]: n for n in sub["nodes"]}
        loader = next((n for n in sub["nodes"] if n["type"] in ("UNETLoader", "UNETLoaderGGUF")), None)
        if loader is None:
            log("no UNETLoader inside the I2V subgraph — LoRA injection skipped", "warn")
            return 0

        consumers: List[Tuple[int, int, int]] = []  # (link_id, target_id, target_slot)
        for l in sub["links"]:
            if l["origin_id"] == loader["id"] and l["type"] == "MODEL":
                consumers.append((l["id"], l["target_id"], l["target_slot"]))
        if not consumers:
            log("UNETLoader has no MODEL consumers — LoRA injection skipped", "warn")
            return 0

        next_node, next_link = self._sub_ids(sub)
        sub["links"] = [l for l in sub["links"] if l["id"] not in {c[0] for c in consumers}]

        prev_id, prev_slot = loader["id"], 0
        loader["outputs"][0]["links"] = []
        base_x, base_y = loader.get("pos", [650, 6200])

        for i, (name, strength) in enumerate(loras):
            node = {
                "id": next_node,
                "type": "LoraLoaderModelOnly",
                "pos": [base_x + 60 + i * 40, base_y + 170 + i * 110],
                "size": [380, 82],
                "flags": {},
                "order": 41 + i,
                "mode": 0,
                "inputs": [{"localized_name": "model", "name": "model", "type": "MODEL", "link": next_link}],
                "outputs": [{"localized_name": "MODEL", "name": "MODEL", "type": "MODEL", "links": []}],
                "title": f"Cast LoRA :: {name}",
                "properties": {"Node name for S&R": "LoraLoaderModelOnly", "cnr_id": "comfy-core"},
                "widgets_values": [name, float(strength)],
            }
            sub["links"].append({
                "id": next_link, "origin_id": prev_id, "origin_slot": prev_slot,
                "target_id": next_node, "target_slot": 0, "type": "MODEL",
            })
            if prev_id == loader["id"]:
                loader["outputs"][0]["links"].append(next_link)
            else:
                nodes[prev_id]["outputs"][0]["links"].append(next_link)
            sub["nodes"].append(node)
            nodes[next_node] = node
            prev_id, prev_slot = next_node, 0
            next_node += 1
            next_link += 1

        for _, target_id, target_slot in consumers:
            sub["links"].append({
                "id": next_link, "origin_id": prev_id, "origin_slot": prev_slot,
                "target_id": target_id, "target_slot": target_slot, "type": "MODEL",
            })
            nodes[prev_id]["outputs"][0]["links"].append(next_link)
            if target_id in nodes:
                nodes[target_id]["inputs"][target_slot]["link"] = next_link
            next_link += 1

        sub["state"]["lastNodeId"] = next_node - 1
        sub["state"]["lastLinkId"] = next_link - 1
        log(f"injected {len(loras)} cast LoRA(s) into '{sub.get('name')}' "
            f"across {len(consumers)} sampler pass(es)")
        return len(loras)

    def bypass_director(self, scene_def: Dict[str, Any]) -> None:
        """Wire the Subject input straight to the sampler and mute the in-graph llava node."""
        director = next(
            (n for n in scene_def["nodes"]
             if self._defs().get(n["type"], {}).get("name") == DIRECTOR_DEF_NAME),
            None,
        )
        i2v = next(
            (n for n in scene_def["nodes"]
             if self._defs().get(n["type"], {}).get("name") == I2V_DEF_NAME),
            None,
        )
        if director is None or i2v is None:
            log("director bypass: could not identify both nodes — leaving graph as-is", "warn")
            return

        prompt_slot = next(
            (i for i, inp in enumerate(i2v["inputs"]) if inp.get("name") == "value"), 1
        )
        # kill director -> i2v
        scene_def["links"] = [
            l for l in scene_def["links"]
            if not (l["origin_id"] == director["id"] and l["target_id"] == i2v["id"])
        ]
        # retarget subgraph Subject input -> i2v prompt
        for l in scene_def["links"]:
            if l["origin_id"] == -10 and l["target_id"] == director["id"] and l["type"] == "STRING":
                l["target_id"] = i2v["id"]
                l["target_slot"] = prompt_slot
                i2v["inputs"][prompt_slot]["link"] = l["id"]
                break
        # drop the image feed into the director and mute it
        scene_def["links"] = [
            l for l in scene_def["links"] if l["target_id"] != director["id"]
        ]
        for inp in director.get("inputs", []):
            inp["link"] = None
        for out in director.get("outputs", []):
            out["links"] = []
        director["mode"] = 4  # bypass
        director["title"] = "Ollama director (bypassed — prompts composed in Python)"
        log("director bypassed: composed prompt now reaches the sampler unmodified")

    # ------------------------------------------------------------------ #
    # top-level graph construction
    # ------------------------------------------------------------------ #

    def _clone(self, type_name: str, pos: List[float], title: Optional[str] = None,
               widgets: Optional[List[Any]] = None) -> Dict[str, Any]:
        exemplar = self.exemplars.get(type_name)
        if exemplar is None:
            raise KeyError(f"No exemplar node of type '{type_name}' in the template")
        node = copy.deepcopy(exemplar)
        node["id"] = self._next_node_id
        self._next_node_id += 1
        node["pos"] = pos
        if title:
            node["title"] = title
        if widgets is not None:
            node["widgets_values"] = widgets
        for inp in node.get("inputs", []) or []:
            inp["link"] = None
        for out in node.get("outputs", []) or []:
            out["links"] = []
        return node

    def _link(self, src: Dict[str, Any], src_slot: int, dst: Dict[str, Any], dst_slot: int,
              type_name: str) -> None:
        lid = self._next_link_id
        self._next_link_id += 1
        src["outputs"][src_slot].setdefault("links", []).append(lid)
        dst["inputs"][dst_slot]["link"] = lid
        self.wf["links"].append([lid, src["id"], src_slot, dst["id"], dst_slot, type_name])

    # ------------------------------------------------------------------ #

    def _cast_loras(self) -> List[Tuple[str, float]]:
        if not self.cfg.get("lora.enabled", True):
            return []
        strength = float(self.cfg.get("lora.ltx_lora_strength", 0.75))
        out_dir = Path(self.cfg.get("lora.output_dir", "./loras"))
        loras: List[Tuple[str, float]] = []
        for ch in self.registry.all():
            name = ch.get("lora", {}).get("video")
            if not name:
                continue
            if not (out_dir / name).exists():
                log(f"{ch['id']}: LTX LoRA '{name}' not found on disk — not wired in", "warn")
                continue
            loras.append((f"cast/{name}" if self.cfg.get("lora.output_dir", "").endswith("cast") else name,
                          strength))
        return loras

    def _upload_keyframe(self, path_or_name: str) -> str:
        p = Path(path_or_name)
        if self.comfy and p.exists():
            try:
                return self.comfy.upload_image(p, subfolder="castforge/keyframes")
            except Exception as exc:  # noqa: BLE001
                log(f"keyframe upload failed ({exc}); referencing filename directly", "warn")
        return p.name

    def _info_note(self, bible: Dict[str, Any], lock_sig: str, loras: List[Tuple[str, float]]) -> str:
        lines = [
            f"# {bible.get('title','Untitled')}",
            "",
            f"**{bible.get('logline','')}**",
            "",
            f"- shots: {len(bible['scenes'])} x {self.cfg.get('project.scene_duration')}s "
            f"@ {self.cfg.get('project.fps')}fps",
            f"- identity lock signature: `{lock_sig}` (identical in every shot)",
            f"- director mode: `{self.cfg.get('movie.director_mode')}`",
            f"- anchor mode: `{self.cfg.get('movie.anchor_mode')}` "
            f"(re-anchor every {self.cfg.get('movie.reanchor_every')})",
            "",
            "## Cast",
        ]
        for ch in self.registry.all():
            v = ch.get("voice", {})
            lines.append(
                f"- **{ch['name']}** `{ch['trigger_token']}` — {ch['identity_line']}  \n"
                f"  voice: {v.get('timbre','?')}, {v.get('pitch','?')}, "
                f"{v.get('accent','?')} | seed `{ch.get('audio_seed')}`"
            )
        lines += ["", "## LoRAs on the transformer"]
        lines += [f"- `{n}` @ {s}" for n, s in loras] or ["- none wired (train first)"]
        lines += [
            "",
            "## Before you queue",
            "```",
            "set CUSTOM_WAN_COMFYUI_ARGS=--disable-pinned-memory",
            "```",
            "Generated by CastForge.",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #

    def build(
        self,
        bible: Dict[str, Any],
        anchors: Dict[int, Dict[str, Any]],
        prompts: Dict[int, str],
        lock_sig: str = "",
    ) -> Path:
        banner("STAGE 6 — building the movie workflow")
        self._load()

        scene_def = self._def_by_name(SCENE_DEF_NAME)
        i2v_def = self._def_by_name(I2V_DEF_NAME)
        merge_def = self._def_by_name(MERGE_DEF_NAME)
        director_def = self._def_by_name(DIRECTOR_DEF_NAME)
        self._prune_definitions([scene_def["id"], i2v_def["id"], merge_def["id"], director_def["id"]])

        loras = self._cast_loras()
        self.inject_loras_subgraph(i2v_def, loras)

        if str(self.cfg.get("movie.director_mode", "python")).lower() == "python":
            self.bypass_director(scene_def)
        else:
            log("director_mode=graph — the in-graph llava director stays in circuit")

        # fresh top level
        self.wf["nodes"] = []
        self.wf["links"] = []
        self.wf["groups"] = []

        width = int(self.cfg.get("project.width", 768))
        height = int(self.cfg.get("project.height", 1344))
        fps = int(self.cfg.get("project.fps", 25))
        duration = int(self.cfg.get("project.scene_duration", 10))
        project = self.cfg.get("project.name", "film")
        scenes = bible["scenes"]

        note = self._clone("MarkdownNote", [-620, 120], "CastForge :: film sheet",
                           [self._info_note(bible, lock_sig, loras)])
        note["size"] = [560, 900]
        self.wf["nodes"].append(note)

        prev_scene_node: Optional[Dict[str, Any]] = None
        prev_last_frame: Optional[Dict[str, Any]] = None
        merged: Optional[Dict[str, Any]] = None
        save_nodes: List[Dict[str, Any]] = []

        for sc in scenes:
            i = sc["index"]
            x = 100 + (i - 1) * 900
            anchor = anchors.get(i, {"mode": "chain"})
            title = f"Shot {i:02d} — {sc.get('title','')}"

            # ---- prompt node
            prompt_node = self._clone("PrimitiveStringMultiline", [x, 640],
                                      f"Prompt :: shot {i:02d}", [prompts[i]])
            prompt_node["size"] = [460, 420]
            self.wf["nodes"].append(prompt_node)

            # ---- scene controller
            scene_node = self._clone(scene_def["id"], [x + 500, 300], title,
                                     [width, height, prompts[i], duration, fps])
            self.wf["nodes"].append(scene_node)

            # ---- first frame source
            if anchor.get("mode") in ("first", "reanchor"):
                image_ref = anchor.get("path") or anchor.get("image") or self.cfg.get("movie.first_image")
                comfy_name = (anchor["image"] if anchor.get("mode") == "first"
                              else self._upload_keyframe(str(image_ref)))
                load = self._clone("LoadImage", [x, 180],
                                   f"Anchor :: shot {i:02d}", [comfy_name, "image"])
                self.wf["nodes"].append(load)
                self._link(load, 0, scene_node, 0, "IMAGE")
                log(f"shot {i:02d}: anchored on {comfy_name}")
            else:
                if prev_last_frame is None:
                    raise RuntimeError(f"shot {i} is chained but there is no previous shot")
                self._link(prev_last_frame, 0, scene_node, 0, "IMAGE")
                log(f"shot {i:02d}: chained from shot {i-1} last frame")

            self._link(prompt_node, 0, scene_node, 3, "STRING")

            # ---- last frame tap for the next shot
            last_frame = self._clone("GetImagesFromBatchIndexed", [x + 500, 180],
                                     f"Last frame :: shot {i:02d}", ["-1"])
            self.wf["nodes"].append(last_frame)
            self._link(scene_node, 1, last_frame, 0, "IMAGE")

            preview = self._clone("PreviewImage", [x + 500, 60], f"Preview :: shot {i:02d}", [])
            self.wf["nodes"].append(preview)
            self._link(last_frame, 0, preview, 0, "IMAGE")

            # ---- per-shot save
            save = self._clone("SaveVideo", [x + 500, 1080], f"Save :: shot {i:02d}",
                               [f"video/{project}/shot_{i:02d}", "auto", "auto"])
            self.wf["nodes"].append(save)
            self._link(scene_node, 0, save, 0, "VIDEO")
            save_nodes.append(save)

            # ---- rolling merge, same wiring pattern as your template
            if merged is None and len(save_nodes) == 1:
                merged = None
            else:
                merge = self._clone(merge_def["id"], [x + 500, 1260],
                                    f"Merge :: through shot {i:02d}", [False, "lanczos", "white", False])
                self.wf["nodes"].append(merge)
                source = merged if merged is not None else save_nodes[0]
                self._link(source, 0, merge, 0, "VIDEO")
                self._link(save, 0, merge, 1, "VIDEO")
                merged = merge

            prev_scene_node = scene_node
            prev_last_frame = last_frame

        # ---- final movie
        if merged is not None:
            final = self._clone("SaveVideo", [100 + len(scenes) * 900, 1260],
                                "FULL MOVIE", [f"video/{project}/FULL_MOVIE", "auto", "auto"])
            self.wf["nodes"].append(final)
            self._link(merged, 0, final, 0, "VIDEO")

        self.wf["last_node_id"] = self._next_node_id - 1
        self.wf["last_link_id"] = self._next_link_id - 1
        self.wf["id"] = f"castforge-{project}"
        self.wf.setdefault("extra", {})["castforge"] = {
            "identity_lock_signature": lock_sig,
            "scenes": len(scenes),
            "cast": [c["id"] for c in self.registry.all()],
            "loras": [n for n, _ in loras],
            "seed": stable_seed("movie", project, base=int(self.cfg.get("project.seed", 0))),
        }
        self.wf["extra"].pop("prompt", None)  # stale API cache from the template

        out = self.cfg.movie_dir / f"{project}_castforge_movie.json"
        write_json(out, self.wf)
        log(f"movie workflow -> {out}")
        log(f"nodes: {len(self.wf['nodes'])}, links: {len(self.wf['links'])}, "
            f"shots: {len(scenes)}, total runtime ~{len(scenes) * duration}s")
        return out
