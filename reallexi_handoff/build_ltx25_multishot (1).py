#!/usr/bin/env python3
"""
build_ltx25_multishot.py
------------------------
Rebuilds "Clean Multi Video Clips together (Sam Ayoub)" for LTX-2.5.

WHAT IT DOES
  * Collapses 8 duplicated subgraph definitions (4 wrappers + 4 generators,
    204 duplicated nodes) into ONE shared "LTX-2.5 Shot" definition that all
    shots instance.  Edit the model once -> all shots change.
  * Migrates the 2.3 model stack to the 2.5 split pack (UNETLoader + VAELoader
    x2 + CLIPLoader x2) and swaps CFGGuider -> LTXVDualCFGGuider.
  * Preserves the exact original flow:
        reference image -> Shot 1 -> Shot 2 -> Shot 3 -> Shot 4
        (each shot's last decoded frame becomes the next shot's first frame)
    including the audio-locked A/V latent path driven off the source MP3.
  * Rebuilds the graph layout into labelled, colour-coded groups.
  * Keeps every orphaned prompt preset in a collapsed "PROMPT LIBRARY" group
    instead of deleting them.

USAGE
    python build_ltx25_multishot.py --src <original.json> --out <new.json>
    python build_ltx25_multishot.py --shots 6            # N shots, one flag
    python build_ltx25_multishot.py --emit-config shots.json
    python build_ltx25_multishot.py --config shots.json  # per-shot prompts
    python build_ltx25_multishot.py --clip-seconds 10  # shorter shots
"""

from __future__ import annotations

import argparse
import copy
import json
import uuid
from pathlib import Path

# --------------------------------------------------------------------------
# LTX-2.5 model files (docs.comfy.org/tutorials/video/ltx/ltx-2-5)
# --------------------------------------------------------------------------
M_UNET = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
M_VAE_VIDEO = "ltx-2.5-video-vae-bf16.safetensors"
M_VAE_AUDIO = "ltx-2.5-audio-vae-bf16.safetensors"
M_CLIP_MAIN = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
M_CLIP_ENH = "gemma4_e2b_it_bf16.safetensors"
M_UPSCALER = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"

HF = "https://huggingface.co/Lightricks/LTX-2.5/resolve/main"
MODEL_META = {
    M_UNET: (f"{HF}/diffusion_models/{M_UNET}", "diffusion_models"),
    M_VAE_VIDEO: (f"{HF}/vae/{M_VAE_VIDEO}", "vae"),
    M_VAE_AUDIO: (f"{HF}/vae/{M_VAE_AUDIO}", "vae"),
    M_CLIP_MAIN: (f"{HF}/text_encoders/{M_CLIP_MAIN}", "text_encoders"),
    M_UPSCALER: (f"{HF}/latent_upscale_models/{M_UPSCALER}", "latent_upscale_models"),
    M_CLIP_ENH: (
        f"https://huggingface.co/Comfy-Org/gemma-4/resolve/main/text_encoders/{M_CLIP_ENH}",
        "text_encoders",
    ),
}

# Sigma schedules carry over unchanged from 2.3 -> 2.5.
SIGMAS_LOW = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
SIGMAS_HIGH = "0.85, 0.7250, 0.4219, 0.0"

# Group palette
C_INPUT, C_SETTINGS, C_MODEL = "#3f789e", "#88A", "#8A8"
C_PROMPT, C_PREP, C_AUDIO = "#a1309b", "#b58b2a", "#429"
C_LOW, C_UP, C_HIGH = "#3f789e", "#653", "#232"
C_OUT, C_LIB = "#8AA", "#444"

CORE = {"cnr_id": "comfy-core", "ver": "0.32.0"}

# Ollama writes the ACTION only. Identity comes from the fixed cast block and
# is never regenerated, so the LLM cannot paraphrase the character away.
DIRECTOR_SYSTEM = """You are a shot director for a video generation model.

You are given the FIRST FRAME of the next shot and a short note describing what
should happen in it. Write ONE flowing sentence, at most two, covering only:
  - what the subjects DO next: movement, gesture, interaction
  - how the camera moves

Rules:
  - Never describe appearance. No colours, clothing, hair, fur, species, body,
    face, or age. Those are handled elsewhere.
  - Refer to the subjects only by the words used in the note.
  - Plain prose, present tense. No lists, headings, markdown, emoji or quotes.
  - Never use negations: no "no", "not", "avoid", "without", "don\'t".
  - Output the sentence only. No preamble, no explanation."""

DIRECTOR_USER_HINT = ("Write the next shot's action and camera move. "
                      "Note for this shot: ")

# Read once off the reference image, then pinned to every shot. Deliberately
# forbidden from mentioning subjects - identity is the cast block's job.
LOOK_SYSTEM = """You describe the LOOK of a frame so it can be reproduced.

Write ONE flowing sentence covering only:
  - the medium: live-action footage, 3D animated film, 2D animation, stop motion
  - the colour palette and grade, named concretely
  - the lighting quality and direction
  - lens and depth-of-field character, and grain or texture

Rules:
  - Never mention subjects, characters, people, animals, actions or story.
  - State the medium first and state it plainly.
  - Plain prose. No lists, headings, markdown, emoji or quotes.
  - Never use negations.
  - Output the sentence only."""

LOOK_USER_HINT = ("Describe the medium, palette, lighting and lens of this "
                  "image so every later shot can match it.")

# Whatever medium the look declares, the OTHER mediums go in the negative.
# This is what stops a shot flipping from live action to 3D between cuts.
MEDIUM_NEGATIVES = {
    "live action": ["3d render", "cgi", "computer animation", "cartoon",
                    "anime", "claymation", "video game screenshot",
                    "game cutscene", "stylised illustration", "plastic shading",
                    "toy-like", "unreal engine render", "rendered look"],
    "3d": ["live-action footage", "photographic", "documentary footage",
           "real photograph", "camera phone video", "photoreal skin"],
    "2d": ["3d render", "cgi", "live-action footage", "photographic",
           "photoreal", "volumetric rendering"],
    "stop motion": ["3d render", "cgi", "live-action footage", "photographic",
                    "smooth cg animation"],
}
STYLE_NEGATIVES = ["style change", "medium change", "inconsistent art style",
                   "colour grade shift", "palette shift", "different film stock",
                   "mismatched lighting between shots"]


def medium_negatives(look_text: str) -> list:
    """Pick the anti-medium terms from whatever the look block declares."""
    t = (look_text or "").lower()
    if "stop motion" in t or "stop-motion" in t:
        key = "stop motion"
    elif "2d" in t or "hand-drawn" in t or "traditional animation" in t:
        key = "2d"
    elif "3d" in t or "cgi" in t or "animated film" in t or "render" in t:
        key = "3d"
    elif t.strip():
        key = "live action"
    else:
        return list(STYLE_NEGATIVES)
    return MEDIUM_NEGATIVES[key] + STYLE_NEGATIVES


# --------------------------------------------------------------------------
# graph builder
# --------------------------------------------------------------------------
class Graph:
    """Minimal LiteGraph emitter shared by the root graph and the subgraph."""

    def __init__(self, link_style: str):
        self.link_style = link_style  # "array" (root) | "object" (subgraph)
        self.nodes: list[dict] = []
        self.links: list = []
        self.groups: list[dict] = []
        self._nid = 0
        self._lid = 0
        self._gid = 0

    # -- ids ---------------------------------------------------------------
    def nid(self) -> int:
        self._nid += 1
        return self._nid

    def lid(self) -> int:
        self._lid += 1
        return self._lid

    # -- nodes -------------------------------------------------------------
    def node(self, ntype, pos, size, *, title=None, widgets=None, inputs=None,
             outputs=None, props=None, flags=None, mode=0) -> dict:
        n = {
            "id": self.nid(),
            "type": ntype,
            "pos": list(pos),
            "size": list(size),
            "flags": flags or {},
            "order": 0,
            "mode": mode,
            "inputs": inputs or [],
            "outputs": outputs or [],
            "properties": {"Node name for S&R": ntype, **CORE, **(props or {})},
        }
        if title:
            n["title"] = title
        if widgets is not None:
            n["widgets_values"] = widgets
        self.nodes.append(n)
        return n

    @staticmethod
    def sock_in(name, typ, *, widget=False, label=None) -> dict:
        d = {"localized_name": label or name, "name": name, "type": typ, "link": None}
        if widget:
            d["widget"] = {"name": name}
        return d

    @staticmethod
    def sock_out(name, typ) -> dict:
        return {"localized_name": name, "name": name, "type": typ, "links": []}

    # -- links -------------------------------------------------------------
    def link(self, src, src_slot, dst, dst_slot, typ) -> int:
        """src/dst are node dicts, or ints for the -10/-20 IO pseudo-nodes."""
        lid = self.lid()
        sid = src if isinstance(src, int) else src["id"]
        did = dst if isinstance(dst, int) else dst["id"]
        if self.link_style == "array":
            self.links.append([lid, sid, src_slot, did, dst_slot, typ])
        else:
            self.links.append({"id": lid, "origin_id": sid, "origin_slot": src_slot,
                               "target_id": did, "target_slot": dst_slot, "type": typ})
        if not isinstance(src, int):
            src["outputs"][src_slot]["links"].append(lid)
        if not isinstance(dst, int):
            dst["inputs"][dst_slot]["link"] = lid
        return lid

    # -- groups ------------------------------------------------------------
    def group(self, title, bounding, color) -> dict:
        self._gid += 1
        g = {"id": self._gid, "title": title, "bounding": list(bounding),
             "color": color, "font_size": 24, "flags": {}}
        self.groups.append(g)
        return g


def loader_props(ntype, filename):
    url, directory = MODEL_META[filename]
    return {"Node name for S&R": ntype,
            "models": [{"name": filename, "url": url, "directory": directory}]}


# --------------------------------------------------------------------------
# the shared shot subgraph
# --------------------------------------------------------------------------
def _assert_sequential(io_spec) -> None:
    """Guard the invariant: value_N names must ascend with array position.

    Getting this wrong is silent in the JSON and only shows up at execution as
    "Return type mismatch between linked nodes" plus a zero-width latent.
    """
    seen = []
    for idx, (name, _t, label, _tg) in enumerate(io_spec):
        if name == "value":
            seen.append((0, idx, label))
        elif name.startswith("value_"):
            seen.append((int(name.split("_")[1]), idx, label))
    for a, b in zip(seen, seen[1:]):
        if b[0] != a[0] + 1 or b[1] < a[1]:
            raise AssertionError(
                f"subgraph input numbering broken: '{a[2]}' then '{b[2]}' "
                f"-> value_{a[0]} at slot {a[1]}, value_{b[0]} at slot {b[1]}. "
                f"ComfyUI maps these by name, so widgets would land on the "
                f"wrong nodes.")


def build_shot_subgraph(sub_id: str, negative_prompt: str, img_compression: int,
                        strength_low: float, second_upscale: bool,
                        cfg_a: float = 1, cfg_b: float = 1,
                        handoff_index: str = "-1",
                        ollama_model: str = "llava:13b-v1.6 (Q4_0, 8.0GB)",
                        ollama_url: str = "http://localhost:11434",
                        use_ollama: bool = False,
                        save_frames: bool = True,
                        frame_check: bool = True,
                        gate: bool = False,
                        window: int = 12,
                        recency: float = 0.6,
                        color_match: bool = True,
                        color_strength: float = 0.85,
                        simple_prompt: bool = False) -> dict:
    g = Graph("object")
    IN, OUT = -10, -20

    # ---- 00 MODELS (shared by every shot - edit once, applies to all) ----
    x = 40
    unet = g.node("UNETLoader", (x, 60), (430, 82), title="LTX-2.5 Transformer",
                  widgets=[M_UNET, "default"],
                  inputs=[g.sock_in("unet_name", "COMBO", widget=True)],
                  outputs=[g.sock_out("MODEL", "MODEL")],
                  props=loader_props("UNETLoader", M_UNET))
    vae_v = g.node("VAELoader", (x, 180), (430, 58), title="Video VAE",
                   widgets=[M_VAE_VIDEO],
                   inputs=[g.sock_in("vae_name", "COMBO", widget=True)],
                   outputs=[g.sock_out("VAE", "VAE")],
                   props=loader_props("VAELoader", M_VAE_VIDEO))
    vae_a = g.node("VAELoader", (x, 275), (430, 58), title="Audio VAE",
                   widgets=[M_VAE_AUDIO],
                   inputs=[g.sock_in("vae_name", "COMBO", widget=True)],
                   outputs=[g.sock_out("VAE", "VAE")],
                   props=loader_props("VAELoader", M_VAE_AUDIO))
    clip_main = g.node("CLIPLoader", (x, 370), (430, 106), title="Gemma 4 12B (LTX-2.5)",
                       widgets=[M_CLIP_MAIN, "ltxv", "default"],
                       inputs=[g.sock_in("clip_name", "COMBO", widget=True)],
                       outputs=[g.sock_out("CLIP", "CLIP")],
                       props=loader_props("CLIPLoader", M_CLIP_MAIN))
    ups_a = g.node("LatentUpscaleModelLoader", (x, 515), (430, 58), title="Spatial Upscaler x2",
                   widgets=[M_UPSCALER],
                   inputs=[g.sock_in("model_name", "COMBO", widget=True)],
                   outputs=[g.sock_out("LATENT_UPSCALE_MODEL", "LATENT_UPSCALE_MODEL")],
                   props=loader_props("LatentUpscaleModelLoader", M_UPSCALER))
    ups_b = None
    if second_upscale:
        ups_b = g.node("LatentUpscaleModelLoader", (x, 610), (430, 58),
                       title="Spatial Upscaler x2 (finish)", widgets=[M_UPSCALER],
                       inputs=[g.sock_in("model_name", "COMBO", widget=True)],
                       outputs=[g.sock_out("LATENT_UPSCALE_MODEL", "LATENT_UPSCALE_MODEL")],
                       props=loader_props("LatentUpscaleModelLoader", M_UPSCALER))
    g.group("00 · MODELS  (shared by all shots)", (20, -10, 470, 715), C_MODEL)

    # ---- 01 SHOT SETTINGS -------------------------------------------------
    x = 530
    w_node = g.node("PrimitiveInt", (x, 60), (330, 82), title="Width",
                    widgets=[800, "fixed"],
                    inputs=[g.sock_in("value", "INT", widget=True)],
                    outputs=[g.sock_out("INT", "INT")])
    h_node = g.node("PrimitiveInt", (x, 180), (330, 82), title="Height",
                    widgets=[1024, "fixed"],
                    inputs=[g.sock_in("value", "INT", widget=True)],
                    outputs=[g.sock_out("INT", "INT")])
    fps_node = g.node("PrimitiveInt", (x, 300), (330, 82), title="Frame Rate",
                      widgets=[25, "fixed"],
                      inputs=[g.sock_in("value", "INT", widget=True)],
                      outputs=[g.sock_out("INT", "INT")])
    dur_node = g.node("PrimitiveFloat", (x, 420), (330, 58), title="Duration (sec)",
                      widgets=[20],
                      inputs=[g.sock_in("value", "FLOAT", widget=True)],
                      outputs=[g.sock_out("FLOAT", "FLOAT")])
    t2v = g.node("PrimitiveBoolean", (x, 515), (330, 58), title="Switch to Text to Video?",
                 widgets=[False], outputs=[g.sock_out("BOOLEAN", "BOOLEAN")])
    g.group("01 · SHOT SETTINGS", (510, -10, 370, 625), C_SETTINGS)

    # ---- 02 PROMPT --------------------------------------------------------
    x = 920
    if simple_prompt:
        # compat: everything is composed in Python and lands in one text node,
        # so the graph needs no StringConcatenate and no runtime rewriting.
        look_txt = beat_txt = use_llm = ollama = None
        cat_static = cat_llm = switch = cat_look = None
        cast_txt = g.node("PrimitiveStringMultiline", (x, 60), (520, 560),
                          title="PROMPT  (look + cast + beat, pre-composed)",
                          widgets=[""],
                          inputs=[g.sock_in("value", "STRING", widget=True)],
                          outputs=[g.sock_out("STRING", "STRING")])
        prompt_src, prompt_slot = cast_txt, 0
    else:
        look_txt = g.node("PrimitiveStringMultiline", (x, -300), (520, 200),
                          title="LOOK / medium + grade  (pinned for all shots)",
                          widgets=[""],
                          inputs=[g.sock_in("value", "STRING", widget=True)],
                          outputs=[g.sock_out("STRING", "STRING")])
        cast_txt = g.node("PrimitiveStringMultiline", (x, 60), (520, 300),
                          title="CAST / identity  (never regenerated)", widgets=[""],
                          inputs=[g.sock_in("value", "STRING", widget=True)],
                          outputs=[g.sock_out("STRING", "STRING")])
        beat_txt = g.node("PrimitiveStringMultiline", (x, 400), (520, 150),
                          title="BEAT  (what happens this shot)", widgets=[""],
                          inputs=[g.sock_in("value", "STRING", widget=True)],
                          outputs=[g.sock_out("STRING", "STRING")])
        use_llm = g.node("PrimitiveBoolean", (x, 590), (520, 58),
                         title="Use Ollama for the action?", widgets=[use_ollama],
                         inputs=[g.sock_in("value", "BOOLEAN", widget=True)],
                         outputs=[g.sock_out("BOOLEAN", "BOOLEAN")])

        ollama = g.node("OllamaImageDescriber", (x, 690), (520, 420),
                        title="Shot director (sees this shot's first frame)",
                        widgets=[ollama_model, "", ollama_url, 300, 0.2, 40, 0.9,
                                 1.1, 42, 2048, 200, 5, DIRECTOR_SYSTEM,
                                 DIRECTOR_USER_HINT],
                        inputs=[g.sock_in("images", "IMAGE"),
                                g.sock_in("structured_output_format", "STRING"),
                                g.sock_in("system_context", "STRING", widget=True),
                                g.sock_in("prompt", "STRING", widget=True)],
                        outputs=[g.sock_out("result", "STRING")],
                        props={"cnr_id": "comfyui-ollama-describer", "ver": "1.1.0"})

        cat_static = g.node("StringConcatenate", (x + 560, 60), (300, 130),
                            title="cast + beat", widgets=["", "", " "],
                            inputs=[g.sock_in("string_a", "STRING", widget=True),
                                    g.sock_in("string_b", "STRING", widget=True),
                                    g.sock_in("delimiter", "STRING", widget=True)],
                            outputs=[g.sock_out("STRING", "STRING")])
        cat_llm = g.node("StringConcatenate", (x + 560, 220), (300, 130),
                         title="cast + director", widgets=["", "", " "],
                         inputs=[g.sock_in("string_a", "STRING", widget=True),
                                 g.sock_in("string_b", "STRING", widget=True),
                                 g.sock_in("delimiter", "STRING", widget=True)],
                         outputs=[g.sock_out("STRING", "STRING")])
        switch = g.node("ComfySwitchNode", (x + 560, 380), (300, 106),
                        title="Prompt source", widgets=[False],
                        inputs=[g.sock_in("on_false", "STRING"),
                                g.sock_in("on_true", "STRING"),
                                g.sock_in("switch", "BOOLEAN", widget=True)],
                        outputs=[g.sock_out("output", "STRING")])
        cat_look = g.node("StringConcatenate", (x + 560, 860), (300, 130),
                          title="look + prompt", widgets=["", "", " "],
                          inputs=[g.sock_in("string_a", "STRING", widget=True),
                                  g.sock_in("string_b", "STRING", widget=True),
                                  g.sock_in("delimiter", "STRING", widget=True)],
                          outputs=[g.sock_out("STRING", "STRING")])

        prompt_src, prompt_slot = cat_look, 0

    prev_prompt = g.node("PreviewAny", (x + 1480, 60), (300, 260),
                         title="Final prompt sent to the model", widgets=[],
                         inputs=[g.sock_in("source", "*")],
                         outputs=[g.sock_out("STRING", "STRING")])
    pos_enc = g.node("CLIPTextEncode", (x + 1480, 340), (300, 130), title="Positive",
                     widgets=[""],
                     inputs=[g.sock_in("clip", "CLIP"), g.sock_in("text", "STRING", widget=True)],
                     outputs=[g.sock_out("CONDITIONING", "CONDITIONING")])
    neg_enc = g.node("CLIPTextEncode", (x + 1480, 500), (300, 130), title="Negative",
                     widgets=[negative_prompt],
                     inputs=[g.sock_in("clip", "CLIP"), g.sock_in("text", "STRING", widget=True)],
                     outputs=[g.sock_out("CONDITIONING", "CONDITIONING")])
    cond = g.node("LTXVConditioning", (x + 1480, 660), (300, 106),
                  widgets=[25],
                  inputs=[g.sock_in("positive", "CONDITIONING"),
                          g.sock_in("negative", "CONDITIONING"),
                          g.sock_in("frame_rate", "FLOAT", widget=True)],
                  outputs=[g.sock_out("positive", "CONDITIONING"),
                           g.sock_out("negative", "CONDITIONING")])
    g.group("02 · PROMPT", (900, -10, 1900, 1140), C_PROMPT)

    # ---- 03 IMAGE PREPROCESS (identity path) ------------------------------
    x = 2290
    resize = g.node("ResizeImageMaskNode", (x, 60), (400, 178), title="Lock Aspect (W x H)",
                    widgets=["scale dimensions", 800, 1024, "disabled", "lanczos"],
                    inputs=[g.sock_in("input", "IMAGE,MASK"),
                            g.sock_in("resize_type.width", "INT", widget=True, label="width"),
                            g.sock_in("resize_type.height", "INT", widget=True, label="height")],
                    outputs=[g.sock_out("resized", "IMAGE,MASK")])
    longedge = g.node("ResizeImagesByLongerEdge", (x, 280), (400, 58),
                      widgets=[1536], inputs=[g.sock_in("images", "IMAGE")],
                      outputs=[g.sock_out("images", "IMAGE")])
    pre = g.node("LTXVPreprocess", (x, 380), (400, 58), title="LTXVPreprocess (compression)",
                 widgets=[img_compression], inputs=[g.sock_in("image", "IMAGE")],
                 outputs=[g.sock_out("output_image", "IMAGE")])
    g.group("03 · IMAGE PREPROCESS  (identity path)", (2270, -10, 440, 470), C_PREP)

    # ---- 04 AUDIO CONDITIONING (locked to source track) -------------------
    x = 2290
    trim = g.node("TrimAudioDuration", (x, 560), (400, 106), title="Trim to shot window",
                  widgets=[0, 20],
                  inputs=[g.sock_in("audio", "AUDIO"),
                          g.sock_in("start_index", "FLOAT", widget=True),
                          g.sock_in("duration", "FLOAT", widget=True)],
                  outputs=[g.sock_out("AUDIO", "AUDIO")])
    a_enc = g.node("LTXVAudioVAEEncode", (x, 700), (400, 58), widgets=[],
                   inputs=[g.sock_in("audio", "AUDIO"), g.sock_in("audio_vae", "VAE")],
                   outputs=[g.sock_out("Audio Latent", "LATENT")])
    solid = g.node("SolidMask", (x, 800), (400, 106), title="Audio lock mask (0 = keep)",
                   widgets=[0, 1024, 1024],
                   inputs=[g.sock_in("width", "INT", widget=True),
                           g.sock_in("height", "INT", widget=True)],
                   outputs=[g.sock_out("MASK", "MASK")])
    a_mask = g.node("SetLatentNoiseMask", (x, 945), (400, 58), widgets=[],
                    inputs=[g.sock_in("samples", "LATENT"), g.sock_in("mask", "MASK")],
                    outputs=[g.sock_out("LATENT", "LATENT")])
    g.group("04 · AUDIO CONDITIONING  (locked to source track)",
            (2270, 510, 440, 520), C_AUDIO)

    # ---- 05 NUMBER CONVERSION + EMPTY LATENT ------------------------------
    x = 2760
    m_w = g.node("ComfyMathExpression", (x, 60), (300, 106), title="width / 2",
                 widgets=["a/2"],
                 inputs=[g.sock_in("values.a", "FLOAT,INT,BOOLEAN", label="a"),
                         g.sock_in("values.b", "FLOAT,INT,BOOLEAN", label="b")],
                 outputs=[g.sock_out("FLOAT", "FLOAT"), g.sock_out("INT", "INT"),
                          g.sock_out("BOOL", "BOOLEAN")])
    m_h = g.node("ComfyMathExpression", (x, 200), (300, 106), title="height / 2",
                 widgets=["a/2"],
                 inputs=[g.sock_in("values.a", "FLOAT,INT,BOOLEAN", label="a"),
                         g.sock_in("values.b", "FLOAT,INT,BOOLEAN", label="b")],
                 outputs=[g.sock_out("FLOAT", "FLOAT"), g.sock_out("INT", "INT"),
                          g.sock_out("BOOL", "BOOLEAN")])
    m_fps = g.node("ComfyMathExpression", (x, 340), (300, 106), title="fps",
                   widgets=["a"],
                   inputs=[g.sock_in("values.a", "FLOAT,INT,BOOLEAN", label="a"),
                           g.sock_in("values.b", "FLOAT,INT,BOOLEAN", label="b")],
                   outputs=[g.sock_out("FLOAT", "FLOAT"), g.sock_out("INT", "INT"),
                            g.sock_out("BOOL", "BOOLEAN")])
    m_len = g.node("ComfyMathExpression", (x, 480), (300, 106), title="duration x fps + 1",
                   widgets=["a * b + 1"],
                   inputs=[g.sock_in("values.a", "FLOAT,INT,BOOLEAN", label="a"),
                           g.sock_in("values.b", "FLOAT,INT,BOOLEAN", label="b")],
                   outputs=[g.sock_out("FLOAT", "FLOAT"), g.sock_out("INT", "INT"),
                            g.sock_out("BOOL", "BOOLEAN")])
    empty = g.node("EmptyLTXVLatentVideo", (x, 640), (300, 130),
                   widgets=[768, 512, 97, 1],
                   inputs=[g.sock_in("width", "INT", widget=True),
                           g.sock_in("height", "INT", widget=True),
                           g.sock_in("length", "INT", widget=True)],
                   outputs=[g.sock_out("LATENT", "LATENT")])
    g.group("05 · NUMBERS + EMPTY LATENT", (2740, -10, 340, 800), C_SETTINGS)

    # ---- 06 GENERATE LOW RESOLUTION ---------------------------------------
    x = 3130
    i2v_low = g.node("LTXVImgToVideoInplace", (x, 60), (400, 106),
                     title="First-frame lock (low res)",
                     widgets=[strength_low, False],
                     inputs=[g.sock_in("vae", "VAE"), g.sock_in("image", "IMAGE"),
                             g.sock_in("latent", "LATENT"),
                             g.sock_in("bypass", "BOOLEAN", widget=True)],
                     outputs=[g.sock_out("latent", "LATENT")])
    cat_low = g.node("LTXVConcatAVLatent", (x, 210), (400, 58), widgets=[],
                     inputs=[g.sock_in("video_latent", "LATENT"),
                             g.sock_in("audio_latent", "LATENT")],
                     outputs=[g.sock_out("latent", "LATENT")])
    noise_low = g.node("RandomNoise", (x, 310), (400, 82), title="Seed",
                       widgets=[0, "fixed"],
                       inputs=[g.sock_in("noise_seed", "INT", widget=True)],
                       outputs=[g.sock_out("NOISE", "NOISE")])
    ks_low = g.node("KSamplerSelect", (x, 430), (400, 58), widgets=["euler_ancestral"],
                    outputs=[g.sock_out("SAMPLER", "SAMPLER")])
    sig_low = g.node("ManualSigmas", (x, 530), (400, 82), widgets=[SIGMAS_LOW],
                     outputs=[g.sock_out("SIGMAS", "SIGMAS")])
    guide_low = g.node("LTXVDualCFGGuider", (x, 650), (400, 106),
                       widgets=[cfg_a, cfg_b],
                       inputs=[g.sock_in("model", "MODEL"),
                               g.sock_in("positive", "CONDITIONING"),
                               g.sock_in("negative", "CONDITIONING")],
                       outputs=[g.sock_out("GUIDER", "GUIDER")])
    samp_low = g.node("SamplerCustomAdvanced", (x + 440, 310), (300, 154), widgets=[],
                      inputs=[g.sock_in("noise", "NOISE"), g.sock_in("guider", "GUIDER"),
                              g.sock_in("sampler", "SAMPLER"), g.sock_in("sigmas", "SIGMAS"),
                              g.sock_in("latent_image", "LATENT")],
                      outputs=[g.sock_out("output", "LATENT"),
                               g.sock_out("denoised_output", "LATENT")])
    sep_low = g.node("LTXVSeparateAVLatent", (x + 440, 500), (300, 82), widgets=[],
                     inputs=[g.sock_in("av_latent", "LATENT")],
                     outputs=[g.sock_out("video_latent", "LATENT"),
                              g.sock_out("audio_latent", "LATENT")])
    g.group("06 · GENERATE LOW RESOLUTION", (3110, -10, 800, 800), C_LOW)

    # ---- 07 LATENT UPSCALE -------------------------------------------------
    x = 3960
    upscale = g.node("LTXVLatentUpsampler", (x, 310), (330, 106), title="Latent Upscale x2",
                     widgets=[],
                     inputs=[g.sock_in("samples", "LATENT"),
                             g.sock_in("upscale_model", "LATENT_UPSCALE_MODEL"),
                             g.sock_in("vae", "VAE")],
                     outputs=[g.sock_out("LATENT", "LATENT")])
    g.group("07 · LATENT UPSCALE", (3940, -10, 370, 800), C_UP)

    # ---- 08 GENERATE HIGH RESOLUTION --------------------------------------
    x = 4360
    i2v_high = g.node("LTXVImgToVideoInplace", (x, 60), (400, 106),
                      title="First-frame lock (high res)",
                      widgets=[1, False],
                      inputs=[g.sock_in("vae", "VAE"), g.sock_in("image", "IMAGE"),
                              g.sock_in("latent", "LATENT"),
                              g.sock_in("bypass", "BOOLEAN", widget=True)],
                      outputs=[g.sock_out("latent", "LATENT")])
    cat_high = g.node("LTXVConcatAVLatent", (x, 210), (400, 58), widgets=[],
                      inputs=[g.sock_in("video_latent", "LATENT"),
                              g.sock_in("audio_latent", "LATENT")],
                      outputs=[g.sock_out("latent", "LATENT")])
    noise_high = g.node("RandomNoise", (x, 310), (400, 82), title="Refiner noise",
                        widgets=[42, "fixed"],
                        inputs=[g.sock_in("noise_seed", "INT", widget=True)],
                        outputs=[g.sock_out("NOISE", "NOISE")])
    ks_high = g.node("KSamplerSelect", (x, 430), (400, 58), widgets=["euler_ancestral"],
                     outputs=[g.sock_out("SAMPLER", "SAMPLER")])
    sig_high = g.node("ManualSigmas", (x, 530), (400, 82), widgets=[SIGMAS_HIGH],
                      outputs=[g.sock_out("SIGMAS", "SIGMAS")])
    guide_high = g.node("LTXVDualCFGGuider", (x, 650), (400, 106),
                        widgets=[cfg_a, cfg_b],
                        inputs=[g.sock_in("model", "MODEL"),
                                g.sock_in("positive", "CONDITIONING"),
                                g.sock_in("negative", "CONDITIONING")],
                        outputs=[g.sock_out("GUIDER", "GUIDER")])
    samp_high = g.node("SamplerCustomAdvanced", (x + 440, 310), (300, 154), widgets=[],
                       inputs=[g.sock_in("noise", "NOISE"), g.sock_in("guider", "GUIDER"),
                               g.sock_in("sampler", "SAMPLER"), g.sock_in("sigmas", "SIGMAS"),
                               g.sock_in("latent_image", "LATENT")],
                       outputs=[g.sock_out("output", "LATENT"),
                                g.sock_out("denoised_output", "LATENT")])
    sep_high = g.node("LTXVSeparateAVLatent", (x + 440, 500), (300, 82), widgets=[],
                      inputs=[g.sock_in("av_latent", "LATENT")],
                      outputs=[g.sock_out("video_latent", "LATENT"),
                               g.sock_out("audio_latent", "LATENT")])
    g.group("08 · GENERATE HIGH RESOLUTION", (4340, -10, 800, 800), C_HIGH)

    # ---- 09 DECODE / SAVE / HANDOFF ---------------------------------------
    x = 5190
    finish = None
    if second_upscale:
        finish = g.node("LTXVLatentUpsampler", (x, 60), (360, 106),
                        title="Finish Upscale x2 (optional)", widgets=[],
                        inputs=[g.sock_in("samples", "LATENT"),
                                g.sock_in("upscale_model", "LATENT_UPSCALE_MODEL"),
                                g.sock_in("vae", "VAE")],
                        outputs=[g.sock_out("LATENT", "LATENT")])
    dec_v = g.node("VAEDecodeTiled", (x, 210), (360, 154), title="Decode Video",
                   widgets=[512, 64, 64, 16],
                   inputs=[g.sock_in("samples", "LATENT"), g.sock_in("vae", "VAE")],
                   outputs=[g.sock_out("IMAGE", "IMAGE")])
    dec_a = g.node("LTXVAudioVAEDecode", (x, 400), (360, 58), title="Decode Audio",
                   widgets=[],
                   inputs=[g.sock_in("samples", "LATENT"), g.sock_in("audio_vae", "VAE")],
                   outputs=[g.sock_out("Audio", "AUDIO")])
    mkvid = g.node("CreateVideo", (x, 500), (360, 106), widgets=[24, 8],
                   inputs=[g.sock_in("images", "IMAGE"), g.sock_in("audio", "AUDIO"),
                           g.sock_in("fps", "FLOAT", widget=True)],
                   outputs=[g.sock_out("VIDEO", "VIDEO")])
    save = g.node("SaveVideo", (x, 640), (360, 130),
                  widgets=["video/ltx25_shot", "auto", "auto"],
                  inputs=[g.sock_in("video", "VIDEO"),
                          g.sock_in("filename_prefix", "STRING", widget=True)],
                  outputs=[g.sock_out("video", "VIDEO")])
    handoff = None
    if not frame_check:
        handoff = g.node("GetImagesFromBatchIndexed", (x, 810), (360, 82),
                         title="Handoff frame -> next shot",
                         widgets=[handoff_index],
                         inputs=[g.sock_in("images", "IMAGE")],
                         outputs=[g.sock_out("IMAGE", "IMAGE")])
    prev_hand = g.node("PreviewImage", (x, 930), (360, 246), title="Continuity check",
                       widgets=[], inputs=[g.sock_in("images", "IMAGE")],
                       outputs=[g.sock_out("images", "IMAGE")])

    # Persist the identity frames. Without these the only record of what each
    # shot received and produced is a PreviewImage in temp/, which is wiped.
    name_in = name_out = save_in = save_out = None
    if save_frames and simple_prompt:
        save_in = g.node("SaveImage", (x + 400, 220), (330, 270),
                         title="Shot input frame → output/",
                         widgets=["castrefs/shot_first"],
                         inputs=[g.sock_in("images", "IMAGE"),
                                 g.sock_in("filename_prefix", "STRING", widget=True)],
                         outputs=[])
        save_out = g.node("SaveImage", (x + 400, 520), (330, 270),
                          title="Handoff frame → output/",
                          widgets=["castrefs/shot_handoff"],
                          inputs=[g.sock_in("images", "IMAGE"),
                                  g.sock_in("filename_prefix", "STRING", widget=True)],
                          outputs=[])
    elif save_frames:
        name_in = g.node("StringConcatenate", (x + 400, 60), (330, 130),
                         title="prefix + _first", widgets=["", "_first", ""],
                         inputs=[g.sock_in("string_a", "STRING", widget=True),
                                 g.sock_in("string_b", "STRING", widget=True),
                                 g.sock_in("delimiter", "STRING", widget=True)],
                         outputs=[g.sock_out("STRING", "STRING")])
        save_in = g.node("SaveImage", (x + 400, 220), (330, 270),
                         title="Shot input frame → output/",
                         widgets=["castrefs/shot"],
                         inputs=[g.sock_in("images", "IMAGE"),
                                 g.sock_in("filename_prefix", "STRING", widget=True)],
                         outputs=[])
        name_out = g.node("StringConcatenate", (x + 400, 520), (330, 130),
                          title="prefix + _handoff", widgets=["", "_handoff", ""],
                          inputs=[g.sock_in("string_a", "STRING", widget=True),
                                  g.sock_in("string_b", "STRING", widget=True),
                                  g.sock_in("delimiter", "STRING", widget=True)],
                          outputs=[g.sock_out("STRING", "STRING")])
        save_out = g.node("SaveImage", (x + 400, 680), (330, 270),
                          title="Handoff frame → output/",
                          widgets=["castrefs/shot"],
                          inputs=[g.sock_in("images", "IMAGE"),
                                  g.sock_in("filename_prefix", "STRING", widget=True)],
                          outputs=[])
    g.group("09 · DECODE · SAVE · HANDOFF", (5170, -10, 400, 1210), C_OUT)

    # ---- 09b HANDOFF QUALITY CHECK ----------------------------------------
    # Replaces the blind "take frame -1" with a scored pick across the tail.
    # Needs custom_nodes/reallexi_handoff installed.
    sel = qgate = rep_sel = rep_gate = None
    if frame_check:
        sel = g.node("HandoffFrameSelect", (x + 810, 60), (360, 210),
                     title="Pick the best seed frame",
                     widgets=[window, 1.0, 1.0, recency, 2.0],
                     inputs=[g.sock_in("images", "IMAGE"),
                             g.sock_in("window", "INT", widget=True),
                             g.sock_in("weight_sharpness", "FLOAT", widget=True),
                             g.sock_in("weight_identity", "FLOAT", widget=True),
                             g.sock_in("weight_recency", "FLOAT", widget=True),
                             g.sock_in("clipping_penalty", "FLOAT", widget=True),
                             g.sock_in("reference", "IMAGE")],
                     outputs=[g.sock_out("frame", "IMAGE"),
                              g.sock_out("index", "INT"),
                              g.sock_out("score", "FLOAT"),
                              g.sock_out("report", "STRING")],
                     props={"Node name for S&R": "HandoffFrameSelect",
                            "cnr_id": "reallexi-handoff", "ver": "1.0.0"})
        rep_sel = g.node("PreviewAny", (x + 810, 300), (360, 260),
                         title="Why this frame was chosen", widgets=[],
                         inputs=[g.sock_in("source", "*")],
                         outputs=[g.sock_out("STRING", "STRING")])
        if gate:
            qgate = g.node("HandoffQualityGate", (x + 810, 590), (360, 160),
                           title="Reject the frame if it drifted",
                           widgets=[0.72, 0.35, "fall back to reference"],
                           inputs=[g.sock_in("frame", "IMAGE"),
                                   g.sock_in("reference", "IMAGE"),
                                   g.sock_in("min_similarity", "FLOAT", widget=True),
                                   g.sock_in("max_clipping", "FLOAT", widget=True),
                                   g.sock_in("on_fail", "COMBO", widget=True)],
                           outputs=[g.sock_out("frame", "IMAGE"),
                                    g.sock_out("passed", "BOOLEAN"),
                                    g.sock_out("report", "STRING")],
                           props={"Node name for S&R": "HandoffQualityGate",
                                  "cnr_id": "reallexi-handoff", "ver": "1.0.0"})
            rep_gate = g.node("PreviewAny", (x + 810, 780), (360, 200),
                              title="Gate verdict", widgets=[],
                              inputs=[g.sock_in("source", "*")],
                              outputs=[g.sock_out("STRING", "STRING")])
        cmatch = None
        if color_match:
            cmatch = g.node("HandoffColorMatch", (x + 810, 1030), (360, 180),
                            title="Pin the palette to the reference",
                            widgets=["mean/std (gentle)", color_strength, False],
                            inputs=[g.sock_in("frame", "IMAGE"),
                                    g.sock_in("reference", "IMAGE"),
                                    g.sock_in("mode", "COMBO", widget=True),
                                    g.sock_in("strength", "FLOAT", widget=True),
                                    g.sock_in("preserve_luminance", "BOOLEAN",
                                              widget=True)],
                            outputs=[g.sock_out("frame", "IMAGE"),
                                     g.sock_out("report", "STRING")],
                            props={"Node name for S&R": "HandoffColorMatch",
                                   "cnr_id": "reallexi-handoff", "ver": "1.0.0"})
        g.group("09b · HANDOFF QUALITY CHECK", (x + 790, -10, 400,
                                                1240 if color_match else 1010), C_UP)

    if save_frames:
        g.group("10 · CAST LIBRARY  (reusable identity frames)",
                (5590, -10, 370, 990), C_LIB)

    # ======================= WIRING =======================================
    L = g.link
    # models
    L(unet, 0, guide_low, 0, "MODEL")
    L(unet, 0, guide_high, 0, "MODEL")
    L(clip_main, 0, pos_enc, 0, "CLIP")
    L(clip_main, 0, neg_enc, 0, "CLIP")

    # prompt: identity is fixed, action is either the beat or Ollama's rewrite
    L(prompt_src, prompt_slot, prev_prompt, 0, "STRING")
    L(prompt_src, prompt_slot, pos_enc, 1, "STRING")
    if not simple_prompt:
        L(pre, 0, ollama, 0, "IMAGE")
        L(beat_txt, 0, ollama, 3, "STRING")
        L(cast_txt, 0, cat_static, 0, "STRING")
        L(beat_txt, 0, cat_static, 1, "STRING")
        L(cast_txt, 0, cat_llm, 0, "STRING")
        L(ollama, 0, cat_llm, 1, "STRING")
        L(cat_static, 0, switch, 0, "STRING")
        L(cat_llm, 0, switch, 1, "STRING")
        L(use_llm, 0, switch, 2, "BOOLEAN")
        L(look_txt, 0, cat_look, 0, "STRING")
        L(switch, 0, cat_look, 1, "STRING")
    L(pos_enc, 0, cond, 0, "CONDITIONING")
    L(neg_enc, 0, cond, 1, "CONDITIONING")
    L(m_fps, 0, cond, 2, "FLOAT")

    # preprocess
    L(w_node, 0, resize, 1, "INT")
    L(h_node, 0, resize, 2, "INT")
    L(resize, 0, longedge, 0, "IMAGE")
    L(longedge, 0, pre, 0, "IMAGE")

    # audio
    L(dur_node, 0, trim, 2, "FLOAT")
    L(trim, 0, a_enc, 0, "AUDIO")
    L(vae_a, 0, a_enc, 1, "VAE")
    L(w_node, 0, solid, 0, "INT")
    L(h_node, 0, solid, 1, "INT")
    L(a_enc, 0, a_mask, 0, "LATENT")
    L(solid, 0, a_mask, 1, "MASK")

    # numbers
    L(w_node, 0, m_w, 0, "INT")
    L(h_node, 0, m_h, 0, "INT")
    L(fps_node, 0, m_fps, 0, "INT")
    L(dur_node, 0, m_len, 0, "FLOAT")
    L(fps_node, 0, m_len, 1, "INT")
    L(m_w, 1, empty, 0, "INT")
    L(m_h, 1, empty, 1, "INT")
    L(m_len, 1, empty, 2, "INT")

    # low-res pass
    L(vae_v, 0, i2v_low, 0, "VAE")
    L(pre, 0, i2v_low, 1, "IMAGE")
    L(empty, 0, i2v_low, 2, "LATENT")
    L(t2v, 0, i2v_low, 3, "BOOLEAN")
    L(i2v_low, 0, cat_low, 0, "LATENT")
    L(a_mask, 0, cat_low, 1, "LATENT")
    L(cond, 0, guide_low, 1, "CONDITIONING")
    L(cond, 1, guide_low, 2, "CONDITIONING")
    L(noise_low, 0, samp_low, 0, "NOISE")
    L(guide_low, 0, samp_low, 1, "GUIDER")
    L(ks_low, 0, samp_low, 2, "SAMPLER")
    L(sig_low, 0, samp_low, 3, "SIGMAS")
    L(cat_low, 0, samp_low, 4, "LATENT")
    L(samp_low, 0, sep_low, 0, "LATENT")

    # upscale
    L(sep_low, 0, upscale, 0, "LATENT")
    L(ups_a, 0, upscale, 1, "LATENT_UPSCALE_MODEL")
    L(vae_v, 0, upscale, 2, "VAE")

    # high-res pass
    L(vae_v, 0, i2v_high, 0, "VAE")
    L(pre, 0, i2v_high, 1, "IMAGE")
    L(upscale, 0, i2v_high, 2, "LATENT")
    L(t2v, 0, i2v_high, 3, "BOOLEAN")
    L(i2v_high, 0, cat_high, 0, "LATENT")
    L(sep_low, 1, cat_high, 1, "LATENT")
    L(cond, 0, guide_high, 1, "CONDITIONING")
    L(cond, 1, guide_high, 2, "CONDITIONING")
    L(noise_high, 0, samp_high, 0, "NOISE")
    L(guide_high, 0, samp_high, 1, "GUIDER")
    L(ks_high, 0, samp_high, 2, "SAMPLER")
    L(sig_high, 0, samp_high, 3, "SIGMAS")
    L(cat_high, 0, samp_high, 4, "LATENT")
    L(samp_high, 0, sep_high, 0, "LATENT")

    # finish / decode
    if second_upscale:
        L(sep_high, 0, finish, 0, "LATENT")
        L(ups_b, 0, finish, 1, "LATENT_UPSCALE_MODEL")
        L(vae_v, 0, finish, 2, "VAE")
        L(finish, 0, dec_v, 0, "LATENT")
    else:
        L(sep_high, 0, dec_v, 0, "LATENT")
    L(vae_v, 0, dec_v, 1, "VAE")
    L(sep_high, 1, dec_a, 0, "LATENT")
    L(vae_a, 0, dec_a, 1, "VAE")
    L(dec_v, 0, mkvid, 0, "IMAGE")
    L(dec_a, 0, mkvid, 1, "AUDIO")
    L(m_fps, 0, mkvid, 2, "FLOAT")
    L(mkvid, 0, save, 0, "VIDEO")
    # handoff: blind index, or scored pick, optionally gated
    if frame_check:
        L(dec_v, 0, sel, 0, "IMAGE")
        L(sel, 3, rep_sel, 0, "STRING")
        if gate:
            L(sel, 0, qgate, 0, "IMAGE")
            L(qgate, 2, rep_gate, 0, "STRING")
            seed_frame, seed_slot = qgate, 0
        else:
            seed_frame, seed_slot = sel, 0
        if color_match:
            # after the gate: the gate must judge the raw pick, not a frame
            # already rescued by a grade correction
            L(seed_frame, seed_slot, cmatch, 0, "IMAGE")
            seed_frame, seed_slot = cmatch, 0
    else:
        L(dec_v, 0, handoff, 0, "IMAGE")
        seed_frame, seed_slot = handoff, 0
    L(seed_frame, seed_slot, prev_hand, 0, "IMAGE")
    if save_frames:
        # pre-compression copy: highest-fidelity version of what came in
        L(longedge, 0, save_in, 0, "IMAGE")
        L(seed_frame, seed_slot, save_out, 0, "IMAGE")
        if name_in is not None:
            L(name_in, 0, save_in, 1, "STRING")
            L(name_out, 0, save_out, 1, "STRING")

    # ---- subgraph IO ------------------------------------------------------
    # order here defines widgets_values order on every instance node
    # ComfyUI resolves generic subgraph inputs by NAME (value, value_1, ...),
    # not by array position. If the numbering is not monotonic in array order,
    # widgets and links land on the wrong internal nodes. So the names are
    # assigned here, from the array, and never written by hand.
    #   "*" -> auto-numbered value / value_1 / value_2 / ...
    io_draft = [
        ("input", "IMAGE,MASK", "first_frame", [(resize, 0)]),
    ] + ([
        ("identity_ref", "IMAGE", "identity_ref",
         [(sel, 6)] + ([(qgate, 1)] if gate else [])
         + ([(cmatch, 1)] if color_match else [])),
    ] if frame_check else []) + [
        ("audio", "AUDIO", "audio", [(trim, 0)]),
    ] + ([
        ("*", "STRING", "prompt", [(cast_txt, 0)]),
    ] if simple_prompt else [
        ("*", "STRING", "look", [(look_txt, 0)]),
        ("*", "STRING", "cast", [(cast_txt, 0)]),
        ("*", "STRING", "beat", [(beat_txt, 0)]),
        ("*", "BOOLEAN", "use_ollama", [(use_llm, 0)]),
    ]) + [
        ("*", "INT", "width", [(w_node, 0)]),
        ("*", "INT", "height", [(h_node, 0)]),
        ("*", "INT", "fps", [(fps_node, 0)]),
        ("*", "FLOAT", "duration_sec", [(dur_node, 0)]),
        ("start_index", "FLOAT", "audio_start_sec", [(trim, 1)]),
        ("noise_seed", "INT", "seed", [(noise_low, 0)]),
        ("filename_prefix", "STRING", "save_prefix",
         [(save, 1)] + ([(name_in, 0), (name_out, 0)]
                        if (save_frames and name_in is not None) else [])),
    ]

    io_spec, generic = [], 0
    for name, typ, label, targets in io_draft:
        if name == "*":
            name = "value" if generic == 0 else f"value_{generic}"
            generic += 1
        io_spec.append((name, typ, label, targets))

    _assert_sequential(io_spec)

    inputs_def, widget_order = [], []
    for slot, (name, typ, label, targets) in enumerate(io_spec):
        link_ids = []
        for tnode, tslot in targets:
            link_ids.append(L(IN, slot, tnode, tslot, typ))
        inputs_def.append({
            "id": str(uuid.uuid4()), "name": name, "type": typ,
            "linkIds": link_ids, "localized_name": name, "label": label,
            "pos": [-34.0, 60 + slot * 20],
        })
        # every input except the two pure-socket ones drives a widget
        if name not in ("input", "audio", "identity_ref"):
            widget_order.append((name, label))

    out_spec = [("VIDEO", "VIDEO", mkvid, 0),
                ("IMAGE", "IMAGE", seed_frame, seed_slot)]
    outputs_def = []
    for slot, (name, typ, snode, sslot) in enumerate(out_spec):
        lid = L(snode, sslot, OUT, slot, typ)
        outputs_def.append({
            "id": str(uuid.uuid4()), "name": name, "type": typ, "linkIds": [lid],
            "localized_name": name, "pos": [5600, 60 + slot * 20],
        })

    definition = {
        "id": sub_id,
        "version": 1,
        "state": {"lastGroupId": g._gid, "lastNodeId": g._nid,
                  "lastLinkId": g._lid, "lastRerouteId": 0},
        "revision": 0,
        "config": {},
        "name": "LTX-2.5 Shot  (A/V · image-chained)",
        "inputNode": {"id": IN, "bounding": [-160, 40, 125, 240]},
        "outputNode": {"id": OUT, "bounding": [5600, 40, 120, 80]},
        "inputs": inputs_def,
        "outputs": outputs_def,
        "widgets": [],
        "nodes": g.nodes,
        "groups": g.groups,
        "links": g.links,
        "extra": {"workflowRendererVersion": "Vue-corrected"},
    }
    return definition, io_spec, widget_order


# --------------------------------------------------------------------------
# root graph
# --------------------------------------------------------------------------
def build_root(src: dict, sub_id: str, io_spec, widget_order, *,
               shot_table: list, width: int, height: int, fps: int,
               look_text: str = "",
               ollama_model: str = "llava:13b-v1.6 (Q4_0, 8.0GB)",
               ollama_url: str = "http://localhost:11434") -> dict:
    g = Graph("array")
    src_nodes = {n["id"]: n for n in src["nodes"]}

    def carried(nid, key="widgets_values"):
        return copy.deepcopy(src_nodes[nid].get(key))

    # ---- README -----------------------------------------------------------
    readme = (
        "# LTX-2.5 · Multi-Shot Character-Locked Pipeline\n\n"
        "**Flow (unchanged from your 2.3 build):**\n"
        "`Reference image → Shot 1 → Shot 2 → Shot 3 → Shot 4`\n"
        "Each shot's last decoded frame becomes the next shot's first frame.\n\n"
        "## One definition, many shots\n"
        "All shot nodes share **one** subgraph definition. Open any shot, change the "
        "checkpoint / sampler / sigmas, and **every** shot changes. That is the point — "
        "the old file had four divergent copies.\n\n"
        "## Per-shot knobs (on the shot node itself)\n"
        "`prompt` · `duration_sec` · `audio_start_sec` · `seed` · `save_prefix`\n\n"
        "## Shared knobs (root, wired to every shot)\n"
        "`width` · `height` · `fps` · reference image · audio track\n\n"
        "Seed is deliberately **per-shot** (base, base+1, base+2 …) so the shots don't "
        "repeat the same motion. Set them all equal if you want maximum uniformity.\n\n"
        "## Character-consistency checklist\n"
        "1. **Keep the character description verbatim** in every shot prompt. Change "
        "only the action/camera sentence.\n"
        "2. **`LTXVPreprocess` compression** is inside the shot subgraph. It re-applies "
        "every shot, so degradation compounds down the chain. Lower it (8–10) if faces "
        "soften by shot 3–4.\n"
        "3. **First-frame lock strength** — low-res `LTXVImgToVideoInplace` is set to "
        "0.7 (the LTX-2.5 template value). Raise toward 0.9 for a tighter lock, lower "
        "for more motion freedom.\n"
        "4. **Handoff frame** is `-1` (last frame). If the last frame is motion-blurred, "
        "try `-3` or `-5`.\n"
        "5. Preview each handoff frame before committing to the next shot.\n"
    )
    g.node("MarkdownNote", (-1750, -560), (620, 700), title="README — read me first",
           widgets=[readme], props={})

    models_note = (
        "# LTX-2.5 model files\n\n"
        "Gated repo — accept the licence at `huggingface.co/Lightricks/LTX-2.5` first.\n\n"
        "| File | Folder |\n|---|---|\n"
        f"| `{M_UNET}` | `models/diffusion_models/` |\n"
        f"| `{M_CLIP_MAIN}` | `models/text_encoders/` |\n"
        f"| `{M_CLIP_ENH}` | `models/text_encoders/` |\n"
        f"| `{M_VAE_VIDEO}` | `models/vae/` |\n"
        f"| `{M_VAE_AUDIO}` | `models/vae/` |\n"
        f"| `{M_UPSCALER}` | `models/latent_upscale_models/` |\n\n"
        "**Requires ComfyUI 0.32.0+.**\n\n"
        "⚠️ Your 2.3 distilled LoRA (`ltx-2.3-22b-distilled-lora-384`) is **not** "
        "compatible with the 2.5 transformer and has been removed. The 2.5 distilled "
        "checkpoint already carries the distillation."
    )
    g.node("MarkdownNote", (-1750, 180), (620, 560), title="Model downloads",
           widgets=[models_note], props={})

    # ---- 01 SOURCE --------------------------------------------------------
    x = -1080
    img = g.node("LoadImage", (x, -520), (400, 380), title="Reference / Actor",
                 widgets=carried(922),
                 outputs=[g.sock_out("IMAGE", "IMAGE"), g.sock_out("MASK", "MASK")],
                 props={"cnr_id": "comfy-core", "ver": "0.19.3"})
    aud = g.node("LoadAudio", (x, -110), (400, 136), title="Music / dialogue track",
                 widgets=carried(921), outputs=[g.sock_out("AUDIO", "AUDIO")],
                 props={"cnr_id": "comfy-core", "ver": "0.19.3"})
    cache = g.node("CacheCleaner", (x, 60), (400, 154), title="VRAM cache cleaner",
                   widgets=carried(2556),
                   inputs=[g.sock_in("anything", "*"), g.sock_in("image_pass", "IMAGE"),
                           g.sock_in("model_pass", "MODEL")],
                   outputs=[g.sock_out("anything", "*"), g.sock_out("image_pass", "IMAGE"),
                            g.sock_out("model_pass", "MODEL"),
                            g.sock_out("status", "STRING")],
                   props={"cnr_id": "cache-cleaner", "ver": "1.0.1"})
    g.group("01 · SOURCE", (-1100, -570, 440, 800), C_INPUT)

    # ---- 02 MASTER SETTINGS ----------------------------------------------
    x = -600
    w_n = g.node("PrimitiveInt", (x, -520), (300, 82), title="width",
                 widgets=[width, "fixed"], outputs=[g.sock_out("INT", "INT")])
    h_n = g.node("PrimitiveInt", (x, -400), (300, 82), title="height",
                 widgets=[height, "fixed"], outputs=[g.sock_out("INT", "INT")])
    f_n = g.node("PrimitiveInt", (x, -280), (300, 82), title="fps",
                 widgets=[fps, "fixed"], outputs=[g.sock_out("INT", "INT")])
    scale = g.node("ImageScale", (x, -160), (300, 154), title="Reference → shot resolution",
                   widgets=["lanczos", width, height, "disabled"],
                   inputs=[g.sock_in("image", "IMAGE"),
                           g.sock_in("width", "INT", widget=True),
                           g.sock_in("height", "INT", widget=True)],
                   outputs=[g.sock_out("IMAGE", "IMAGE")],
                   props={"cnr_id": "comfy-core", "ver": "0.20.3"})
    ref_save = g.node("SaveImage", (x, 30), (300, 270),
                      title="Canonical reference → output/",
                      widgets=["castrefs/00_reference"],
                      inputs=[g.sock_in("images", "IMAGE"),
                              g.sock_in("filename_prefix", "STRING", widget=True)],
                      outputs=[],
                      props={"cnr_id": "comfy-core", "ver": "0.19.3"})
    g.group("02 · MASTER SETTINGS", (-620, -570, 340, 900), C_SETTINGS)

    # ---- 03 REFERENCE ANALYSIS (Ollama) -----------------------------------
    x = -240
    sysctx = g.node("PrimitiveStringMultiline", (x, -520), (380, 300),
                    title="System context", widgets=carried(2539),
                    outputs=[g.sock_out("STRING", "STRING")])
    usrp = g.node("PrimitiveStringMultiline", (x, -190), (380, 300),
                  title="Prompt user", widgets=carried(2540),
                  outputs=[g.sock_out("STRING", "STRING")])
    ollama = g.node("OllamaImageDescriber", (x + 420, -520), (420, 560),
                    title="Describe reference", widgets=carried(2525),
                    inputs=[g.sock_in("images", "IMAGE"),
                            g.sock_in("structured_output_format", "STRING"),
                            g.sock_in("system_context", "STRING", widget=True),
                            g.sock_in("prompt", "STRING", widget=True)],
                    outputs=[g.sock_out("result", "STRING")],
                    props={"cnr_id": "comfyui-ollama-describer", "ver": "1.1.0"})
    ollama_prev = g.node("PreviewAny", (x + 420, 80), (420, 230),
                         title="Describer output", widgets=[],
                         inputs=[g.sock_in("source", "*")],
                         outputs=[g.sock_out("STRING", "STRING")])
    look_llm = g.node("OllamaImageDescriber", (x + 420, 330), (420, 420),
                      title="Look reader (reads the reference once)",
                      widgets=[ollama_model, "", ollama_url, 300, 0.15, 40, 0.9,
                               1.1, 42, 2048, 200, 5, LOOK_SYSTEM, LOOK_USER_HINT],
                      inputs=[g.sock_in("images", "IMAGE"),
                              g.sock_in("structured_output_format", "STRING"),
                              g.sock_in("system_context", "STRING", widget=True),
                              g.sock_in("prompt", "STRING", widget=True)],
                      outputs=[g.sock_out("result", "STRING")],
                      props={"cnr_id": "comfyui-ollama-describer", "ver": "1.1.0"})
    look_fixed = g.node("PrimitiveStringMultiline", (x, 330), (380, 240),
                        title="LOOK (hand-written)", widgets=[look_text],
                        outputs=[g.sock_out("STRING", "STRING")])
    look_use_llm = g.node("PrimitiveBoolean", (x, 600), (380, 58),
                          title="Read the look from the image instead?",
                          widgets=[False],
                          outputs=[g.sock_out("BOOLEAN", "BOOLEAN")])
    look_switch = g.node("ComfySwitchNode", (x + 880, 330), (300, 106),
                         title="Look source", widgets=[False],
                         inputs=[g.sock_in("on_false", "STRING"),
                                 g.sock_in("on_true", "STRING"),
                                 g.sock_in("switch", "BOOLEAN", widget=True)],
                         outputs=[g.sock_out("STRING", "STRING")])
    look_prev = g.node("PreviewAny", (x + 880, 480), (300, 270),
                       title="Look pinned to every shot", widgets=[],
                       inputs=[g.sock_in("source", "*")],
                       outputs=[g.sock_out("STRING", "STRING")])
    g.group("03 · REFERENCE ANALYSIS + LOOK LOCK", (-260, -570, 1460, 1360),
            C_PROMPT)

    # ---- 04 PROMPT LIBRARY (preserved, collapsed) -------------------------
    lib_ids = [1247, 1248, 1249, 1250, 1252, 1253, 1254, 1547, 2517, 2548, 3231]
    lib_ids = [i for i in lib_ids if i in src_nodes]
    lx, ly = -1750, 800
    for k, nid in enumerate(lib_ids):
        s = src_nodes[nid]
        g.node("PrimitiveStringMultiline", (lx + (k % 4) * 460, ly + (k // 4) * 90),
               (440, 60), title=f"[preset] {s.get('title') or nid}",
               widgets=s.get("widgets_values"),
               outputs=[g.sock_out("STRING", "STRING")],
               flags={"collapsed": True})
    rows = (len(lib_ids) + 3) // 4
    g.group("04 · PROMPT LIBRARY  (unused presets — kept for reference)",
            (lx - 20, ly - 50, 4 * 460 + 20, rows * 90 + 70), C_LIB)

    # ---- 05.. SHOTS -------------------------------------------------------
    shot_nodes = []
    sx0, sy = 1240, -520
    for i, shot in enumerate(shot_table):
        sx = sx0 + i * 620
        inst_inputs = []
        for name, typ, label, _ in io_spec:
            d = {"label": label, "name": name, "type": typ, "link": None}
            if name not in ("input", "audio", "identity_ref"):
                d["widget"] = {"name": name}
            inst_inputs.append(d)
        values = {"look": shot["look"], "cast": shot["cast"], "beat": shot["beat"],
                  "use_ollama": shot["use_ollama"],
                  "prompt": " ".join(x for x in (shot["look"], shot["cast"],
                                                 shot["beat"]) if x),
                  "width": width, "height": height, "fps": fps,
                  "duration_sec": shot["duration_sec"],
                  "audio_start_sec": shot["audio_start_sec"],
                  "seed": shot["seed"], "save_prefix": shot["save_prefix"]}
        wv = [values[label] for _name, label in widget_order]
        n = g.node(sub_id, (sx, sy), (560, 780), title=shot["title"],
                   widgets=wv, inputs=inst_inputs,
                   outputs=[g.sock_out("VIDEO", "VIDEO"), g.sock_out("IMAGE", "IMAGE")],
                   props={"cnr_id": "comfy-core", "ver": "0.32.0", "previewExposures": []})
        n["properties"].pop("Node name for S&R", None)
        prev = g.node("PreviewImage", (sx, sy + 830), (560, 320),
                      title=f"{shot['title']} → handoff frame", widgets=[],
                      inputs=[g.sock_in("images", "IMAGE")],
                      outputs=[g.sock_out("images", "IMAGE")],
                      props={"cnr_id": "comfy-core", "ver": "0.20.3"})
        g.group(f"{i + 5:02d} · {shot['title']}"
                f"   [{shot['audio_start_sec']:g}s → "
                f"{shot['audio_start_sec'] + shot['duration_sec']:g}s]"
                f"   anchor: {shot['anchor']}",
                (sx - 20, sy - 50, 600, 1230),
                C_LOW if i % 2 == 0 else C_HIGH)
        shot_nodes.append((n, prev))

    # ---- wiring -----------------------------------------------------------
    L = g.link
    slot = {name: k for k, (name, *_rest) in enumerate(io_spec)}
    # label-keyed too: input NAMES shift between build modes, labels do not
    slot.update({lbl: k for k, (_n, _t, lbl, *_r) in enumerate(io_spec)})
    L(img, 0, cache, 1, "IMAGE")
    L(img, 0, ollama, 0, "IMAGE")
    L(sysctx, 0, ollama, 2, "STRING")
    L(usrp, 0, ollama, 3, "STRING")
    L(ollama, 0, ollama_prev, 0, "STRING")
    L(scale, 0, look_llm, 0, "IMAGE")
    L(look_fixed, 0, look_switch, 0, "STRING")
    L(look_llm, 0, look_switch, 1, "STRING")
    L(look_use_llm, 0, look_switch, 2, "BOOLEAN")
    L(look_switch, 0, look_prev, 0, "STRING")
    L(img, 0, scale, 0, "IMAGE")
    L(w_n, 0, scale, 1, "INT")
    L(h_n, 0, scale, 2, "INT")
    L(scale, 0, ref_save, 0, "IMAGE")

    for i, (n, prev) in enumerate(shot_nodes):
        # anchor "reference" restarts from the clean original and wipes all
        # accumulated drift; "previous" continues the chain from the last frame
        if i == 0 or shot_table[i]["anchor"] == "reference":
            L(scale, 0, n, slot["first_frame"], "IMAGE")
        else:
            L(shot_nodes[i - 1][0], 1, n, slot["first_frame"], "IMAGE")
        if "look" in slot:
            L(look_switch, 0, n, slot["look"], "STRING")
        if "identity_ref" in slot:
            L(scale, 0, n, slot["identity_ref"], "IMAGE")
        L(aud, 0, n, slot["audio"], "AUDIO")
        L(w_n, 0, n, slot["width"], "INT")
        L(h_n, 0, n, slot["height"], "INT")
        L(f_n, 0, n, slot["fps"], "INT")
        L(n, 1, prev, 0, "IMAGE")

    return g


# --------------------------------------------------------------------------
# prompt assembly
# --------------------------------------------------------------------------
# Used only when no per-shot `action` is supplied. Deliberately DISTINCT:
# cloning one prompt across shots makes every shot re-run the same motion arc
# from a slightly different start frame, so the sequence stalls and loops.
DEFAULT_BEATS = [
    "The subject settles into the opening pose and begins moving with the beat",
    "The subject turns toward camera and performs the vocal line",
    "The subject opens up into fuller movement as the energy lifts",
    "The subject eases back down, weight shifting slowly as the section resolves",
    "The subject steps to one side, head tracking the camera",
    "The subject holds still for a beat, then picks the rhythm back up",
]
DEFAULT_CAMERAS = [
    "slow push-in", "handheld drift", "slow orbit to the right",
    "static", "gentle crane up", "slow pull-back",
]

HUMAN_KINDS = {"man", "woman", "boy", "girl", "person", "human", "child",
               "guy", "lady", "male", "female", "singer", "dancer", "drummer",
               "guitarist", "musician", "performer"}

# Hybrid failure modes, phrased for the NEGATIVE branch where negation works.
BASE_HYBRID_NEG = [
    "hybrid creature", "chimera", "merged anatomy", "fused subjects",
    "morphing between characters", "one character turning into another",
    "swapped heads", "extra limbs", "anthropomorphic",
]


SCENE_HOLDER = {}
LOOK_HOLDER = {}


def build_look_block(look) -> str:
    """One sentence pinning medium, grade, lighting and lens.

    Placed FIRST in every prompt: text encoders weight early tokens heavily,
    and the medium is the thing that must never drift between shots.
    """
    if not look:
        return ""
    if isinstance(look, str):
        return look.strip()
    order = ["medium", "grade", "lighting", "lens", "texture"]
    parts = [str(look[k]).strip().rstrip(".") for k in order
             if look.get(k)]
    for k, v in look.items():
        if k not in order and v:
            parts.append(str(v).strip().rstrip("."))
    if not parts:
        return ""
    body = ", ".join(parts)
    return (f"{body[0].upper()}{body[1:]}. "
            f"Every shot keeps this same medium, palette and lighting.")


def build_cast_block(cast: list) -> tuple:
    """Return (positive_prefix, extra_negative_terms).

    Each character gets ONE closed clause: placement, look, anatomy, and a
    POSITIVE persistence assertion ("stays entirely a cat"). Attributes are
    never pooled into a shared list - pooling is what produces cat-head-on-
    human-body. The kind noun is repeated rather than pronouned, because the
    repetition is what binds the attributes to the right subject.
    """
    if not cast:
        return "", []

    clauses = []
    scene = (SCENE_HOLDER.get("scene") or "").strip().rstrip(".")
    if scene:
        clauses.append(f"{scene[0].upper()}{scene[1:]}.")
    for c in cast:
        kind = (c.get("kind") or "subject").strip()
        look = (c.get("look") or c.get("description") or "").strip().rstrip(".")
        anatomy = (c.get("anatomy") or "").strip().rstrip(".")
        where = (c.get("where") or c.get("position") or "").strip().rstrip(".")

        parts = []
        if where:
            parts.append(f"{where[0].upper()}{where[1:]}, {look}")
        else:
            parts.append(f"{look[0].upper()}{look[1:]}" if look else f"A {kind}")
        if anatomy:
            parts.append(f"The {kind} has {anatomy}")
        parts.append(f"The {kind} stays entirely a {kind} for the whole shot")
        clauses.append(". ".join(p for p in parts if p) + ".")

    kinds = [(c.get("kind") or "subject").strip() for c in cast]
    if len(kinds) > 1:
        listed = ", ".join(f"the {k}" for k in kinds[:-1]) + f" and the {kinds[-1]}"
        clauses.append(
            f"{listed[0].upper()}{listed[1:]} are separate beings that keep "
            f"their own separate bodies throughout the shot."
        )

    # cross-kind hybrids, generated from the actual cast
    neg = list(BASE_HYBRID_NEG)
    humans = [k for k in kinds if k.lower() in HUMAN_KINDS]
    animals = [k for k in kinds if k.lower() not in HUMAN_KINDS]
    for a in animals:
        for h in humans:
            neg += [f"{a} head on a {h} body", f"{h} head on a {a} body",
                    f"{a}-{h} hybrid", f"{h} with {a} features",
                    f"{a} with {h} features"]
    if animals and humans:
        neg += ["fur on human skin", "human face on an animal",
                "animal ears on a person"]
    seen, dedup = set(), []
    for t in neg:
        if t.lower() not in seen:
            seen.add(t.lower())
            dedup.append(t)
    return " ".join(clauses), dedup


def _anchor_for(i: int, mode: str | None) -> str:
    """Shot 1 always starts from the reference. After that:
      previous  - pure chain (max continuity, max drift)
      reference - star topology (zero drift, no continuity)
      alternate - re-anchor every other shot (drift capped at one hop)
    """
    if i == 0:
        return "reference"
    if mode == "reference":
        return "reference"
    if mode == "alternate":
        return "reference" if i % 2 == 0 else "previous"
    return "previous"


def compose_beat(cfg: dict, shot: dict) -> str:
    """The per-shot ACTION only - what happens and how the camera moves.

    Kept separate from the cast block so the identity text is never rewritten,
    and so Ollama can replace this half at runtime without touching identity.
    """
    bits = []
    body = (shot.get("action") or shot.get("prompt") or "").strip().rstrip(".")
    if body:
        bits.append(f"{body[0].upper()}{body[1:]}.")
    cam = (shot.get("camera") or cfg.get("camera") or "").strip().rstrip(".")
    if cam:
        bits.append(f"Camera: {cam}.")
    audio = (shot.get("audio") or cfg.get("audio_description") or "").strip().rstrip(".")
    if audio:
        bits.append(f"{audio[0].upper()}{audio[1:]}.")
    return " ".join(bits)


# --------------------------------------------------------------------------
# shot table
# --------------------------------------------------------------------------
def resolve_shots(cfg: dict, fallback_prompt: str, fallback_prefix: str) -> tuple:
    """Turn a compact config into one fully-specified dict per shot.

    Every field is optional per shot and falls back to `defaults`.
    `audio_start_sec` is CUMULATIVE unless a shot pins it explicitly, so
    variable-length shots still line up end-to-end on the source track.
    """
    d = cfg.get("defaults", {})
    SCENE_HOLDER["scene"] = cfg.get("scene", "")
    LOOK_HOLDER["text"] = build_look_block(cfg.get("look"))
    cast_block, hybrid_neg = build_cast_block(cfg.get("cast") or [])
    lock = (cfg.get("character_lock") or "").rstrip()
    joiner = d.get("lock_joiner", "\n\n")
    prefix = d.get("save_prefix", fallback_prefix)

    raw = cfg.get("shots")
    if isinstance(raw, int):                    # "shots": 6
        raw = [{} for _ in range(raw)]
    elif not raw:
        raw = [{}]

    cursor = float(d.get("audio_start", 0.0))
    table, used_defaults = [], False
    for i, s in enumerate(raw):
        dur = float(s.get("duration_sec", d.get("duration_sec", 20.0)))
        start = float(s["audio_start_sec"]) if "audio_start_sec" in s else cursor
        cursor = start + dur

        s = dict(s)
        if not s.get("action") and not s.get("prompt"):
            s["action"] = DEFAULT_BEATS[i % len(DEFAULT_BEATS)]
            s.setdefault("camera", DEFAULT_CAMERAS[i % len(DEFAULT_CAMERAS)])
            used_defaults = True

        beat = compose_beat(cfg, s)
        identity = cast_block or lock or ""
        if not identity:
            identity, beat = fallback_prompt, ""
        body = (identity + " " + beat).strip()

        table.append({
            "title": s.get("title", f"SHOT {i + 1}"),
            "prompt": body,
            "look": LOOK_HOLDER.get("text", ""),
            "cast": identity,
            "beat": beat,
            "use_ollama": bool(s.get("use_ollama", d.get("use_ollama", False))),
            "anchor": (s.get("anchor") or _anchor_for(i, d.get("anchor"))),
            "duration_sec": dur,
            "audio_start_sec": start,
            "seed": int(s.get("seed", int(d.get("seed", 0)) + i)),
            "save_prefix": s.get("save_prefix", f"{prefix}shot{i + 1:02d}"),
            "action": (s.get("action") or s.get("prompt") or "")[:58],
        })
    return table, hybrid_neg, used_defaults


STARTER_CONFIG = {
    "_comment": "Give each character ONE closed block in 'cast'. Never pool "
                "attributes across characters - that is what produces "
                "cat-head-on-human-body. Keep 'action' to 1-2 beats; for "
                "image-to-video the scene already exists in frame 0.",
    "defaults": {
        "duration_sec": 20,
        "audio_start": 40,
        "seed": 0,
        "save_prefix": "newVideoParts/sam_new_",
    },
    "look": {
        "medium": "live-action footage shot on a full-frame cinema camera",
        "grade": "warm golden-hour palette with amber highlights, soft neutral "
                 "shadows and gentle film contrast",
        "lighting": "low sunlight from a window on the right, soft falloff",
        "lens": "35mm lens, shallow depth of field",
        "texture": "fine natural film grain",
    },
    "scene": "a sunlit living room with warm late-afternoon light",
    "camera": "slow push-in, handheld drift",
    "cast": [
        {
            "id": "milo",
            "kind": "cat",
            "look": "a small orange tabby cat with green eyes and short fur",
            "anatomy": "four legs, a full feline body, a long tail, whiskers "
                       "and pointed ears",
            "where": "on the left, on the wooden floor",
        },
        {
            "id": "sara",
            "kind": "woman",
            "look": "a woman in her thirties with long dark hair, wearing a "
                    "red jacket",
            "anatomy": "a human face, human hands and human proportions",
            "where": "on the right, seated on the sofa",
        },
    ],
    "shots": [
        {"title": "SHOT 1", "action": "The cat lifts its head and blinks slowly "
                                      "while the woman turns toward it"},
        {"title": "SHOT 2", "action": "The woman reaches down and the cat steps "
                                      "toward her hand",
         "camera": "slow orbit to the right"},
        {"title": "SHOT 3", "action": "The cat settles beside her and she smiles",
         "camera": "static"},
        {"title": "SHOT 4", "action": "The woman leans back as the cat curls up",
         "camera": "gentle crane up"},
    ],
}


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mnt/user-data/uploads/"
                                     "Clean_Multi_Video_Clips_together_Sam_Ayoub__3_.json")
    ap.add_argument("--out", default="/mnt/user-data/outputs/"
                                     "LTX25_Multi_Shot_Character_Lock_Sam_Ayoub.json")
    ap.add_argument("--config", help="shot table JSON (see --emit-config)")
    ap.add_argument("--emit-config", metavar="PATH",
                    help="write a starter shot-table config and exit")
    ap.add_argument("--shots", type=int,
                    help="number of shots (ignored when --config supplies a list)")
    ap.add_argument("--clip-seconds", type=float, default=20.0)
    ap.add_argument("--audio-start", type=float, default=40.0)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img-compression", type=int, default=10)
    ap.add_argument("--strength-low", type=float, default=0.7)
    ap.add_argument("--no-finish-upscale", action="store_true")
    ap.add_argument("--cfg", type=float, nargs=2, default=[1, 1],
                    metavar=("A", "B"),
                    help="LTXVDualCFGGuider values. 1 1 is the distilled "
                         "default; the negative prompt has little effect "
                         "there. Raise only on the dev transformer.")
    ap.add_argument("--handoff-index", default="-6",
                    help="frame pulled from each clip to seed the next shot. "
                         "-1 is the literal last frame and is usually the "
                         "most degraded; -6 backs off the tail.")
    ap.add_argument("--compat", action="store_true",
                    help="build using ONLY node types your original workflow "
                         "already used. No custom pack, no StringConcatenate, "
                         "no runtime prompt assembly. Guaranteed to load.")
    ap.add_argument("--no-color-match", action="store_true",
                    help="skip pinning each seed frame's palette to the reference")
    ap.add_argument("--color-strength", type=float, default=0.85,
                    help="how hard to pin the palette (0-1)")
    ap.add_argument("--no-frame-check", action="store_true",
                    help="use the blind GetImagesFromBatchIndexed pick instead "
                         "of the scored selector")
    ap.add_argument("--gate", action="store_true",
                    help="add HandoffQualityGate: fall back to the reference "
                         "when the chosen frame still drifts too far")
    ap.add_argument("--window", type=int, default=12,
                    help="how many tail frames the selector considers")
    ap.add_argument("--recency", type=float, default=0.6,
                    help="bias toward the final frame. Higher protects motion "
                         "continuity, lower prioritises image quality.")
    ap.add_argument("--no-save-frames", action="store_true",
                    help="skip writing the reusable identity frames")
    ap.add_argument("--ollama", action="store_true",
                    help="default the per-shot director switch to ON")
    ap.add_argument("--ollama-model", default="llava:13b-v1.6 (Q4_0, 8.0GB)")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--anchor", choices=["previous", "reference", "alternate"],
                    default="previous",
                    help="default first-frame source for shots 2..N")
    ap.add_argument("--print-prompts", action="store_true",
                    help="print the composed prompts and exit")
    args = ap.parse_args()

    if args.compat:
        args.no_frame_check = True
        args.no_color_match = True
        args.ollama = False

    if args.emit_config:
        p = Path(args.emit_config)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(STARTER_CONFIG, indent=2, ensure_ascii=False))
        print(f"wrote starter config -> {p}")
        return

    src = json.loads(Path(args.src).read_text())
    src_nodes = {n["id"]: n for n in src["nodes"]}
    src_subs = {s["id"]: s for s in src["definitions"]["subgraphs"]}
    ltx = src_subs["950249ca-aac2-4f72-9468-1a7e2a4c5391"]
    ltx_nodes = {n["id"]: n for n in ltx["nodes"]}

    negative = ltx_nodes[314]["widgets_values"][0]
    shot_prompt = src_nodes[920]["widgets_values"][0]
    save_prefix = src_nodes[1246]["widgets_values"][0]

    cfg = json.loads(Path(args.config).read_text()) if args.config else {}
    cfg.setdefault("defaults", {})
    cfg["defaults"].setdefault("duration_sec", args.clip_seconds)
    cfg["defaults"].setdefault("audio_start", args.audio_start)
    cfg["defaults"].setdefault("seed", args.seed)
    cfg["defaults"].setdefault("save_prefix", save_prefix)
    if args.ollama:
        cfg["defaults"].setdefault("use_ollama", True)
    if args.anchor != "previous":
        cfg["defaults"].setdefault("anchor", args.anchor)
    if args.shots is not None and not isinstance(cfg.get("shots"), list):
        cfg["shots"] = args.shots
    cfg.setdefault("shots", 4)

    shot_table, hybrid_neg, used_defaults = resolve_shots(
        cfg, shot_prompt, save_prefix)
    look_text = LOOK_HOLDER.get("text", "")
    extra_neg = list(hybrid_neg) + medium_negatives(look_text)
    seen, dedup = set(), []
    for t in extra_neg:
        if t.lower() not in seen:
            seen.add(t.lower()); dedup.append(t)
    if dedup:
        negative = negative.rstrip().rstrip(',') + ',\n\n' + ', '.join(dedup)

    if args.print_prompts:
        for sh in shot_table:
            print(f"--- {sh['title']}  (anchor: {sh['anchor']}, "
                  f"ollama: {sh['use_ollama']}) ---")
            print("[LOOK — pinned first, wired from root]")
            print(sh['look'] or "(none)")
            print()
            print("[CAST — fixed, never regenerated]")
            print(sh['cast'])
            print()
            print("[BEAT — replaced by Ollama when the switch is on]")
            print(sh['beat'] or "(none)")
            print()
        print('--- NEGATIVE (shared) ---')
        print(negative)
        return

    sub_id = str(uuid.uuid4())
    definition, io_spec, widget_order = build_shot_subgraph(
        sub_id, negative, args.img_compression, args.strength_low,
        second_upscale=not args.no_finish_upscale,
        cfg_a=args.cfg[0], cfg_b=args.cfg[1],
        handoff_index=args.handoff_index,
        ollama_model=args.ollama_model, ollama_url=args.ollama_url,
        use_ollama=args.ollama, save_frames=not args.no_save_frames,
        frame_check=not args.no_frame_check, gate=args.gate,
        window=args.window, recency=args.recency,
        color_match=not args.no_color_match,
        color_strength=args.color_strength,
        simple_prompt=args.compat)

    root = build_root(src, sub_id, io_spec, widget_order,
                      shot_table=shot_table, width=args.width,
                      height=args.height, fps=args.fps,
                      look_text=look_text,
                      ollama_model=args.ollama_model,
                      ollama_url=args.ollama_url)

    wf = {
        "id": str(uuid.uuid4()),
        "revision": 0,
        "last_node_id": root._nid,
        "last_link_id": root._lid,
        "nodes": root.nodes,
        "links": root.links,
        "groups": root.groups,
        "definitions": {"subgraphs": [definition]},
        "config": {},
        "extra": {
            "ds": {"scale": 0.35, "offset": [1900, 900]},
            "frontendVersion": "1.48.7",
            "VHS_latentpreview": False,
            "VHS_latentpreviewrate": 0,
            "VHS_MetadataImage": True,
            "VHS_KeepIntermediate": True,
        },
        "version": 0.4,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(wf, indent=2, ensure_ascii=False))
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB)")
    print(f"  root nodes : {len(root.nodes)}   root links : {len(root.links)}")
    print(f"  subgraphs  : 1   definition nodes : {len(definition['nodes'])}")
    print()
    print(f"  {len(shot_table)} shots  ·  chain: "
          + " → ".join(s["title"] for s in shot_table))
    print(f"  {'shot':<10} {'audio window':<16} {'dur':>5} {'anchor':<9} action")
    for s in shot_table:
        win = f"{s['audio_start_sec']:g}s → {s['audio_start_sec']+s['duration_sec']:g}s"
        print(f"  {s['title']:<10} {win:<16} {s['duration_sec']:>4g}s "
              f"{s['anchor']:<9} {s['action']}")

    # duplicate-prompt guard: identical shots re-run the same motion arc
    from collections import Counter
    dupes = [h for h, c in Counter(x["prompt"] for x in shot_table).items()
             if c > 1]
    if dupes:
        names = [x["title"] for x in shot_table if x["prompt"] in dupes]
        print()
        print(f"  !! {len(names)} shots share an identical prompt: "
              f"{', '.join(names)}")
        print("     Each shot starts from the previous shot's last frame. Same")
        print("     instruction = same motion arc replayed, so the sequence stalls.")
        print("     Give each shot its own 'action' in the config.")
    if used_defaults:
        print()
        print("  note: generic placeholder actions were used for shots with no")
        print("        'action' set. Run --emit-config and write your own.")
    print()
    if args.no_frame_check:
        print(f"  handoff: blind pick at index {args.handoff_index}")
    else:
        print(f"  handoff: scored pick over the last {args.window} frames "
              f"(recency {args.recency})")
        print(f"           gate: {'on — falls back to reference on drift' if args.gate else 'off'}")
        print("           requires custom_nodes/reallexi_handoff")
    if args.compat:
        print("  COMPAT MODE: no custom nodes, no StringConcatenate.")
        print("               prompt is pre-composed; the shot director is off.")
    print(f"  look lock: {'palette pinned at ' + str(args.color_strength) if not args.no_color_match else 'colour match OFF'}")
    if look_text:
        print(f"             {look_text[:66]}…")
    print(f"             medium negatives: "
          f"{', '.join(medium_negatives(look_text)[:4])}…")
    if not args.no_save_frames:
        print("  cast library -> ComfyUI/output/")
        print(f"    {'':4}castrefs/00_reference_00001_.png"
              f"        (canonical anchor)")
        for s in shot_table[:2]:
            print(f"    {'':4}{s['save_prefix']}_first_00001_.png")
            print(f"    {'':4}{s['save_prefix']}_handoff_00001_.png")
        if len(shot_table) > 2:
            print(f"    {'':4}… and the same pair for the remaining "
                  f"{len(shot_table) - 2} shots")
    total = sum(s["duration_sec"] for s in shot_table)
    print(f"  total {total:g}s  ·  ~{int(total*args.fps)} frames "
          f"({args.fps} fps)")


if __name__ == "__main__":
    main()
