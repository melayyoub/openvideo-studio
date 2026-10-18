"""Stage 1 — the story engine.

You give it three things: subject, mood, age. Ollama writes everything else — title,
theme, the full cast bible (appearance down to eye colour, wardrobe, palette, motion
signature and a complete voice profile), the beat sheet, and then every scene with
action, camera, dialogue, ambience and explicit continuity notes.

Two passes on purpose:
  pass 1  cast + world   — high temperature, this is where invention belongs
  pass 2  scenes         — lower temperature, constrained by the cast that already exists

The cast ids minted in pass 1 are the only ids pass 2 is allowed to reference, and the
validator enforces it. That is the first link in the identity chain.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .config import Config
from .ollama_client import Ollama
from .schema import (
    SCENE_LIST_SCHEMA,
    STORY_SKELETON_SCHEMA,
    make_scene_validator,
    validate_skeleton,
)
from .util import banner, log, slugify, stable_seed, trigger_token, write_json

SKELETON_SYSTEM = """You are a film development executive and character designer.

You are given only a subject, a mood and a target audience age. From that, you invent a
complete, original short film and its cast.

HARD RULES
- Output ONLY JSON that matches the requested schema. No markdown, no commentary.
- Every character must be described so precisely that two different artists drawing from
  your description alone would produce the same character. Vague is failure.
- 'identity_line' is the single most important field. One dense sentence, at least 12
  words, naming species, apparent age, hair, eyes, skin, and the exact wardrobe with
  colours. It will be pasted verbatim into every shot of the film and must never need
  editing.
- 'never' lists things that must NEVER change or appear for that character across the
  whole film (e.g. "never changes hair colour", "never appears as an adult",
  "never removes the green scarf").
- 'voice' must be castable: describe how the character SOUNDS, not what they say.
  Include a signature_sound, a non-verbal noise that belongs to them alone.
- 'id' must be lowercase snake_case, short, derived from the name.
- Keep the cast small: 2 to 4 characters. Small casts hold identity; large casts drift.
- Content must be appropriate for the stated audience age. No violence, no peril that
  is not resolved, nothing frightening for young ages.
"""

SCENES_SYSTEM = """You are a director breaking a story into shots for an AI video model.

You are given a story bible with a FIXED cast. You may not invent new characters, rename
anyone, or alter any appearance.

HARD RULES
- Output ONLY JSON matching the schema. No markdown, no commentary.
- 'characters' may contain ONLY the exact character ids given to you.
- NEVER describe what a character looks like. Appearance is handled elsewhere and is
  locked. Describe only what they DO.
- 'action' is the shot's motion, at least 40 words, concrete and filmable, covering the
  full shot duration from start to end. Continuous motion, no cuts inside a shot.
- The very first frame of each shot is a still that already exists. Write the action so
  that motion BEGINS after that first frame and flows out of that exact pose.
- 'motion_beats' are 3 to 5 short ordered beats of escalating energy for that shot.
- 'dialogue' may be empty, but when present the speaker must be a cast id and 'line'
  must be short enough to speak inside the shot duration. 'delivery' says how it sounds.
- 'ambience' lists concrete diegetic sounds present in the shot.
- 'continuity_from_previous' states what physically carries over from the previous shot:
  positions, held objects, lighting, weather, emotional state.
- Consecutive shots must connect. The end pose of one shot is the start pose of the next.
"""


class StoryEngine:
    def __init__(self, cfg: Config, ollama: Ollama):
        self.cfg = cfg
        self.ollama = ollama

    # ------------------------------------------------------------------ #

    def _skeleton(self, subject: str, mood: str, age: int, extra: str, language: str) -> Dict[str, Any]:
        user = f"""SUBJECT: {subject}
MOOD: {mood}
AUDIENCE AGE: {age}
LANGUAGE FOR ALL WRITTEN TEXT: {language}
{('ADDITIONAL DIRECTION: ' + extra) if extra else ''}

Invent the film. Return JSON with:
  title, logline, theme, visual_style, colour_palette (4-6 colours),
  world (one paragraph describing the setting),
  age_rating_note (why this is safe for a {age} year old),
  beats (5-9 short story beats, in order, the spine of the film),
  cast (2-4 characters, each fully specified).

For each character supply exactly these fields:
  id, name, role, species, age_appearance, gender_presentation, build, skin, hair, eyes,
  face_marks, wardrobe, signature_prop, palette (2-5 colours), identity_line, never (2-5
  rules), motion_signature (how this character moves, one sentence),
  voice {{timbre, pitch, pace, accent, quirk, default_emotion, signature_sound}}.
"""
        return self.ollama.chat_json(
            system=SKELETON_SYSTEM,
            user=user,
            schema=STORY_SKELETON_SCHEMA,
            validator=validate_skeleton,
            temperature=float(self.cfg.get("ollama.temperature_story", 0.85)),
            label="story-skeleton",
        )

    def _scenes(self, skeleton: Dict[str, Any], count: int, duration: int, age: int) -> List[Dict[str, Any]]:
        roster = [
            {
                "id": c["id"],
                "name": c["name"],
                "role": c["role"],
                "motion_signature": c["motion_signature"],
                "voice_pace": c["voice"]["pace"],
            }
            for c in skeleton["cast"]
        ]
        user = f"""STORY BIBLE
title: {skeleton['title']}
logline: {skeleton['logline']}
theme: {skeleton['theme']}
world: {skeleton['world']}
visual style: {skeleton['visual_style']}
beats: {json.dumps(skeleton['beats'], ensure_ascii=False)}

FIXED CAST (these ids are the ONLY ids you may use):
{json.dumps(roster, indent=2, ensure_ascii=False)}

Break this into EXACTLY {count} shots, each {duration} seconds long, for an audience of
age {age}. Shot 1 opens the film, shot {count} resolves it. Distribute the beats across
the shots so the story completes.

Return JSON: {{"scenes": [ ... ]}} where each scene has:
  index (1-based), title, beat, location, time_of_day, weather, characters (cast ids),
  keyframe_prompt (one sentence describing the STILL first frame: who is where, doing
  what, framing and light — no appearance details, no camera movement),
  action (>= 40 words of continuous motion), motion_beats (3-5),
  camera (one sentence of camera movement that starts static then moves),
  dialogue (0-2 lines, speaker must be a cast id, short enough for {duration}s),
  ambience (2-5 concrete sounds), music (mood cue),
  continuity_from_previous (for scene 1 write "opening shot"),
  transition_out (how this shot hands off to the next).
"""
        obj = self.ollama.chat_json(
            system=SCENES_SYSTEM,
            user=user,
            schema=SCENE_LIST_SCHEMA,
            validator=make_scene_validator([c["id"] for c in skeleton["cast"]], count),
            temperature=float(self.cfg.get("ollama.temperature_director", 0.35)),
            label="story-scenes",
        )
        scenes = obj["scenes"]
        for i, sc in enumerate(scenes, start=1):
            sc["index"] = i
            sc["id"] = f"s{i:02d}_{slugify(sc.get('title', 'scene'), 24)}"
            sc["duration_seconds"] = duration
        return scenes

    # ------------------------------------------------------------------ #

    def build(self, force: bool = False) -> Dict[str, Any]:
        out_path = self.cfg.story_bible_path
        if out_path.exists() and not force:
            log(f"story bible already exists, reusing: {out_path}")
            return json.loads(out_path.read_text(encoding="utf-8"))

        banner("STAGE 1 — story engine (Ollama)")
        subject = self.cfg.require("story_input.subject")
        mood = self.cfg.get("story_input.mood", "warm and wondrous")
        age = int(self.cfg.get("story_input.age", 8))
        extra = self.cfg.get("story_input.extra", "")
        language = self.cfg.get("story_input.language", "en")
        count = int(self.cfg.get("project.scene_count", 6))
        duration = int(self.cfg.get("project.scene_duration", 10))
        base_seed = int(self.cfg.get("project.seed", 0))

        log(f"subject: {subject}")
        log(f"mood: {mood} | age: {age} | scenes: {count} x {duration}s")

        skeleton = self._skeleton(subject, mood, age, extra, language)
        log(f"title: {skeleton['title']}")
        log(f"cast: {', '.join(c['name'] + ' (' + c['id'] + ')' for c in skeleton['cast'])}")

        # Mint the deterministic identity handles every later stage depends on.
        for ch in skeleton["cast"]:
            ch["id"] = slugify(ch["id"], 24)
            ch["trigger_token"] = trigger_token(ch["id"])
            ch["lora_name"] = f"castforge_{slugify(self.cfg.get('project.name','film'),24)}_{ch['id']}"
            ch["base_seed"] = stable_seed("character", ch["id"], base=base_seed)
            ch["audio_seed"] = stable_seed("voice", ch["id"], base=base_seed)

        scenes = self._scenes(skeleton, count, duration, age)
        log(f"scenes: {len(scenes)} written")

        bible: Dict[str, Any] = {
            "schema_version": 1,
            "project": self.cfg.get("project.name"),
            "input": {"subject": subject, "mood": mood, "age": age, "extra": extra},
            "seed": base_seed,
            **skeleton,
            "scenes": scenes,
            "continuity_rules": [
                "Character appearance is defined once in the cast bible and never re-described.",
                "Every shot receives the identity block verbatim; the director model may not edit it.",
                "Every shot receives the voice block verbatim for each character present.",
                "The end pose of a shot is the start pose of the next shot.",
                "No character may appear who is not listed in that shot's characters array.",
            ],
        }
        write_json(out_path, bible)
        log(f"wrote {out_path}")
        return bible

    # ------------------------------------------------------------------ #

    def direct_scene(
        self,
        bible: Dict[str, Any],
        scene: Dict[str, Any],
        keyframe_path: Optional[str],
        previous_summary: str,
    ) -> str:
        """Vision pass: look at the actual keyframe and write ONLY the motion paragraph.

        This is the in-graph Ollama director lifted out into Python, where its output can
        be validated before it ever reaches the sampler. It is deliberately forbidden from
        describing appearance — that text is injected verbatim by prompt_composer.
        """
        system = (
            "You are a motion director for a video model. You write ONE paragraph of "
            "continuous physical motion and nothing else.\n"
            "RULES:\n"
            "- Output exactly one paragraph. No lists, no headings, no preamble, no quotes.\n"
            "- NEVER describe appearance, clothing, colours, species, age or face. Those are "
            "already locked and describing them again causes drift.\n"
            "- The first frame is fixed and must not change: no motion at frame 0. All motion "
            "starts after frame 0 and flows out of the exact pose you can see.\n"
            "- Give continuous, visible, escalating motion: head, shoulders, torso, arms, and "
            "camera. Never write 'minimal', 'subtle only' or 'static'.\n"
            "- Include mouth movement consistent with any dialogue so lips can sync.\n"
            "- One continuous shot. No cuts, no new characters, no text or logos."
        )
        user = f"""SHOT {scene['index']}: {scene['title']}
LOCATION: {scene['location']}, {scene['time_of_day']}, {scene['weather']}
CONTINUITY FROM PREVIOUS SHOT: {previous_summary or scene.get('continuity_from_previous', 'opening shot')}
INTENDED ACTION: {scene['action']}
MOTION BEATS: {' -> '.join(scene.get('motion_beats', []))}
CAMERA: {scene['camera']}
DIALOGUE: {json.dumps(scene.get('dialogue', []), ensure_ascii=False)}
DURATION: {scene.get('duration_seconds', 10)} seconds

Write the single motion paragraph for this shot, starting from the pose visible in the image."""

        if keyframe_path:
            text = self.ollama.describe_image(
                keyframe_path, system=system, user=user,
                temperature=float(self.cfg.get("ollama.temperature_director", 0.35)),
            )
        else:
            text = self.ollama.chat_text(
                system, user, temperature=float(self.cfg.get("ollama.temperature_director", 0.35))
            )

        words = len(text.split())
        if words < 35:
            log(f"scene {scene['index']}: director paragraph thin ({words}w), falling back to bible action", "warn")
            from .util import one_paragraph
            text = one_paragraph(f"{scene['action']} {' '.join(scene.get('motion_beats', []))} {scene['camera']}")
        return text
