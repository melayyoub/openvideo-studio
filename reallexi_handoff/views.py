"""The canonical 10-view cast coverage sheet.

Ten views chosen so that together they cover a full 360 degrees of the body plus the
close-range facial detail a video model needs for lip-sync and expression. This is the
minimum set that makes a character LoRA hold identity from any camera angle in a movie.

    V01 front            0 deg    full body   silhouette + wardrobe front
    V02 three_quarter_l  45 deg   full body   the workhorse cinematic angle
    V03 profile_l        90 deg   full body   nose/chin/jaw line
    V04 back_quarter_l   135 deg  full body   rear costume + hair back
    V05 back             180 deg  full body   full rear coverage
    V06 three_quarter_r  315 deg  full body   mirror-side asymmetry
    V07 profile_r        270 deg  full body   mirror-side jaw line
    V08 face_front       0 deg    head+shldr  identity anchor, highest weight
    V09 face_expression  30 deg   head+shldr  emotion + open mouth for lip-sync
    V10 action_detail    varies   full body   motion pose + hands/props

``repeats`` controls how many times a view is duplicated in the LoRA dataset. Face views
carry more weight because facial drift is what audiences actually notice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class ViewSpec:
    id: str
    name: str
    angle: str
    framing: str
    pose: str
    purpose: str
    repeats: int = 1
    tags: List[str] = field(default_factory=list)

    def edit_instruction(self, identity_line: str, background: str) -> str:
        """Instruction handed to the image-edit model. Identity is stated, never invented."""
        return (
            f"Keep the exact same character from the reference image with zero changes to "
            f"face, body type, species, skin tone, hair, eye colour, wardrobe, colours and props. "
            f"{identity_line} "
            f"Re-render this identical character as a character-sheet view: {self.framing}, "
            f"{self.angle}, {self.pose}. "
            f"Background: {background}. Even neutral studio lighting, no harsh shadows, "
            f"sharp focus, full subject inside frame, no cropping of head or feet, "
            f"no text, no watermark, no logo, no border, single character only."
        )

    def caption(self, token: str, name: str, identity_line: str) -> str:
        """LoRA training caption. Trigger token always leads."""
        return (
            f"{token}, {name}, {identity_line} "
            f"{self.framing}, {self.angle}, {self.pose}, neutral studio background, "
            f"even lighting, character reference sheet."
        )


VIEW_SPECS: List[ViewSpec] = [
    ViewSpec(
        id="v01_front",
        name="Front",
        angle="camera dead-on at 0 degrees, eye level",
        framing="full body head to toe, centred",
        pose="standing relaxed A-pose, arms slightly away from the body, feet shoulder width apart, looking straight into camera",
        purpose="silhouette and front wardrobe",
        repeats=2,
        tags=["turnaround", "full_body"],
    ),
    ViewSpec(
        id="v02_three_quarter_left",
        name="Three-quarter left",
        angle="camera 45 degrees to the character's left, eye level",
        framing="full body head to toe",
        pose="standing relaxed, weight even, head turned slightly toward camera",
        purpose="primary cinematic angle",
        repeats=2,
        tags=["turnaround", "full_body"],
    ),
    ViewSpec(
        id="v03_profile_left",
        name="Left profile",
        angle="camera 90 degrees to the character's left, strict side view, eye level",
        framing="full body head to toe",
        pose="standing straight, arms at sides, facing left",
        purpose="jaw, nose and posture line",
        repeats=1,
        tags=["turnaround", "full_body"],
    ),
    ViewSpec(
        id="v04_back_quarter_left",
        name="Rear three-quarter left",
        angle="camera 135 degrees behind and to the left, eye level",
        framing="full body head to toe",
        pose="standing relaxed, back mostly to camera, head not turned",
        purpose="rear costume seams and hair mass",
        repeats=1,
        tags=["turnaround", "full_body"],
    ),
    ViewSpec(
        id="v05_back",
        name="Back",
        angle="camera directly behind at 180 degrees, eye level",
        framing="full body head to toe",
        pose="standing relaxed A-pose seen from behind",
        purpose="full rear coverage",
        repeats=1,
        tags=["turnaround", "full_body"],
    ),
    ViewSpec(
        id="v06_three_quarter_right",
        name="Three-quarter right",
        angle="camera 45 degrees to the character's right, eye level",
        framing="full body head to toe",
        pose="standing relaxed, head turned slightly toward camera",
        purpose="mirror-side asymmetry",
        repeats=2,
        tags=["turnaround", "full_body"],
    ),
    ViewSpec(
        id="v07_profile_right",
        name="Right profile",
        angle="camera 90 degrees to the character's right, strict side view, eye level",
        framing="full body head to toe",
        pose="standing straight, arms at sides, facing right",
        purpose="mirror-side jaw line",
        repeats=1,
        tags=["turnaround", "full_body"],
    ),
    ViewSpec(
        id="v08_face_front",
        name="Face, front",
        angle="camera dead-on at 0 degrees, eye level, close range",
        framing="head and shoulders portrait, face filling most of the frame",
        pose="neutral calm expression, mouth closed, eyes open looking into camera",
        purpose="the identity anchor the whole movie is judged against",
        repeats=4,
        tags=["face", "anchor"],
    ),
    ViewSpec(
        id="v09_face_expression",
        name="Face, expressive",
        angle="camera 30 degrees to the left, eye level, close range",
        framing="head and shoulders portrait",
        pose="bright open smile with mouth open showing natural speech shape, eyebrows raised, eyes lively",
        purpose="emotion range and lip-sync mouth interior",
        repeats=3,
        tags=["face", "lipsync"],
    ),
    ViewSpec(
        id="v10_action_detail",
        name="Action and detail",
        angle="camera slightly low at 20 degrees to the left, dynamic",
        framing="full body in motion with hands clearly visible",
        pose="mid-stride walking forward with natural arm swing, hands open and clearly readable, any signature prop held as in the reference",
        purpose="motion silhouette, hands and props",
        repeats=2,
        tags=["action", "hands"],
    ),
]

VIEW_BY_ID: Dict[str, ViewSpec] = {v.id: v for v in VIEW_SPECS}

DEFAULT_BACKGROUND = (
    "plain flat neutral light grey seamless studio backdrop, no props, no scenery, no floor line"
)


def total_dataset_images(views: int = len(VIEW_SPECS), variants: int = 1) -> int:
    """How many training images a single character will contribute."""
    return sum(v.repeats for v in VIEW_SPECS[:views]) * max(1, variants)
