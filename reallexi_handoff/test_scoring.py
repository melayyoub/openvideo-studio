"""Synthetic tests for the handoff scorer and colour match.

Builds a fake clip whose tail degrades the way real video-model output does,
then checks the selector rejects the bad frames and still picks a late one.

    python -m reallexi_handoff.tests.test_scoring
"""

import sys

import numpy as np

from ..nodes import HandoffColorMatch, HandoffFrameSelect, HandoffQualityGate
from ..scoring import _gray, exposure_penalty, sharpness, similarity


def _gray_mean(img):
    return _gray(img).mean()

rng = np.random.default_rng(7)
H = W = 128


def base_scene(shift=0, hue=(1.0, 0.85, 0.6)):
    """A crisp scene: textured background, a bright subject block, edges."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    bg = 0.35 + 0.10 * np.sin((x + shift) / 5.0) * np.cos(y / 7.0)
    img = np.stack([bg * hue[0], bg * hue[1], bg * hue[2]], -1)
    img[30:90, 25 + shift:75 + shift] = 0.85          # subject
    img[45:60, 35 + shift:65 + shift] = 0.12          # high-contrast detail
    return np.clip(img, 0, 1).astype(np.float32)


def blur(img, k):
    out = img.copy()
    for _ in range(k):
        p = np.pad(out, ((1, 1), (1, 1), (0, 0)), mode="edge")
        out = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
               + out) / 5.0
    return out.astype(np.float32)


def build_clip():
    """20 frames. Sharp until 13, progressively smeared after, frame 18 flared,
    frame 19 (the literal last frame) heavily blurred - the realistic case."""
    frames, labels = [], []
    for i in range(20):
        f = base_scene(shift=i // 3)
        if i == 18:
            f = np.clip(blur(f, 2) * 2.6, 0, 1)       # blown highlights
            labels.append("FLARED")
        elif i >= 14:
            f = blur(f, (i - 13) * 2)
            labels.append(f"blur x{(i - 13) * 2}")
        else:
            labels.append("clean")
        frames.append(f)
    return np.stack(frames), labels


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    return cond


clip, labels = build_clip()
ref = base_scene(shift=0)
ok = True

print("frame quality profile")
print(f"  {'idx':>4} {'label':>10} {'sharp':>10} {'clip':>7}")
for i, f in enumerate(clip):
    print(f"  {i:>4} {labels[i]:>10} {sharpness(f):>10.5f} "
          f"{exposure_penalty(f):>7.3f}")

print()
print("metric sanity")
ok &= check("sharp frame scores higher than blurred",
            sharpness(clip[10]) > sharpness(clip[19]))
ok &= check("flared frame flagged by exposure penalty",
            exposure_penalty(clip[18]) > exposure_penalty(clip[10]))
ok &= check("identical image self-similarity is high",
            similarity(ref, ref) > 0.95)
ok &= check("drifted palette scores lower than matching palette",
            similarity(base_scene(0, (0.2, 0.4, 1.0)), ref)
            < similarity(base_scene(2), ref))

sel = HandoffFrameSelect()
print()
print("selector — default weights, reference connected")
frame, idx, score, report = sel.select(clip, 12, 1.0, 1.0, 0.6, 2.0,
                                       reference=ref[None])
print("\n".join("    " + line for line in report.splitlines()))
ok &= check("did NOT pick the literal last frame (19)", idx != 19)
ok &= check("did NOT pick the flared frame (18)", idx != 18)
ok &= check("picked from the clean tail region (<=13)", idx <= 13)
ok &= check("still picked a late frame, not the window start",
            idx >= len(clip) - 12)
ok &= check("returned a single frame", frame.shape[0] == 1)

print()
print("selector — recency cranked up (continuity prioritised)")
_, idx_hi, _, _ = sel.select(clip, 12, 1.0, 1.0, 3.0, 2.0, reference=ref[None])
print(f"    picked index {idx_hi}")
ok &= check("higher recency weight moves the pick later or equal",
            idx_hi >= idx)

print()
print("selector — no reference (identity term inert)")
_, idx_nr, _, rep_nr = sel.select(clip, 12, 1.0, 1.0, 0.6, 2.0)
print(f"    picked index {idx_nr}")
ok &= check("still avoids the blurred last frame", idx_nr != 19)
ok &= check("reports the missing reference", "identity term is inert" in rep_nr)

print()
print("selector — window larger than the clip")
_, idx_big, _, _ = sel.select(clip, 500, 1.0, 1.0, 0.6, 2.0, reference=ref[None])
ok &= check("clamps to clip length without error", 0 <= idx_big < len(clip))

print()
print("selector — window of 1 degenerates to last-frame behaviour")
_, idx_one, _, _ = sel.select(clip, 1, 1.0, 1.0, 0.6, 2.0, reference=ref[None])
ok &= check("window=1 returns the final frame", idx_one == 19)

gate = HandoffQualityGate()
print()
print("quality gate")
_, passed_good, rep_good = gate.gate(clip[10][None], ref[None], 0.72, 0.35,
                                     "fall back to reference")
ok &= check("clean frame passes the gate", passed_good)
out_bad, passed_bad, rep_bad = gate.gate(
    base_scene(0, (0.15, 0.2, 1.0))[None], ref[None], 0.72, 0.35,
    "fall back to reference")
ok &= check("badly drifted frame fails the gate", not passed_bad)
ok &= check("failed gate forwards the reference",
            np.allclose(out_bad, ref[None]))
out_ov, _, _ = gate.gate(base_scene(0, (0.15, 0.2, 1.0))[None], ref[None],
                         0.72, 0.35, "pass through anyway")
ok &= check("override forwards the frame instead of the reference",
            not np.allclose(out_ov, ref[None]))
_, passed_flare, _ = gate.gate(clip[18][None], ref[None], 0.0, 0.10,
                               "fall back to reference")
ok &= check("flared frame fails on the clipping threshold", not passed_flare)


# ---------------------------------------------------------------------------
# colour match
# ---------------------------------------------------------------------------

def chan_offset(a, b):
    return float(np.mean(np.abs(a.mean((0, 1)) - b.mean((0, 1)))))

cm = HandoffColorMatch()
ref_look = base_scene(0, (1.0, 0.85, 0.6))          # warm golden reference
drifted  = base_scene(0, (0.55, 0.7, 1.15))         # shot 3 has gone cold/blue
drifted  = np.clip(drifted, 0, 1).astype(np.float32)

print()
print("colour match")
print(f"    reference channel means: {ref_look.mean((0,1)).round(3)}")
print(f"    drifted   channel means: {drifted.mean((0,1)).round(3)}")

out_ms, rep_ms = cm.match(drifted[None], ref_look[None], "mean/std (gentle)", 1.0, False)
print(f"    after mean/std         : {out_ms[0].mean((0,1)).round(3)}")
ok &= check("mean/std pulls the palette toward the reference",
            chan_offset(out_ms[0], ref_look) < chan_offset(drifted, ref_look))
ok &= check("mean/std at strength 1 lands close to the reference",
            chan_offset(out_ms[0], ref_look) < 0.02)

out_h, _ = cm.match(drifted[None], ref_look[None], "histogram (strong)", 1.0, False)
print(f"    after histogram        : {out_h[0].mean((0,1)).round(3)}")
ok &= check("histogram mode also converges",
            chan_offset(out_h[0], ref_look) < 0.02)

out_0, _ = cm.match(drifted[None], ref_look[None], "mean/std (gentle)", 0.0, False)
ok &= check("strength 0 is a no-op", np.allclose(out_0[0], drifted, atol=1e-6))

out_half, _ = cm.match(drifted[None], ref_look[None], "mean/std (gentle)", 0.5, False)
ok &= check("strength 0.5 sits between the two",
            chan_offset(drifted, ref_look) > chan_offset(out_half[0], ref_look)
            > chan_offset(out_ms[0], ref_look))

out_self, _ = cm.match(ref_look[None], ref_look[None], "mean/std (gentle)", 1.0, False)
ok &= check("matching to itself is a near no-op",
            np.allclose(out_self[0], ref_look, atol=1e-4))

out_lum, _ = cm.match(drifted[None], ref_look[None], "mean/std (gentle)", 1.0, True)
ok &= check("preserve_luminance keeps brightness closer to the source",
            abs(float(_gray_mean(out_lum[0])) - float(_gray_mean(drifted)))
            < abs(float(_gray_mean(out_ms[0])) - float(_gray_mean(drifted))))

ok &= check("output stays inside 0..1",
            out_h.min() >= 0.0 and out_h.max() <= 1.0)

batch = np.stack([drifted, base_scene(1, (0.6, 0.6, 1.2))]).astype(np.float32)
out_b, _ = cm.match(batch, ref_look[None], "mean/std (gentle)", 1.0, False)
ok &= check("handles multi-frame batches", out_b.shape == batch.shape)

print()
print("RESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
