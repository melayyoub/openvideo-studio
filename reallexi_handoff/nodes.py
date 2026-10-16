"""ComfyUI node classes. Thin wrappers around `scoring`."""

from __future__ import annotations

import numpy as np

from .scoring import (_gray, _hist_match, _meanstd_match, exposure_penalty,
                      score_window, similarity)

try:
    import torch
except ImportError:                                   # allows offline testing
    torch = None


def _to_np(t):
    return t.detach().cpu().numpy() if torch is not None and hasattr(t, "detach") else np.asarray(t)


def _like(src, arr):
    if torch is not None and hasattr(src, "detach"):
        return torch.from_numpy(np.ascontiguousarray(arr)).to(src.device).to(src.dtype)
    return arr


class HandoffFrameSelect:
    """Pick the best seed frame from the tail of a generated clip."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "window": ("INT", {"default": 12, "min": 1, "max": 240,
                                   "tooltip": "How many frames back from the "
                                              "end to consider. Keep small so "
                                              "the cut stays tight."}),
                "weight_sharpness": ("FLOAT", {"default": 1.0, "min": 0.0,
                                               "max": 5.0, "step": 0.05}),
                "weight_identity": ("FLOAT", {"default": 1.0, "min": 0.0,
                                              "max": 5.0, "step": 0.05}),
                "weight_recency": ("FLOAT", {"default": 0.6, "min": 0.0,
                                             "max": 5.0, "step": 0.05,
                                             "tooltip": "Bias toward the last "
                                                        "frame. Raise to protect "
                                                        "motion continuity, lower "
                                                        "to prioritise quality."}),
                "clipping_penalty": ("FLOAT", {"default": 2.0, "min": 0.0,
                                               "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "reference": ("IMAGE", {"tooltip": "Known-good identity frame. "
                                                   "Enables the drift score."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("frame", "index", "score", "report")
    FUNCTION = "select"
    CATEGORY = "Reallexi/handoff"

    def select(self, images, window, weight_sharpness, weight_identity,
               weight_recency, clipping_penalty, reference=None):
        arr = _to_np(images)
        n_total = arr.shape[0]
        w = int(min(max(1, window), n_total))
        tail = arr[n_total - w:]
        ref = _to_np(reference)[0] if reference is not None else None

        scores, parts = score_window(tail, ref, weight_sharpness,
                                     weight_identity, weight_recency,
                                     clipping_penalty)
        best_local = int(np.argmax(scores))
        best_global = n_total - w + best_local

        lines = [
            f"window: last {w} of {n_total} frames "
            f"(indices {n_total - w}..{n_total - 1})",
            f"picked: index {best_global}  "
            f"({best_global - (n_total - 1)} from the end)  "
            f"score {float(scores[best_local]):.4f}",
            "",
            f"{'idx':>6} {'score':>8} {'sharp':>8} {'ident':>7} {'clip':>7}",
        ]
        for i in range(w):
            mark = " <-" if i == best_local else ""
            lines.append(
                f"{n_total - w + i:>6} {float(scores[i]):>8.4f} "
                f"{float(parts['sharp_norm'][i]):>8.3f} "
                f"{float(parts['ident'][i]):>7.3f} "
                f"{float(parts['clip'][i]):>7.3f}{mark}")
        if ref is None:
            lines.append("")
            lines.append("note: no reference connected - identity term is inert.")

        frame = _like(images, tail[best_local:best_local + 1])
        return (frame, best_global, float(scores[best_local]), "\n".join(lines))


class HandoffQualityGate:
    """Last line of defence: if the chosen frame still drifts too far from a
    known-good reference, pass the reference through instead.

    ComfyUI cannot branch mid-graph, so this is a pure selection between two
    already-computed inputs rather than a conditional. Both paths execute; only
    one is forwarded. Falling back restores identity at the cost of continuity,
    so it should trip rarely - read `passed` and the report before trusting a run.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame": ("IMAGE",),
                "reference": ("IMAGE",),
                "min_similarity": ("FLOAT", {"default": 0.72, "min": 0.0,
                                             "max": 1.0, "step": 0.01}),
                "max_clipping": ("FLOAT", {"default": 0.35, "min": 0.0,
                                           "max": 1.0, "step": 0.01}),
                "on_fail": (["fall back to reference", "pass through anyway"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN", "STRING")
    RETURN_NAMES = ("frame", "passed", "report")
    FUNCTION = "gate"
    CATEGORY = "Reallexi/handoff"

    def gate(self, frame, reference, min_similarity, max_clipping, on_fail):
        f = _to_np(frame)[0]
        r = _to_np(reference)[0]
        sim = similarity(f, r)
        clip = exposure_penalty(f)
        ok = (sim >= min_similarity) and (clip <= max_clipping)

        report = (f"similarity {sim:.3f} (min {min_similarity:.2f})  "
                  f"clipping {clip:.3f} (max {max_clipping:.2f})\n"
                  f"verdict: {'PASS' if ok else 'FAIL'}")
        if not ok and on_fail == "fall back to reference":
            report += "\naction: forwarding the REFERENCE - this shot starts as a cut."
            return (reference, False, report)
        if not ok:
            report += "\naction: forwarding the frame anyway (on_fail override)."
        return (frame, bool(ok), report)


class HandoffColorMatch:
    """Force the seed frame's palette back onto the reference's palette.

    Prompt text alone cannot hold a look across shots - the model re-grades
    every generation slightly, and over four shots the palette walks. This is
    the mechanical backstop: it transfers the reference's colour statistics
    onto the frame before it seeds the next shot, so grade drift cannot
    accumulate no matter what the sampler did.

    It fixes colour and exposure. It does NOT fix render medium - a frame that
    came out as 3D animation stays 3D animation with the right palette. Medium
    is held by the prompt's look block and the medium negatives.

    Place it AFTER the quality gate: the gate should judge the raw pick, so
    that a frame is not rescued by a grade correction that hides real drift.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame": ("IMAGE",),
                "reference": ("IMAGE",),
                "mode": (["mean/std (gentle)", "histogram (strong)"],),
                "strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0,
                                       "step": 0.05,
                                       "tooltip": "1.0 pins the palette hard. "
                                                  "Lower if shots start to look "
                                                  "flat or identical."}),
                "preserve_luminance": ("BOOLEAN", {"default": False,
                                                   "tooltip": "Correct hue and "
                                                              "saturation but keep "
                                                              "the frame's own "
                                                              "brightness."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frame", "report")
    FUNCTION = "match"
    CATEGORY = "Reallexi/handoff"

    def match(self, frame, reference, mode, strength, preserve_luminance):
        f = _to_np(frame).astype(np.float32)
        r = _to_np(reference)[0].astype(np.float32)
        out = np.empty_like(f)
        before, after = [], []

        for b in range(f.shape[0]):
            img = f[b]
            before.append(float(np.mean(np.abs(img.mean((0, 1)) - r.mean((0, 1))))))
            lum0 = _gray(img)
            if mode.startswith("histogram"):
                fixed = _hist_match(img, r)
            else:
                fixed = _meanstd_match(img, r)
            fixed = img + (fixed - img) * float(strength)
            if preserve_luminance:
                lum1 = _gray(fixed)
                scale = np.divide(lum0, np.maximum(lum1, 1e-5))[..., None]
                fixed = fixed * scale
            fixed = np.clip(fixed, 0.0, 1.0)
            after.append(float(np.mean(np.abs(fixed.mean((0, 1)) - r.mean((0, 1))))))
            out[b] = fixed

        report = (f"mode: {mode}  strength {strength:.2f}  "
                  f"preserve_luminance {preserve_luminance}\n"
                  f"mean channel offset from reference: "
                  f"{np.mean(before):.4f} -> {np.mean(after):.4f}")
        return (_like(frame, out), report)


NODE_CLASS_MAPPINGS = {
    "HandoffFrameSelect": HandoffFrameSelect,
    "HandoffQualityGate": HandoffQualityGate,
    "HandoffColorMatch": HandoffColorMatch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HandoffFrameSelect": "Handoff Frame Select",
    "HandoffQualityGate": "Handoff Quality Gate",
    "HandoffColorMatch": "Handoff Color Match",
}
