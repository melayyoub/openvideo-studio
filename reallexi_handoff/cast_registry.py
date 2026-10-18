"""Stage 2 — the cast registry.

One file, ``cast/cast_registry.json``, is the single source of truth for who is in this
film. Everything downstream — turnaround views, LoRA datasets, keyframes, scene prompts,
voice locks — reads from here and nothing else.

Three properties that matter:

* **Append-only by default.** ``upsert`` merges into existing entries; it never silently
  replaces one. A character that already exists keeps its ids, seeds and views.
* **Lockable.** Once ``locked`` is true, the entry is frozen. Re-running the whole
  pipeline will not regenerate or overwrite it. This is what stops you losing a cast you
  already like.
* **Backed up.** Every write snapshots the previous file into ``cast/_backups/``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config
from .util import backup, log, read_json, stable_seed, trigger_token, write_json


class CastRegistry:
    def __init__(self, path: str | Path, project: str = "film"):
        self.path = Path(path)
        data = read_json(self.path, None)
        if data is None:
            data = {"schema_version": 1, "project": project, "characters": {}, "order": []}
        self.data: Dict[str, Any] = data

    # ------------------------------------------------------------------ #
    # access
    # ------------------------------------------------------------------ #

    @property
    def characters(self) -> Dict[str, Any]:
        return self.data.setdefault("characters", {})

    @property
    def order(self) -> List[str]:
        return self.data.setdefault("order", [])

    def get(self, cid: str) -> Dict[str, Any]:
        if cid not in self.characters:
            raise KeyError(f"Unknown cast member '{cid}'. Known: {sorted(self.characters)}")
        return self.characters[cid]

    def all(self) -> List[Dict[str, Any]]:
        return [self.characters[c] for c in self.order if c in self.characters]

    def present(self, ids: List[str]) -> List[Dict[str, Any]]:
        """Cast members present in a scene, in registry order (stable prompt ordering)."""
        want = set(ids)
        return [c for c in self.all() if c["id"] in want]

    def is_locked(self, cid: str) -> bool:
        return bool(self.characters.get(cid, {}).get("locked"))

    def unlocked_ids(self) -> List[str]:
        return [c for c in self.order if not self.is_locked(c)]

    # ------------------------------------------------------------------ #
    # mutation
    # ------------------------------------------------------------------ #

    def upsert(self, character: Dict[str, Any], *, force: bool = False) -> Dict[str, Any]:
        cid = character["id"]
        existing = self.characters.get(cid)

        if existing and existing.get("locked") and not force:
            log(f"cast '{cid}' is locked — keeping existing entry untouched")
            return existing

        merged = dict(existing or {})
        for key, value in character.items():
            # Never let a regeneration blank out detail we already have.
            if value in (None, "", [], {}) and key in merged:
                continue
            merged[key] = value

        merged.setdefault("views", {})
        merged.setdefault("source_images", [])
        merged.setdefault("locked", False)
        merged.setdefault("lora", {"image": None, "video": None, "trained": False})
        merged.setdefault("trigger_token", trigger_token(cid))
        merged.setdefault("base_seed", stable_seed("character", cid))
        merged.setdefault("audio_seed", stable_seed("voice", cid))

        self.characters[cid] = merged
        if cid not in self.order:
            self.order.append(cid)
        return merged

    def attach_view(self, cid: str, view_id: str, path: str | Path, caption: str, meta: Optional[Dict] = None) -> None:
        ch = self.get(cid)
        ch.setdefault("views", {})[view_id] = {
            "path": str(Path(path).as_posix()),
            "caption": caption,
            **(meta or {}),
        }

    def attach_source(self, cid: str, path: str | Path, note: str = "") -> None:
        ch = self.get(cid)
        entry = {"path": str(Path(path).as_posix()), "note": note}
        if entry not in ch.setdefault("source_images", []):
            ch["source_images"].append(entry)

    def set_lora(self, cid: str, kind: str, name: str, trained: bool = True) -> None:
        ch = self.get(cid)
        ch.setdefault("lora", {})[kind] = name
        ch["lora"]["trained"] = trained or ch["lora"].get("trained", False)

    def lock(self, cid: Optional[str] = None) -> None:
        targets = [cid] if cid else list(self.order)
        for t in targets:
            if t in self.characters:
                self.characters[t]["locked"] = True
        log(f"locked {len(targets)} cast member(s)")

    def unlock(self, cid: Optional[str] = None) -> None:
        targets = [cid] if cid else list(self.order)
        for t in targets:
            if t in self.characters:
                self.characters[t]["locked"] = False

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #

    def save(self) -> Path:
        snap = backup(self.path)
        if snap:
            log(f"registry snapshot -> {snap.name}", "debug")
        write_json(self.path, self.data)
        return self.path

    # ------------------------------------------------------------------ #
    # reporting
    # ------------------------------------------------------------------ #

    def report(self, expected_views: int = 10) -> str:
        rows = []
        for ch in self.all():
            views = len(ch.get("views", {}))
            flags = []
            if ch.get("locked"):
                flags.append("locked")
            if ch.get("lora", {}).get("trained"):
                flags.append("lora")
            status = "ok" if views >= expected_views else f"{views}/{expected_views}"
            rows.append(
                f"  {ch['id']:<18} {ch.get('name','?'):<16} views={status:<7} "
                f"token={ch.get('trigger_token','?'):<14} {' '.join(flags)}"
            )
        return "\n".join(rows) or "  (empty)"


def registry_from_bible(cfg: Config, bible: Dict[str, Any], force: bool = False) -> CastRegistry:
    """Seed the registry from the story bible without disturbing locked members."""
    reg = CastRegistry(cfg.cast_registry_path, project=str(cfg.get("project.name", "film")))
    for ch in bible.get("cast", []):
        reg.upsert(ch, force=force)
    reg.save()
    log(f"registry now holds {len(reg.order)} cast member(s)")
    return reg
