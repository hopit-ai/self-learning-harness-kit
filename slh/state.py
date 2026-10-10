"""Content-addressed harness state; saved versions are never overwritten."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any


def _canonical(value: Any) -> str:
    # Reject lossy JSON key coercion recursively, including mixed-key objects.
    def validate(item: Any) -> None:
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("JSON object keys must be strings")
            for child in item.values():
                validate(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                validate(child)
    validate(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def check_base(base: Any) -> None:
    """A base-arm binding is {"arm": name, "config_sha256": 64 hex} or None."""
    if base is None:
        return
    if (not isinstance(base, dict) or set(base) != {"arm", "config_sha256"}
            or not isinstance(base["arm"], str) or not base["arm"]
            or not isinstance(base["config_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", base["config_sha256"])):
        raise ValueError("base arm binding must be {arm, config_sha256}")


class HarnessState:
    """Notes, skill text and guard hook configuration at one point in time.

    Values returned by properties are copies. Construct a new state to propose
    an edit; save publishes a version directory, load verifies its contents.
    Note and guard order is significant, while dictionary order is not.
    Canonical JSON preserves numeric types: 1 and 1.0 have different hashes.

    ``rehearsal``: a state produced by a learner rehearsal (stub evidence) or
    derived from one. The flag is part of the canonical content, so it changes
    the version id and cannot be dropped without failing the load-time hash
    check. Derived states must pass ``rehearsal=parent.rehearsal or ...``. The
    runner refuses such a state unless the run is fully stub, and marks the
    run evidence-ineligible. A non-rehearsal state's content (and version id)
    is unchanged by this field.

    ``base``: the fixed arm this state was learned on and is valid for
    (``slh.execution.base_arm_fingerprint``: arm name and the cap-free resolved
    config hash). B0 (baseline) and B1 (think) share the empty, unbound state's
    hash, so identity is always the pair (base arm, version). A bound state's
    canonical content includes the binding (different version id per base);
    the runner refuses it on any other base arm. Unbound states (``None``,
    every state before learning system v2) keep their content and version id.
    """

    def __init__(
        self, notes: list[dict[str, Any]] | None = None,
        skills: dict[str, str] | None = None,
        guards: list[dict[str, Any]] | None = None, *, rehearsal: bool = False,
        base: dict[str, str] | None = None,
    ) -> None:
        if not isinstance(rehearsal, bool):
            raise ValueError("rehearsal must be a boolean")
        check_base(base)
        content = {"schema_version": 1, "notes": [] if notes is None else notes,
                   "skills": {} if skills is None else skills,
                   "guards": [] if guards is None else guards}
        if rehearsal:
            content["rehearsal"] = True
        if base is not None:
            content["base"] = dict(base)
        if not isinstance(content["notes"], list) or any(
            not isinstance(note, dict) or not isinstance(note.get("content"), str)
            for note in content["notes"]
        ):
            raise ValueError("notes must be a list of objects with string content")
        if not isinstance(content["skills"], dict) or any(
            not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", name)
            or not isinstance(text, str) for name, text in content["skills"].items()
        ):
            raise ValueError("skills must map safe directory names to SKILL.md text")
        names = [name.casefold() for name in content["skills"]]
        if len(set(names)) != len(names):
            raise ValueError("skill names must be unique case-insensitively")
        if not isinstance(content["guards"], list) or any(
            not isinstance(guard, dict) for guard in content["guards"]
        ):
            raise ValueError("guards must be a list of hook configuration objects")
        self._canonical_content = _canonical(content)

    @classmethod
    def empty(cls) -> HarnessState:
        """A0: no added prompt section and no guard hooks."""
        return cls()

    @property
    def version_id(self) -> str:
        return hashlib.sha256(self._canonical_content.encode("utf-8")).hexdigest()[:16]

    @property
    def content_sha256(self) -> str:
        """Full SHA-256 of the canonical content (the version id is its prefix):
        the candidate-artifact hash the v2 screen's tie-break uses."""
        return hashlib.sha256(self._canonical_content.encode("utf-8")).hexdigest()

    @property
    def rehearsal(self) -> bool:
        return json.loads(self._canonical_content).get("rehearsal") is True

    @property
    def base(self) -> dict[str, str] | None:
        """The base arm binding, or None for an unbound (legacy or empty) state."""
        value = json.loads(self._canonical_content).get("base")
        return None if value is None else dict(value)

    def bound_to(self, base: dict[str, str]) -> HarnessState:
        """The same content bound to ``base``; refuses a state bound elsewhere."""
        check_base(base)
        if self.base is not None and self.base != base:
            raise ValueError("harness state is bound to a different base arm")
        return HarnessState(self.notes, self.skills, self.guards, rehearsal=self.rehearsal,
                            base=base)

    def identity(self, base: dict[str, str] | None = None) -> dict[str, Any]:
        """(base arm, state version): what every identity check compares."""
        return {"harness_version_id": self.version_id, "state_base_arm": self.base,
                "base_arm": dict(base) if base is not None else self.base}

    @property
    def notes(self) -> list[dict[str, Any]]:
        return json.loads(self._canonical_content)["notes"]

    @property
    def skills(self) -> dict[str, str]:
        return json.loads(self._canonical_content)["skills"]

    @property
    def guards(self) -> list[dict[str, Any]]:
        """The single hook list injected into the guard layer."""
        return json.loads(self._canonical_content)["guards"]

    def system_prompt_section(self, *, max_chars_per_skill: int = 12000,
                              max_total_skill_chars: int = 48000) -> str:
        """Inject skill bodies; limits count body characters, excluding labels.

        Every skill gets a label, even when the total body budget is exhausted.
        Omitted body text is explicitly marked; full text is the default within
        these configurable limits. Guard configurations are injected separately.
        """
        for limit in (max_chars_per_skill, max_total_skill_chars):
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
                raise ValueError("skill character limits must be nonnegative integers")
        if not self.notes and not self.skills:
            return ""
        lines = ["## Harness state"]
        if self.notes:
            lines.extend(["", "### Notes"])
            lines.extend(f"- {note['content']}" for note in self.notes)
        if self.skills:
            lines.extend(["", "### Skills"])
            remaining = max_total_skill_chars
            for name, text in sorted(self.skills.items()):
                length = min(len(text), max_chars_per_skill, remaining)
                lines.extend(["", f"#### {name}: skills/{name}/SKILL.md", text[:length]])
                if length < len(text):
                    lines.append("[SKILL.md truncated]")
                remaining -= length
        return "\n".join(lines)

    def save(self, root: str | Path = "state") -> Path:
        """Publish once; repeated saves verify rather than rewrite the version."""
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        destination = root / self.version_id
        if destination.exists():
            existing = self.load(destination)
            if existing._canonical_content != self._canonical_content:
                raise ValueError(f"state version collision: {self.version_id}")
            return destination
        staging = Path(tempfile.mkdtemp(prefix=".state-", dir=root))
        try:
            manifest = {"schema_version": 1, "version_id": self.version_id,
                        "content_sha256": hashlib.sha256(
                            self._canonical_content.encode("utf-8")).hexdigest(),
                        "skill_names": sorted(self.skills)}
            if self.rehearsal:
                manifest["rehearsal"] = True
            if self.base is not None:
                manifest["base"] = self.base
            (staging / "manifest.json").write_text(_canonical(manifest) + "\n", encoding="utf-8")
            (staging / "notes.jsonl").write_text(
                "".join(_canonical(note) + "\n" for note in self.notes), encoding="utf-8")
            (staging / "guards.json").write_text(_canonical(self.guards) + "\n", encoding="utf-8")
            (staging / "skills").mkdir()
            for name, text in self.skills.items():
                directory = staging / "skills" / name
                directory.mkdir()
                (directory / "SKILL.md").write_bytes(text.encode("utf-8"))
            try:
                staging.rename(destination)
            except OSError:
                if not destination.exists():
                    raise
                existing = self.load(destination)
                if existing._canonical_content != self._canonical_content:
                    raise ValueError(f"state version collision: {self.version_id}") from None
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return destination

    @classmethod
    def load(cls, path: str | Path, version_id: str | None = None) -> HarnessState:
        """Load a version directory, or a root and a validated version id."""
        path = Path(path)
        if version_id is not None:
            if not re.fullmatch(r"[0-9a-f]{16}", version_id):
                raise ValueError("invalid state version id")
            path = path / version_id
        if path.is_symlink():
            raise ValueError("state versions must not be symlinks")
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported state schema")
        names = manifest["skill_names"]
        # Validate paths before opening any skill files.
        if not isinstance(names, list) or len(names) != len(set(names)):
            raise ValueError("invalid skill index")
        cls(skills=dict.fromkeys(names, ""))
        expected = {"manifest.json", "notes.jsonl", "guards.json"}
        expected.update(f"skills/{name}/SKILL.md" for name in names)
        actual = set()
        for entry in path.rglob("*"):
            if entry.is_symlink():
                raise ValueError("state entries must not be symlinks")
            if entry.is_file():
                actual.add(entry.relative_to(path).as_posix())
        if actual != expected or not (path / "skills").is_dir():
            raise ValueError("unexpected or missing state files")
        rehearsal = manifest.get("rehearsal", False)
        if not isinstance(rehearsal, bool):
            raise ValueError("invalid state rehearsal flag")
        base = manifest.get("base")
        check_base(base)
        notes = [json.loads(line) for line in
                 (path / "notes.jsonl").read_text(encoding="utf-8").split("\n") if line]
        state = cls(notes=notes,
                    skills={name: (path / "skills" / name / "SKILL.md").read_bytes().decode(
                        "utf-8") for name in names},
                    guards=json.loads((path / "guards.json").read_text(encoding="utf-8")),
                    rehearsal=rehearsal, base=base)
        digest = hashlib.sha256(state._canonical_content.encode("utf-8")).hexdigest()
        if (manifest.get("content_sha256") != digest
                or manifest.get("version_id") != state.version_id or path.name != state.version_id):
            raise ValueError("state content hash/version mismatch")
        return state

    def diff(self, other: HarnessState) -> dict[str, Any]:
        """Describe changes from this state to another, including removed content."""
        before, after = self.skills, other.skills
        return {"from_version": self.version_id, "to_version": other.version_id,
                "notes": None if self.notes == other.notes else
                    {"before": self.notes, "after": other.notes},
                "guards": None if self.guards == other.guards else
                    {"before": self.guards, "after": other.guards},
                "skills": {
                    "added": {name: after[name] for name in sorted(after.keys() - before.keys())},
                    "removed": {name: before[name] for name in sorted(before.keys() - after.keys())},
                    "changed": {name: {"before": before[name], "after": after[name]}
                                for name in sorted(before.keys() & after.keys())
                                if before[name] != after[name]}}}
