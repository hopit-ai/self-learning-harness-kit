"""Freeze task roles without splitting connected retail scenario variants."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

TAU2_COMMIT = "672227c6b6676edc20d57ea53b7000262aae77b9"
RETAIL_TASKS_BLOB = "e95a9f898b2d1f6568ea8faec78edc3767ce747d"

# Copied from continual_learning/continual/tau2_preflight.py, VARIANT_FAMILIES.
# Singleton task ids are their own family; the grouping is conservative.
VARIANT_FAMILIES: tuple[tuple[str, ...], ...] = (
    ("0", "1"), ("2", "3", "4"), ("5", "6", "7", "8", "9"),
    ("10", "11"), ("12", "13"), ("25", "26"), ("30", "31", "32"),
    ("33", "34"), ("36", "37", "38"), ("41", "42"), ("46", "47"),
    ("51", "52"), ("54", "55"), ("56", "57"), ("60", "61"),
    ("62", "63"), ("67", "68"), ("71", "72"), ("82", "83", "84"),
    ("85", "86"), ("89", "90"), ("91", "92"), ("93", "94", "95"),
    ("96", "97"), ("98", "99"), ("101", "102"), ("103", "104"),
    ("109", "110"), ("111", "112"),
)


def git_blob_sha(path: str | Path) -> str:
    """Hash the exact source bytes with Git's SHA-1 blob framing."""
    body = Path(path).read_bytes()
    return hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()


def family_map(task_ids: list[str]) -> dict[str, str]:
    known: dict[str, str] = {}
    for group in VARIANT_FAMILIES:
        for member in group:
            if member in known:
                raise ValueError(f"task id appears in two families: {member}")
            if member not in task_ids:
                raise ValueError(f"family task id absent from tasks.json: {member}")
            known[member] = group[0]
    return {task_id: "retail-family-" + known.get(task_id, task_id).zfill(3)
            for task_id in sorted(task_ids)}


def _sources(tasks_path: str | Path, splits_path: str | Path) -> tuple[list[str], dict]:
    tasks = json.loads(Path(tasks_path).read_text(encoding="utf-8"))
    ids = [row["id"] for row in tasks]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("task ids must be unique nonempty strings")
    official = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    for role in ("train", "test"):
        members = official[role]
        if not isinstance(members, list) or any(not isinstance(i, str) for i in members):
            raise ValueError(f"official {role} must be a list of string ids")
        if len(set(members)) != len(members) or not set(members) <= set(ids):
            raise ValueError(f"invalid official {role} task ids")
    if set(official["train"]) & set(official["test"]):
        raise ValueError("official train/test overlap")
    return ids, official


def _metadata(tasks_path: str | Path, splits_path: str | Path, domain: str) -> dict[str, str]:
    return {"tau2_commit": TAU2_COMMIT,
            "tasks_path": f"data/tau2/domains/{domain}/tasks.json",
            "split_tasks_path": f"data/tau2/domains/{domain}/split_tasks.json",
            "tasks_git_blob_sha1": git_blob_sha(tasks_path),
            "split_tasks_git_blob_sha1": git_blob_sha(splits_path)}


def build_retail_splits(
    tasks_json_path: str | Path, split_tasks_json_path: str | Path, seed: int
) -> dict[str, Any]:
    """Return sorted roles plus the source hashes and complete family map.

    Put every train family shared with official test into dev first. Shuffle
    remaining families with the recorded seed and fill dev to at least 24
    (or the closest larger attainable size), without fragmenting a family.
    Official test remains untouched; stream has zero family overlap with it.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    ids, official = _sources(tasks_json_path, split_tasks_json_path)
    fmap = family_map(ids)
    groups: dict[str, list[str]] = {}
    for task_id in sorted(official["train"]):
        groups.setdefault(fmap[task_id], []).append(task_id)
    test_families = {fmap[i] for i in official["test"]}
    mandatory = set(groups) & test_families
    families = sorted(set(groups) - mandatory)
    random.Random(seed).shuffle(families)
    target = min(24, len(official["train"]))
    mandatory_size = sum(len(groups[f]) for f in mandatory)
    attainable: dict[int, tuple[str, ...]] = {0: ()}
    for family in families:
        for size, chosen in list(attainable.items()):
            attainable.setdefault(size + len(groups[family]), (*chosen, family))
    remaining = max(0, target - mandatory_size)
    closest = min(size for size in attainable if size >= remaining)
    dev_families = mandatory | set(attainable[closest])
    # Carve a separate confirmation pool from the former stream, whole families.
    confirmation_seed = seed
    stream_families = sorted(set(groups) - dev_families)
    random.Random(confirmation_seed).shuffle(stream_families)
    attainable = {0: ()}
    for family in stream_families:
        for size, chosen in list(attainable.items()):
            attainable.setdefault(size + len(groups[family]), (*chosen, family))
    confirm_target = min(8, sum(len(groups[f]) for f in stream_families))
    confirm_size = min(size for size in attainable if size >= confirm_target)
    confirm_families = set(attainable[confirm_size])
    roles = {
        "stream": sorted(i for i in official["train"] if fmap[i] not in dev_families | confirm_families),
        "confirm": sorted(i for i in official["train"] if fmap[i] in confirm_families),
        "dev": sorted(i for i in official["train"] if fmap[i] in dev_families),
        "test": sorted(official["test"]),
    }
    return {"schema_version": 2, "domain": "retail", "seed": seed,
            "confirmation_seed": confirmation_seed, "target_confirm_size": confirm_target,
            "source": _metadata(tasks_json_path, split_tasks_json_path, "retail"),
            "family_map": fmap, "target_dev_size": target,
            "sizes": {role: len(members) for role, members in roles.items()},
            "labels": {"stream": "learning", "dev": "gate", "confirm": "post-acceptance-report-only",
                       "test": "confirmatory"},
            **roles}


def build_airline_split(tasks_path: str | Path, splits_path: str | Path) -> dict[str, Any]:
    """Record only the official test role for transfer; never a training role."""
    _, official = _sources(tasks_path, splits_path)
    transfer = sorted(official["test"])
    return {"schema_version": 1, "domain": "airline", "transfer": transfer,
            "labels": {"transfer": "transfer-only"}, "sizes": {"transfer": len(transfer)},
            "source": _metadata(tasks_path, splits_path, "airline")}


def verify(tau2_root: str | Path, split_dir: str | Path = "splits") -> dict[str, dict[str, int]]:
    """Verify frozen manifests by rebuilding from the pinned vendor bytes.

    Full structural equality checks roles, coverage, families, seed, labels,
    sizes and provenance. Source paths are relative to tau2_root.
    """
    root, directory = Path(tau2_root), Path(split_dir)
    domains = root / "data/tau2/domains"
    if git_blob_sha(domains / "retail/tasks.json") != RETAIL_TASKS_BLOB:
        raise ValueError("retail tasks.json does not match the pinned Git blob")
    # Check the checkout revision when Git metadata is available (archives may
    # have no .git directory; their exact source bytes are still checked).
    if (root / ".git").exists():
        import subprocess
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if commit != TAU2_COMMIT:
            raise ValueError("tau2 checkout does not match the pinned commit")
    retail = json.loads((directory / "retail.json").read_text(encoding="utf-8"))
    expected = {
        "retail": build_retail_splits(domains / "retail/tasks.json",
                                      domains / "retail/split_tasks.json", retail["seed"]),
        "airline": build_airline_split(domains / "airline/tasks.json",
                                       domains / "airline/split_tasks.json"),
    }
    for domain, manifest in expected.items():
        actual = json.loads((directory / f"{domain}.json").read_text(encoding="utf-8"))
        if actual != manifest:
            raise ValueError(f"{domain}.json differs from the pinned vendor sources or split rules")
    return {domain: manifest["sizes"] for domain, manifest in expected.items()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--tau2-root", type=Path, required=True)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--output-dir", type=Path, default=Path("splits"))
    check = commands.add_parser("verify")
    check.add_argument("--tau2-root", type=Path, required=True)
    check.add_argument("--split-dir", type=Path, default=Path("splits"))
    args = parser.parse_args(argv)
    if args.command == "verify":
        for domain, sizes in verify(args.tau2_root, args.split_dir).items():
            print(f"{domain}: verified {sizes}")
        return
    domains = args.tau2_root / "data/tau2/domains"
    # Read both before writing anything: incomplete sources must fail closed.
    retail = build_retail_splits(domains / "retail/tasks.json",
                                 domains / "retail/split_tasks.json", args.seed)
    airline = build_airline_split(domains / "airline/tasks.json",
                                  domains / "airline/split_tasks.json")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for domain, manifest in (("retail", retail), ("airline", airline)):
        path = args.output_dir / f"{domain}.json"
        content = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to replace frozen split: {path}")
    for domain, manifest in (("retail", retail), ("airline", airline)):
        path = args.output_dir / f"{domain}.json"
        if not path.exists():
            with path.open("x", encoding="utf-8") as output:
                output.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"{path}: {manifest['sizes']}")
    if retail["sizes"]["stream"] < 40:
        print("Warning: retail stream has fewer than 40 tasks")


if __name__ == "__main__":
    main()
