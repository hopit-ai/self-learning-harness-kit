"""Partner side, run by ``setup_hermes_offline.sh`` with the Hermes venv's interpreter:
``pip check`` of the venv, accepting ONLY the mismatches Hermes' own lock resolved on
purpose.

Hermes resolves its dependencies with ``[tool.uv] override-dependencies`` (its
``pyproject.toml``): an override replaces a dependency's requirement during resolution,
but the installed distribution still declares the original, so ``pip check`` reports it
(e.g. ``browser-harness 0.1.13 has requirement pillow==12.3.0, but you have pillow
13.0.0``). The same rule as Hermes' ``pm/pip_check.py``: a reported line is expected
exactly when it names an overridden dependency installed at a version the override
allows (or, for an override whose marker is false here, an overridden dependency that is
not installed). Everything else fails.

    python hermes_pip_check.py HERMES_CLONE/pyproject.toml   # runs pip check itself
    python hermes_pip_check.py PYPROJECT --report FILE         # checks a saved report

Prints the allowlist and a JSON verdict; exit 0 only when every line is expected and pip
itself ran cleanly, else 4 (``verdict``).
Stdlib + pip's vendored ``packaging`` (always present with pip).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

try:
    from packaging.requirements import Requirement
except ImportError:  # pragma: no cover - the venv's pip always vendors it
    from pip._vendor.packaging.requirements import Requirement

_MISMATCH = re.compile(r"^(\S+) (\S+) has requirement (.+), but you have (\S+) (\S+)\.$")
_MISSING = re.compile(r"^(\S+) (\S+) requires (\S+), which is not installed\.$")


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def overrides(pyproject: Path) -> dict[str, Requirement]:
    d = tomllib.loads(Path(pyproject).read_text())
    out = {}
    for line in d.get("tool", {}).get("uv", {}).get("override-dependencies", []):
        r = Requirement(line)
        out[_norm(r.name)] = r
    return out


def unexpected(report: str, ovr: dict[str, Requirement]) -> tuple[list[str], list[str]]:
    """(accepted, unexpected) report lines."""
    ok, bad = [], []
    for line in (x.strip() for x in report.splitlines()):
        if not line or line.startswith("No broken requirements"):
            continue
        m = _MISMATCH.match(line)
        if m:
            dep, have = _norm(m[4]), m[5]
            r = ovr.get(dep)
            if r is not None and (r.marker is None or r.marker.evaluate()) \
                    and r.specifier.contains(have, prereleases=True):
                ok.append(line)
                continue
        m = _MISSING.match(line)
        if m:
            r = ovr.get(_norm(m[3]))
            if r is not None and r.marker is not None and not r.marker.evaluate():
                ok.append(line)  # the override removes it on this platform
                continue
        bad.append(line)
    return ok, bad


def verdict(stdout: str, stderr: str, returncode: int,
            ovr: dict[str, Requirement]) -> tuple[list[str], list[str]]:
    """(accepted, problems) of one ``pip check`` run. A problem is: a report line no Hermes
    override explains; anything on stderr that shows pip itself failed (a traceback, an
    import or other error); a non-zero return code with nothing reported (pip failed
    before checking). pip's own success line ("No broken requirements found.") with
    return code 0 is fine; return code 1 with only accepted lines is the expected case."""
    ok, bad = unexpected(stdout, ovr)
    problems = [f"unexpected: {x}" for x in bad]
    err = (stderr or "").strip()
    if re.search(r"Traceback|Error|error:|No module named", err):
        problems.append(f"pip check failed: {err.splitlines()[-1][:300]}")
    if returncode != 0 and not ok and not bad:
        problems.append(f"pip check exited {returncode} without a report"
                        + (f": {err.splitlines()[-1][:300]}" if err else ""))
    if returncode == 0 and bad:
        problems.append("pip check exited 0 yet reported unexpected lines")
    return ok, problems


def main(argv: list[str]) -> int:
    """Exit 0 when every reported line is a Hermes override; 4 otherwise. ``--report FILE
    [--returncode N] [--stderr FILE]`` checks a saved run (tests)."""
    pyproject = Path(argv[1])
    ovr = overrides(pyproject)
    if len(argv) > 3 and argv[2] == "--report":
        stdout = Path(argv[3]).read_text()
        rest = argv[4:]
        rc = int(rest[rest.index("--returncode") + 1]) if "--returncode" in rest else (
            1 if stdout.strip() and "No broken requirements" not in stdout else 0)
        stderr = Path(rest[rest.index("--stderr") + 1]).read_text() \
            if "--stderr" in rest else ""
    else:
        r = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True,
                           text=True, check=False)
        stdout, stderr, rc = r.stdout, r.stderr, r.returncode
    print("override-dependencies (the allowlist):")
    for req in ovr.values():
        print(f"  {req}")
    ok, problems = verdict(stdout, stderr, rc, ovr)
    for line in ok:
        print(f"accepted (a Hermes override): {line}")
    for line in problems:
        print(f"PROBLEM: {line}")
    print(json.dumps({"pip_check_ok": not problems, "returncode": rc, "accepted": len(ok),
                      "problems": len(problems)}))
    return 0 if not problems else 4


if __name__ == "__main__":
    sys.exit(main(sys.argv))
