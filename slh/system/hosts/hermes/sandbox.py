"""Process containment for one Hermes episode: macOS (Seatbelt) and Linux (bubblewrap).

Both forms give the same two properties, plus an in-process egress guard in the runner, plus
0444/0555 modes, plus the post-run re-hash (the hard incident check):

* file writes are denied everywhere except the episode root (HERMES_HOME's mutable areas,
  the workspace, the out dir, a private TMPDIR); the learned-state paths of HERMES_HOME are
  read-only again inside it;
* network: the model endpoint and this episode's Unix sockets (the bridge) only.

**macOS** (``backend() == "seatbelt"``, the laptop): ``sandbox-exec`` profiles (``profile``):
writes only under the root (plus ``/dev/null``), the learned-state paths denied again (the
last matching rule wins); outbound TCP only to ``localhost:<model port>``.

**Linux** (``backend() == "bwrap"``, the GPU partner's node; ``LinuxSpec``): bubblewrap with
every namespace unshared (user, mount, pid, ipc, uts, cgroup and **network**), all
capabilities dropped, ``--die-with-parent``, ``--new-session``. The filesystem view is built
from binds only: a short list of system directories and the declared readable roots
read-only, the episode root read-write, and the learned-state paths bound read-only again
on top of it. Nothing else exists inside (the home directory, ``~/.ssh``, the kit's sealed
folders, other projects). The network namespace has only its own loopback, so there is NO
route to any host address, the host's loopback included. The model is reached through a
Unix socket in the episode root (``UnixRelay`` on the host side forwards it to the episode's
spend gate; the runner listens on ``127.0.0.1:<port>`` inside the namespace and forwards
to that socket). The only ways out of the sandbox are therefore the bridge socket and the
model relay socket.

What the Linux form does NOT enforce (docs/system/mh1-design.md s. 6): CPU, memory and
process-count limits (only the wall-clock kill), a syscall filter beyond bubblewrap's
defaults, and reads of the bound system directories (``/usr``, parts of ``/etc``).
``backend()`` runs a live self-test once per process (``linux_selftest``: a write outside the
root, a write to a read-only bind, a read of a file outside every bind, the home directory,
a TCP connection to a listener on the host's loopback, and the relay path); if bubblewrap is
missing or any probe disagrees, there is no backend and every caller refuses to run (fail
closed).

``run_contained`` starts the runner in its own process group with a scrubbed environment and
kills the whole group at the wall-clock deadline.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SANDBOX_EXEC = "/usr/bin/sandbox-exec"
BWRAP_CANDIDATES = ("/usr/bin/bwrap", "/usr/local/bin/bwrap", "/bin/bwrap")
_BACKEND: dict[str, Any] = {}


def bwrap_path() -> str | None:
    env = os.environ.get("SLH_BWRAP")
    for p in ((env,) if env else ()) + BWRAP_CANDIDATES:
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def backend() -> str | None:
    """``"seatbelt"`` (macOS), ``"bwrap"`` (Linux, only after its live self-test passed) or
    None (no containment: every caller refuses)."""
    if "name" not in _BACKEND:
        name = None
        if sys.platform == "darwin" and os.path.exists(SANDBOX_EXEC):
            name = "seatbelt"
        elif sys.platform.startswith("linux") and bwrap_path():
            rep = linux_selftest()
            _BACKEND["selftest"] = rep
            name = "bwrap" if rep.get("ok") else None
        _BACKEND["name"] = name
    return _BACKEND["name"]


def backend_report() -> dict[str, Any]:
    name = backend()
    return {"backend": name, "platform": sys.platform, "bwrap": bwrap_path(),
            "selftest": _BACKEND.get("selftest")}


def sandbox_available() -> bool:
    return backend() is not None


def _q(p: str | Path) -> str:
    s = os.path.realpath(str(p))
    if '"' in s or "\\" in s:
        raise ValueError(f"unsupported path for a sandbox profile: {s!r}")
    return f'"{s}"'


# System paths an interpreter needs to read (libraries, the dyld cache, zoneinfo, locale,
# certificates). Everything else is unreadable: the user's home (``~/.slh-xp``, ``~/.ssh``),
# other repositories, ``learner_runs/.seal`` (review fix 6).
SYSTEM_READ = ("/usr/lib", "/usr/share", "/System", "/Library/Frameworks", "/private/etc",
               "/private/var/db/timezone", "/private/var/db/dyld", "/dev", "/opt/homebrew/opt",
               "/opt/homebrew/Cellar", "/opt/homebrew/lib", "/opt/homebrew/etc")
SECRET_DENY = ("~/.slh-xp", "~/.ssh", "~/.aws", "~/.config", "~/.netrc", "~/.hermes")
# Mach services a Python process and Hermes use in a sandbox (name resolution of the local
# user, the system log, notifications, the trust store for TLS). All others are denied.
MACH_ALLOW = ("com.apple.system.opendirectoryd.libinfo", "com.apple.system.logger",
              "com.apple.system.notification_center", "com.apple.logd",
              "com.apple.diagnosticd", "com.apple.trustd", "com.apple.trustd.agent",
              "com.apple.SecurityServer", "com.apple.system.opendirectoryd.membership",
              "com.apple.CoreServices.coreservicesd", "com.apple.lsd.mapdb")


def python_roots(python: Path) -> list[Path]:
    """The real installation the venv's interpreter runs from (macOS form)."""
    real = Path(os.path.realpath(python))
    return [real.parents[2]] if len(real.parents) > 2 else [real.parent]


def profile(*, writable: Sequence[Path], readonly: Sequence[Path], model_port: int | None,
            readable: Sequence[Path] | None = None, sockets: Sequence[str] = (),
            secret_roots: Sequence[Path] = ()) -> str:
    """Seatbelt profile text (last matching rule wins).

    * writes: only under ``writable``; ``readonly`` paths inside them are denied again;
    * reads (file data): only ``readable`` + ``SYSTEM_READ`` when ``readable`` is given;
      metadata stays readable (path resolution); ``SECRET_DENY`` and ``secret_roots`` are
      denied explicitly whatever else allows;
    * network: TCP only to ``localhost:<model_port>``; Unix sockets only the listed paths
      (none during replay);
    * mach lookups: only ``MACH_ALLOW``.
    """
    lines = ["(version 1)", "(allow default)", "(deny network*)"]
    for sp in sockets:
        lines.append(f"(allow network-outbound (remote unix-socket (path-literal {_q(sp)})))")
    if model_port is not None:
        lines.append(f'(allow network-outbound (remote ip "localhost:{int(model_port)}"))')
    lines += ["(deny file-write*)",
              "(allow file-write* " + " ".join(f"(subpath {_q(p)})" for p in writable)
              + ' (literal "/dev/null") (literal "/dev/zero") (literal "/dev/dtracehelper"))']
    if readonly:
        lines.append("(deny file-write* " + " ".join(
            f"(subpath {_q(p)})" if Path(p).is_dir() else f"(literal {_q(p)})"
            for p in readonly) + ")")
    if readable is not None:
        roots = [*readable, *SYSTEM_READ]
        lines += ["(deny file-read-data)",
                  '(allow file-read-data (literal "/") ' + " ".join(
                      f"(subpath {_q(p)})" for p in roots if os.path.exists(p)) + ")"]
    secrets = [Path(os.path.expanduser(p)) for p in SECRET_DENY] + list(secret_roots)
    lines.append("(deny file-read* file-write* " + " ".join(
        f"(subpath {_q(p)})" for p in secrets) + ")")
    lines += ["(deny mach-lookup)",
              "(allow mach-lookup " + " ".join(f'(global-name "{n}")' for n in MACH_ALLOW)
              + ")"]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- Linux


# Read-only system paths inside the Linux sandbox (when they exist). Top-level symlinks
# (merged /usr: /bin -> usr/bin) are recreated as symlinks, not bound.
LINUX_SYSTEM_RO = ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/lib32", "/etc/ssl",
                   "/etc/pki", "/etc/ca-certificates", "/etc/alternatives", "/etc/ld.so.cache",
                   "/etc/ld.so.conf", "/etc/ld.so.conf.d", "/etc/localtime",
                   "/etc/nsswitch.conf", "/etc/passwd", "/etc/group", "/etc/hosts",
                   "/etc/os-release", "/etc/mime.types")


def linux_python_roots(python: Path) -> list[Path]:
    """The base installation a venv's interpreter runs from (Linux form): the prefix of the
    ``home`` recorded in ``pyvenv.cfg`` (``/usr/bin`` -> ``/usr``), or of the resolved
    interpreter. Never ``/``, the home directory or one of its ancestors (refused)."""
    python = Path(python)
    cfg = python.parent.parent / "pyvenv.cfg"
    home_bin = None
    if cfg.exists():
        for line in cfg.read_text(errors="replace").splitlines():
            k, _, v = line.partition("=")
            if k.strip() == "home" and v.strip():
                home_bin = Path(os.path.realpath(v.strip()))
    if home_bin is None:
        home_bin = Path(os.path.realpath(python)).parent
    prefix = home_bin.parent if home_bin.name == "bin" else home_bin
    _check_bindable(prefix)
    return [prefix]


def _check_bindable(p: Path) -> None:
    """Refuse a bind that would expose the whole filesystem or the home directory."""
    rp = Path(os.path.realpath(p))
    home = Path(os.path.realpath(os.path.expanduser("~")))
    if rp == Path("/") or rp == home or rp in home.parents:
        raise ValueError(f"refusing to expose {rp} inside the sandbox (/, the home directory "
                         "or one of its ancestors)")
    for s in SECRET_DENY:
        sp = Path(os.path.realpath(os.path.expanduser(s)))
        if sp == rp or rp in sp.parents:
            raise ValueError(f"refusing to expose {rp}: it contains the secret path {sp}")


def symlink_chain(path: str | Path) -> list[tuple[str, str]]:
    """The symlinks a path goes through, as ``(target, link)`` pairs, outermost first. The
    binds are made at REAL paths, so every path the process uses unresolved (its argv, a
    venv's interpreter symlink, ``pyvenv.cfg`` home, e.g. ``/home -> /storage/home`` on the
    partner's nodes) gets the same links inside the namespace (``--symlink``)."""
    out: list[tuple[str, str]] = []
    p = os.path.normpath(os.path.abspath(str(path)))
    seen: set[str] = set()
    for _ in range(40):  # bounded: a symlink loop cannot hang the harness
        parts = p.split(os.sep)
        hit = None
        for k in range(2, len(parts) + 1):
            pre = os.sep.join(parts[:k]) or os.sep
            if os.path.islink(pre):
                hit = (pre, os.sep.join(parts[k:]))
                break
        if hit is None or hit[0] in seen:
            break
        link, rest = hit
        seen.add(link)
        target = os.readlink(link)
        out.append((target, link))
        real = os.path.realpath(link)
        p = os.path.normpath(os.path.join(real, rest)) if rest else real
    return out


def _venv_links(p: str) -> list[str]:
    """For an interpreter inside a venv: its symlink target and the ``home`` of its
    ``pyvenv.cfg`` (both recorded UNRESOLVED by ``python -m venv``)."""
    extra: list[str] = []
    cfg = Path(p).parent.parent / "pyvenv.cfg"
    if os.path.islink(p):
        t = os.readlink(p)
        extra.append(t if os.path.isabs(t) else os.path.join(os.path.dirname(p), t))
    if cfg.exists():
        for line in cfg.read_text(errors="replace").splitlines():
            k, _, v = line.partition("=")
            if k.strip() in ("home", "executable") and v.strip():
                extra.append(v.strip())
    return extra


@dataclass(frozen=True)
class LinuxSpec:
    """A bubblewrap invocation: binds only (see the module docstring)."""

    writable: tuple[str, ...]
    readonly: tuple[str, ...]
    readable: tuple[str, ...]
    sockets: tuple[str, ...] = ()
    secret_roots: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for p in (*self.writable, *self.readable):
            _check_bindable(Path(p))
        for sp in self.secret_roots:
            s = Path(os.path.realpath(sp))
            for p in (*self.writable, *self.readable):
                rp = Path(os.path.realpath(p))
                if rp == s or rp in s.parents or s in rp.parents:
                    raise ValueError(f"bind {rp} would expose the secret root {s}")
        for sp in self.sockets:
            rs = Path(os.path.realpath(sp))
            if not any(Path(os.path.realpath(w)) in rs.parents for w in self.writable):
                raise ValueError(f"socket {rs} is not inside a writable bind")

    def argv(self, cmd: Sequence[str], cwd: str | Path) -> list[str]:
        bw = bwrap_path()
        if bw is None:
            raise RuntimeError("bubblewrap (bwrap) not found; refusing to run uncontained")
        a = [bw, "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL",
             "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
        for d in LINUX_SYSTEM_RO:
            if os.path.islink(d) and os.path.dirname(d) == "/":
                a += ["--symlink", os.readlink(d), d]
            elif os.path.exists(d):
                a += ["--ro-bind", d, d]
        seen: set[str] = set()
        for p in self.readable:
            rp = os.path.realpath(p)
            if os.path.exists(rp) and rp not in seen:
                seen.add(rp)
                a += ["--ro-bind", rp, rp]
        for p in self.writable:
            rp = os.path.realpath(p)
            a += ["--bind", rp, rp]
        # learned state read-only AGAIN on top of the writable root (later mounts win);
        # a path that does not exist yet cannot be bound (a file created there later is an
        # extra file for the post-run hash: an incident)
        for p in self.readonly:
            rp = os.path.realpath(p)
            if os.path.exists(rp):
                a += ["--ro-bind", rp, rp]
        # symlinked ancestors of every path used unresolved (argv, binds, venv links),
        # recreated inside: the binds are at real paths (``symlink_chain``)
        used = [c for c in cmd if isinstance(c, str) and os.path.isabs(c)]
        used += [x for c in list(used) for x in _venv_links(c)]
        links: dict[str, str] = {}
        system = set(LINUX_SYSTEM_RO)
        bound = [os.path.realpath(b) for b in (*self.readable, *self.writable,
                                               *self.readonly)] + list(LINUX_SYSTEM_RO)

        def exposed(d: str) -> bool:  # the directory is already visible through a bind
            rd = os.path.realpath(d)
            return any(rd == b or rd.startswith(b.rstrip("/") + "/") for b in bound)

        for p in (*used, *self.readable, *self.writable, *self.readonly, str(cwd)):
            for target, link in symlink_chain(p):
                # a link inside a bound tree is already there (bound as it is on the host)
                if link not in system and not exposed(os.path.dirname(link)):
                    links.setdefault(link, target)
        for link in sorted(links):
            a += ["--symlink", links[link], link]
        a += ["--chdir", os.path.realpath(cwd), "--", *cmd]
        return a


def contain(*, writable: Sequence[Path], readonly: Sequence[Path], model_port: int | None,
            readable: Sequence[Path] | None = None, sockets: Sequence[str] = (),
            secret_roots: Sequence[Path] = ()) -> str | LinuxSpec:
    """The containment for this platform's backend: Seatbelt profile text (macOS) or a
    ``LinuxSpec`` (bubblewrap). Refuses (RuntimeError) when there is no backend."""
    name = backend()
    if name == "seatbelt":
        return profile(writable=writable, readonly=readonly, model_port=model_port,
                       readable=readable, sockets=sockets, secret_roots=secret_roots)
    if name == "bwrap":
        return LinuxSpec(writable=tuple(str(p) for p in writable),
                         readonly=tuple(str(p) for p in readonly),
                         readable=tuple(str(p) for p in (readable or ())),
                         sockets=tuple(str(s) for s in sockets),
                         secret_roots=tuple(str(p) for p in secret_roots))
    raise RuntimeError("no sandbox backend (sandbox-exec on macOS, a self-tested bubblewrap "
                       f"on Linux): {backend_report()}; refusing to run uncontained")


def interpreter_roots(python: Path) -> list[Path]:
    """Readable roots of an interpreter for the current backend."""
    if backend() == "bwrap":
        return linux_python_roots(python)
    return python_roots(python)


class UnixRelay:
    """Host side of the Linux model path: accepts on a Unix socket inside the episode root
    and forwards every connection to ``host:port`` (the episode's spend gate)."""

    def __init__(self, path: str | Path, host: str, port: int) -> None:
        if host not in ("127.0.0.1", "localhost"):
            raise ValueError("the relay forwards to a loopback endpoint only")
        self.path, self.host, self.port = str(path), host, int(port)
        self._srv: socket.socket | None = None
        self._stop = threading.Event()
        self.connections = 0

    def start(self) -> UnixRelay:
        if len(self.path.encode()) > 104:
            raise ValueError(f"Unix socket path too long: {self.path}")
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(self.path)
        s.listen(64)
        s.settimeout(0.5)
        self._srv = s
        threading.Thread(target=self._loop, daemon=True, name="slh-unix-relay").start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                c, _ = self._srv.accept()  # type: ignore[union-attr]
            except TimeoutError:
                continue
            except OSError:
                return
            self.connections += 1
            threading.Thread(target=self._pair, args=(c,), daemon=True).start()

    def _pair(self, c: socket.socket) -> None:
        try:
            u = socket.create_connection((self.host, self.port), timeout=30)
        except OSError:
            c.close()
            return
        pump_pair(c, u)

    def stop(self) -> None:
        self._stop.set()
        if self._srv is not None:
            try:
                self._srv.close()
            finally:
                try:
                    os.unlink(self.path)
                except OSError:
                    pass


def pump_pair(a: socket.socket, b: socket.socket) -> None:
    """Copy bytes both ways until either side closes."""
    def pump(src: socket.socket, dst: socket.socket) -> None:
        try:
            while True:
                data = src.recv(65536)
                if not data:
                    break
                dst.sendall(data)
        except OSError:
            pass
        finally:
            for s in (src, dst):
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    for s in (a, b):
        s.settimeout(None)
    t = threading.Thread(target=pump, args=(b, a), daemon=True)
    t.start()
    pump(a, b)
    t.join(5)
    a.close()
    b.close()


_PROBE = r"""
import json, os, socket, sys
root, ro, secret, port, relay, home = sys.argv[1:7]
out = {}
def tryit(name, fn):
    try:
        fn(); out[name] = "ok"
    except Exception as e:
        out[name] = "denied:" + type(e).__name__
tryit("write_root", lambda: open(os.path.join(root, "w.txt"), "w").write("x"))
tryit("write_readonly", lambda: open(ro, "w").write("x"))
tryit("write_outside", lambda: open("/var/tmp/slh-probe-" + str(os.getpid()), "w").write("x"))
tryit("read_secret", lambda: open(secret).read())
try:
    out["list_home"] = sorted(os.listdir(home))
except Exception as e:
    out["list_home"] = "denied:" + type(e).__name__
def tcp():
    s = socket.create_connection(("127.0.0.1", int(port)), timeout=3); s.close()
tryit("tcp_host_loopback", tcp)
def ext():
    s = socket.create_connection(("1.1.1.1", 53), timeout=3); s.close()
tryit("tcp_external", ext)
def via_relay():
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.settimeout(5)
    s.connect(relay); data = s.recv(16); s.close()
    if data != b"slh-probe": raise RuntimeError(repr(data))
tryit("relay", via_relay)
print(json.dumps(out))
"""

_EXPECT = {"write_root": "ok", "write_readonly": "denied", "write_outside": "denied",
           "read_secret": "denied", "tcp_host_loopback": "denied", "tcp_external": "denied",
           "relay": "ok"}


def linux_selftest(python: str | None = None, timeout_s: float = 60.0) -> dict[str, Any]:
    """Live probes of the Linux form on THIS node (fail closed on any surprise)."""
    rep: dict[str, Any] = {"ok": False, "bwrap": bwrap_path()}
    if rep["bwrap"] is None:
        rep["reason"] = "bwrap not found"
        return rep
    py = Path(python or sys.executable)
    base = Path(tempfile.mkdtemp(prefix="slh-st-", dir=os.environ.get("SLH_SANDBOX_TMP",
                                                                         "/tmp")))
    other = Path(tempfile.mkdtemp(prefix="slh-st-x-", dir=str(base.parent)))
    lst = socket.socket()
    relay = None
    try:
        root = base / "root"
        root.mkdir()
        ro = root / "ro.txt"
        ro.write_text("read-only")
        os.chmod(ro, 0o644)  # writable by mode: only the bind may stop the write
        secret = other / "secret.txt"
        secret.write_text("secret")
        probe = root / "probe.py"
        probe.write_text(_PROBE)
        lst.bind(("127.0.0.1", 0))
        lst.listen(8)
        port = lst.getsockname()[1]

        def serve() -> None:
            lst.settimeout(timeout_s)
            while True:
                try:
                    c, _ = lst.accept()
                except OSError:
                    return
                try:
                    c.sendall(b"slh-probe")
                finally:
                    c.close()

        threading.Thread(target=serve, daemon=True).start()
        relay = UnixRelay(root / "m.sock", "127.0.0.1", port).start()
        real_home = os.path.realpath(os.path.expanduser("~"))
        spec = LinuxSpec(writable=(str(root),), readonly=(str(ro),),
                         readable=tuple(str(p) for p in linux_python_roots(py)) + (
                             str(Path(py).parent.parent),),
                         sockets=(str(root / "m.sock"),))
        cmd = spec.argv([str(py), "-I", "-B", str(probe), str(root), str(ro), str(secret),
                         str(port), str(root / "m.sock"), real_home], cwd=root)
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False,
                           env={"PATH": "/usr/bin:/bin", "HOME": str(root),
                                "LANG": "C.UTF-8"})
        rep["returncode"], rep["stderr"] = p.returncode, p.stderr[-1500:]
        try:
            got = json.loads(p.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            rep["reason"] = "probe produced no result"
            return rep
        rep["probes"] = got
        bad = {k: got.get(k) for k, v in _EXPECT.items()
               if not str(got.get(k, "")).startswith(v)}
        # the home directory must not exist inside, or hold only the directories bubblewrap
        # created to reach a bind below it (never the real contents)
        lh = got.get("list_home")
        if isinstance(lh, list) and not set(lh) <= _home_components(spec, real_home):
            bad["list_home"] = lh
        if ro.read_text() != "read-only":
            bad["readonly_changed"] = True
        rep["mismatches"] = bad
        rep["ok"] = not bad and p.returncode == 0
        if bad:
            rep["reason"] = f"probe mismatches: {bad}"
        return rep
    except Exception as e:  # noqa: BLE001 - any failure means no backend (fail closed)
        rep["reason"] = f"{type(e).__name__}: {e}"
        return rep
    finally:
        if relay is not None:
            relay.stop()
        lst.close()
        shutil.rmtree(base, ignore_errors=True)
        shutil.rmtree(other, ignore_errors=True)


def _home_components(spec: LinuxSpec, home: str) -> set[str]:
    """Names bubblewrap creates directly under ``home`` to reach binds below it."""
    out = set()
    for p in (*spec.writable, *spec.readable, *LINUX_SYSTEM_RO):
        rp = os.path.realpath(p)
        if rp.startswith(home.rstrip("/") + "/"):
            out.add(rp[len(home.rstrip("/")) + 1:].split("/")[0])
    return out


# ----------------------------------------------------------------------------- running


_LIVE: set[int] = set()  # process groups of contained runs in flight (``kill_all``)
_LIVE_LOCK = threading.Lock()


def kill_all() -> int:
    """SIGKILL every contained run still in flight (an interrupted dispatch)."""
    with _LIVE_LOCK:
        pids = list(_LIVE)
    for pid in pids:
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    return len(pids)


@dataclass
class Completed:
    returncode: int | None
    killed: bool
    seconds: float
    stdout: str
    stderr: str


def scrubbed_env(extra: Mapping[str, str], *, home: Path, tmp: Path) -> dict[str, str]:
    """A minimal environment: no inherited credentials, proxies or Python settings."""
    env = {"PATH": "/usr/bin:/bin", "HOME": str(home), "TMPDIR": str(tmp), "LANG": "C.UTF-8",
           "LC_ALL": "C.UTF-8", "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
           "HERMES_DISABLE_LAZY_INSTALLS": "1", "HERMES_GATEWAY_NO_SUPERVISE": "1",
           "HERMES_CUA_TELEMETRY": "0", "DO_NOT_TRACK": "1", "PYTHONHASHSEED": "0"}
    env.update({k: str(v) for k, v in extra.items()})
    return env


def run_contained(argv: Sequence[str], *, env: Mapping[str, str], cwd: Path,
                  profile_text: str | LinuxSpec | None, deadline_s: float,
                  log_dir: Path) -> Completed:
    """Run ``argv`` contained (a Seatbelt profile text or a ``LinuxSpec``; None only when the
    caller explicitly disabled the sandbox), killing its process group at the deadline.
    Output goes to files (no pipe can block the child)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = list(argv)
    if isinstance(profile_text, LinuxSpec):
        if backend() != "bwrap":
            raise RuntimeError("a Linux containment spec needs the bubblewrap backend")
        (log_dir / "bwrap.json").write_text(json.dumps(profile_text.argv(cmd, cwd)))
        cmd = profile_text.argv(cmd, cwd)
    elif profile_text is not None:
        if backend() != "seatbelt":
            raise RuntimeError("a Seatbelt profile needs sandbox-exec (macOS)")
        pf = log_dir / "sandbox.sb"
        pf.write_text(profile_text)
        cmd = [SANDBOX_EXEC, "-f", str(pf), *cmd]
    out_p, err_p = log_dir / "stdout.txt", log_dir / "stderr.txt"
    t = time.time()
    killed = False
    with open(out_p, "wb") as so, open(err_p, "wb") as se:
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=dict(env), stdin=subprocess.DEVNULL,
                                stdout=so, stderr=se, start_new_session=True)
        with _LIVE_LOCK:
            _LIVE.add(proc.pid)
        try:
            proc.wait(timeout=deadline_s)
        except subprocess.TimeoutExpired:
            killed = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        finally:
            with _LIVE_LOCK:
                _LIVE.discard(proc.pid)
    return Completed(proc.returncode, killed, time.time() - t,
                     out_p.read_text(errors="replace")[-4000:],
                     err_p.read_text(errors="replace")[-8000:])


def rmtree(p: Path) -> None:
    from slh.system.hosts.hermes.materialize import make_writable

    if p.exists():
        make_writable(p)
        shutil.rmtree(p, ignore_errors=True)
