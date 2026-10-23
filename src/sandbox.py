"""Build sandboxed command lines for shell strings and argv lists.

This module only constructs commands; it does not execute them. Given a
command and a :class:`SandboxPolicy`, it returns an equivalent command whose
filesystem writes are limited to the policy's roots plus temporary and device
paths. Two backends are supported:

* **macOS** — ``sandbox-exec`` (seatbelt) with a generated, closed-by-default
  ``.sbpl`` profile. The profile denies everything and then allows the
  process, sysctl, IPC, PTY, and preference operations commonly needed by
  build tools. File reads are unrestricted; writes are limited to temporary
  locations, device files, and the policy's roots. The binary is invoked by
  absolute path to avoid PATH injection.
* **Linux** — ``bwrap`` (bubblewrap). The filesystem is mounted read-only,
  policy roots are rebound read-write, ``/tmp`` is a fresh tmpfs, and the
  network namespace is unshared unless the policy allows networking.

Native Windows is unsupported. When no backend is available,
:func:`sandbox_backend` returns ``"none"`` and the wrapping functions return
the original command unchanged, leaving the caller to apply its own fallback
policy.

Seccomp-bpf and Landlock are not used on Linux: Python cannot install those
kernel policies from userspace, so isolation relies on bubblewrap namespaces.

``SandboxPolicy`` denies networking by default. :func:`build_exec_command`
opts into it by default because coding tasks commonly need package downloads.
"""

from __future__ import annotations

import os
import platform
import shutil
import tempfile
from dataclasses import dataclass, field

# Invoke by absolute path so a hostile PATH cannot substitute another binary.
_MACOS_SANDBOX_EXEC = "/usr/bin/sandbox-exec"


@dataclass(frozen=True)
class SandboxPolicy:
    """Filesystem and network restrictions for a sandboxed command.

    ``writable_roots`` lists additional absolute directories the command may
    modify. The sandbox also permits writes to temporary and device paths such
    as ``/tmp``. Reads are unrestricted so builds can access headers,
    dependencies, and toolchains. Network access is denied unless
    ``allow_network`` is true.
    """

    writable_roots: tuple[str, ...] = ()
    allow_network: bool = False

    @classmethod
    def for_workspace(
        cls, workspace: str | os.PathLike[str], *, allow_network: bool = False
    ) -> "SandboxPolicy":
        """Return a policy whose only writable root is ``workspace``."""
        root = os.path.abspath(str(workspace))
        return cls(writable_roots=(root,), allow_network=allow_network)

    def normalized_roots(self) -> list[str]:
        """Return absolute paths for non-empty writable roots."""
        return [os.path.abspath(r) for r in self.writable_roots if r]


def sandbox_backend() -> str:
    """Return the active sandbox backend name.

    The result is ``"seatbelt"``, ``"bwrap"``, or ``"none"``. macOS uses
    seatbelt only when ``/usr/bin/sandbox-exec`` exists. Linux uses bubblewrap
    only when ``bwrap`` is on ``PATH``. Other platforms, or a missing binary,
    degrade to ``"none"``.
    """
    system = platform.system()
    if system == "Darwin" and os.path.exists(_MACOS_SANDBOX_EXEC):
        return "seatbelt"
    if system == "Linux" and shutil.which("bwrap"):
        return "bwrap"
    return "none"


# Closed-by-default seatbelt policy. Unlike an ``(allow default)`` profile
# with a write fence, this denies everything and re-grants only what common
# build tools need: process creation, selected sysctls, IPC for Python
# multiprocessing and OpenMP, PTYs, and user preferences. Reads, writable
# roots, and optional network access are added by ``_seatbelt_profile``.
_SEATBELT_BASE_POLICY = r"""(version 1)

; Closed-by-default sandbox policy.

; start with closed-by-default
(deny default)

; child processes inherit the policy of their parent
(allow process-exec)
(allow process-fork)
(allow signal (target same-sandbox))

; process-info
(allow process-info* (target same-sandbox))

(allow file-write-data
  (require-all
    (path "/dev/null")
    (vnode-type CHARACTER-DEVICE)))

; sysctls permitted.
(allow sysctl-read
  (sysctl-name "hw.activecpu")
  (sysctl-name "hw.busfrequency_compat")
  (sysctl-name "hw.byteorder")
  (sysctl-name "hw.cacheconfig")
  (sysctl-name "hw.cachelinesize_compat")
  (sysctl-name "hw.cpufamily")
  (sysctl-name "hw.cpufrequency_compat")
  (sysctl-name "hw.cputype")
  (sysctl-name "hw.l1dcachesize_compat")
  (sysctl-name "hw.l1icachesize_compat")
  (sysctl-name "hw.l2cachesize_compat")
  (sysctl-name "hw.l3cachesize_compat")
  (sysctl-name "hw.logicalcpu_max")
  (sysctl-name "hw.machine")
  (sysctl-name "hw.model")
  (sysctl-name "hw.memsize")
  (sysctl-name "hw.ncpu")
  (sysctl-name "hw.nperflevels")
  ; CPU feature detection; fingerprinting is not a concern here.
  (sysctl-name-prefix "hw.optional.arm.")
  (sysctl-name-prefix "hw.optional.armv8_")
  (sysctl-name "hw.packages")
  (sysctl-name "hw.pagesize_compat")
  (sysctl-name "hw.pagesize")
  (sysctl-name "hw.physicalcpu")
  (sysctl-name "hw.physicalcpu_max")
  (sysctl-name "hw.logicalcpu")
  (sysctl-name "hw.cpufrequency")
  (sysctl-name "hw.tbfrequency_compat")
  (sysctl-name "hw.vectorunit")
  (sysctl-name "machdep.cpu.brand_string")
  (sysctl-name "kern.argmax")
  (sysctl-name "kern.hostname")
  (sysctl-name "kern.maxfilesperproc")
  (sysctl-name "kern.maxproc")
  (sysctl-name "kern.osproductversion")
  (sysctl-name "kern.osrelease")
  (sysctl-name "kern.ostype")
  (sysctl-name "kern.osvariant_status")
  (sysctl-name "kern.osversion")
  (sysctl-name "kern.secure_kernel")
  (sysctl-name "kern.usrstack64")
  (sysctl-name "kern.version")
  (sysctl-name "sysctl.proc_cputype")
  (sysctl-name "vm.loadavg")
  (sysctl-name-prefix "hw.perflevel")
  (sysctl-name-prefix "kern.proc.pgrp.")
  (sysctl-name-prefix "kern.proc.pid.")
  (sysctl-name-prefix "net.routetable.")
)

; Allow Java to read some CPU info (misclassified as a "write").
(allow sysctl-write
  (sysctl-name "kern.grade_cputype"))

; IOKit
(allow iokit-open
  (iokit-registry-entry-class "RootDomainUserClient")
)

; needed to look up user info
(allow mach-lookup
  (global-name "com.apple.system.opendirectoryd.libinfo")
)

; Needed for Python multiprocessing on macOS for the SemLock
(allow ipc-posix-sem)

; Needed for PyTorch/libomp on macOS to register OpenMP runtimes.
(allow ipc-posix-shm-read-data
  ipc-posix-shm-write-create
  ipc-posix-shm-write-unlink
  (ipc-posix-name-regex #"^/__KMP_REGISTERED_LIB_[0-9]+$"))

(allow mach-lookup
  (global-name "com.apple.PowerManagement.control")
)

; allow openpty()
(allow pseudo-tty)
(allow file-read* file-write* file-ioctl (literal "/dev/ptmx"))
(allow file-read* file-write*
  (require-all
    (regex #"^/dev/ttys[0-9]+")
    (extension "com.apple.sandbox.pty")))
(allow file-ioctl (regex #"^/dev/ttys[0-9]+"))

; allow readonly user preferences
(allow ipc-posix-shm-read* (ipc-posix-name-prefix "apple.cfprefs."))
(allow mach-lookup
  (global-name "com.apple.cfprefsd.daemon")
  (global-name "com.apple.cfprefsd.agent")
  (local-name "com.apple.cfprefsd.agent"))
(allow user-preference-read)
"""

# Appended only when networking is allowed. Grants outbound and inbound
# traffic plus the DNS, TLS, and system-service lookups needed to establish
# connections under a deny-by-default profile.
_SEATBELT_NETWORK_POLICY = r"""
; network (only when allow_network)
(allow network-outbound)
(allow network-inbound)
(allow system-socket)
(allow mach-lookup
  (global-name "com.apple.SystemConfiguration.DNSConfiguration")
  (global-name "com.apple.SystemConfiguration.configd")
  (global-name "com.apple.SecurityServer")
  (global-name "com.apple.networkd")
  (global-name "com.apple.ocspd")
  (global-name "com.apple.trustd.agent")
  (global-name "com.apple.bsd.dirhelper")
  (global-name "com.apple.system.opendirectoryd.membership"))
"""


def _seatbelt_profile(policy: SandboxPolicy) -> str:
    """Build the seatbelt profile for ``policy``.

    Combines the closed-by-default base with broad reads, writes to temporary
    and device paths, writes under the policy roots, and optional networking.
    """
    parts = [
        _SEATBELT_BASE_POLICY.rstrip(),
        "",
        "; Broad read-only filesystem access for headers and dependencies.",
        "(allow file-read*)",
        "",
        "; Temporary and device paths that builds commonly write to.",
        '(allow file-write* (subpath "/tmp"))',
        '(allow file-write* (subpath "/private/tmp"))',
        '(allow file-write* (subpath "/private/var/folders"))',
        '(allow file-write* (literal "/dev/null"))',
        '(allow file-write* (literal "/dev/stdout"))',
        '(allow file-write* (literal "/dev/stderr"))',
        '(allow file-write-data (regex #"^/dev/tty"))',
        "",
        "; Policy-provided writable roots.",
    ]
    for root in policy.normalized_roots():
        # Escape characters that are special inside an SBPL string literal.
        escaped = root.replace("\\", "\\\\").replace('"', '\\"')
        parts.append(f'(allow file-write* (subpath "{escaped}"))')
    if policy.allow_network:
        parts.append(_SEATBELT_NETWORK_POLICY.rstrip())
    return "\n".join(parts) + "\n"


def _write_seatbelt_profile(policy: SandboxPolicy) -> str:
    """Write a seatbelt profile to a temporary file and return its path.

    The caller owns the file and must remove it after execution, normally via
    :meth:`WrappedCommand.cleanup`.
    """
    fd, path = tempfile.mkstemp(prefix="minicode-sb-", suffix=".sbpl")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(_seatbelt_profile(policy))
    except Exception:
        os.unlink(path)
        raise
    return path


@dataclass
class WrappedCommand:
    """A command plus the backend state needed to execute and clean it up.

    ``argv`` is the command to launch, ``cleanup_path`` is a temporary file to
    remove afterwards, ``backend`` records which backend produced the command,
    and ``extra_env`` holds environment overrides to apply.
    """

    argv: list[str]
    cleanup_path: str | None = None
    backend: str = "none"
    extra_env: dict[str, str] = field(default_factory=dict)

    def cleanup(self) -> None:
        """Remove the temporary artifact, if present.

        Missing files and unlink errors are ignored.
        """
        if self.cleanup_path and os.path.exists(self.cleanup_path):
            try:
                os.unlink(self.cleanup_path)
            except OSError:
                pass


def wrap_argv_command(
    inner_argv: list[str],
    policy: SandboxPolicy,
) -> WrappedCommand:
    """Wrap a tokenized command in the active sandbox backend.

    Use this for commands that are not shell strings, such as
    ``[sys.executable, "/tmp/script.py"]``. When no backend is available, the
    original argv is returned unchanged with ``backend="none"``.

    Callers must call :meth:`WrappedCommand.cleanup` after execution when
    ``cleanup_path`` is set.
    """
    backend = sandbox_backend()

    if backend == "seatbelt":
        profile_path = _write_seatbelt_profile(policy)
        argv = [_MACOS_SANDBOX_EXEC, "-f", profile_path, *inner_argv]
        return WrappedCommand(argv=argv, cleanup_path=profile_path, backend=backend)

    if backend == "bwrap":
        argv = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc"]
        argv += ["--tmpfs", "/tmp"]
        for root in policy.normalized_roots():
            # bwrap rejects bind sources that do not exist.
            if os.path.exists(root):
                argv += ["--bind", root, root]
        if not policy.allow_network:
            argv += ["--unshare-net"]
        argv += ["--die-with-parent", *inner_argv]
        return WrappedCommand(argv=argv, backend=backend)

    return WrappedCommand(argv=list(inner_argv), backend="none")


def wrap_shell_command(
    command: str,
    policy: SandboxPolicy,
    *,
    shell: str = "/bin/bash",
) -> WrappedCommand:
    """Wrap a shell command string in the active sandbox backend.

    The command is passed to ``shell -c``. When no backend is available, the
    result falls back to ``[shell, "-c", command]`` with ``backend="none"``;
    the caller is responsible for applying a safe fallback policy.
    """
    return wrap_argv_command([shell, "-c", command], policy)


def sandbox_available() -> bool:
    """Return whether a sandbox backend is currently available."""
    return sandbox_backend() != "none"


def sandbox_enabled() -> bool:
    """Return whether command sandboxing is enabled.

    Sandboxing is on by default. Setting ``MINICODE_SANDBOX`` to ``0``,
    ``false``, ``off``, or ``no`` disables command wrapping. Disabling only
    bypasses wrapping; it does not change any other tool policy.
    """
    raw = os.environ.get("MINICODE_SANDBOX", "").strip().lower()
    return raw not in ("0", "false", "off", "no")


def build_exec_command(
    *,
    command: str | None = None,
    argv: list[str] | None = None,
    workspace: str | os.PathLike[str],
    allow_network: bool = True,
    shell: str = "/bin/bash",
) -> WrappedCommand:
    """Build the command a tool executor should run.

    Exactly one of ``command`` (a shell string) or ``argv`` (a token list)
    must be provided. When sandboxing is enabled and a backend is available,
    the command is wrapped with ``workspace`` as its only writable root.
    Otherwise the command is returned unchanged with ``backend`` set to
    ``"disabled"`` or ``"none"``.

    ``allow_network`` defaults to ``True`` because coding tasks commonly need
    package downloads. Pass ``False`` for strict network isolation; it does not
    change the filesystem restriction applied by the sandbox.
    """
    if (command is None) == (argv is None):
        raise ValueError("provide exactly one of command= or argv=")

    if not sandbox_enabled():
        bare = [shell, "-c", command] if command is not None else list(argv or [])
        return WrappedCommand(argv=bare, backend="disabled")

    policy = SandboxPolicy.for_workspace(workspace, allow_network=allow_network)
    if command is not None:
        return wrap_shell_command(command, policy, shell=shell)
    return wrap_argv_command(list(argv or []), policy)


def describe_backend() -> str:
    """Return a short backend description for logs and prompts."""
    backend = sandbox_backend()
    return {
        "seatbelt": "macOS seatbelt (sandbox-exec)",
        "bwrap": "Linux bubblewrap (bwrap)",
        "none": "no sandbox (degraded: approval-first)",
    }[backend]
