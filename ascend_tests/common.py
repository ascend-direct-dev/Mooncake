"""Shared helpers for ascend_tests."""

from __future__ import annotations

import glob
import os
import socket
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_build_dir() -> Path:
    env = os.getenv("MOONCAKE_BUILD")
    if env:
        return Path(env)
    return repo_root() / "build"


def host_ip() -> str:
    return os.getenv("HOST_IP", socket.gethostbyname(socket.gethostname()))


def master_addr() -> str:
    port = os.getenv("MASTER_PORT", "50051")
    host = os.getenv("MASTER_HOST", "127.0.0.1")
    return f"{host}:{port}"


def metadata_server() -> str:
    return os.getenv("MC_METADATA_SERVER", "P2PHANDSHAKE")


def find_master_binary(build_dir: Path) -> Path | None:
    candidates = [
        build_dir / "mooncake-store/src/mooncake_master",
        build_dir / "mooncake-store/mooncake_master",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def find_pybind_modules(build_dir: Path) -> tuple[Path | None, Path | None]:
    integration = build_dir / "mooncake-integration"
    engine = glob.glob(str(integration / "engine*.so"))
    store = glob.glob(str(integration / "store*.so"))
    engine_path = Path(engine[0]) if engine else None
    store_path = Path(store[0]) if store else None
    return engine_path, store_path


def setup_python_env(build_dir: Path) -> None:
    integration = build_dir / "mooncake-integration"
    pkg = integration / "mooncake"
    pkg.mkdir(parents=True, exist_ok=True)

    engine_so, store_so = find_pybind_modules(build_dir)
    if engine_so:
        link = pkg / engine_so.name
        if not link.exists():
            link.symlink_to(os.path.relpath(engine_so, pkg))
    if store_so:
        link = pkg / store_so.name
        if not link.exists():
            link.symlink_to(os.path.relpath(store_so, pkg))

    libasio = build_dir / "mooncake-asio/libasio.so"
    if libasio.is_file():
        link = pkg / "libasio.so"
        if not link.exists():
            link.symlink_to(os.path.relpath(libasio, pkg))

    ld_paths = [
        str(pkg),
        str(build_dir / "mooncake-transfer-engine/src/transport/ascend_transport"),
        str(build_dir / "mooncake-transfer-engine/src"),
        str(build_dir / "mooncake-store/src"),
        str(build_dir / "mooncake-asio"),
    ]
    existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(ld_paths + ([existing_ld] if existing_ld else []))

    py_path = str(integration)
    existing_py = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = ":".join([py_path] + ([existing_py] if existing_py else []))

    if str(integration) not in sys.path:
        sys.path.insert(0, str(integration))


def has_hccn_conf() -> bool:
    return os.path.isfile("/etc/hccn.conf")


def has_npu_runtime() -> bool:
    try:
        import torch

        return torch.npu.is_available()
    except Exception:
        return False


def artifacts_ready(build_dir: Path) -> tuple[bool, str]:
    engine_so, store_so = find_pybind_modules(build_dir)
    master = find_master_binary(build_dir)
    missing = []
    if not engine_so:
        missing.append("engine*.so")
    if not store_so:
        missing.append("store*.so")
    if not master:
        missing.append("mooncake_master")
    if missing:
        return False, "missing build artifacts: " + ", ".join(missing)
    return True, ""
