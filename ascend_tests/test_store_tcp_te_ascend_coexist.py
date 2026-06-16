"""
Single-process Store (TCP) + TE (ascend_direct) coexistence test.

Store and TE initiator run in the main Python process. ADXL target runs in a
child process (same pattern as transfer_engine_ascend_direct_perf) because
two ADXL engines connecting in one process is not supported reliably.

Device must be set via torch.npu.set_device() before TransferEngine.initialize().
"""

from __future__ import annotations

import multiprocessing as mp
import os
import subprocess
import time
import unittest

from ascend_tests.common import (
    artifacts_ready,
    default_build_dir,
    has_hccn_conf,
    has_npu_runtime,
    host_ip,
    master_addr,
    metadata_server,
    setup_python_env,
    find_master_binary,
)

BLOCK_SIZE = int(os.getenv("ASCEND_TEST_BLOCK_SIZE", str(32 * 1024)))
GLOBAL_SEGMENT_SIZE = int(os.getenv("ASCEND_TEST_GLOBAL_SEGMENT", str(256 * 1024 * 1024)))
LOCAL_BUFFER_SIZE = int(os.getenv("ASCEND_TEST_LOCAL_BUFFER", str(64 * 1024 * 1024)))
TARGET_DEVICE = int(os.getenv("TARGET_DEVICE", "0"))
INITIATOR_DEVICE = int(os.getenv("INITIATOR_DEVICE", "1"))
TRANSFER_RETRIES = int(os.getenv("ASCEND_TRANSFER_RETRIES", "5"))
TRANSFER_RETRY_SLEEP_SEC = float(os.getenv("ASCEND_TRANSFER_RETRY_SLEEP", "2"))
TARGET_WARMUP_SEC = int(os.getenv("ASCEND_TARGET_WARMUP_SEC", "10"))

BUILD_DIR = default_build_dir()
READY, ARTIFACT_MSG = artifacts_ready(BUILD_DIR)
setup_python_env(BUILD_DIR)

_master_proc = None
_target_proc: mp.Process | None = None
_result_queue: mp.Queue | None = None
_target_cmd_queue: mp.Queue | None = None
_target_rsp_queue: mp.Queue | None = None


def _run_adxl_target(
    host: str, device: int, out_queue: mp.Queue, cmd_queue: mp.Queue, rsp_queue: mp.Queue
) -> None:
    setup_python_env(BUILD_DIR)
    if "HCCL_INTRA_ROCE_ENABLE" in os.environ:
        del os.environ["HCCL_INTRA_ROCE_ENABLE"]

    import torch
    import torch_npu  # noqa: F401
    from mooncake.engine import TransferEngine

    torch.npu.set_device(device)
    te = TransferEngine()
    rc = te.initialize(
        f"{host}:12345",
        metadata_server(),
        "ascend_direct",
        "",
    )
    if rc != 0:
        out_queue.put({"init_rc": rc})
        return

    buf_tensor = torch.zeros(BLOCK_SIZE, dtype=torch.uint8, device=f"npu:{device}")
    reg_rc = te.register_memory(buf_tensor.data_ptr(), BLOCK_SIZE)
    if reg_rc != 0:
        out_queue.put({"init_rc": 0, "reg_rc": reg_rc})
        return

    out_queue.put(
        {
            "init_rc": 0,
            "reg_rc": 0,
            "segment_id": f"{host}:{te.get_rpc_port()}",
        }
    )
    while True:
        cmd = cmd_queue.get()
        if cmd is None:
            break
        if cmd.get("cmd") == "verify":
            torch.npu.synchronize()
            pattern = int(cmd["pattern"])
            ok = bool((buf_tensor == pattern).all().item())
            rsp_queue.put({"verify_ok": ok, "pattern": pattern})


def _start_master() -> None:
    global _master_proc

    master_bin = find_master_binary(BUILD_DIR)
    if master_bin is None:
        raise RuntimeError("mooncake_master not found")
    port = master_addr().split(":")[1]
    _master_proc = subprocess.Popen(
        [str(master_bin), f"--port={port}", "--eviction_high_watermark_ratio=0.95"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    if _master_proc.poll() is not None:
        raise RuntimeError(f"mooncake_master exited early with code {_master_proc.returncode}")


def _stop_master() -> None:
    global _master_proc
    if _master_proc is None:
        return
    _master_proc.terminate()
    try:
        _master_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _master_proc.kill()
        _master_proc.wait(timeout=5)
    _master_proc = None


def setUpModule() -> None:
    if "HCCL_INTRA_ROCE_ENABLE" in os.environ:
        del os.environ["HCCL_INTRA_ROCE_ENABLE"]
    _start_master()


def tearDownModule() -> None:
    global _target_proc
    if _target_proc is not None and _target_proc.is_alive():
        _target_proc.terminate()
        _target_proc.join(timeout=5)
        if _target_proc.is_alive():
            _target_proc.kill()
    _stop_master()


@unittest.skipUnless(READY, ARTIFACT_MSG or "build artifacts missing")
@unittest.skipUnless(has_hccn_conf(), "/etc/hccn.conf required for HCCS")
@unittest.skipUnless(has_npu_runtime(), "torch.npu not available")
class TestStoreTcpTeAscendCoexist(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        cls.host = host_ip()
        cls.local_hostname = os.getenv("LOCAL_HOSTNAME", f"{cls.host}:17820")
        cls._torch = torch

    def _setup_store(self):
        from mooncake.store import MooncakeDistributedStore

        store = MooncakeDistributedStore()
        rc = store.setup(
            self.local_hostname,
            metadata_server(),
            GLOBAL_SEGMENT_SIZE,
            LOCAL_BUFFER_SIZE,
            "tcp",
            "",
            master_addr(),
        )
        self.assertEqual(rc, 0, "Store setup with protocol=tcp failed")
        return store

    def _store_put_get(self, store, tag: str) -> None:
        key = f"ascend_tests_{tag}"
        value = f"store-tcp-{tag}".encode()
        self.assertEqual(store.put(key, value), 0, f"store put failed: {tag}")
        got = store.get(key)
        self.assertEqual(got, value, f"store get mismatch: {tag}")

    def test_store_tcp_and_te_ascend_direct_coexist(self) -> None:
        global _target_proc, _result_queue, _target_cmd_queue, _target_rsp_queue
        from mooncake.engine import TransferEngine

        store = self._setup_store()
        self._store_put_get(store, "phase1_baseline")

        ctx = mp.get_context("spawn")
        _result_queue = ctx.Queue()
        _target_cmd_queue = ctx.Queue()
        _target_rsp_queue = ctx.Queue()
        _target_proc = ctx.Process(
            target=_run_adxl_target,
            args=(self.host, TARGET_DEVICE, _result_queue, _target_cmd_queue, _target_rsp_queue),
            daemon=True,
        )
        _target_proc.start()

        try:
            target_info = _result_queue.get(timeout=60)
        except Exception as exc:
            self.fail(f"ADXL target process did not become ready: {exc}")

        self.assertEqual(target_info.get("init_rc"), 0, "target TE initialize failed")
        self.assertEqual(target_info.get("reg_rc"), 0, "target register_memory failed")
        segment_id = target_info.get("segment_id")
        self.assertTrue(segment_id)

        self._store_put_get(store, "phase3_with_adxl_target")

        time.sleep(TARGET_WARMUP_SEC)

        import torch_npu  # noqa: F401

        self._torch.npu.set_device(INITIATOR_DEVICE)
        initiator = TransferEngine()
        rc = initiator.initialize(
            f"{self.host}:12346",
            metadata_server(),
            "ascend_direct",
            "",
        )
        self.assertEqual(rc, 0, "initiator TE initialize failed")

        src_tensor = self._torch.zeros(
            BLOCK_SIZE, dtype=self._torch.uint8, device=f"npu:{INITIATOR_DEVICE}"
        )
        src_tensor.fill_(0xAB)
        rc = initiator.register_memory(src_tensor.data_ptr(), BLOCK_SIZE)
        self.assertEqual(rc, 0, "initiator register_memory failed")

        dst = initiator.get_first_buffer_address(segment_id)
        self.assertNotEqual(dst, 0, f"no remote buffer on segment {segment_id}")

        last_rc = -1
        for _ in range(TRANSFER_RETRIES):
            last_rc = initiator.transfer_sync_write(
                segment_id, src_tensor.data_ptr(), dst, BLOCK_SIZE
            )
            if last_rc == 0:
                break
            time.sleep(TRANSFER_RETRY_SLEEP_SEC)
        self.assertEqual(last_rc, 0, "transfer_sync_write failed after retries")

        pattern = 0xAB
        _target_cmd_queue.put({"cmd": "verify", "pattern": pattern})
        try:
            verify_result = _target_rsp_queue.get(timeout=30)
        except Exception as exc:
            self.fail(f"ADXL target verify did not respond: {exc}")
        self.assertTrue(
            verify_result.get("verify_ok"),
            f"ADXL transfer data mismatch on target npu:{TARGET_DEVICE}, "
            f"expected all bytes=0x{pattern:02x}",
        )

        store.close()


if __name__ == "__main__":
    unittest.main()
