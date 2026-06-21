"""PCIe DMA round-trip test for ``store.alloc_host_pin_mem`` on Ascend NPU.

Verifies that host memory allocated via
``MooncakeDistributedStore.alloc_host_pin_mem`` (mmap + ``aclrtHostRegister``)
is usable for NPU PCIe DMA: fill the host buffer with a known pattern,
``aclrtMemcpy`` H2D into device memory, scribble over the host buffer, then
``aclrtMemcpy`` D2H back and check the data round-trips intact.

Unlike ``aclrtMallocHost`` (torch ``pin_memory=True``), this buffer is ordinary
pageable memory registered to the NPU, so it is ALSO registrable by RDMA
transports. RDMA registration is NOT exercised here (no NIC on this machine);
see ``docs/mds/host-pinned-rdma-registrable-buffer.md``.

Run via ``ascend_tests/run.sh`` (which sets PYTHONPATH/LD_LIBRARY_PATH), or:

    export PYTHONPATH=build/mooncake-integration
    export LD_LIBRARY_PATH=build/mooncake-integration/mooncake:\
build/mooncake-transfer-engine/src:build/mooncake-store/src:\
build/mooncake-common/src:\
build/mooncake-transfer-engine/src/transport/ascend_transport
    python3 -m unittest ascend_tests.test_alloc_host_pin_mem_pcie_dma -v
"""

from __future__ import annotations

import ctypes
import os
import unittest

from .common import default_build_dir, setup_python_env

# ACL constants (acl/acl_rt.h)
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEM_MALLOC_HUGE_FIRST = 0

DEVICE_ID = int(os.getenv("ASCEND_DEVICE_ID", "0"))


def _acl_available() -> bool:
    try:
        import acl  # noqa: F401

        return True
    except Exception:
        return False


@unittest.skipUnless(_acl_available(), "pyACL (acl) not available")
class TestAllocHostPinMemPcieDma(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_python_env(default_build_dir())

        import acl

        cls.acl = acl
        ret = acl.init()
        assert ret == 0, f"acl.init failed: {ret}"
        ret = acl.rt.set_device(DEVICE_ID)
        assert ret == 0, f"aclrtSetDevice failed: {ret}"
        cls.context, ret = acl.rt.create_context(DEVICE_ID)
        assert ret == 0, f"aclrtCreateContext failed: {ret}"

        from mooncake.store import MooncakeDistributedStore

        cls.store = MooncakeDistributedStore()

    @classmethod
    def tearDownClass(cls):
        acl = getattr(cls, "acl", None)
        if acl is None:
            return
        if getattr(cls, "context", None):
            acl.rt.destroy_context(cls.context)
        acl.rt.reset_device(DEVICE_ID)
        acl.finalize()

    def _roundtrip(self, size: int, use_huge: bool, huge_1gb: bool = False):
        acl = self.acl
        host_ptr = self.store.alloc_host_pin_mem(size, use_huge, huge_1gb)
        if use_huge and host_ptr == 0:
            self.skipTest(
                "huge pages not reserved "
                "(echo N > /proc/sys/vm/nr_hugepages)"
            )
        self.assertNotEqual(host_ptr, 0, "alloc_host_pin_mem returned 0")

        try:
            pattern = os.urandom(size)
            ctypes.memmove(host_ptr, pattern, size)

            dev_ptr, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST)
            self.assertEqual(ret, 0, "aclrtMalloc failed")
            try:
                # Host -> Device PCIe DMA
                ret = acl.rt.memcpy(dev_ptr, size, host_ptr, size,
                                    ACL_MEMCPY_HOST_TO_DEVICE)
                self.assertEqual(ret, 0, "H2D aclrtMemcpy failed")

                # Prove D2H actually repopulates the host buffer.
                ctypes.memset(host_ptr, 0, size)

                # Device -> Host PCIe DMA
                ret = acl.rt.memcpy(host_ptr, size, dev_ptr, size,
                                    ACL_MEMCPY_DEVICE_TO_HOST)
                self.assertEqual(ret, 0, "D2H aclrtMemcpy failed")

                back = ctypes.string_at(host_ptr, size)
                self.assertEqual(back, pattern,
                                 "PCIe DMA round-trip data mismatch")
            finally:
                acl.rt.free(dev_ptr)
        finally:
            self.assertTrue(self.store.free_host_pin_mem(host_ptr),
                            "free_host_pin_mem failed")

    def test_pcie_dma_roundtrip_normal_pages(self):
        self._roundtrip(4 * 1024 * 1024, use_huge=False)

    def test_pcie_dma_roundtrip_huge_pages(self):
        self._roundtrip(4 * 1024 * 1024, use_huge=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
