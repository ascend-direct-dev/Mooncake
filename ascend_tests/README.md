# ascend_tests: Store TCP + TE Ascend Direct (Python API)

Single-process integration test for `-DUSE_ASCEND_DIRECT=ON -DUSE_TCP=ON` builds.

Store and TE initiator share the main Python process. ADXL target runs in a
child process (same layout as `transfer_engine_ascend_direct_perf`) so HCCS
connect is reliable across two NPU devices.

## What it verifies

| Component | Python API | Transport |
|-----------|------------|-----------|
| Mooncake Store | `MooncakeDistributedStore.setup(..., "tcp", ...)` | TCP |
| Transfer Engine | `TransferEngine.initialize(..., "ascend_direct", "")` | ADXL / HCCS |

Store and TE initiator use the main process; ADXL target uses a child process.

## Prerequisites

1. Build with Ascend Direct and TCP:

   ```bash
   cmake -B build -DUSE_ASCEND_DIRECT=ON -DUSE_TCP=ON -DWITH_STORE=ON -DWITH_TE=ON
   cmake --build build -j
   ```

2. CANN / `torch_npu` installed; NPU visible to `torch.npu.is_available()`.

3. `/etc/hccn.conf` present (same-node HCCS). Dual-card example:

   ```
   address_0=172.31.255.238
   address_1=172.31.255.239
   ```

4. Set NPU device in **user code** before `TransferEngine.initialize()`:

   ```python
   import torch
   torch.npu.set_device(0)  # or 1 for initiator on second card
   ```

   Mooncake reads the current device via `aclrtGetDevice()`; it does not call `aclrtSetDevice()`.

5. Do not set `HCCL_INTRA_ROCE_ENABLE` unless testing RoCE explicitly.

## Run

```bash
bash ascend_tests/run.sh
```

Environment overrides:

| Variable | Default | Meaning |
|----------|---------|---------|
| `MOONCAKE_BUILD` | `build/` | CMake build directory |
| `HOST_IP` | auto | Host IP for TE segments |
| `TARGET_DEVICE` | `0` | `torch.npu.set_device` for ADXL target child process |
| `INITIATOR_DEVICE` | `1` | `torch.npu.set_device` for ADXL initiator (main process) |
| `MASTER_PORT` | `50051` | `mooncake_master` port (started by test) |

## Expected log keywords

- Store: `Transfer engine auto discovery is disabled for protocol: tcp`, `TcpTransport: listen`
- TE: `install AscendDirectTransport`, `Success to initialize adxl engine`
- Transfer: `Connected to segment` (may take ~1–2s on HCCS)

## Notes

- `segment_id` for initiator is target **RPC listening port** (`host:rpc_port`), not the ADXL port.
- Same-machine target/initiator must use different `TARGET_DEVICE` / `INITIATOR_DEVICE` to avoid ADXL `103900`.
- Process teardown may abort with `corrupted size vs. prev_size`; `run.sh` treats unittest `OK` as pass.
