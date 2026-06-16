#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${MOONCAKE_BUILD:-$ROOT/build}"

if [[ -f "${ASCEND_HOME:-}/setenv.bash" ]]; then
  # shellcheck disable=SC1091
  source "${ASCEND_HOME}/setenv.bash"
elif [[ -f /home/developer/Ascend/cann/bin/setenv.bash ]]; then
  # shellcheck disable=SC1091
  source /home/developer/Ascend/cann/bin/setenv.bash
fi

unset HCCL_INTRA_ROCE_ENABLE
export ASCEND_CONNECT_TIMEOUT="${ASCEND_CONNECT_TIMEOUT:-60000}"

echo "=== build pybind + master ==="
cmake --build "$BUILD" --target engine store mooncake_master -j"$(nproc)"

PKG="$BUILD/mooncake-integration/mooncake"
mkdir -p "$PKG"
ENGINE_SO="$(ls -1 "$BUILD"/mooncake-integration/engine*.so | head -1)"
STORE_SO="$(ls -1 "$BUILD"/mooncake-integration/store*.so | head -1)"
ln -sfn "$(realpath --relative-to="$PKG" "$ENGINE_SO")" "$PKG/$(basename "$ENGINE_SO")"
ln -sfn "$(realpath --relative-to="$PKG" "$STORE_SO")" "$PKG/$(basename "$STORE_SO")"
if [[ -f "$BUILD/mooncake-asio/libasio.so" ]]; then
  ln -sfn "$(realpath --relative-to="$PKG" "$BUILD/mooncake-asio/libasio.so")" "$PKG/libasio.so"
fi

export MOONCAKE_BUILD="$BUILD"
export PYTHONPATH="$BUILD/mooncake-integration:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$PKG:$BUILD/mooncake-transfer-engine/src/transport/ascend_transport:$BUILD/mooncake-transfer-engine/src:$BUILD/mooncake-store/src:$BUILD/mooncake-asio:${LD_LIBRARY_PATH:-}"

echo "=== run ascend_tests ==="
cd "$ROOT"
set +e
python3 -m unittest ascend_tests.test_store_tcp_te_ascend_coexist -v 2>&1 | tee /tmp/ascend_tests_last.log
rc=${PIPESTATUS[0]}
set -e
if grep -q "^OK" /tmp/ascend_tests_last.log; then
  echo "=== ascend_tests PASS ==="
  exit 0
fi
echo "=== ascend_tests FAIL (rc=$rc) ==="
exit "${rc:-1}"
