#include "tent/runtime/platform.h"
#include "tent/transport/rdma/ibv_loader.h"

#include <acl/acl.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>

extern "C" int mooncake_tent_platform_free_memory(void* addr, size_t size) {
    if (addr == nullptr) {
        return -1;
    }
    auto status = mooncake::tent::Platform::getLoader().free(addr, size);
    return status.ok() ? 0 : -1;
}

extern "C" uint64_t mooncake_tent_probe_rdma_max_mr_size() {
    auto& loader = mooncake::tent::IbvLoader::Instance();
    if (!loader.available()) {
        return 0;
    }

    const auto& verbs = loader.sym();
    int device_count = 0;
    ibv_device** devices = verbs.ibv_get_device_list(&device_count);
    if (devices == nullptr || device_count <= 0) {
        if (devices != nullptr) {
            verbs.ibv_free_device_list(devices);
        }
        return 0;
    }

    uint64_t limit = std::numeric_limits<uint64_t>::max();
    bool found = false;
    for (int index = 0; index < device_count; ++index) {
        ibv_context* context = verbs.ibv_open_device(devices[index]);
        if (context == nullptr) {
            continue;
        }
        ibv_device_attr attr = {};
        if (verbs.ibv_query_device(context, &attr) == 0 &&
            attr.max_mr_size > 0) {
            limit = std::min(limit, static_cast<uint64_t>(attr.max_mr_size));
            found = true;
        }
        verbs.ibv_close_device(context);
    }
    verbs.ibv_free_device_list(devices);
    return found ? limit : 0;
}

// Bring up ACL for a plain Rust process before a TENT engine is created.
//
// TENT's ascend_direct transport installs one HIXL engine per device and its
// initEngines() starts with aclrtGetCurrentContext/aclrtGetDevice
// (tent/src/transport/ascend/ascend_direct_transport.cpp:313-350). Store-RS
// creates the TENT engine from a process where nothing has called
// aclInit/aclrtSetDevice, so the transport is skipped with
// "Get device context failed, device may be not set." and later host-memory
// allocation/registration fails. aclInit is process-wide (once); the device
// binding is per-thread, so callers must run this on the thread that creates
// the engine.
extern "C" int mooncake_tent_prepare_ascend(int device_id) {
    static std::once_flag acl_once;
    static aclError acl_init_status = ACL_ERROR_NONE;
    std::call_once(acl_once, []() {
        aclError ret = aclInit(nullptr);
        if (ret != ACL_ERROR_NONE) {
            // Another component may have initialised ACL already; that is
            // fine as long as the runtime is usable.
            uint32_t device_count = 0;
            if (aclrtGetDeviceCount(&device_count) != ACL_ERROR_NONE) {
                acl_init_status = ret;
                LOG(ERROR) << "aclInit failed, ret=" << ret
                           << ", errmsg: " << aclGetRecentErrMsg();
            }
        }
    });
    if (acl_init_status != ACL_ERROR_NONE) {
        return -1;
    }
    if (aclrtSetDevice(device_id) != ACL_ERROR_NONE) {
        LOG(ERROR) << "aclrtSetDevice(" << device_id
                   << ") failed, errmsg: " << aclGetRecentErrMsg();
        return -1;
    }
    return 0;
}
