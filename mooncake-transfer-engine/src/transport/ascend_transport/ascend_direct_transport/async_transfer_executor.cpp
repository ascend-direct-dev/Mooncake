// Copyright 2025 Huawei Technologies Co., Ltd
// Copyright 2024 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "transport/ascend_transport/ascend_direct_transport/async_transfer_executor.h"
#include "transport/ascend_transport/ascend_direct_transport/utils.h"

#include <glog/logging.h>

#include <chrono>

#include "common.h"
#include "transfer_metadata.h"

namespace mooncake {
namespace {
constexpr int64_t kMillisToNano = 1000000;
constexpr size_t kAsyncTaskLimit = 100U;
constexpr int32_t kDefaultDisconnectTime = 1000;
constexpr int64_t kQueryPollIntervalMicros = 10;
}  // namespace

AsyncTransferExecutor::AsyncTransferExecutor(const InitParams& params)
    : TransferExecutorBase(params),
      transfer_timeout_in_nano_(params.transfer_timeout * kMillisToNano) {}

AsyncTransferExecutor::~AsyncTransferExecutor() {
    AsyncTransferExecutor::finalize();
}

int AsyncTransferExecutor::initialize() {
    int ret = initEngines();
    if (ret != 0) {
        return ret;
    }
    running_ = true;
    query_thread_ = std::thread(&AsyncTransferExecutor::queryThreadLoop, this);
    LOG(INFO) << "AsyncTransferExecutor query thread started";
    return 0;
}

void AsyncTransferExecutor::finalize() {
    if (finalized_) {
        return;
    }
    running_ = false;
    query_cv_.notify_all();
    async_task_cv_.notify_all();
    if (query_thread_.joinable()) {
        query_thread_.join();
    }
    LOG(INFO) << "AsyncTransferExecutor query thread stopped";
    cleanupConnections();
    finalizeEngines();
    finalized_ = true;
}

TransferExecutorBase::ExecuteResult AsyncTransferExecutor::execute(
    size_t local_engine_idx, const std::string& target_adxl_engine_name,
    adxl::TransferOp operation,
    const std::vector<Transport::Slice*>& slice_list) {
    if (local_engine_idx >= adxl_engines_.size()) {
        LOG(ERROR) << "Invalid local_engine_idx: " << local_engine_idx;
        return {.ret = -1, .status = adxl::FAILED, .retryable = false};
    }

    if (!params_.auto_connect) {
        int ret = checkAndConnect(local_engine_idx, target_adxl_engine_name);
        if (ret != 0) {
            return {.ret = -1, .status = adxl::FAILED, .retryable = true};
        }
    }

    auto start_time = getCurrentTimeInNano();
    for (auto* slice : slice_list) {
        slice->ascend_direct.start_time = start_time;
    }

    {
        std::unique_lock<std::mutex> lock(async_task_mutex_);
        async_task_cv_.wait(lock, [this] {
            return !running_ || active_async_tasks_ < kAsyncTaskLimit;
        });
        if (!running_) {
            return {.ret = -1, .status = adxl::FAILED, .retryable = false};
        }
        active_async_tasks_++;
    }

    std::vector<adxl::TransferOpDesc> op_descs;
    op_descs.reserve(slice_list.size());
    for (const auto* slice : slice_list) {
        adxl::TransferOpDesc op_desc{};
        op_desc.local_addr = reinterpret_cast<uintptr_t>(slice->source_addr);
        op_desc.remote_addr =
            reinterpret_cast<uintptr_t>(slice->ascend_direct.dest_addr);
        op_desc.len = slice->length;
        op_descs.emplace_back(op_desc);
    }

    adxl::TransferReq req_handle;
    auto status = adxl_engines_[local_engine_idx]->TransferAsync(
        target_adxl_engine_name.c_str(), operation, op_descs,
        adxl::TransferArgs(), req_handle);

    if (status == adxl::SUCCESS) {
        {
            std::lock_guard<std::mutex> lock(async_handle_map_mutex_);
            const uintptr_t handle_key =
                reinterpret_cast<uintptr_t>(req_handle);
            async_handle_to_engine_idx_[handle_key] = local_engine_idx;
            async_handle_to_target_engine_[handle_key] =
                target_adxl_engine_name;
        }
        for (auto* slice : slice_list) {
            slice->ascend_direct.handle = req_handle;
        }
        {
            std::unique_lock<std::mutex> lock(query_mutex_);
            query_slice_queue_.push(slice_list);
        }
        query_cv_.notify_one();
        return {.ret = 0, .status = status, .retryable = false};
    }
    {
        std::lock_guard<std::mutex> lock(async_task_mutex_);
        if (active_async_tasks_ > 0) {
            active_async_tasks_--;
        }
        async_task_cv_.notify_one();
    }
    disconnect(local_engine_idx, target_adxl_engine_name,
               kDefaultDisconnectTime);
    return {.ret = -1, .status = status, .retryable = true};
}

void AsyncTransferExecutor::queryThreadLoop() {
    std::vector<std::vector<Transport::Slice*>> pending_batches;
    while (running_) {
        fetchPendingBatches(pending_batches);
        if (pending_batches.empty()) {
            continue;
        }

        auto it = pending_batches.begin();
        while (it != pending_batches.end()) {
            auto& slice_list = *it;
            if (processOneBatch(slice_list)) {
                it = pending_batches.erase(it);
            } else {
                ++it;
            }
        }

        if (!pending_batches.empty()) {
            std::this_thread::sleep_for(
                std::chrono::microseconds(kQueryPollIntervalMicros));
        }
    }
}

void AsyncTransferExecutor::fetchPendingBatches(
    std::vector<std::vector<Transport::Slice*>>& out_batches) {
    std::unique_lock<std::mutex> lock(query_mutex_);
    if (out_batches.empty()) {
        query_cv_.wait(
            lock, [this] { return !running_ || !query_slice_queue_.empty(); });
    }
    if (!running_) {
        return;
    }
    while (!query_slice_queue_.empty()) {
        out_batches.emplace_back(std::move(query_slice_queue_.front()));
        query_slice_queue_.pop();
    }
}

bool AsyncTransferExecutor::processOneBatch(
    std::vector<Transport::Slice*>& slice_list) {
    if (slice_list.empty()) {
        return true;
    }

    auto handle =
        static_cast<adxl::TransferReq>(slice_list[0]->ascend_direct.handle);
    size_t engine_idx = 0;
    bool found_engine_idx = false;
    std::string target_adxl_engine_name;
    {
        std::lock_guard<std::mutex> lock(async_handle_map_mutex_);
        const uintptr_t handle_key = reinterpret_cast<uintptr_t>(handle);
        auto it_handle = async_handle_to_engine_idx_.find(handle_key);
        if (it_handle != async_handle_to_engine_idx_.end()) {
            engine_idx = it_handle->second;
            found_engine_idx = true;
        }
        auto it_target = async_handle_to_target_engine_.find(handle_key);
        if (it_target != async_handle_to_target_engine_.end()) {
            target_adxl_engine_name = it_target->second;
        }
    }

    auto fail_batch = [&](const std::string& message,
                          const std::string& target_adxl_engine_name) {
        LOG(ERROR) << message;
        for (auto* slice : slice_list) {
            slice->markFailed();
        }
        handleTaskFinished(handle);
        if (engine_idx < adxl_engines_.size() &&
            !target_adxl_engine_name.empty()) {
            disconnect(engine_idx, target_adxl_engine_name,
                       params_.connect_timeout);
        }
        return true;
    };

    if (!found_engine_idx) {
        return fail_batch("Cannot resolve async handle to engine index", "");
    }

    if (engine_idx >= adxl_engines_.size() ||
        engine_idx >= local_engine_contexts_.size()) {
        return fail_batch(
            "Invalid async engine index: " + std::to_string(engine_idx), "");
    }

    if (target_adxl_engine_name.empty()) {
        auto target_segment_desc =
            metadata_->getSegmentDescByID(slice_list[0]->target_id);
        if (!target_segment_desc) {
            return fail_batch(
                "Cannot find target segment descriptor for target_id: " +
                    std::to_string(slice_list[0]->target_id),
                "");
        }
        size_t remote_engine_idx = engine_idx;
        if (slice_list[0]->ascend_direct.dst_engine_id >= 0) {
            remote_engine_idx = static_cast<size_t>(
                slice_list[0]->ascend_direct.dst_engine_id);
        }
        target_adxl_engine_name =
            resolveTargetAdxlEngineName(target_segment_desc, remote_engine_idx);
        if (target_adxl_engine_name.empty()) {
            return fail_batch("Cannot resolve target adxl engine name", "");
        }
    }

    auto context_ret =
        aclrtSetCurrentContext(local_engine_contexts_[engine_idx]);
    if (context_ret != ACL_ERROR_NONE) {
        return fail_batch("Call aclrtSetCurrentContext failed, ret: " +
                              std::to_string(context_ret) +
                              ", engine_idx: " + std::to_string(engine_idx),
                          target_adxl_engine_name);
    }

    adxl::TransferStatus task_status;
    auto ret =
        adxl_engines_[engine_idx]->GetTransferStatus(handle, task_status);

    if (ret != adxl::SUCCESS || task_status == adxl::TransferStatus::FAILED) {
        return fail_batch("Get transfer status failed, ret: " +
                              std::to_string(static_cast<int>(ret)) +
                              ", errmsg: " + aclGetRecentErrMsg(),
                          target_adxl_engine_name);
    }

    if (task_status == adxl::TransferStatus::COMPLETED) {
        for (auto* slice : slice_list) {
            slice->markSuccess();
        }
        handleTaskFinished(handle);
        if (params_.use_short_connection) {
            disconnect(engine_idx, target_adxl_engine_name,
                       params_.connect_timeout);
        }
        return true;
    }

    auto now = getCurrentTimeInNano();
    if (now - slice_list[0]->ascend_direct.start_time >
        transfer_timeout_in_nano_) {
        return fail_batch("Transfer timeout", target_adxl_engine_name);
    }

    return false;
}

void AsyncTransferExecutor::handleTaskFinished(adxl::TransferReq handle) {
    std::lock_guard<std::mutex> handle_lock(async_handle_map_mutex_);
    const uintptr_t handle_key = reinterpret_cast<uintptr_t>(handle);
    async_handle_to_engine_idx_.erase(handle_key);
    async_handle_to_target_engine_.erase(handle_key);
    std::lock_guard<std::mutex> lock(async_task_mutex_);
    if (active_async_tasks_ > 0) {
        active_async_tasks_--;
    }
    async_task_cv_.notify_one();
}

}  // namespace mooncake
