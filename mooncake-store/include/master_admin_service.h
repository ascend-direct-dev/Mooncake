#pragma once

#include <csignal>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include <ylt/coro_http/coro_http_server.hpp>

#include "ha/ha_types.h"

namespace mooncake {

extern const uint64_t kMetricReportIntervalSeconds;

class WrappedMasterService;

class MasterAdminServer {
   public:
    MasterAdminServer(uint16_t http_port, bool enable_metric_reporting);

    ~MasterAdminServer();

    bool Start();

    void Stop();

    void SetRuntimeState(ha::MasterRuntimeState state);

    void SetObservedLeader(const std::optional<ha::MasterView>& leader_view);

    void SetServiceDelegate(std::shared_ptr<WrappedMasterService> service);

    void SetServiceAvailable(bool available);

   private:
    struct RuntimeSnapshot {
        ha::MasterRuntimeState state = ha::MasterRuntimeState::kStarting;
        std::optional<ha::MasterView> leader_view;
        std::shared_ptr<WrappedMasterService> service;
        bool service_available = false;
    };

    RuntimeSnapshot SnapshotState() const;

    std::string BuildMetricsText() const;

    std::string BuildTenantQuotaMetricsText() const;

    std::string BuildMetricsSummaryText() const;

    std::shared_ptr<WrappedMasterService> GetActiveService() const;

    template <typename Handler>
    void WithActiveService(coro_http::coro_http_response& resp,
                           Handler&& handler) const;

    void HandleMetrics(coro_http::coro_http_request& req,
                       coro_http::coro_http_response& resp);
    void HandleMetricsSummary(coro_http::coro_http_request& req,
                              coro_http::coro_http_response& resp);
    void HandleHealth(coro_http::coro_http_request& req,
                      coro_http::coro_http_response& resp);
    void HandleRole(coro_http::coro_http_request& req,
                    coro_http::coro_http_response& resp);
    void HandleHaStatus(coro_http::coro_http_request& req,
                        coro_http::coro_http_response& resp);
    void HandleLeader(coro_http::coro_http_request& req,
                      coro_http::coro_http_response& resp);
    void HandleQueryKey(coro_http::coro_http_request& req,
                        coro_http::coro_http_response& resp);
    void HandleGetAllKeys(coro_http::coro_http_request& req,
                          coro_http::coro_http_response& resp);
    void HandleGetAllSegments(coro_http::coro_http_request& req,
                              coro_http::coro_http_response& resp);
    void HandleGetSegmentsDetail(coro_http::coro_http_request& req,
                                 coro_http::coro_http_response& resp);
    void HandleQuerySegment(coro_http::coro_http_request& req,
                            coro_http::coro_http_response& resp);
    void HandleCreateDrainJob(coro_http::coro_http_request& req,
                              coro_http::coro_http_response& resp);
    void HandleQueryDrainJob(coro_http::coro_http_request& req,
                             coro_http::coro_http_response& resp);
    void HandleCancelDrainJob(coro_http::coro_http_request& req,
                              coro_http::coro_http_response& resp);
    void HandleSegmentStatus(coro_http::coro_http_request& req,
                             coro_http::coro_http_response& resp);
    void HandleBatchQueryKeys(coro_http::coro_http_request& req,
                              coro_http::coro_http_response& resp);
    void HandleGetTenantQuotas(coro_http::coro_http_request& req,
                               coro_http::coro_http_response& resp);
    void HandleUpsertTenantQuota(coro_http::coro_http_request& req,
                                 coro_http::coro_http_response& resp);
    void HandleDeleteTenantQuota(coro_http::coro_http_request& req,
                                 coro_http::coro_http_response& resp);
    void HandleGetDefaultTenantQuota(coro_http::coro_http_request& req,
                                     coro_http::coro_http_response& resp);
    void HandleSetDefaultTenantQuota(coro_http::coro_http_request& req,
                                     coro_http::coro_http_response& resp);

    void RegisterHandler();

    uint16_t http_port_;
    bool enable_metric_reporting_ = false;
    coro_http::coro_http_server http_server_;
    std::thread metric_report_thread_;
    std::atomic<bool> metric_report_running_{false};
    // Used to wake the metric-report thread early when the server stops.
    // Replaces a std::binary_semaphore (C++20) so the code builds under
    // gcc 10 / C++17.
    std::mutex metric_report_mutex_;
    std::condition_variable metric_report_cv_;
    bool metric_report_stop_requested_ = false;
    std::atomic<bool> started_{false};
    mutable std::mutex state_mutex_;
    ha::MasterRuntimeState state_{ha::MasterRuntimeState::kStarting};
    std::optional<ha::MasterView> leader_view_;
    std::shared_ptr<WrappedMasterService> service_;
    bool service_available_ = false;
};

}  // namespace mooncake
