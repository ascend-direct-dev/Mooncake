#pragma once

#include <atomic>
#include <charconv>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>

namespace mooncake {

using RandomEngine = std::mt19937_64;

namespace detail {

template <typename Result, std::integral DistributionInteger,
          std::uniform_random_bit_generator Generator>
Result sampleUniform(Result lower_bound, Result upper_bound,
                     Generator& generator) {
    std::uniform_int_distribution<DistributionInteger> distribution(
        static_cast<DistributionInteger>(lower_bound),
        static_cast<DistributionInteger>(upper_bound));
    return static_cast<Result>(distribution(generator));
}

}  // namespace detail

// Returns the pseudo-random engine shared by random helpers on this thread.
// The engine is seeded once per thread and is not safe to use from another
// thread.
inline RandomEngine& threadLocalRandomEngine() {
    thread_local RandomEngine engine = [] {
        std::random_device device;
        std::seed_seq seed{device(), device(), device(), device(),
                           device(), device(), device(), device()};
        return RandomEngine(seed);
    }();
    return engine;
}

namespace detail {

struct RandomSeedState {
    std::mutex mu;
    bool configured = false;
    bool rejected_env = false;
    // 0 unknown, 1 unseeded, 2 seeded. Published after configuration so the
    // hot path can avoid the mutex.
    std::atomic<int> mode{0};
    std::optional<uint64_t> seed;
    std::optional<RandomEngine> engine;
};

inline void PublishRandomSeedMode(RandomSeedState& state) {
    state.mode.store(state.seed.has_value() ? 2 : 1, std::memory_order_release);
}

inline RandomSeedState& randomSeedState() {
    static RandomSeedState state;
    return state;
}

inline void EnsureRandomSeedConfigured(RandomSeedState& state) {
    if (state.configured) {
        return;
    }
    state.configured = true;
    const char* env = std::getenv("MOONCAKE_RANDOM_SEED");
    if (env == nullptr || env[0] == '\0') {
        return;
    }
    uint64_t value = 0;
    const char* end = env + std::strlen(env);
    const auto parsed = std::from_chars(env, end, value);
    if (parsed.ec == std::errc{} && parsed.ptr == end) {
        state.seed = value;
        return;
    }
    state.rejected_env = true;
}

}  // namespace detail

// Fixes the allocation RNG. Heartbeats, shard scans, and port picks keep
// using threadLocalRandomEngine(), so their timing cannot shift placement.
// Every allocationRandomIndex() call then draws from one process-wide
// mt19937_64 in call order. One writer that allocates keys in order therefore
// repeats the same placement decisions. Callers must also sort segment names
// before sampling: the seed does not freeze mount arrival order.
// MOONCAKE_RANDOM_SEED is read on first use when SetRandomSeed was not called.
inline void SetRandomSeed(uint64_t seed) {
    auto& state = detail::randomSeedState();
    std::lock_guard<std::mutex> lock(state.mu);
    state.configured = true;
    state.rejected_env = false;
    state.seed = seed;
    state.engine.reset();
    detail::PublishRandomSeedMode(state);
}

inline void ClearRandomSeed() {
    auto& state = detail::randomSeedState();
    std::lock_guard<std::mutex> lock(state.mu);
    state.configured = false;
    state.rejected_env = false;
    state.seed.reset();
    state.engine.reset();
    state.mode.store(0, std::memory_order_release);
}

inline std::optional<uint64_t> FixedRandomSeed() {
    auto& state = detail::randomSeedState();
    std::lock_guard<std::mutex> lock(state.mu);
    detail::EnsureRandomSeedConfigured(state);
    detail::PublishRandomSeedMode(state);
    return state.seed;
}

inline bool HasFixedRandomSeed() {
    auto& state = detail::randomSeedState();
    const int mode = state.mode.load(std::memory_order_acquire);
    if (mode != 0) {
        return mode == 2;
    }
    return FixedRandomSeed().has_value();
}

// True once, if MOONCAKE_RANDOM_SEED was set but was not an unsigned integer.
inline bool ConsumeRejectedRandomSeedEnv() {
    auto& state = detail::randomSeedState();
    std::lock_guard<std::mutex> lock(state.mu);
    detail::EnsureRandomSeedConfigured(state);
    detail::PublishRandomSeedMode(state);
    const bool rejected = state.rejected_env;
    state.rejected_env = false;
    return rejected;
}

template <std::uniform_random_bit_generator Generator>
size_t randomIndex(size_t upper_bound, Generator& generator) {
    if (upper_bound == 0) {
        throw std::invalid_argument("randomIndex upper bound must be positive");
    }
    std::uniform_int_distribution<size_t> distribution(0, upper_bound - 1);
    return distribution(generator);
}

// Returns an unbiased random index in [0, upper_bound).
inline size_t randomIndex(size_t upper_bound) {
    return randomIndex(upper_bound, threadLocalRandomEngine());
}

// When a seed is fixed, return stable_index instead of drawing. Shard scans
// and single-replica picks then do not depend on which RPC thread runs.
inline size_t stableIndex(size_t upper_bound, size_t stable_index) {
    if (upper_bound == 0) {
        throw std::invalid_argument("stableIndex upper bound must be positive");
    }
    if (HasFixedRandomSeed()) {
        return stable_index % upper_bound;
    }
    return randomIndex(upper_bound);
}

// Allocation draws. With a fixed seed this is one locked stream; without one
// it stays on the calling thread's engine.
inline size_t allocationRandomIndex(size_t upper_bound) {
    auto& state = detail::randomSeedState();
    std::unique_lock<std::mutex> lock(state.mu);
    detail::EnsureRandomSeedConfigured(state);
    detail::PublishRandomSeedMode(state);
    if (state.seed.has_value()) {
        if (!state.engine.has_value()) {
            state.engine.emplace(*state.seed);
        }
        return randomIndex(upper_bound, *state.engine);
    }
    lock.unlock();
    return randomIndex(upper_bound);
}

template <std::integral Integer, std::uniform_random_bit_generator Generator>
Integer randomUniform(Integer lower_bound, Integer upper_bound,
                      Generator& generator) {
    if (lower_bound > upper_bound) {
        throw std::invalid_argument(
            "randomUniform lower bound must not exceed upper bound");
    }
    if constexpr (std::signed_integral<Integer>) {
        return detail::sampleUniform<Integer, long long>(
            lower_bound, upper_bound, generator);
    } else {
        return detail::sampleUniform<Integer, unsigned long long>(
            lower_bound, upper_bound, generator);
    }
}

// Returns an unbiased random integer in [lower_bound, upper_bound].
template <std::integral Integer>
Integer randomUniform(Integer lower_bound, Integer upper_bound) {
    return randomUniform(lower_bound, upper_bound, threadLocalRandomEngine());
}

}  // namespace mooncake
