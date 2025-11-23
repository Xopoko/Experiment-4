// Emergence detection engine for lattice dynamics (3D Ising oriented).
// Provides trajectory sampling, coarse-graining, and basic information metrics.
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace emergence {

using Spin = int8_t; // expected to be -1 or +1
using State = std::vector<Spin>;
using MacroState = std::vector<int>;

struct LatticeConfig {
    std::array<int, 3> dims{3, 3, 3}; // x, y, z sizes
    double J = 1.0;
    double h = 0.0;
    bool periodic = true;
};

struct SamplerParams {
    int burn_in = 200;
    int steps = 128;
    int sample_interval = 1;
    int trajectories = 32;
    uint64_t seed = 1;
    double beta = 0.25; // 1 / temperature
};

struct MacroSpec {
    enum class Type {
        Magnetization,
        Energy,
        BlockMajority,
        LayerMagnetization,
        DomainWalls
    };

    Type type = Type::Magnetization;
    std::string name;
    std::array<int, 3> block_shape{2, 2, 2}; // used for BlockMajority
    int axis = 2;                              // used for LayerMagnetization (0=x,1=y,2=z)
};

struct PartitionSpec {
    enum class Type { AxisHalf, Explicit };

    Type type = Type::AxisHalf;
    int axis = 2; // for AxisHalf
    std::vector<int> maskA; // explicit indices belonging to part A; others go to B
};

struct Trajectory {
    std::vector<State> steps; // includes the initial state
};

struct EmergenceMetrics {
    double entropy = 0.0;
    double mutual_information = 0.0;
    double effective_information = 0.0;
    double phi = 0.0;
    double synergy = 0.0;
    double ppmi = 0.0; // persistent mutual information at a fixed lag
};

struct MacroReport {
    MacroSpec spec;
    EmergenceMetrics metrics;
    double emergence_score = 0.0; // heuristic combination vs micro baseline
};

struct EmergenceReport {
    LatticeConfig lattice;
    SamplerParams sampler;
    PartitionSpec partition;
    EmergenceMetrics micro;
    std::vector<MacroReport> macros;
};

class Lattice3D {
public:
    explicit Lattice3D(const LatticeConfig& cfg);

    int volume() const { return volume_; }
    const std::array<int, 3>& dims() const { return cfg_.dims; }
    const LatticeConfig& config() const { return cfg_; }
    int index(int x, int y, int z) const;
    std::array<int, 3> coords(int idx) const;
    const std::vector<std::vector<int>>& neighbors() const { return neighbors_; }

    double energy(const State& state) const;

private:
    LatticeConfig cfg_;
    int volume_ = 0;
    std::vector<std::vector<int>> neighbors_;

    void build_neighbors();
};

class TrajectorySampler {
public:
    TrajectorySampler(Lattice3D lattice, SamplerParams params);

    std::vector<Trajectory> run();

private:
    Lattice3D lattice_;
    SamplerParams params_;
    std::mt19937_64 rng_;

    void metropolis_step(State& state);
    State random_state();
};

// Core computation: evaluate emergence metrics for micro and macro levels.
class EmergenceAnalyzer {
public:
    EmergenceAnalyzer(const Lattice3D& lattice,
                      const PartitionSpec& partition,
                      int ppmi_lag = 5);

    EmergenceReport analyze(const std::vector<Trajectory>& trajectories,
                            const std::vector<MacroSpec>& macros,
                            const EmergenceMetrics& micro_baseline_hint = {}) const;

private:
    const Lattice3D& lattice_;
    PartitionSpec partition_;
    int ppmi_lag_;

    EmergenceMetrics compute_micro(const std::vector<Trajectory>& trajectories) const;
    MacroReport compute_macro(const std::vector<Trajectory>& trajectories,
                              const MacroSpec& spec,
                              const EmergenceMetrics& micro_baseline) const;
};

// Helpers to emit a small JSON without external deps.
std::string to_json(const EmergenceReport& report);

} // namespace emergence
