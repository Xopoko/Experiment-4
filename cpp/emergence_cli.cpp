#include "emergence_engine.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace emergence;

namespace {

std::string get_arg(int argc, char** argv, const std::string& flag, const std::string& fallback) {
    for (int i = 1; i < argc - 1; ++i) {
        if (flag == argv[i]) {
            return argv[i + 1];
        }
    }
    return fallback;
}

bool has_flag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i) {
        if (flag == argv[i]) return true;
    }
    return false;
}

std::array<int, 3> parse_dims(const std::string& s, const std::array<int, 3>& fallback) {
    std::array<int, 3> dims = fallback;
    char sep = 'x';
    if (s.find('x') == std::string::npos && s.find('X') == std::string::npos) sep = ',';
    std::stringstream ss(s);
    std::string token;
    int idx = 0;
    while (std::getline(ss, token, sep) && idx < 3) {
        dims[idx++] = std::atoi(token.c_str());
    }
    return dims;
}

void print_usage() {
    std::cout << "Usage: emergence_cli [options]\n"
                 "Options:\n"
                 "  --dims 3x3x3        Lattice dimensions (default 3x3x3)\n"
                 "  --beta 0.25         Inverse temperature beta\n"
                 "  --J 1.0             Coupling J\n"
                 "  --h 0.0             External field h\n"
                 "  --burn 200          Burn-in steps\n"
                 "  --steps 128         Recorded steps per trajectory\n"
                 "  --interval 1        Metropolis sweeps between samples\n"
                 "  --trajectories 32   Number of trajectories\n"
                 "  --seed 1            RNG seed\n"
                 "  --ppmi-lag 5        Lag for persistent MI\n"
                 "  --macro-set baseline|full   Macro set (baseline: magnetization+layer_z+block_majority; full adds energy/domain_walls)\n"
                 "  --no-pbc            Use open boundaries instead of periodic\n"
                 "  --help              Show this message\n";
}

} // namespace

int main(int argc, char** argv) {
    if (has_flag(argc, argv, "--help")) {
        print_usage();
        return 0;
    }

    LatticeConfig cfg;
    cfg.dims = parse_dims(get_arg(argc, argv, "--dims", "3x3x3"), cfg.dims);
    cfg.J = std::atof(get_arg(argc, argv, "--J", "1.0").c_str());
    cfg.h = std::atof(get_arg(argc, argv, "--h", "0.0").c_str());
    cfg.periodic = !has_flag(argc, argv, "--no-pbc");

    SamplerParams params;
    params.beta = std::atof(get_arg(argc, argv, "--beta", "0.25").c_str());
    params.burn_in = std::atoi(get_arg(argc, argv, "--burn", "200").c_str());
    params.steps = std::atoi(get_arg(argc, argv, "--steps", "128").c_str());
    params.sample_interval = std::atoi(get_arg(argc, argv, "--interval", "1").c_str());
    params.trajectories = std::atoi(get_arg(argc, argv, "--trajectories", "32").c_str());
    params.seed = static_cast<uint64_t>(std::atoll(get_arg(argc, argv, "--seed", "1").c_str()));

    int ppmi_lag = std::atoi(get_arg(argc, argv, "--ppmi-lag", "5").c_str());

    Lattice3D lattice(cfg);
    TrajectorySampler sampler(lattice, params);
    auto trajectories = sampler.run();

    PartitionSpec partition;
    partition.type = PartitionSpec::Type::AxisHalf;
    partition.axis = 2; // z-axis split by default

    EmergenceAnalyzer analyzer(lattice, partition, ppmi_lag);

    std::string macro_set = get_arg(argc, argv, "--macro-set", "baseline");
    std::transform(macro_set.begin(), macro_set.end(), macro_set.begin(), [](unsigned char c) { return std::tolower(c); });

    std::vector<MacroSpec> macros;
    if (macro_set == "baseline" || macro_set == "base") {
        macros.push_back(MacroSpec{MacroSpec::Type::Magnetization, "magnetization"});
        MacroSpec layers{MacroSpec::Type::LayerMagnetization, "layer_z"};
        layers.axis = 2;
        macros.push_back(layers);
        MacroSpec block{MacroSpec::Type::BlockMajority, "block_majority"};
        macros.push_back(block);
    } else if (macro_set == "full") {
        macros.push_back(MacroSpec{MacroSpec::Type::Magnetization, "magnetization"});
        macros.push_back(MacroSpec{MacroSpec::Type::Energy, "energy"});
        MacroSpec block{MacroSpec::Type::BlockMajority, "block_majority"};
        macros.push_back(block);
        MacroSpec layers{MacroSpec::Type::LayerMagnetization, "layer_z"};
        layers.axis = 2;
        macros.push_back(layers);
        macros.push_back(MacroSpec{MacroSpec::Type::DomainWalls, "domain_walls"});
    } else {
        std::cerr << "Unknown macro-set: " << macro_set << " (expected baseline|full)\n";
        return EXIT_FAILURE;
    }

    auto report = analyzer.analyze(trajectories, macros);
    report.sampler = params;

    std::cout << to_json(report) << std::endl;
    return 0;
}
