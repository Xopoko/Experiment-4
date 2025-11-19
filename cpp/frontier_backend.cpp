#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr int SOURCE_VERTEX = 0;
constexpr std::size_t MAX_VERTICES = 64;  // mask + frontier array limit

int g_vertex_count = 0;

struct Dimensions {
    int nx = 2;
    int ny = 2;
    int nz = 5;
};

struct Options {
    Dimensions dims;
    int max_edges = 10;
};

struct Vec3 {
    int x;
    int y;
    int z;
};

struct StateKey {
    std::array<std::uint8_t, MAX_VERTICES> frontier{};  // (remaining_future << 1) | parity
    std::array<std::uint8_t, MAX_VERTICES> connected{}; // 1 if connected to source
    std::uint8_t odd_past = 0;
    std::uint8_t edges_used = 0;
    std::uint64_t mask = 0;
};

struct StateHasher {
    std::size_t operator()(const StateKey &state) const noexcept {
        std::size_t h = std::hash<std::uint64_t>{}(state.mask);
        h ^= (static_cast<std::size_t>(state.edges_used) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
        h ^= (static_cast<std::size_t>(state.odd_past) + 0x9e3779b97f4a7c55ULL + (h << 6) + (h >> 2));
        for (int i = 0; i < g_vertex_count; ++i) {
            h ^= static_cast<std::size_t>(state.frontier[i]) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= static_cast<std::size_t>(state.connected[i]) + 0x9e3779b97f4a7c55ULL + (h << 6) + (h >> 2);
        }
        return h;
    }
};

struct StateEq {
    bool operator()(const StateKey &a, const StateKey &b) const noexcept {
        if (a.odd_past != b.odd_past || a.edges_used != b.edges_used || a.mask != b.mask) {
            return false;
        }
        for (int i = 0; i < g_vertex_count; ++i) {
            if (a.frontier[i] != b.frontier[i] || a.connected[i] != b.connected[i]) {
                return false;
            }
        }
        return true;
    }
};

struct GraphData {
    std::vector<Vec3> coords;
    std::vector<std::vector<int>> past_neighbors;
    std::vector<std::vector<int>> future_neighbors;
    std::vector<std::vector<int>> adjacency;
};

Options parse_args(int argc, char **argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        auto parse_value = [&](const std::string &name) -> std::string {
            auto pos = arg.find('=');
            if (pos != std::string::npos) {
                return arg.substr(pos + 1);
            }
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                std::exit(EXIT_FAILURE);
            }
            return std::string(argv[++i]);
        };

        if (arg == "--nx" || arg.rfind("--nx=", 0) == 0) {
            opts.dims.nx = std::stoi(parse_value("--nx"));
        } else if (arg == "--ny" || arg.rfind("--ny=", 0) == 0) {
            opts.dims.ny = std::stoi(parse_value("--ny"));
        } else if (arg == "--nz" || arg.rfind("--nz=", 0) == 0) {
            opts.dims.nz = std::stoi(parse_value("--nz"));
        } else if (arg == "--max-edges" || arg.rfind("--max-edges=", 0) == 0) {
            opts.max_edges = std::stoi(parse_value("--max-edges"));
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: frontier_backend [--nx N] [--ny N] [--nz N] [--max-edges M]\n";
            std::exit(EXIT_SUCCESS);
        }
    }
    return opts;
}

inline std::uint8_t make_entry(std::uint8_t remaining_future, std::uint8_t parity) {
    return static_cast<std::uint8_t>((remaining_future << 1) | (parity & 1u));
}

GraphData build_graph(const Dimensions &dims) {
    const int nx = dims.nx;
    const int ny = dims.ny;
    const int nz = dims.nz;
    const int total = nx * ny * nz;
    auto to_index = [=](int x, int y, int z) {
        return ((z * ny) + y) * nx + x;
    };

    GraphData data;
    data.coords.reserve(total);
    data.past_neighbors.assign(total, {});
    data.future_neighbors.assign(total, {});
    data.adjacency.assign(total, {});

    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                data.coords.push_back({x, y, z});
            }
        }
    }

    const std::array<Vec3, 6> dirs = {{{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}}};
    for (int idx = 0; idx < total; ++idx) {
        Vec3 v = data.coords[idx];
        for (const auto &dir : dirs) {
            int nx_ = v.x + dir.x;
            int ny_ = v.y + dir.y;
            int nz_ = v.z + dir.z;
            if (nx_ < 0 || nx_ >= nx || ny_ < 0 || ny_ >= ny || nz_ < 0 || nz_ >= nz) {
                continue;
            }
            int nb_idx = to_index(nx_, ny_, nz_);
            data.adjacency[idx].push_back(nb_idx);
            if (nb_idx < idx) {
                data.past_neighbors[idx].push_back(nb_idx);
            } else if (nb_idx > idx) {
                data.future_neighbors[idx].push_back(nb_idx);
            }
        }
    }

    return data;
}

bool is_connected(std::uint64_t mask, const std::vector<std::vector<int>> &adjacency) {
    std::vector<int> vertices;
    for (int idx = 0; idx < static_cast<int>(adjacency.size()); ++idx) {
        if (mask & (1ULL << idx)) {
            vertices.push_back(idx);
        }
    }
    if (vertices.empty()) return false;
    if (!(mask & (1ULL << SOURCE_VERTEX))) return false;

    std::vector<char> visited(adjacency.size(), 0);
    std::vector<int> stack = {SOURCE_VERTEX};
    while (!stack.empty()) {
        int v = stack.back();
        stack.pop_back();
        if (visited[v] || !(mask & (1ULL << v))) continue;
        visited[v] = 1;
        for (int nb : adjacency[v]) {
            if (mask & (1ULL << nb)) {
                stack.push_back(nb);
            }
        }
    }

    for (int v : vertices) {
        if (!visited[v]) return false;
    }
    return true;
}

struct FrontierResult {
    std::vector<std::size_t> states_per_step;
    std::vector<long long> counts_by_length;
    std::size_t final_state_count = 0;
};

void unite_connected(StateKey &state, int a, int b) {
    if (state.connected[a] || state.connected[b]) {
        state.connected[a] = state.connected[b] = 1;
    }
}

FrontierResult run_frontier_dp(const GraphData &graph, int max_edges) {
    const int total_vertices = static_cast<int>(graph.coords.size());
    FrontierResult result;
    result.states_per_step.reserve(total_vertices);
    result.counts_by_length.assign(max_edges + 1, 0);

    std::unordered_map<StateKey, long long, StateHasher, StateEq> states;
    states.reserve(1024);
    StateKey initial;
    initial.connected[SOURCE_VERTEX] = 1;
    states.emplace(initial, 1LL);

    for (int idx = 0; idx < total_vertices; ++idx) {
        std::unordered_map<StateKey, long long, StateHasher, StateEq> next_states;
        next_states.reserve(states.size() * 2 + 16);
        const auto &past = graph.past_neighbors[idx];
        const auto &future = graph.future_neighbors[idx];
        for (const auto &entry : states) {
            const StateKey &state = entry.first;
            long long ways = entry.second;

            std::vector<int> processed;
            processed.reserve(past.size());
            for (int nb : past) {
                if (state.frontier[nb] != 0) {
                    processed.push_back(nb);
                }
            }

            int combos = 1 << static_cast<int>(processed.size());
            for (int subset = 0; subset < combos; ++subset) {
                int add_edges = __builtin_popcount(static_cast<unsigned int>(subset));
                int new_edges = state.edges_used + add_edges;
                if (new_edges > max_edges) {
                    continue;
                }

                StateKey new_state = state;
                new_state.edges_used = static_cast<std::uint8_t>(new_edges);
                std::uint64_t mask = state.mask;
                std::uint8_t odd_past = state.odd_past;
                bool valid = true;

                for (std::size_t pos = 0; pos < processed.size(); ++pos) {
                    int nb = processed[pos];
                    bool take_edge = (subset >> pos) & 1;
                    std::uint8_t value = new_state.frontier[nb];
                    std::uint8_t parity = value & 1u;
                    std::uint8_t remaining = value >> 1;
                    if (remaining == 0) {
                        valid = false;
                        break;
                    }
                    remaining -= 1;
                    if (take_edge) {
                        parity ^= 1u;
                        mask |= (1ULL << nb);
                        mask |= (1ULL << idx);
                        unite_connected(new_state, nb, idx);
                    }
                    if (remaining == 0) {
                        new_state.frontier[nb] = 0;
                        if (parity) {
                            odd_past += 1;
                        }
                    } else {
                        new_state.frontier[nb] = make_entry(remaining, parity);
                    }
                }
                if (!valid) {
                    continue;
                }

                std::uint8_t parity_current = static_cast<std::uint8_t>(add_edges & 1);
                if (!future.empty()) {
                    std::uint8_t remaining_future = static_cast<std::uint8_t>(future.size());
                    new_state.frontier[idx] = make_entry(remaining_future, parity_current);
                    new_state.connected[idx] = (idx == SOURCE_VERTEX || new_state.connected[idx]);
                } else {
                    new_state.frontier[idx] = 0;
                    if (parity_current) {
                        odd_past += 1;
                    }
                }

                new_state.mask = mask;
                new_state.odd_past = odd_past;
                auto it = next_states.find(new_state);
                if (it == next_states.end()) {
                    next_states.emplace(std::move(new_state), ways);
                } else {
                    it->second += ways;
                }
            }
        }
        states.swap(next_states);
        result.states_per_step.push_back(states.size());
    }

    result.final_state_count = states.size();

    for (const auto &entry : states) {
        const StateKey &state = entry.first;
        long long ways = entry.second;
        bool frontier_empty = true;
        for (int i = 0; i < g_vertex_count; ++i) {
            if (state.frontier[i] != 0) {
                frontier_empty = false;
                break;
            }
        }
        if (!frontier_empty) {
            continue;
        }
        if (state.edges_used == 0 || state.odd_past != 2) {
            continue;
        }
        if (!state.connected[SOURCE_VERTEX]) {
            continue;
        }
        if (state.edges_used < static_cast<int>(result.counts_by_length.size()) && is_connected(state.mask, graph.adjacency)) {
            result.counts_by_length[state.edges_used] += ways;
        }
    }

    return result;
}

void dump_json(const Options &opts, const FrontierResult &res) {
    std::cout << "{\n";
    std::cout << "  \"dims\": [" << opts.dims.nx << ", " << opts.dims.ny << ", " << opts.dims.nz << "],\n";
    std::cout << "  \"max_edges\": " << opts.max_edges << ",\n";
    std::cout << "  \"counts_by_length\": {\n";
    bool first = true;
    for (int length = 1; length < static_cast<int>(res.counts_by_length.size()); ++length) {
        long long count = res.counts_by_length[length];
        if (count == 0) continue;
        if (!first) {
            std::cout << ",\n";
        }
        first = false;
        std::cout << "    \"" << length << "\": " << count;
    }
    if (!first) {
        std::cout << "\n";
    }
    std::cout << "  },\n";
    std::cout << "  \"states_per_step\": [";
    for (std::size_t i = 0; i < res.states_per_step.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << res.states_per_step[i];
    }
    std::cout << "],\n";
    std::cout << "  \"final_state_count\": " << res.final_state_count << ",\n";
    std::cout << "  \"notes\": \"Frontier DP with source-connectivity tags\"\n";
    std::cout << "}\n";
}

}  // namespace

int main(int argc, char **argv) {
    Options opts = parse_args(argc, argv);
    if (opts.dims.nx <= 0 || opts.dims.ny <= 0 || opts.dims.nz <= 0) {
        std::cerr << "Dimensions must be positive integers\n";
        return EXIT_FAILURE;
    }
    g_vertex_count = opts.dims.nx * opts.dims.ny * opts.dims.nz;
    if (g_vertex_count <= 0 || g_vertex_count > static_cast<int>(MAX_VERTICES)) {
        std::cerr << "This backend currently supports up to " << MAX_VERTICES << " vertices (got "
                  << g_vertex_count << ")\n";
        return EXIT_FAILURE;
    }

    const GraphData graph = build_graph(opts.dims);
    const FrontierResult res = run_frontier_dp(graph, opts.max_edges);
    dump_json(opts, res);
    return EXIT_SUCCESS;
}
