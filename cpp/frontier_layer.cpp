#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr int SOURCE_VERTEX = 0;
constexpr std::size_t MAX_VERTICES = 64;

int g_vertex_count = 0;
int g_nx = 0, g_ny = 0, g_nz = 0;

struct Dimensions {
    int nx = 2;
    int ny = 2;
    int nz = 2;
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
    std::array<std::uint8_t, MAX_VERTICES> frontier{};
    std::array<std::uint8_t, MAX_VERTICES> connected{};
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
        if (a.odd_past != b.odd_past || a.edges_used != b.edges_used || a.mask != b.mask) return false;
        for (int i = 0; i < g_vertex_count; ++i) {
            if (a.frontier[i] != b.frontier[i] || a.connected[i] != b.connected[i]) return false;
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
            std::cout << "Usage: frontier_layer [--nx N] [--ny N] [--nz N] [--max-edges M]\n";
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
    auto to_index = [=](int x, int y, int z) { return ((z * ny) + y) * nx + x; };

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
            if (nx_ < 0 || nx_ >= nx || ny_ < 0 || ny_ >= ny || nz_ < 0 || nz_ >= nz) continue;
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
        if (mask & (1ULL << idx)) vertices.push_back(idx);
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
            if (mask & (1ULL << nb)) stack.push_back(nb);
        }
    }
    for (int v : vertices) if (!visited[v]) return false;
    return true;
}

inline bool less_state(const StateKey &a, const StateKey &b) {
    for (int i = 0; i < g_vertex_count; ++i) {
        if (a.frontier[i] != b.frontier[i]) return a.frontier[i] < b.frontier[i];
    }
    for (int i = 0; i < g_vertex_count; ++i) {
        if (a.connected[i] != b.connected[i]) return a.connected[i] < b.connected[i];
    }
    if (a.edges_used != b.edges_used) return a.edges_used < b.edges_used;
    if (a.odd_past != b.odd_past) return a.odd_past < b.odd_past;
    return a.mask < b.mask;
}

std::vector<std::vector<int>> build_layer_permutations_full(const Dimensions &dims) {
    const int nx = dims.nx;
    const int ny = dims.ny;
    const int nz = dims.nz;
    const bool square = (nx == ny);
    using TF = std::function<std::pair<int, int>(int, int)>;
    std::vector<TF> tfs;
    tfs.push_back([&](int x, int y) { return std::make_pair(x, y); });                        // id
    tfs.push_back([&](int x, int y) { return std::make_pair(nx - 1 - x, ny - 1 - y); });      // rot180
    tfs.push_back([&](int x, int y) { return std::make_pair(nx - 1 - x, y); });               // refl x
    tfs.push_back([&](int x, int y) { return std::make_pair(x, ny - 1 - y); });               // refl y
    if (square) {
        tfs.push_back([&](int x, int y) { return std::make_pair(y, nx - 1 - x); });           // rot90
        tfs.push_back([&](int x, int y) { return std::make_pair(ny - 1 - y, x); });           // rot270
        tfs.push_back([&](int x, int y) { return std::make_pair(y, x); });                    // refl diag
        tfs.push_back([&](int x, int y) { return std::make_pair(nx - 1 - y, ny - 1 - x); });  // refl anti
    }

    auto to_idx = [&](int x, int y, int z) { return ((z * ny) + y) * nx + x; };
    std::vector<std::vector<int>> perms;
    for (const auto &tf : tfs) {
        std::vector<int> p(g_vertex_count, -1);
        bool ok = true;
        for (int z = 0; z < nz && ok; ++z) {
            for (int y = 0; y < ny && ok; ++y) {
                for (int x = 0; x < nx && ok; ++x) {
                    auto [xn, yn] = tf(x, y);
                    if (xn < 0 || xn >= nx || yn < 0 || yn >= ny) { ok = false; break; }
                    int old_idx = to_idx(x, y, z);
                    int new_idx = to_idx(xn, yn, z);
                    if (p[old_idx] != -1) { ok = false; break; }
                    p[old_idx] = new_idx;
                }
            }
        }
        if (!ok) continue;
        if (std::any_of(p.begin(), p.end(), [](int v){return v==-1;})) continue;
        bool dup = false;
        for (auto &q : perms) if (q == p) { dup = true; break; }
        if (!dup) perms.push_back(std::move(p));
    }
    if (perms.empty()) {
        perms.emplace_back(g_vertex_count);
        for (int i = 0; i < g_vertex_count; ++i) perms[0][i] = i;
    }
    return perms;
}

StateKey apply_perm(const StateKey &src, const std::vector<int> &perm) {
    StateKey dst{};
    dst.edges_used = src.edges_used;
    dst.odd_past = src.odd_past;
    for (int i = 0; i < g_vertex_count; ++i) {
        int j = perm[i];
        dst.frontier[j] = src.frontier[i];
        dst.connected[j] = src.connected[i];
    }
    std::uint64_t m = src.mask;
    while (m) {
        int bit = __builtin_ctzll(m);
        m &= (m - 1);
        dst.mask |= (1ULL << perm[bit]);
    }
    return dst;
}

StateKey canonicalize_full(const StateKey &state, const std::vector<std::vector<int>> &perms) {
    StateKey best = state;
    for (const auto &p : perms) {
        StateKey tmp = apply_perm(state, p);
        if (less_state(tmp, best)) best = std::move(tmp);
    }
    return best;
}

struct FrontierResult {
    std::vector<std::size_t> states_per_layer;
    std::vector<long long> counts_by_length;
    std::size_t final_state_count = 0;
};

void unite_connected(StateKey &state, int a, int b) {
    if (state.connected[a] || state.connected[b]) {
        state.connected[a] = state.connected[b] = 1;
    }
}

FrontierResult run_frontier_layer(const Options &opts) {
    const int nx = opts.dims.nx;
    const int ny = opts.dims.ny;
    const int nz = opts.dims.nz;
    const int total = nx * ny * nz;
    const int layer_size = nx * ny;

    GraphData graph = build_graph(opts.dims);
    auto layer_perms = build_layer_permutations_full(opts.dims);

    FrontierResult res;
    res.counts_by_length.assign(opts.max_edges + 1, 0);

    std::unordered_map<StateKey, long long, StateHasher, StateEq> states;
    states.reserve(1024);
    StateKey init;
    init.connected[SOURCE_VERTEX] = 1;
    states.emplace(init, 1LL);

    for (int idx = 0; idx < total; ++idx) {
        std::unordered_map<StateKey, long long, StateHasher, StateEq> next_states;
        next_states.reserve(states.size() * 2 + 16);
        const auto &past = graph.past_neighbors[idx];
        const auto &future = graph.future_neighbors[idx];
        for (const auto &entry : states) {
            const StateKey &state = entry.first;
            long long ways = entry.second;
            if (state.odd_past > 2) continue;

            std::vector<int> processed;
            processed.reserve(past.size());
            for (int nb : past) {
                if (state.frontier[nb] != 0) processed.push_back(nb);
            }

            int combos = 1 << static_cast<int>(processed.size());
            for (int subset = 0; subset < combos; ++subset) {
                int add_edges = __builtin_popcount(static_cast<unsigned int>(subset));
                int new_edges = state.edges_used + add_edges;
                if (new_edges > opts.max_edges) continue;

                StateKey ns = state;
                ns.edges_used = static_cast<std::uint8_t>(new_edges);
                std::uint64_t mask = state.mask;
                std::uint8_t odd_past = state.odd_past;
                bool valid = true;

                for (std::size_t pos = 0; pos < processed.size(); ++pos) {
                    int nb = processed[pos];
                    bool take = (subset >> pos) & 1;
                    std::uint8_t val = ns.frontier[nb];
                    std::uint8_t parity = val & 1u;
                    std::uint8_t remaining = val >> 1;
                    if (remaining == 0) { valid = false; break; }
                    remaining -= 1;
                    if (take) {
                        parity ^= 1u;
                        mask |= (1ULL << nb);
                        mask |= (1ULL << idx);
                        unite_connected(ns, nb, idx);
                    }
                    if (remaining == 0) {
                        ns.frontier[nb] = 0;
                        if (parity) odd_past += 1;
                    } else {
                        ns.frontier[nb] = make_entry(remaining, parity);
                    }
                }
                if (!valid) continue;

                std::uint8_t parity_current = static_cast<std::uint8_t>(add_edges & 1u);
                if (!future.empty()) {
                    std::uint8_t remaining_future = static_cast<std::uint8_t>(future.size());
                    ns.frontier[idx] = make_entry(remaining_future, parity_current);
                    ns.connected[idx] = (idx == SOURCE_VERTEX || ns.connected[idx]);
                } else {
                    ns.frontier[idx] = 0;
                    if (parity_current) odd_past += 1;
                }

                ns.mask = mask;
                ns.odd_past = odd_past;
                auto it = next_states.find(ns);
                if (it == next_states.end()) {
                    next_states.emplace(std::move(ns), ways);
                } else {
                    it->second += ways;
                }
            }
        }
        states.swap(next_states);

        // канонизация на границе слоя
        if ((idx + 1) % layer_size == 0) {
            std::unordered_map<StateKey, long long, StateHasher, StateEq> canon;
            canon.reserve(states.size());
            for (const auto &entry : states) {
                StateKey can = canonicalize_full(entry.first, layer_perms);
                auto it = canon.find(can);
                if (it == canon.end()) canon.emplace(std::move(can), entry.second);
                else it->second += entry.second;
            }
            states.swap(canon);
            res.states_per_layer.push_back(states.size());
        }
    }

    res.final_state_count = states.size();
    for (const auto &entry : states) {
        const StateKey &state = entry.first;
        long long ways = entry.second;
        bool frontier_empty = true;
        for (int i = 0; i < g_vertex_count; ++i) {
            if (state.frontier[i] != 0) { frontier_empty = false; break; }
        }
        if (!frontier_empty) continue;
        if (state.edges_used == 0 || state.odd_past != 2) continue;
        if (!state.connected[SOURCE_VERTEX]) continue;
        if (state.edges_used < static_cast<int>(res.counts_by_length.size()) && is_connected(state.mask, graph.adjacency)) {
            res.counts_by_length[state.edges_used] += ways;
        }
    }
    return res;
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
        if (!first) std::cout << ",\n";
        first = false;
        std::cout << "    \"" << length << "\": " << count;
    }
    if (!first) std::cout << "\n";
    std::cout << "  },\n";
    std::cout << "  \"states_per_layer\": [";
    for (std::size_t i = 0; i < res.states_per_layer.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << res.states_per_layer[i];
    }
    std::cout << "],\n";
    std::cout << "  \"final_state_count\": " << res.final_state_count << ",\n";
    std::cout << "  \"notes\": \"Layer canonicalization by xy-permutations after each layer\"\n";
    std::cout << "}\n";
}

}  // namespace

int main(int argc, char **argv) {
    Options opts = parse_args(argc, argv);
    if (opts.dims.nx <= 0 || opts.dims.ny <= 0 || opts.dims.nz <= 0) {
        std::cerr << "Dimensions must be positive integers\n";
        return EXIT_FAILURE;
    }
    g_nx = opts.dims.nx;
    g_ny = opts.dims.ny;
    g_nz = opts.dims.nz;
    g_vertex_count = g_nx * g_ny * g_nz;
    if (g_vertex_count <= 0 || g_vertex_count > static_cast<int>(MAX_VERTICES)) {
        std::cerr << "Supports up to " << MAX_VERTICES << " vertices (got " << g_vertex_count << ")\n";
        return EXIT_FAILURE;
    }
    FrontierResult res = run_frontier_layer(opts);
    dump_json(opts, res);
    return EXIT_SUCCESS;
}
