#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Vec3 {
    int x;
    int y;
    int z;
};

struct Vec3Less {
    bool operator()(const Vec3 &a, const Vec3 &b) const {
        if (a.x != b.x) return a.x < b.x;
        if (a.y != b.y) return a.y < b.y;
        return a.z < b.z;
    }
};

struct Config {
    int max_edges = 6;
    int bound = -1;
    int first_direction = -1;
    bool collect_targets = true;
    std::vector<int> prefix_dirs;
};

constexpr Vec3 DIRS[6] = {
    {1, 0, 0},
    {-1, 0, 0},
    {0, 1, 0},
    {0, -1, 0},
    {0, 0, 1},
    {0, 0, -1},
};

constexpr int OPPOSITE_DIR[6] = {1, 0, 3, 2, 5, 4};

class ClusterEnumerator {
  public:
    explicit ClusterEnumerator(const Config &cfg)
        : max_edges_(cfg.max_edges),
          bound_((cfg.bound > 0) ? cfg.bound : cfg.max_edges),
          span_(2 * bound_ + 1),
          total_vertices_(span_ * span_ * span_),
          first_direction_(cfg.first_direction),
          collect_targets_(cfg.collect_targets),
          prefix_dirs_(cfg.prefix_dirs),
          origin_id_(encode_vertex(0, 0, 0)),
          id_to_coord_(total_vertices_),
          degrees_(total_vertices_, 0),
          parity_(total_vertices_, 0),
          active_pos_(total_vertices_, -1),
          odd_pos_(total_vertices_, -1),
          vertex_min_dir_(total_vertices_, 6),
          counts_(cfg.max_edges + 1, 0) {
        precompute_coords();
        add_active(origin_id_);
        vertex_min_dir_[origin_id_] = (first_direction_ >= 0) ? first_direction_ : 6;
    }

    void run() {
        dfs();
    }

    std::size_t state_count() const { return state_counter_; }

    const std::vector<long long> &counts() const { return counts_; }

    const std::map<Vec3, long long, Vec3Less> &targets() const { return target_hist_; }

  private:
    int max_edges_;
    int bound_;
    int span_;
    int total_vertices_;
    int first_direction_;
    bool collect_targets_;
    std::vector<int> prefix_dirs_;
    int origin_id_;

    std::vector<Vec3> id_to_coord_;
    std::vector<int> degrees_;
    std::vector<std::uint8_t> parity_;
    std::vector<int> active_vertices_;
    std::vector<int> active_pos_;
    std::vector<int> odd_vertices_;
    std::vector<int> odd_pos_;
    std::vector<int> vertex_min_dir_;
    std::vector<std::uint64_t> edges_;
    std::vector<long long> counts_;
    std::map<Vec3, long long, Vec3Less> target_hist_;
    std::size_t state_counter_ = 0;

    void precompute_coords() {
        for (int x = -bound_; x <= bound_; ++x) {
            for (int y = -bound_; y <= bound_; ++y) {
                for (int z = -bound_; z <= bound_; ++z) {
                    int idx = encode_vertex_raw(x, y, z);
                    id_to_coord_[idx] = {x, y, z};
                }
            }
        }
    }

    int encode_vertex_raw(int x, int y, int z) const {
        return ((x + bound_) * span_ + (y + bound_)) * span_ + (z + bound_);
    }

    int encode_vertex(int x, int y, int z) const {
        return encode_vertex_raw(x, y, z);
    }

    int encode_vertex(const Vec3 &v) const {
        return encode_vertex_raw(v.x, v.y, v.z);
    }

    bool within_bounds(const Vec3 &v) const {
        return v.x >= -bound_ && v.x <= bound_ && v.y >= -bound_ && v.y <= bound_ && v.z >= -bound_ && v.z <= bound_;
    }

    std::uint64_t encode_edge(int a, int b) const {
        if (a > b) std::swap(a, b);
        return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32) |
               static_cast<std::uint32_t>(b);
    }

    void add_active(int vertex) {
        if (active_pos_[vertex] != -1) return;
        active_pos_[vertex] = static_cast<int>(active_vertices_.size());
        active_vertices_.push_back(vertex);
    }

    void remove_active(int vertex) {
        if (vertex == origin_id_) return;
        int pos = active_pos_[vertex];
        if (pos == -1) return;
        int last = active_vertices_.back();
        active_vertices_[pos] = last;
        active_pos_[last] = pos;
        active_vertices_.pop_back();
        active_pos_[vertex] = -1;
    }

    void add_odd(int vertex) {
        if (odd_pos_[vertex] != -1) return;
        odd_pos_[vertex] = static_cast<int>(odd_vertices_.size());
        odd_vertices_.push_back(vertex);
    }

    void remove_odd(int vertex) {
        int pos = odd_pos_[vertex];
        if (pos == -1) return;
        int last = odd_vertices_.back();
        odd_vertices_[pos] = last;
        odd_pos_[last] = pos;
        odd_vertices_.pop_back();
        odd_pos_[vertex] = -1;
    }

    void toggle_parity(int vertex) {
        parity_[vertex] ^= 1u;
        if (parity_[vertex]) {
            add_odd(vertex);
        } else {
            remove_odd(vertex);
        }
    }

    void increase_degree(int vertex) {
        int before = degrees_[vertex];
        degrees_[vertex] = before + 1;
        if (before == 0) {
            add_active(vertex);
            if (vertex != origin_id_) {
                vertex_min_dir_[vertex] = 6;
            }
        }
    }

    void decrease_degree(int vertex) {
        int before = degrees_[vertex];
        degrees_[vertex] = before - 1;
        if (vertex != origin_id_ && degrees_[vertex] == 0) {
            remove_active(vertex);
            vertex_min_dir_[vertex] = 6;
        }
    }

    bool edge_exists(std::uint64_t edge) const {
        return std::binary_search(edges_.begin(), edges_.end(), edge);
    }

    std::size_t insert_edge(std::uint64_t edge) {
        auto it = std::lower_bound(edges_.begin(), edges_.end(), edge);
        std::size_t pos = static_cast<std::size_t>(it - edges_.begin());
        edges_.insert(it, edge);
        return pos;
    }

    void remove_edge_at(std::size_t pos) {
        edges_.erase(edges_.begin() + static_cast<std::ptrdiff_t>(pos));
    }

    void record_cluster(int num_edges) {
        if (num_edges <= 0) return;
        if (odd_vertices_.size() != 2) return;
        if (odd_pos_[origin_id_] == -1) return;
        int other = (odd_vertices_[0] == origin_id_) ? odd_vertices_[1]
                                                     : (odd_vertices_[1] == origin_id_ ? odd_vertices_[0] : -1);
        if (other < 0) return;
        counts_[num_edges] += 1;
        if (collect_targets_) {
            target_hist_[id_to_coord_[other]] += 1;
        }
    }

    void dfs() {
        state_counter_ += 1;
        int num_edges = static_cast<int>(edges_.size());
        record_cluster(num_edges);
        int remaining = max_edges_ - num_edges;
        if (remaining <= 0) {
            return;
        }
        int odd_count = static_cast<int>(odd_vertices_.size());
        int min_needed = (odd_count == 0) ? 1 : std::max(0, (odd_count - 2) / 2);
        if (odd_pos_[origin_id_] == -1) {
            min_needed = std::max(min_needed, 1);
        }
        if (min_needed > remaining) {
            return;
        }
        std::vector<int> snapshot = active_vertices_;
        for (int vertex_id : snapshot) {
            Vec3 base = id_to_coord_[vertex_id];
            for (int dir_idx = 0; dir_idx < 6; ++dir_idx) {
                int depth = static_cast<int>(edges_.size());
                if (!prefix_dirs_.empty() && depth < static_cast<int>(prefix_dirs_.size())) {
                    if (prefix_dirs_[depth] != dir_idx) {
                        continue;
                    }
                }
                Vec3 nxt = {base.x + DIRS[dir_idx].x, base.y + DIRS[dir_idx].y, base.z + DIRS[dir_idx].z};
                if (!within_bounds(nxt)) {
                    continue;
                }
                int nxt_id = encode_vertex(nxt);
                if (vertex_id != origin_id_ && nxt_id == origin_id_) {
                    continue;
                }
                int degree_vertex = degrees_[vertex_id];
                if (vertex_id == origin_id_) {
                    if (degree_vertex == 0) {
                        if (first_direction_ >= 0 && dir_idx != first_direction_) {
                            continue;
                        }
                    } else if (dir_idx < vertex_min_dir_[vertex_id]) {
                        continue;
                    }
                } else {
                    if (degree_vertex > 0 && dir_idx < vertex_min_dir_[vertex_id]) {
                        continue;
                    }
                }
                std::uint64_t edge = encode_edge(vertex_id, nxt_id);
                if (edge_exists(edge)) {
                    continue;
                }
                int degree_nxt = degrees_[nxt_id];
                int rev_dir = OPPOSITE_DIR[dir_idx];
                if (degree_nxt > 0 && rev_dir < vertex_min_dir_[nxt_id]) {
                    continue;
                }
                std::size_t pos = insert_edge(edge);
                int prev_min_vertex = vertex_min_dir_[vertex_id];
                int prev_min_nxt = vertex_min_dir_[nxt_id];
                if (degree_vertex == 0) {
                    vertex_min_dir_[vertex_id] = dir_idx;
                }
                if (degree_nxt == 0) {
                    vertex_min_dir_[nxt_id] = rev_dir;
                }
                increase_degree(vertex_id);
                increase_degree(nxt_id);
                toggle_parity(vertex_id);
                toggle_parity(nxt_id);
                dfs();
                toggle_parity(vertex_id);
                toggle_parity(nxt_id);
                decrease_degree(vertex_id);
                decrease_degree(nxt_id);
                vertex_min_dir_[vertex_id] = prev_min_vertex;
                vertex_min_dir_[nxt_id] = prev_min_nxt;
                remove_edge_at(pos);
            }
        }
    }
};

std::vector<int> parse_prefix(const std::string &text) {
    std::vector<int> result;
    if (text.empty()) {
        return result;
    }
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) continue;
        result.push_back(std::stoi(token));
    }
    return result;
}

Config parse_args(int argc, char **argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.rfind("--max-edges", 0) == 0) {
            auto pos = arg.find('=');
            if (pos == std::string::npos) {
                if (i + 1 >= argc) {
                    std::cerr << "Missing value for --max-edges\n";
                    std::exit(1);
                }
                cfg.max_edges = std::atoi(argv[++i]);
            } else {
                cfg.max_edges = std::atoi(arg.substr(pos + 1).c_str());
            }
        } else if (arg.rfind("--bound", 0) == 0) {
            auto pos = arg.find('=');
            if (pos == std::string::npos) {
                if (i + 1 >= argc) {
                    std::cerr << "Missing value for --bound\n";
                    std::exit(1);
                }
                cfg.bound = std::atoi(argv[++i]);
            } else {
                cfg.bound = std::atoi(arg.substr(pos + 1).c_str());
            }
        } else if (arg.rfind("--first-direction", 0) == 0) {
            auto pos = arg.find('=');
            if (pos == std::string::npos) {
                if (i + 1 >= argc) {
                    std::cerr << "Missing value for --first-direction\n";
                    std::exit(1);
                }
                cfg.first_direction = std::atoi(argv[++i]);
            } else {
                cfg.first_direction = std::atoi(arg.substr(pos + 1).c_str());
            }
        } else if (arg == "--skip-targets") {
            cfg.collect_targets = false;
        } else if (arg.rfind("--prefix", 0) == 0) {
            auto pos = arg.find('=');
            std::string value;
            if (pos == std::string::npos) {
                if (i + 1 >= argc) {
                    std::cerr << "Missing value for --prefix\n";
                    std::exit(1);
                }
                value = argv[++i];
            } else {
                value = arg.substr(pos + 1);
            }
            cfg.prefix_dirs = parse_prefix(value);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: cluster_enum [--max-edges N] [--bound B] [--first-direction D]"
                         " [--skip-targets] [--prefix d1,d2,...]\n";
            std::exit(0);
        }
    }
    return cfg;
}

void dump_json(const ClusterEnumerator &enumerator, int max_edges, bool collect_targets) {
    std::cout << "{\n";
    std::cout << "  \"max_edges\": " << max_edges << ",\n";
    std::cout << "  \"counts_by_length\": {\n";
    const auto &counts = enumerator.counts();
    bool first = true;
    for (int i = 1; i < static_cast<int>(counts.size()); ++i) {
        if (counts[i] == 0) continue;
        if (!first) std::cout << ",\n";
        first = false;
        std::cout << "    \"" << i << "\": " << counts[i];
    }
    if (!first) std::cout << "\n";
    std::cout << "  },\n";
    std::cout << "  \"target_hist\": [\n";
    if (collect_targets) {
        const auto &targets = enumerator.targets();
        bool first_target = true;
        for (const auto &entry : targets) {
            if (!first_target) std::cout << ",\n";
            first_target = false;
            const auto &vec = entry.first;
            std::cout << "    {\n";
            std::cout << "      \"target\": [" << vec.x << ", " << vec.y << ", " << vec.z << "],\n";
            std::cout << "      \"count\": " << entry.second << "\n";
            std::cout << "    }";
        }
        if (!targets.empty()) {
            std::cout << "\n";
        }
    }
    std::cout << "  ],\n";
    std::cout << "  \"state_count\": " << enumerator.state_count() << ",\n";
    std::cout << "  \"notes\": \"Enumerated via C++ DFS backend\"\n";
    std::cout << "}\n";
}

}  // namespace

int main(int argc, char **argv) {
    Config cfg = parse_args(argc, argv);
    ClusterEnumerator enumerator(cfg);
    enumerator.run();
    dump_json(enumerator, cfg.max_edges, cfg.collect_targets);
    return 0;
}
