#include "emergence_engine.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <unordered_set>

namespace emergence {

namespace {

struct PairHash {
    size_t operator()(const std::pair<std::string, std::string>& p) const noexcept {
        size_t h1 = std::hash<std::string>{}(p.first);
        size_t h2 = std::hash<std::string>{}(p.second);
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

template <class Container>
std::string encode_vec(const Container& v) {
    std::ostringstream oss;
    bool first = true;
    for (auto val : v) {
        if (!first) {
            oss << ',';
        }
        oss << static_cast<int>(val);
        first = false;
    }
    return oss.str();
}

// Combines two encoded strings into one joint key.
std::string join_keys(const std::string& a, const std::string& b) {
    std::string out;
    out.reserve(a.size() + b.size() + 1);
    out.append(a);
    out.push_back('|');
    out.append(b);
    return out;
}

template <class Key>
using Counter = std::unordered_map<Key, size_t>;

template <class Key>
double entropy(const Counter<Key>& counts, double total) {
    if (total == 0.0) return 0.0;
    double h = 0.0;
    for (const auto& [_, c] : counts) {
        double p = static_cast<double>(c) / total;
        if (p > 0) {
            h -= p * std::log(p);
        }
    }
    return h;
}

double mutual_information(const std::vector<std::string>& xs,
                          const std::vector<std::string>& ys) {
    const size_t n = xs.size();
    if (n == 0 || ys.size() != n) return 0.0;

    Counter<std::string> cx, cy;
    std::unordered_map<std::pair<std::string, std::string>, size_t, PairHash> cxy;
    cx.reserve(n * 2);
    cy.reserve(n * 2);
    cxy.reserve(n * 2);

    for (size_t i = 0; i < n; ++i) {
        const auto& x = xs[i];
        const auto& y = ys[i];
        cx[x]++;
        cy[y]++;
        cxy[{x, y}]++;
    }

    double total = static_cast<double>(n);
    double mi = 0.0;
    for (const auto& [xy, c_xy] : cxy) {
        double p_xy = static_cast<double>(c_xy) / total;
        double p_x = static_cast<double>(cx[xy.first]) / total;
        double p_y = static_cast<double>(cy[xy.second]) / total;
        if (p_xy > 0 && p_x > 0 && p_y > 0) {
            mi += p_xy * std::log(p_xy / (p_x * p_y));
        }
    }
    return mi;
}

// Effective information with uniform interventions over X.
double effective_information(const std::vector<std::string>& xs,
                             const std::vector<std::string>& ys) {
    const size_t n = xs.size();
    if (n == 0 || ys.size() != n) return 0.0;

    Counter<std::string> cx;
    std::unordered_map<std::pair<std::string, std::string>, size_t, PairHash> cxy;
    for (size_t i = 0; i < n; ++i) {
        cx[xs[i]]++;
        cxy[{xs[i], ys[i]}]++;
    }

    const double num_x = static_cast<double>(cx.size());
    if (num_x == 0.0) return 0.0;

    // Compute p_y under uniform p(x).
    std::unordered_map<std::string, double> p_y;
    for (const auto& [xy, c_xy] : cxy) {
        const auto& x = xy.first;
        const auto& y = xy.second;
        double p_y_given_x = static_cast<double>(c_xy) / static_cast<double>(cx[x]);
        p_y[y] += (1.0 / num_x) * p_y_given_x;
    }

    double ei = 0.0;
    for (const auto& [xy, c_xy] : cxy) {
        const auto& x = xy.first;
        const auto& y = xy.second;
        double p_xy = (1.0 / num_x) * (static_cast<double>(c_xy) / static_cast<double>(cx[x]));
        double p_x = 1.0 / num_x;
        double p_y_val = p_y[y];
        if (p_xy > 0 && p_y_val > 0) {
            ei += p_xy * std::log(p_xy / (p_x * p_y_val));
        }
    }
    return ei;
}

double synergy_two_sources(const std::vector<std::string>& a,
                           const std::vector<std::string>& b,
                           const std::vector<std::string>& target) {
    if (a.size() != b.size() || a.size() != target.size() || a.empty()) return 0.0;
    std::vector<std::string> joint;
    joint.reserve(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        joint.push_back(join_keys(a[i], b[i]));
    }
    double mi_joint = mutual_information(joint, target);
    double mi_a = mutual_information(a, target);
    double mi_b = mutual_information(b, target);
    return mi_joint - mi_a - mi_b; // "interaction information" style
}

} // namespace

// --- Lattice3D --------------------------------------------------------------

Lattice3D::Lattice3D(const LatticeConfig& cfg) : cfg_(cfg) {
    volume_ = cfg_.dims[0] * cfg_.dims[1] * cfg_.dims[2];
    build_neighbors();
}

int Lattice3D::index(int x, int y, int z) const {
    return (z * cfg_.dims[1] + y) * cfg_.dims[0] + x;
}

std::array<int, 3> Lattice3D::coords(int idx) const {
    int x = idx % cfg_.dims[0];
    int y = (idx / cfg_.dims[0]) % cfg_.dims[1];
    int z = idx / (cfg_.dims[0] * cfg_.dims[1]);
    return {x, y, z};
}

void Lattice3D::build_neighbors() {
    neighbors_.assign(volume_, {});
    const auto wrap = [&](int coord, int dim, int delta) {
        int v = coord + delta;
        if (cfg_.periodic) {
            v = (v % dim + dim) % dim;
        }
        return v;
    };

    for (int z = 0; z < cfg_.dims[2]; ++z) {
        for (int y = 0; y < cfg_.dims[1]; ++y) {
            for (int x = 0; x < cfg_.dims[0]; ++x) {
                int idx = index(x, y, z);
                auto& nb = neighbors_[idx];
                nb.reserve(6);
                // Six nearest neighbors
                const std::array<std::array<int, 3>, 6> deltas = {{
                    {1, 0, 0}, {-1, 0, 0},
                    {0, 1, 0}, {0, -1, 0},
                    {0, 0, 1}, {0, 0, -1},
                }};
                for (const auto& d : deltas) {
                    int nx = wrap(x, cfg_.dims[0], d[0]);
                    int ny = wrap(y, cfg_.dims[1], d[1]);
                    int nz = wrap(z, cfg_.dims[2], d[2]);
                    if (!cfg_.periodic) {
                        if (nx < 0 || nx >= cfg_.dims[0] ||
                            ny < 0 || ny >= cfg_.dims[1] ||
                            nz < 0 || nz >= cfg_.dims[2]) {
                            continue;
                        }
                    }
                    nb.push_back(index(nx, ny, nz));
                }
            }
        }
    }
}

double Lattice3D::energy(const State& state) const {
    double e = 0.0;
    // Field term
    for (int i = 0; i < volume_; ++i) {
        e -= cfg_.h * static_cast<double>(state[i]);
    }
    // Interaction term (each edge counted twice -> multiply by 0.5)
    for (int i = 0; i < volume_; ++i) {
        double nb_sum = 0.0;
        for (int nb : neighbors_[i]) {
            nb_sum += static_cast<double>(state[nb]);
        }
        e -= 0.5 * cfg_.J * static_cast<double>(state[i]) * nb_sum;
    }
    return e;
}

// --- TrajectorySampler ------------------------------------------------------

TrajectorySampler::TrajectorySampler(Lattice3D lattice, SamplerParams params)
    : lattice_(std::move(lattice)), params_(params), rng_(params.seed) {}

State TrajectorySampler::random_state() {
    State s(lattice_.volume());
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    for (auto& v : s) {
        v = (uni(rng_) < 0.5) ? -1 : 1;
    }
    return s;
}

void TrajectorySampler::metropolis_step(State& state) {
    std::uniform_int_distribution<int> site_dist(0, lattice_.volume() - 1);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    int idx = site_dist(rng_);
    double nb_sum = 0.0;
    for (int nb : lattice_.neighbors()[idx]) {
        nb_sum += static_cast<double>(state[nb]);
    }
    const auto& cfg = lattice_.config();
    double delta_e = 2.0 * static_cast<double>(state[idx]) * (params_.beta * cfg.J * nb_sum + params_.beta * cfg.h);
    if (delta_e <= 0.0 || std::exp(-delta_e) > uni(rng_)) {
        state[idx] = -state[idx];
    }
}

std::vector<Trajectory> TrajectorySampler::run() {
    std::vector<Trajectory> out;
    out.reserve(params_.trajectories);

    for (int t = 0; t < params_.trajectories; ++t) {
        State state = random_state();
        for (int i = 0; i < params_.burn_in; ++i) {
            metropolis_step(state);
        }
        Trajectory traj;
        traj.steps.reserve(params_.steps + 1);
        traj.steps.push_back(state);
        for (int step = 0; step < params_.steps; ++step) {
            for (int k = 0; k < params_.sample_interval; ++k) {
                metropolis_step(state);
            }
            traj.steps.push_back(state);
        }
        out.push_back(std::move(traj));
    }
    return out;
}

// --- Macro builders ---------------------------------------------------------

MacroState macro_magnetization(const State& s) {
    int sum = std::accumulate(s.begin(), s.end(), 0);
    return {sum};
}

MacroState macro_energy(const State& s, const Lattice3D& lattice) {
    double e = lattice.energy(s);
    return {static_cast<int>(std::lround(e))};
}

MacroState macro_layer_magnetization(const State& s, const Lattice3D& lattice, int axis) {
    auto dims = lattice.dims();
    MacroState m(dims[axis], 0);
    for (int idx = 0; idx < lattice.volume(); ++idx) {
        auto c = lattice.coords(idx);
        m[c[axis]] += static_cast<int>(s[idx]);
    }
    return m;
}

MacroState macro_block_majority(const State& s, const Lattice3D& lattice, const std::array<int, 3>& block) {
    auto dims = lattice.dims();

    std::vector<int> out;
    for (int z0 = 0; z0 < dims[2]; z0 += block[2]) {
        for (int y0 = 0; y0 < dims[1]; y0 += block[1]) {
            for (int x0 = 0; x0 < dims[0]; x0 += block[0]) {
                int sum = 0;
                for (int dz = 0; dz < block[2] && z0 + dz < dims[2]; ++dz) {
                    for (int dy = 0; dy < block[1] && y0 + dy < dims[1]; ++dy) {
                        for (int dx = 0; dx < block[0] && x0 + dx < dims[0]; ++dx) {
                            int idx = lattice.index(x0 + dx, y0 + dy, z0 + dz);
                            sum += static_cast<int>(s[idx]);
                        }
                    }
                }
                if (sum > 0) out.push_back(1);
                else if (sum < 0) out.push_back(-1);
                else out.push_back(0);
            }
        }
    }
    return out;
}

MacroState macro_domain_walls(const State& s, const Lattice3D& lattice) {
    int walls = 0;
    auto dims = lattice.dims();
    const auto& cfg = lattice.config();
    for (int idx = 0; idx < lattice.volume(); ++idx) {
        auto c = lattice.coords(idx);
        const std::array<std::array<int, 3>, 3> deltas = {{
            {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
        }};
        for (const auto& d : deltas) {
            int nx = c[0] + d[0];
            int ny = c[1] + d[1];
            int nz = c[2] + d[2];
            if (cfg.periodic) {
                nx = (nx % dims[0] + dims[0]) % dims[0];
                ny = (ny % dims[1] + dims[1]) % dims[1];
                nz = (nz % dims[2] + dims[2]) % dims[2];
            } else {
                if (nx < 0 || nx >= dims[0] || ny < 0 || ny >= dims[1] || nz < 0 || nz >= dims[2]) {
                    continue;
                }
            }
            int nb_idx = lattice.index(nx, ny, nz);
            if (s[idx] != s[nb_idx]) ++walls;
        }
    }
    return {walls};
}

MacroState apply_macro(const State& s, const Lattice3D& lattice, const MacroSpec& spec) {
    switch (spec.type) {
        case MacroSpec::Type::Magnetization:
            return macro_magnetization(s);
        case MacroSpec::Type::Energy:
            return macro_energy(s, lattice);
        case MacroSpec::Type::BlockMajority:
            return macro_block_majority(s, lattice, spec.block_shape);
        case MacroSpec::Type::LayerMagnetization:
            return macro_layer_magnetization(s, lattice, spec.axis);
        case MacroSpec::Type::DomainWalls:
            return macro_domain_walls(s, lattice);
        default:
            return macro_magnetization(s);
    }
}

// --- Partition helpers ------------------------------------------------------

struct SplitState {
    MacroState a;
    MacroState b;
};

SplitState split_micro(const State& s, const Lattice3D& lattice, const PartitionSpec& p) {
    SplitState out;
    out.a.reserve(s.size());
    out.b.reserve(s.size());

    if (p.type == PartitionSpec::Type::Explicit && !p.maskA.empty()) {
        std::unordered_set<int> mask(p.maskA.begin(), p.maskA.end());
        for (int idx = 0; idx < static_cast<int>(s.size()); ++idx) {
            if (mask.count(idx)) out.a.push_back(s[idx]);
            else out.b.push_back(s[idx]);
        }
        return out;
    }

    int axis = p.axis;
    int cut = std::max(1, lattice.dims()[axis] / 2);
    for (int idx = 0; idx < static_cast<int>(s.size()); ++idx) {
        auto c = lattice.coords(idx);
        if (c[axis] < cut) out.a.push_back(s[idx]);
        else out.b.push_back(s[idx]);
    }
    return out;
}

SplitState split_macro(const MacroState& m) {
    SplitState out;
    int mid = static_cast<int>(m.size()) / 2;
    out.a.insert(out.a.end(), m.begin(), m.begin() + mid);
    out.b.insert(out.b.end(), m.begin() + mid, m.end());
    return out;
}

// --- EmergenceAnalyzer ------------------------------------------------------

EmergenceAnalyzer::EmergenceAnalyzer(const Lattice3D& lattice,
                                     const PartitionSpec& partition,
                                     int ppmi_lag)
    : lattice_(lattice), partition_(partition), ppmi_lag_(ppmi_lag) {}

EmergenceMetrics EmergenceAnalyzer::compute_micro(const std::vector<Trajectory>& trajectories) const {
    std::vector<std::string> x_t;
    std::vector<std::string> x_next;
    std::vector<std::string> x_lag;

    std::vector<std::string> a_t, a_next, b_t, b_next;

    for (const auto& traj : trajectories) {
        if (traj.steps.size() < 2) continue;
        for (size_t i = 0; i + 1 < traj.steps.size(); ++i) {
            const auto& s = traj.steps[i];
            const auto& s1 = traj.steps[i + 1];
            x_t.push_back(encode_vec(s));
            x_next.push_back(encode_vec(s1));

            auto split0 = split_micro(s, lattice_, partition_);
            auto split1 = split_micro(s1, lattice_, partition_);
            a_t.push_back(encode_vec(split0.a));
            b_t.push_back(encode_vec(split0.b));
            a_next.push_back(encode_vec(split1.a));
            b_next.push_back(encode_vec(split1.b));
        }
        if (traj.steps.size() > static_cast<size_t>(ppmi_lag_)) {
            for (size_t i = 0; i + ppmi_lag_ < traj.steps.size(); ++i) {
                x_lag.push_back(encode_vec(traj.steps[i]));
                x_lag.push_back(encode_vec(traj.steps[i + ppmi_lag_]));
            }
        }
    }

    EmergenceMetrics m;
    Counter<std::string> cx;
    for (const auto& k : x_t) cx[k]++;
    m.entropy = entropy(cx, static_cast<double>(x_t.size()));
    m.mutual_information = mutual_information(x_t, x_next);
    m.effective_information = effective_information(x_t, x_next);
    double mi_a = mutual_information(a_t, a_next);
    double mi_b = mutual_information(b_t, b_next);
    m.phi = m.mutual_information - (mi_a + mi_b);
    m.synergy = synergy_two_sources(a_t, b_t, x_next);

    if (!x_lag.empty()) {
        // x_lag pairs are stored sequentially: (t, t+lag)
        std::vector<std::string> first, second;
        first.reserve(x_lag.size() / 2);
        second.reserve(x_lag.size() / 2);
        for (size_t i = 0; i + 1 < x_lag.size(); i += 2) {
            first.push_back(x_lag[i]);
            second.push_back(x_lag[i + 1]);
        }
        m.ppmi = mutual_information(first, second);
    }
    return m;
}

MacroReport EmergenceAnalyzer::compute_macro(const std::vector<Trajectory>& trajectories,
                                             const MacroSpec& spec,
                                             const EmergenceMetrics& micro_baseline) const {
    std::vector<std::string> m_t, m_next;
    std::vector<std::string> a_t, a_next, b_t, b_next;
    std::vector<std::string> m_lag_first, m_lag_second;

    for (const auto& traj : trajectories) {
        if (traj.steps.size() < 2) continue;
        for (size_t i = 0; i + 1 < traj.steps.size(); ++i) {
            MacroState m0 = apply_macro(traj.steps[i], lattice_, spec);
            MacroState m1 = apply_macro(traj.steps[i + 1], lattice_, spec);
            m_t.push_back(encode_vec(m0));
            m_next.push_back(encode_vec(m1));

            auto split0 = split_macro(m0);
            auto split1 = split_macro(m1);
            a_t.push_back(encode_vec(split0.a));
            b_t.push_back(encode_vec(split0.b));
            a_next.push_back(encode_vec(split1.a));
            b_next.push_back(encode_vec(split1.b));
        }
        if (traj.steps.size() > static_cast<size_t>(ppmi_lag_)) {
            for (size_t i = 0; i + ppmi_lag_ < traj.steps.size(); ++i) {
                MacroState m0 = apply_macro(traj.steps[i], lattice_, spec);
                MacroState m1 = apply_macro(traj.steps[i + ppmi_lag_], lattice_, spec);
                m_lag_first.push_back(encode_vec(m0));
                m_lag_second.push_back(encode_vec(m1));
            }
        }
    }

    EmergenceMetrics metrics;
    Counter<std::string> cm;
    for (const auto& k : m_t) cm[k]++;
    metrics.entropy = entropy(cm, static_cast<double>(m_t.size()));
    metrics.mutual_information = mutual_information(m_t, m_next);
    metrics.effective_information = effective_information(m_t, m_next);

    double mi_a = mutual_information(a_t, a_next);
    double mi_b = mutual_information(b_t, b_next);
    metrics.phi = metrics.mutual_information - (mi_a + mi_b);
    metrics.synergy = synergy_two_sources(a_t, b_t, m_next);
    if (!m_lag_first.empty()) {
        metrics.ppmi = mutual_information(m_lag_first, m_lag_second);
    }

    MacroReport rep;
    rep.spec = spec;
    rep.metrics = metrics;
    rep.emergence_score =
        (metrics.effective_information - micro_baseline.effective_information) +
        (metrics.synergy - micro_baseline.synergy) +
        (metrics.phi - micro_baseline.phi);
    return rep;
}

EmergenceReport EmergenceAnalyzer::analyze(const std::vector<Trajectory>& trajectories,
                                           const std::vector<MacroSpec>& macros,
                                           const EmergenceMetrics& micro_baseline_hint) const {
    EmergenceReport report{};
    report.lattice = lattice_.config();
    report.partition = partition_;
    // We reuse sampler defaults as a placeholder; caller can fill if needed.
    SamplerParams dummy;
    report.sampler = dummy;

    EmergenceMetrics micro_metrics = micro_baseline_hint;
    if (micro_metrics.entropy == 0.0 && trajectories.size() > 0) {
        micro_metrics = compute_micro(trajectories);
    }
    report.micro = micro_metrics;

    for (const auto& spec : macros) {
        report.macros.push_back(compute_macro(trajectories, spec, micro_metrics));
    }
    return report;
}

// --- JSON output ------------------------------------------------------------

std::string to_json(const EmergenceReport& report) {
    std::ostringstream oss;
    oss << "{";
    oss << "\"lattice\":{\"dims\":[" << report.lattice.dims[0] << "," << report.lattice.dims[1]
        << "," << report.lattice.dims[2] << "],\"J\":" << report.lattice.J
        << ",\"h\":" << report.lattice.h << ",\"periodic\":" << (report.lattice.periodic ? "true" : "false") << "},";
    oss << "\"sampler\":{\"burn_in\":" << report.sampler.burn_in << ",\"steps\":" << report.sampler.steps
        << ",\"sample_interval\":" << report.sampler.sample_interval << ",\"trajectories\":" << report.sampler.trajectories
        << ",\"beta\":" << report.sampler.beta << ",\"seed\":" << report.sampler.seed << "},";
    oss << "\"micro\":{"
        << "\"entropy\":" << report.micro.entropy
        << ",\"mi\":" << report.micro.mutual_information
        << ",\"ei\":" << report.micro.effective_information
        << ",\"phi\":" << report.micro.phi
        << ",\"synergy\":" << report.micro.synergy
        << ",\"ppmi\":" << report.micro.ppmi
        << "},";
    oss << "\"macros\":[";
    for (size_t i = 0; i < report.macros.size(); ++i) {
        const auto& m = report.macros[i];
        oss << "{";
        oss << "\"name\":\"" << (m.spec.name.empty() ? "macro_" + std::to_string(i) : m.spec.name) << "\",";
        oss << "\"mi\":" << m.metrics.mutual_information
            << ",\"ei\":" << m.metrics.effective_information
            << ",\"phi\":" << m.metrics.phi
            << ",\"synergy\":" << m.metrics.synergy
            << ",\"ppmi\":" << m.metrics.ppmi
            << ",\"emergence_score\":" << m.emergence_score;
        oss << "}";
        if (i + 1 < report.macros.size()) oss << ",";
    }
    oss << "]";
    oss << "}";
    return oss.str();
}

} // namespace emergence
