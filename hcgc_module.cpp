#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <random>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

// Coarsened graph data produced by CSRGraph::build_coarsened_data()
struct CoarsenedData {
  int num_nodes_new = 0;
  std::vector<int> old_to_new;   // [orig node] -> new supernode ID (-1 if not a root)
  std::vector<int> new_to_old;   // [new supernode ID] -> representative orig node
  std::vector<int> src, dst;
  std::vector<float> edge_weights;
  std::vector<float> features;   // centroid features, contiguous by type
  std::vector<int> type_boundaries_new;
  std::vector<int> feature_dims_new;
};

// Compressed Sparse Row storage for heterogeneous graphs (HCGC path only)
struct CSRGraph {
  // ── Graph structure ───────────────────────────────────────────────────────
  std::vector<int>   node_ptr;
  std::vector<int>   edge_dst;
  std::vector<float> edge_weight;
  int num_nodes;

  // ── Feature data (pointers into external or owned arrays) ────────────────
  const float *node_features_1d;
  const int   *feature_dims;
  const int   *type_boundaries;
  int          num_types;
  std::vector<int> feature_offsets;

  // ── Owned storage for coarsened-level graphs ──────────────────────────────
  std::vector<float> owned_features;
  std::vector<int>   owned_type_boundaries;
  std::vector<int>   owned_feature_dims_vec;

  // ── Coalition tracking (union-find) ──────────────────────────────────────
  std::vector<int>               coalition_map;    // parent pointer; root ↔ coalition_map[v]==v
  std::vector<int>               coalition_size;
  std::vector<std::vector<float>> coalition_feat_sum;  // sum of features per coalition root

  // ── HCGC: per-type feature variance (adaptive merge threshold) ───────────
  // feat_var[t] = average squared L2 distance from type mean.
  // Used as the merge threshold in coarse_graph_hcgc().
  std::vector<float>              feat_var;
  std::vector<std::vector<float>> type_mean;  // [t][d] = centroid of type t

  // ── Hub-anchor: top-k% degree nodes per type are frozen ──────────────────
  float             hub_anchor_percentile = 0.0f;  // 0 = disabled; 0.01 = top 1%
  std::vector<bool> frozen;                        // frozen[v] → cannot be a merge candidate
  std::vector<bool> finalized;                     // finalized[v] → merged by Ball Multi-Merge; locked permanently so rebuilt singletons cannot thrash

  // ── Per-type merge threshold overrides ───────────────────────────────────
  // If non-empty: threshold for type t = feat_var_scale_per_type[t] * feat_var[t]
  // Falls back to the global feat_var_scale scalar when empty or out-of-range.
  std::vector<float> feat_var_scale_per_type;

  // ── Scalability caps ──────────────────────────────────────────────────────
  int              max_candidates_per_mediator = 0;
  int              max_hub_degree = 0;
  std::vector<int>   hub_degree_caps;   // per-type; 0 = unlimited for that type
  std::vector<float> type_mean_deg;     // per-type mean node degree (parameter-free deg weighting)

  // ── Helpers ───────────────────────────────────────────────────────────────

  inline int hub_cap_for_type(int t) const {
    if (!hub_degree_caps.empty() && t >= 0 && t < (int)hub_degree_caps.size())
      return hub_degree_caps[t];
    return max_hub_degree;
  }

  // ── Auto hub-degree caps (mean + k_sigma * std of total degree per type) ──
  void compute_auto_hub_caps(float k_sigma = 3.0f) {
    hub_degree_caps.resize(num_types, 0);
    std::cout << "[C++] Auto hub caps (mean + " << k_sigma << "*std of total degree):\n";
    for (int t = 0; t < num_types; ++t) {
      int start = (t > 0) ? type_boundaries[t - 1] : 0;
      int end   = type_boundaries[t];
      int n     = end - start;
      if (n == 0) continue;
      double sum = 0.0, sum_sq = 0.0;
      for (int u = start; u < end; ++u) {
        int deg = node_ptr[u + 1] - node_ptr[u];
        sum    += deg;
        sum_sq += (double)deg * deg;
      }
      double mean = sum / n;
      double std_ = std::sqrt(std::max(0.0, sum_sq / n - mean * mean));
      int cap = std::max(1, static_cast<int>(std::round(mean + k_sigma * std_)));
      hub_degree_caps[t] = cap;
      std::cout << "  type " << t << "  deg_mean=" << mean
                << "  deg_std=" << std_ << "  cap=" << cap << "\n";
    }
  }

  // ── Per-type mean degree (used for parameter-free degree penalty) ─────────
  // deg_factor = 1 + log1p(max_deg / mean_deg_type)
  // No free hyperparameter — entirely derived from the graph's degree distribution.
  // High-degree nodes (hubs) are naturally penalized more without needing role_alpha.
  void compute_type_mean_deg() {
    type_mean_deg.resize(num_types, 1.0f);
    for (int t = 0; t < num_types; ++t) {
      int start = (t > 0) ? type_boundaries[t - 1] : 0;
      int end   = type_boundaries[t];
      int n     = end - start;
      if (n == 0) { type_mean_deg[t] = 1.0f; continue; }
      double sum = 0.0;
      for (int u = start; u < end; ++u)
        sum += (node_ptr[u + 1] - node_ptr[u]);
      type_mean_deg[t] = static_cast<float>(std::max(1.0, sum / n));
    }
    std::cout << "[C++] HCGC type_mean_deg:";
    for (int t = 0; t < num_types; ++t)
      std::cout << " t" << t << "=" << type_mean_deg[t];
    std::cout << "\n";
    std::cout.flush();
  }

  // ── Hub-anchor: freeze top hub_anchor_percentile fraction per type ────────
  void compute_frozen_nodes() {
    frozen.assign(num_nodes, false);
    if (hub_anchor_percentile <= 0.0f) return;

    int frozen_total = 0;
    for (int t = 0; t < num_types; ++t) {
      int start = (t > 0) ? type_boundaries[t - 1] : 0;
      int end   = type_boundaries[t];
      int n     = end - start;
      if (n == 0) continue;

      std::vector<std::pair<int,int>> deg_nodes;
      deg_nodes.reserve(n);
      for (int u = start; u < end; ++u)
        deg_nodes.push_back({node_ptr[u+1] - node_ptr[u], u});

      std::sort(deg_nodes.begin(), deg_nodes.end(),
                [](const auto &a, const auto &b){ return a.first > b.first; });

      int k = std::max(1, static_cast<int>(std::ceil(n * hub_anchor_percentile)));
      if (deg_nodes[0].first <= 1) continue;  // no meaningful degree variation

      for (int i = 0; i < k && i < n; ++i)
        frozen[deg_nodes[i].second] = true;

      frozen_total += k;
      std::cout << "  [HubAnchor] type " << t
                << "  n=" << n << "  frozen top-" << k
                << "  (deg " << deg_nodes[k-1].first << "+"
                << "  max=" << deg_nodes[0].first << ")\n";
    }
    std::cout << "  [HubAnchor] total frozen: " << frozen_total
              << " / " << num_nodes << " nodes\n";
  }

  inline int get_node_type(int u) const {
    for (int t = 0; t < num_types; ++t)
      if (u < type_boundaries[t]) return t;
    return -1;
  }

  inline const float *get_node_feature(int u) const {
    int t = get_node_type(u);
    if (t < 0) return nullptr;
    int local = u - (t > 0 ? type_boundaries[t - 1] : 0);
    return node_features_1d + feature_offsets[t] + local * feature_dims[t];
  }

  // ── Initialization ────────────────────────────────────────────────────────

  void init_features(const float *features_1d, const int *boundaries,
                     const int *dims, int types) {
    node_features_1d = features_1d;
    type_boundaries  = boundaries;
    feature_dims     = dims;
    num_types        = types;
    feature_offsets.resize(num_types, 0);
    int off = 0;
    for (int t = 0; t < num_types; ++t) {
      feature_offsets[t] = off;
      int cnt = type_boundaries[t] - (t > 0 ? type_boundaries[t - 1] : 0);
      off += cnt * feature_dims[t];
    }
  }

  void init_features_from_owned(std::vector<float> feats,
                                std::vector<int>   boundaries,
                                std::vector<int>   dims) {
    owned_features          = std::move(feats);
    owned_type_boundaries   = std::move(boundaries);
    owned_feature_dims_vec  = std::move(dims);
    num_types        = static_cast<int>(owned_type_boundaries.size());
    node_features_1d = owned_features.data();
    type_boundaries  = owned_type_boundaries.data();
    feature_dims     = owned_feature_dims_vec.data();
    feature_offsets.resize(num_types, 0);
    int cur = 0;
    for (int t = 0; t < num_types; ++t) {
      feature_offsets[t] = cur;
      int cnt = owned_type_boundaries[t] - (t > 0 ? owned_type_boundaries[t - 1] : 0);
      cur += cnt * owned_feature_dims_vec[t];
    }
  }

  void build_from_edgelist(int n, const int *src, const int *dst,
                           const float *weights, int num_edges) {
    num_nodes = n;
    node_ptr.assign(num_nodes + 1, 0);
    edge_dst.resize(num_edges * 2);
    edge_weight.resize(num_edges * 2);

    for (int i = 0; i < num_edges; ++i) {
      node_ptr[src[i]]++;
      node_ptr[dst[i]]++;
    }
    int cur = 0;
    for (int i = 0; i < num_nodes; ++i) {
      int d = node_ptr[i];
      node_ptr[i] = cur;
      cur += d;
    }
    node_ptr[num_nodes] = cur;

    std::vector<int> cursor = node_ptr;
    for (int i = 0; i < num_edges; ++i) {
      int u = src[i], v = dst[i];
      float w = weights[i];
      edge_dst[cursor[u]]   = v;
      edge_weight[cursor[u]++] = w;
      edge_dst[cursor[v]]   = u;
      edge_weight[cursor[v]++] = w;
    }
  }

  // ── HCGC: per-type feature variance (adaptive merge threshold) ───────────
  // Computes the average squared L2 distance of each node from its type mean.
  // Serves as the natural Dirichlet-energy threshold: merge two coalitions
  // whose squared centroid distance is below this "typical spread".
  void compute_feat_var() {
    feat_var.resize(num_types, 1.0f);
    type_mean.resize(num_types);
    for (int t = 0; t < num_types; ++t) {
      int start = (t > 0) ? type_boundaries[t - 1] : 0;
      int end   = type_boundaries[t];
      int n = end - start, dim = feature_dims[t];
      type_mean[t].assign(dim, 0.0f);
      if (n < 2 || dim == 0) { feat_var[t] = 1.0f; continue; }

      std::vector<double> mean(dim, 0.0);
      for (int u = start; u < end; ++u) {
        const float *f = get_node_feature(u);
        for (int d = 0; d < dim; ++d) mean[d] += f[d];
      }
      for (int d = 0; d < dim; ++d) mean[d] /= n;
      for (int d = 0; d < dim; ++d)
        type_mean[t][d] = static_cast<float>(mean[d]);

      double var = 0.0;
      for (int u = start; u < end; ++u) {
        const float *f = get_node_feature(u);
        for (int d = 0; d < dim; ++d) {
          double diff = f[d] - mean[d];
          var += diff * diff;
        }
      }
      float fv = static_cast<float>(var / n);
      if (!std::isfinite(fv) || fv <= 0.0f) {
        feat_var[t] = 1.0f;
        std::cout << "[C++] HCGC type " << t
                  << " feat_var non-finite (raw=" << fv << "), fallback=1.0\n";
      } else {
        feat_var[t] = fv;
        std::cout << "[C++] HCGC type " << t << " feat_var=" << feat_var[t] << "\n";
      }
    }
  }

  // ── Coalition union-find ──────────────────────────────────────────────────

  inline int find_root(int x) {
    while (coalition_map[x] != x) x = coalition_map[x];
    return x;
  }

  // Flatten all nodes to point directly at their root (path compression).
  void normalize_coalition_map() {
    for (int i = 0; i < num_nodes; ++i)
      coalition_map[i] = find_root(i);
  }

  // Merge coalition `from` into coalition `into`.
  inline void merge_coalitions(int from, int into) {
    if (from == into) return;
    int dim = (int)coalition_feat_sum[into].size();
    for (int d = 0; d < dim; ++d)
      coalition_feat_sum[into][d] += coalition_feat_sum[from][d];
    coalition_size[into] += coalition_size[from];
    coalition_size[from]  = 0;
    coalition_map[from]   = into;
  }

  void init_coalition_map() {
    coalition_map.assign(num_nodes, 0);
    coalition_size.assign(num_nodes, 1);
    coalition_feat_sum.resize(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
      coalition_map[i] = i;
      int t = get_node_type(i), dim = feature_dims[t];
      const float *f = get_node_feature(i);
      coalition_feat_sum[i].assign(f, f + dim);
    }
  }

  // ── Coarsened graph construction ──────────────────────────────────────────

  CoarsenedData build_coarsened_data() {
    CoarsenedData cd;
    cd.old_to_new.assign(num_nodes, -1);
    cd.feature_dims_new.assign(feature_dims, feature_dims + num_types);
    cd.type_boundaries_new.resize(num_types);

    int new_id = 0;
    for (int t = 0; t < num_types; ++t) {
      int start = (t > 0) ? type_boundaries[t - 1] : 0;
      int end   = type_boundaries[t];
      for (int u = start; u < end; ++u)
        if (coalition_map[u] == u) {   // u is a root
          cd.old_to_new[u] = new_id++;
          cd.new_to_old.push_back(u);
        }
      cd.type_boundaries_new[t] = new_id;
    }
    cd.num_nodes_new = new_id;

    int total_feat = 0;
    for (int t = 0; t < num_types; ++t) {
      int cnt = cd.type_boundaries_new[t] -
                (t > 0 ? cd.type_boundaries_new[t - 1] : 0);
      total_feat += cnt * feature_dims[t];
    }
    cd.features.resize(total_feat);

    int feat_off = 0;
    for (int t = 0; t < num_types; ++t) {
      int start = (t > 0) ? type_boundaries[t - 1] : 0;
      int end   = type_boundaries[t], dim = feature_dims[t];
      for (int u = start; u < end; ++u) {
        if (coalition_map[u] == u) {   // root: emit centroid
          float inv = 1.0f / static_cast<float>(coalition_size[u]);
          for (int d = 0; d < dim; ++d)
            cd.features[feat_off + d] = coalition_feat_sum[u][d] * inv;
          feat_off += dim;
        }
      }
    }

    // Use unordered_map<int64_t> for O(1) amortised lookup and better cache
    // behaviour vs map<pair<int,int>> — ~20-50x faster on large graphs.
    std::unordered_map<int64_t, float> edge_map;
    edge_map.reserve(static_cast<size_t>(num_nodes) * 4);

    const int progress_step = std::max(1, num_nodes / 20);
    for (int u = 0; u < num_nodes; ++u) {
      if (u % progress_step == 0) {
        std::cout << "[HCGC] build_coarsened_data: " << u << "/" << num_nodes
                  << " (" << (100 * u / num_nodes) << "%)  "
                  << "edges so far: " << edge_map.size() << "\n";
        std::cout.flush();
      }
      int nu = cd.old_to_new[find_root(u)];
      for (int e = node_ptr[u]; e < node_ptr[u + 1]; ++e) {
        int v = edge_dst[e];
        if (u >= v) continue;
        int nv = cd.old_to_new[find_root(v)];
        if (nu == nv) continue;
        int64_t key = (nu < nv)
            ? (((int64_t)nu << 32) | (int64_t)(uint32_t)nv)
            : (((int64_t)nv << 32) | (int64_t)(uint32_t)nu);
        edge_map[key] += edge_weight[e];
      }
    }
    std::cout << "[HCGC] build_coarsened_data: " << num_nodes << "/" << num_nodes
              << " (100%)  total edges: " << edge_map.size() << "\n";
    std::cout.flush();
    cd.src.reserve(edge_map.size());
    cd.dst.reserve(edge_map.size());
    cd.edge_weights.reserve(edge_map.size());
    for (auto &kv : edge_map) {
      cd.src.push_back(static_cast<int>(kv.first >> 32));
      cd.dst.push_back(static_cast<int>(kv.first & 0xFFFFFFFFLL));
      cd.edge_weights.push_back(kv.second);
    }
    return cd;
  }

  // ── Node Reassignment (Jacobi-style best-response) ───────────────────────
  //
  // After Ball Multi-Merge, each node checks whether it would lower its
  // Dirichlet energy by moving to a different coalition it can see via
  // cross-type mediator paths.  All moves are recorded first (read phase),
  // then applied atomically (Jacobi write phase) to avoid order-dependency.
  //
  // Only singleton/non-root nodes can switch: multi-member roots would drag
  // their children along (that is a merge, not a switch).
  //
  // Threshold: same as Ball Multi-Merge (feat_var_scale * feat_var[t]).
  // Using the same energy criterion keeps reassignment consistent with the
  // merge step and prevents cascade collapse from over-aggressive re-routing.
  //
  int reassignment_pass(float feat_var_scale) {
    normalize_coalition_map();  // flatten: coalition_map[v] == root directly

    struct PendingSwitch { int node_v, t_src, old_root, new_root; };
    std::vector<PendingSwitch> pending;

    // Stamped-array deduplication: seen_mark[r] == v means root r was already
    // found while processing v.  O(1) lookup/insert, no per-node heap alloc.
    std::vector<int> seen_mark(num_nodes, -1);

    std::vector<int> avail_roots;
    avail_roots.reserve(64);

    constexpr int max_avail_roots = 32;
    constexpr int max_fruitless   = 30;

    for (int t_src = 0; t_src < num_types; ++t_src) {
      int src_start = (t_src > 0) ? type_boundaries[t_src - 1] : 0;
      int src_end   = type_boundaries[t_src];
      int dim       = feature_dims[t_src];
      float eff_scale = (!feat_var_scale_per_type.empty() &&
                         t_src < (int)feat_var_scale_per_type.size())
                        ? feat_var_scale_per_type[t_src] : feat_var_scale;
      float threshold = 1.0f * eff_scale *
                        ((t_src < (int)feat_var.size()) ? feat_var[t_src] : 1.0f);

      int type_n    = src_end - src_start;
      int prog_step = std::max(1, type_n / 10);
      std::cout << "[HCGC] reassignment_pass type=" << t_src
                << " (" << type_n << " nodes, dim=" << dim << ")...\n";
      std::cout.flush();

      for (int v = src_start; v < src_end; ++v) {
        if ((v - src_start) % prog_step == 0 && v > src_start) {
          int pct = 100 * (v - src_start) / type_n;
          std::cout << "[HCGC]   reassignment type=" << t_src
                    << " " << pct << "%  pending=" << pending.size() << "\n";
          std::cout.flush();
        }

        int old_root = coalition_map[v];

        // Skip any node whose root was ever merged by Ball Multi-Merge
        // (across ALL outers and levels, not just this one).
        // finalized[] grows monotonically and survives rebuild via make_compact_graph.
        // This prevents the cascade where rebuild resets coalition_size=1 and
        // allows previously-merged nodes to thrash in subsequent outers.
        if (!finalized.empty() && finalized[old_root]) continue;

        // Skip hub nodes
        int cap_v = hub_cap_for_type(t_src);
        if (cap_v > 0 && (node_ptr[v+1] - node_ptr[v]) > cap_v) continue;

        // Skip frozen nodes (hub-anchor)
        if (!frozen.empty() && frozen[v]) continue;

        // Collect distinct coalition roots reachable in 2 hops
        seen_mark[old_root] = v;
        avail_roots.clear();

        bool capped = false;
        for (int e = node_ptr[v]; e < node_ptr[v+1] && !capped; ++e) {
          int nb = edge_dst[e];
          if (get_node_type(nb) == t_src) continue;  // only cross-type mediators
          int cap_nb = hub_cap_for_type(get_node_type(nb));
          if (cap_nb > 0 && (node_ptr[nb+1] - node_ptr[nb]) > cap_nb) continue;
          int fruitless = 0;
          for (int e2 = node_ptr[nb]; e2 < node_ptr[nb+1]; ++e2) {
            int u = edge_dst[e2];
            if (get_node_type(u) != t_src) continue;
            int r = coalition_map[u];
            if (seen_mark[r] != v) {
              seen_mark[r] = v;
              avail_roots.push_back(r);
              fruitless = 0;
              if ((int)avail_roots.size() >= max_avail_roots) { capped = true; break; }
            } else {
              if (++fruitless >= max_fruitless) break;
            }
          }
        }
        if (avail_roots.empty()) continue;

        const float *fv = get_node_feature(v);

        // Cost of staying: dist² between v and (old_root \ v) centroid
        float current_cost;
        if (coalition_size[old_root] <= 1) {
          current_cost = std::numeric_limits<float>::max();
        } else {
          int rem = coalition_size[old_root] - 1;
          float inv = 1.0f / static_cast<float>(rem);
          current_cost = 0.0f;
          for (int d = 0; d < dim; ++d) {
            float c = (coalition_feat_sum[old_root][d] - fv[d]) * inv;
            float diff = fv[d] - c;
            current_cost += diff * diff;
          }
        }

        int   best_root = -1;
        float best_cost = current_cost;

        for (int r : avail_roots) {
          if (coalition_size[r] <= 0) continue;
          float inv = 1.0f / static_cast<float>(coalition_size[r]);
          float cand = 0.0f;
          for (int d = 0; d < dim; ++d) {
            float c    = coalition_feat_sum[r][d] * inv;
            float diff = fv[d] - c;
            cand += diff * diff;
          }
          if (cand < best_cost) { best_cost = cand; best_root = r; }
        }

        // Degree penalty (parameter-free): high-degree nodes are harder to reassign.
        // deg_factor ≥ 1 always, so this only tightens the acceptance criterion.
        if (best_root != -1) {
          float deg_v    = static_cast<float>(node_ptr[v + 1] - node_ptr[v]);
          float mean_d   = (t_src < (int)type_mean_deg.size())
                           ? type_mean_deg[t_src] : 1.0f;
          float deg_factor = 1.0f + std::log1p(deg_v / std::max(mean_d, 1.0f));
          if (best_cost * deg_factor <= threshold)
            pending.push_back({v, t_src, old_root, best_root});
        }
      }
    }

    // ── Jacobi write phase ─────────────────────────────────────────────────
    // Guards:
    //  1. old_root check: skip if v was already moved by an earlier switch.
    //  2. new_root check: skip if new_root is no longer a self-pointing root
    //     (prevents coalition_map cycles that make normalize loop infinitely).
    int total_switches = 0;
    for (auto &sw : pending) {
      int v   = sw.node_v;
      int dim = feature_dims[sw.t_src];
      if (coalition_map[v] != sw.old_root)          continue;  // guard 1
      if (coalition_map[sw.new_root] != sw.new_root) continue;  // guard 2
      const float *fv = get_node_feature(v);
      coalition_size[sw.old_root]--;
      for (int d = 0; d < dim; ++d) coalition_feat_sum[sw.old_root][d] -= fv[d];
      coalition_size[sw.new_root]++;
      for (int d = 0; d < dim; ++d) coalition_feat_sum[sw.new_root][d] += fv[d];
      coalition_map[v] = sw.new_root;
      ++total_switches;
    }

    normalize_coalition_map();
    std::cout << "[HCGC] Reassignment: " << total_switches << " switches\n";
    return total_switches;
  }

  // ── HCGC: Heterogeneous CGC (marginal Dirichlet energy) ──────────────────
  //
  // Merge criterion: merge coalitions u and v if their weighted centroid
  // distance would not increase the graph's Dirichlet energy beyond the
  // per-type variance threshold:
  //   ΔDE = (w_um * w_vm) * ||μ_u - μ_v||² <= feat_var_scale * feat_var[t_src]
  //
  // feat_var[t] is recomputed each outer pass from current centroids, so the
  // threshold tightens naturally as coalitions grow — no fixed hyperparameter.
  //
  // Outer loop repeats until neither Ball Multi-Merge nor Node Reassignment
  // produces any change (stable-state convergence).
  //
  // Returns {merges_this_outer, switches_this_outer}.
  std::pair<int,int> coarse_graph_hcgc_once(int inner_passes,
                                             float feat_var_scale,
                                             bool skip_reassignment,
                                             int outer_idx,
                                             int window_size = 20) {
    std::mt19937 rng(42 + outer_idx);
    std::normal_distribution<float> rand_dist(0.0f, 1.0f);

    int merges_this_outer = 0;
    std::vector<bool> matched(num_nodes, false);

    for (int pass = 0; pass < inner_passes; ++pass) {
      for (int t_src = 0; t_src < num_types; ++t_src) {
        int dim = feature_dims[t_src];
        std::vector<float> proj_vec(dim);
        for (int d = 0; d < dim; ++d) proj_vec[d] = rand_dist(rng);

        float eff_scale_ball = (!feat_var_scale_per_type.empty() &&
                                t_src < (int)feat_var_scale_per_type.size())
                               ? feat_var_scale_per_type[t_src] : feat_var_scale;
        float threshold = eff_scale_ball *
                          ((t_src < (int)feat_var.size()) ? feat_var[t_src] : 1.0f);

        for (int t_med = 0; t_med < num_types; ++t_med) {
          int med_start = (t_med > 0) ? type_boundaries[t_med - 1] : 0;
          int med_end   = type_boundaries[t_med];
          int cap_med   = hub_cap_for_type(t_med);

          // Rank mediators by number of t_src neighbours (highest first)
          std::vector<std::pair<int,int>> leaders;
          leaders.reserve(med_end - med_start);
          for (int m = med_start; m < med_end; ++m) {
            if (cap_med > 0 && (node_ptr[m+1] - node_ptr[m]) > cap_med) continue;
            int score = 0;
            for (int e = node_ptr[m]; e < node_ptr[m+1]; ++e)
              if (get_node_type(edge_dst[e]) == t_src) ++score;
            if (score >= 2) leaders.push_back({score, m});
          }
          std::sort(leaders.begin(), leaders.end(),
                    [](const auto &a, const auto &b){ return a.first > b.first; });

          int total_merges = 0;
          std::cout << " [HCGC] outer=" << (outer_idx+1) << " pass=" << (pass+1)
                    << " t_src=" << t_src << " via t_med=" << t_med
                    << " | " << leaders.size() << " mediators\n";

          for (size_t li = 0; li < leaders.size(); ++li) {
            int m = leaders[li].second;

            // Collect unmatched t_src neighbours (excluding frozen nodes)
            std::vector<std::pair<int,float>> candidates;
            for (int e = node_ptr[m]; e < node_ptr[m+1]; ++e) {
              int v = edge_dst[e];
              if (get_node_type(v) == t_src && !matched[v]
                  && (frozen.empty() || !frozen[v]))
                candidates.push_back({v, edge_weight[e]});
            }
            if (candidates.size() < 2) continue;

            // 1D LSH sort: project centroids onto a random unit vector and sort.
            // This places similar coalitions adjacently so the O(window_size)
            // sliding-window comparison below approximates O(N²) pairwise search.
            std::vector<float> proj_vals(candidates.size());
            for (size_t ci2 = 0; ci2 < candidates.size(); ++ci2) {
              int r = find_root(candidates[ci2].first);
              float c = static_cast<float>(std::max(1, coalition_size[r]));
              float p = 0.0f;
              if (!coalition_feat_sum[r].empty())
                for (int d = 0; d < dim; ++d)
                  p += (coalition_feat_sum[r][d] / c) * proj_vec[d];
              proj_vals[ci2] = p;
            }
            std::vector<size_t> order(candidates.size());
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(),
                      [&proj_vals](size_t a, size_t b){
                        return proj_vals[a] < proj_vals[b]; });
            {
              std::vector<std::pair<int,float>> tmp(candidates.size());
              std::vector<float> tmp_proj(candidates.size());
              for (size_t k = 0; k < candidates.size(); ++k) {
                tmp[k]      = candidates[order[k]];
                tmp_proj[k] = proj_vals[order[k]];
              }
              candidates = std::move(tmp);
              proj_vals  = std::move(tmp_proj);
            }

            // Stride-sample if over cap
            if (max_candidates_per_mediator > 0 &&
                (int)candidates.size() > max_candidates_per_mediator) {
              std::vector<std::pair<int,float>> sampled;
              std::vector<float> sampled_proj;
              sampled.reserve(max_candidates_per_mediator);
              sampled_proj.reserve(max_candidates_per_mediator);
              float stride = static_cast<float>(candidates.size()) /
                             static_cast<float>(max_candidates_per_mediator);
              for (int si = 0; si < max_candidates_per_mediator; ++si) {
                int idx = static_cast<int>(si * stride);
                sampled.push_back(candidates[idx]);
                sampled_proj.push_back(proj_vals[idx]);
              }
              candidates = std::move(sampled);
              proj_vals  = std::move(sampled_proj);
            }

            // 3-phase Ball Multi-Merge: avoids the greedy first-come-first-merged
            // bias by electing leaders based on local density before merging.
            // Phase 1: Density Estimation
            std::vector<int> density(candidates.size(), 0);
            for (size_t i = 0; i < candidates.size(); ++i) {
              int cu = find_root(candidates[i].first);
              float cnt_u = static_cast<float>(std::max(1, coalition_size[cu]));
              int start_j = std::max(0, static_cast<int>(i) - window_size);
              int end_j   = std::min(static_cast<int>(candidates.size()),
                                     static_cast<int>(i) + window_size + 1);
              for (int j = start_j; j < end_j; ++j) {
                if (i == (size_t)j) continue;
                int cv = find_root(candidates[j].first);
                if (cu == cv) { density[i]++; continue; }
                float cnt_v = static_cast<float>(std::max(1, coalition_size[cv]));
                float dist_sq = 0.0f;
                for (int d = 0; d < dim; ++d) {
                  float fu = coalition_feat_sum[cu][d] / cnt_u;
                  float fv = coalition_feat_sum[cv][d] / cnt_v;
                  float diff = fu - fv;
                  dist_sq += diff * diff;
                }
                float w_eff = candidates[i].second * candidates[j].second;
                if (w_eff * dist_sq <= threshold) density[i]++;
              }
            }

            // Phase 2: Leader Selection (sort by density descending)
            std::vector<size_t> density_order(candidates.size());
            std::iota(density_order.begin(), density_order.end(), 0);
            std::sort(density_order.begin(), density_order.end(),
                      [&density](size_t a, size_t b){ return density[a] > density[b]; });

            // Phase 3: Ball Multi-Merge (union-find)
            // merge_coalitions() updates coalition_feat_sum[cu] in-place,
            // so each subsequent comparison sees the grown centroid.
            std::vector<bool> local_matched(candidates.size(), false);
            for (size_t idx = 0; idx < candidates.size(); ++idx) {
              size_t ci = density_order[idx];
              if (local_matched[ci]) continue;
              int u = candidates[ci].first;
              if (matched[u]) continue;
              int cu = find_root(u);

              int start_j = std::max(0, static_cast<int>(ci) - window_size);
              int end_j   = std::min(static_cast<int>(candidates.size()),
                                     static_cast<int>(ci) + window_size + 1);

              bool any_merge = false;
              for (int j = start_j; j < end_j; ++j) {
                if (local_matched[j]) continue;
                int v = candidates[j].first;
                if (matched[v]) continue;
                int cv = find_root(v);
                if (cu == cv) continue;

                float cnt_u = static_cast<float>(std::max(1, coalition_size[cu]));
                float cnt_v = static_cast<float>(std::max(1, coalition_size[cv]));
                float dist_sq = 0.0f;
                for (int d = 0; d < dim; ++d) {
                  float fu = coalition_feat_sum[cu][d] / cnt_u;
                  float fv = coalition_feat_sum[cv][d] / cnt_v;
                  float diff = fu - fv;
                  dist_sq += diff * diff;
                }
                float w_eff = candidates[ci].second * candidates[j].second;
                if (w_eff * dist_sq <= threshold) {
                  merge_coalitions(cv, cu);
                  local_matched[j] = true;
                  matched[v]       = true;
                  any_merge        = true;
                  ++total_merges;
                  ++merges_this_outer;
                }
              }

              if (any_merge) {
                matched[u]        = true;
                local_matched[ci] = true;
              }
            }
          }
          std::cout << "   -> " << total_merges << " merges (Ball Multi-Merge)\n";
        } // t_med
      }   // t_src
    }     // inner pass

    std::cout << "[HCGC] Inner passes done: " << merges_this_outer << " merges\n";
    std::cout.flush();

    // ── Finalize merged roots ─────────────────────────────────────────────
    // After Ball Multi-Merge: any root with coalition_size > 1 is permanently
    // locked.  This set grows monotonically across outers AND across rebuild
    // boundaries (make_compact_graph transfers it).
    // Without this, rebuild resets coalition_size=1 for every node, making
    // the singleton-only check vacuous and causing runaway merges at high scales.
    {
      normalize_coalition_map();  // ensure flat so every root is self-pointed
      if (finalized.empty()) finalized.assign(num_nodes, false);
      int newly_finalized = 0;
      for (int v = 0; v < num_nodes; ++v)
        if (coalition_map[v] == v && coalition_size[v] > 1 && !finalized[v]) {
          finalized[v] = true;
          ++newly_finalized;
        }
      std::cout << "[HCGC] Finalized " << newly_finalized
                << " new roots (total locked: "
                << std::count(finalized.begin(), finalized.end(), true)
                << " / " << num_nodes << ")\n";
      std::cout.flush();
    }

    int switches_this_outer = 0;
    if (!skip_reassignment) {
      switches_this_outer = reassignment_pass(feat_var_scale);
    } else {
      std::cout << "[HCGC] Reassignment skipped.\n";
      std::cout.flush();
    }

    return {merges_this_outer, switches_this_outer};
  }
};

// ── HCGC entry point ─────────────────────────────────────────────────────────
//
// Threshold is automatically derived from per-type feature variance and
// tightens each outer pass as coalitions grow — self-regulating compression.
// Iterates until stable-state convergence (no merges and no reassignments).
//
py::array_t<int>
create_graph_hcgc(py::array_t<int>   src_nodes,
                  py::array_t<int>   dst_nodes,
                  py::array_t<float> weights,
                  py::array_t<float> all_features,
                  py::array_t<int>   type_boundaries,
                  py::array_t<int>   feature_dims,
                  int   num_levels  = 1,
                  int   inner_passes = 2,
                  int   max_outer   = 10,
                  float feat_var_scale = 1.0f,
                  int   max_candidates  = 0,
                  int   max_hub_degree  = 0,
                  py::array_t<int> hub_degree_caps = py::array_t<int>(),
                  bool  auto_hub_caps          = true,
                  bool  skip_reassignment      = false,
                  int   window_size            = 20,
                  float hub_anchor_percentile  = 0.0f,
                  py::array_t<float> feat_var_scale_per_type_arr = py::array_t<float>(),
                  float target_comp_ratio = 0.0f) {

  py::buffer_info src_buf   = src_nodes.request();
  py::buffer_info dst_buf   = dst_nodes.request();
  py::buffer_info w_buf     = weights.request();
  py::buffer_info feat_buf  = all_features.request();
  py::buffer_info bound_buf = type_boundaries.request();
  py::buffer_info dims_buf  = feature_dims.request();

  int num_edges  = src_buf.shape[0];
  int num_types  = bound_buf.shape[0];
  const int *boundaries_ptr = static_cast<const int *>(bound_buf.ptr);
  int num_nodes  = boundaries_ptr[num_types - 1];

  CSRGraph graph;
  graph.init_features(static_cast<const float *>(feat_buf.ptr), boundaries_ptr,
                      static_cast<const int *>(dims_buf.ptr), num_types);
  graph.build_from_edgelist(num_nodes,
                            static_cast<const int *>(src_buf.ptr),
                            static_cast<const int *>(dst_buf.ptr),
                            static_cast<const float *>(w_buf.ptr), num_edges);

  std::cout << "[HCGC] Graph: " << num_nodes << " nodes, "
            << num_edges << " edges, " << num_types << " types.\n";

  graph.max_candidates_per_mediator = max_candidates;
  graph.max_hub_degree              = max_hub_degree;

  {
    py::buffer_info caps_buf = hub_degree_caps.request();
    int caps_len = static_cast<int>(caps_buf.shape[0]);
    if (caps_len > 0) {
      const int *caps_ptr = static_cast<const int *>(caps_buf.ptr);
      graph.hub_degree_caps.assign(caps_ptr, caps_ptr + caps_len);
    }
  }
  if (auto_hub_caps) graph.compute_auto_hub_caps();

  // Per-type feat_var_scale (optional; falls back to scalar when empty/short)
  {
    py::buffer_info pt_buf = feat_var_scale_per_type_arr.request();
    int pt_len = static_cast<int>(pt_buf.shape[0]);
    if (pt_len > 0) {
      const float *pt_ptr = static_cast<const float *>(pt_buf.ptr);
      graph.feat_var_scale_per_type.assign(pt_ptr, pt_ptr + pt_len);
      std::cout << "[HCGC] Per-type feat_var_scale: [";
      for (int t = 0; t < pt_len; ++t)
        std::cout << pt_ptr[t] << (t + 1 < pt_len ? ", " : "");
      std::cout << "]\n";
      std::cout.flush();
    }
  }

  graph.hub_anchor_percentile = hub_anchor_percentile;
  graph.init_coalition_map();
  graph.compute_feat_var();
  graph.compute_type_mean_deg();
  if (hub_anchor_percentile > 0.0f) {
    std::cout << "[HubAnchor] Freezing top "
              << (hub_anchor_percentile * 100.0f) << "% hubs per type:\n";
    graph.compute_frozen_nodes();
  }

  int orig_num_nodes = num_nodes;
  std::vector<int> cur_node_of_orig(orig_num_nodes);
  std::iota(cur_node_of_orig.begin(), cur_node_of_orig.end(), 0);
  std::vector<int> orig_rep_of_cur(orig_num_nodes);
  std::iota(orig_rep_of_cur.begin(), orig_rep_of_cur.end(), 0);

  std::vector<std::unique_ptr<CSRGraph>> level_graphs;
  CSRGraph *cur_ptr = &graph;

  // Build a new compact CSRGraph from a CoarsenedData snapshot.
  auto make_compact_graph = [&](CSRGraph *src, CoarsenedData cd)
      -> std::unique_ptr<CSRGraph> {
    auto g = std::make_unique<CSRGraph>();
    g->init_features_from_owned(
        std::move(cd.features),
        std::move(cd.type_boundaries_new),
        std::move(cd.feature_dims_new));
    g->build_from_edgelist(cd.num_nodes_new,
                           cd.src.data(), cd.dst.data(),
                           cd.edge_weights.data(),
                           static_cast<int>(cd.src.size()));
    g->max_candidates_per_mediator = src->max_candidates_per_mediator;
    g->max_hub_degree              = src->max_hub_degree;
    g->hub_anchor_percentile       = src->hub_anchor_percentile;
    g->feat_var_scale_per_type     = src->feat_var_scale_per_type;  // inherit per-type scales
    if (auto_hub_caps) g->compute_auto_hub_caps();
    else               g->hub_degree_caps = src->hub_degree_caps;
    g->init_coalition_map();
    g->compute_feat_var();
    g->compute_type_mean_deg();
    if (g->hub_anchor_percentile > 0.0f) g->compute_frozen_nodes();

    // ── Transfer finalization to the new compact graph ─────────────────────
    // New node j ↔ old root cd.new_to_old[j].  If that old root was finalized
    // (merged in any previous outer) or currently has coalition_size > 1
    // (merged THIS outer, just before rebuild), mark j finalized.
    // This is what makes finalized[] survive rebuild and grow monotonically.
    g->finalized.assign(cd.num_nodes_new, false);
    for (int j = 0; j < cd.num_nodes_new; ++j) {
      int old_root = cd.new_to_old[j];
      if ((!src->finalized.empty() && src->finalized[old_root]) ||
          src->coalition_size[old_root] > 1)
        g->finalized[j] = true;
    }
    {
      int nf = static_cast<int>(std::count(g->finalized.begin(), g->finalized.end(), true));
      std::cout << "[HCGC] Compact graph: " << cd.num_nodes_new
                << " nodes, " << nf << " finalized\n";
      std::cout.flush();
    }

    return g;
  };

  if (target_comp_ratio > 0.0f)
    std::cout << "[HCGC] Target compression: " << target_comp_ratio << "x"
              << " (up to " << num_levels << " levels)\n";

  bool target_reached = false;

  for (int level = 0; level < num_levels; ++level) {
    std::cout << "[HCGC] === Level " << (level + 1) << " / " << num_levels
              << " (" << cur_ptr->num_nodes << " nodes) ===\n";
    std::cout.flush();

    // Outer loop with rebuild-between-outers to keep coalition IDs compact.
    // After outer=1, roots scatter across the original ID space → cache thrashing.
    // build_coarsened_data() reindexes roots to 0..K-1 for cache-friendly access.
    for (int outer = 0; outer < max_outer; ++outer) {
      std::cout << "[HCGC] --- Outer " << (outer+1) << "/" << max_outer
                << "  (" << cur_ptr->num_nodes << " nodes) ---\n";
      std::cout.flush();

      auto [m, s] = cur_ptr->coarse_graph_hcgc_once(
          inner_passes, feat_var_scale, skip_reassignment, outer, window_size);
      std::cout << "[HCGC] Outer " << (outer+1) << ": "
                << m << " merges, " << s << " switches\n";
      std::cout.flush();

      // Update orig→cur mapping
      cur_ptr->normalize_coalition_map();
      for (int u = 0; u < orig_num_nodes; ++u)
        cur_node_of_orig[u] = cur_ptr->coalition_map[cur_node_of_orig[u]];

      if (m == 0 && s == 0) {
        std::cout << "[HCGC] Converged at outer " << (outer+1) << "\n";
        // Check target after convergence (no rebuild yet: count roots manually)
        if (target_comp_ratio > 0.0f) {
          int n_roots = 0;
          for (int u = 0; u < cur_ptr->num_nodes; ++u)
            if (cur_ptr->coalition_map[u] == u) ++n_roots;
          float cur_comp = (float)orig_num_nodes / std::max(1, n_roots);
          std::cout << "[HCGC] Compression after level " << (level+1)
                    << ": " << cur_comp << "x"
                    << " / target " << target_comp_ratio << "x\n";
          std::cout.flush();
          if (cur_comp >= target_comp_ratio) {
            std::cout << "[HCGC] Target reached. Stopping.\n";
            target_reached = true;
          }
        }
        break;
      }

      // Rebuild compact graph for next outer iteration
      if (outer + 1 < max_outer) {
        CoarsenedData cd = cur_ptr->build_coarsened_data();
        if (cd.num_nodes_new >= cur_ptr->num_nodes) {
          std::cout << "[HCGC] No further compression at outer "
                    << (outer+1) << ". Stopping.\n";
          break;
        }
        std::cout << "[HCGC] Rebuild after outer " << (outer+1) << ": "
                  << cur_ptr->num_nodes << " -> " << cd.num_nodes_new
                  << " nodes (compact reindex)\n";
        std::cout.flush();

        for (int u = 0; u < orig_num_nodes; ++u)
          cur_node_of_orig[u] = cd.old_to_new[cur_node_of_orig[u]];

        std::vector<int> new_orig_rep(cd.num_nodes_new);
        for (int j = 0; j < cd.num_nodes_new; ++j)
          new_orig_rep[j] = orig_rep_of_cur[cd.new_to_old[j]];
        orig_rep_of_cur = std::move(new_orig_rep);

        auto next = make_compact_graph(cur_ptr, std::move(cd));
        cur_ptr = next.get();
        level_graphs.push_back(std::move(next));

        // Check target after intra-level rebuild
        if (target_comp_ratio > 0.0f) {
          float cur_comp = (float)orig_num_nodes / (float)cur_ptr->num_nodes;
          if (cur_comp >= target_comp_ratio) {
            std::cout << "[HCGC] Target " << target_comp_ratio
                      << "x reached (" << cur_comp << "x) after outer rebuild. Stopping.\n";
            target_reached = true;
            break;
          }
        }
      }
    } // outer

    if (target_reached) break;
    if (level + 1 >= num_levels) break;

    CoarsenedData cd = cur_ptr->build_coarsened_data();
    if (cd.num_nodes_new >= cur_ptr->num_nodes) {
      std::cout << "[HCGC] Level " << (level+1)
                << ": no further compression. Stopping.\n";
      break;
    }
    std::cout << "[HCGC] Level " << (level+1) << " -> " << (level+2) << ": "
              << cur_ptr->num_nodes << " -> " << cd.num_nodes_new << "\n";
    std::cout.flush();

    for (int u = 0; u < orig_num_nodes; ++u)
      cur_node_of_orig[u] = cd.old_to_new[cur_node_of_orig[u]];

    std::vector<int> new_orig_rep(cd.num_nodes_new);
    for (int j = 0; j < cd.num_nodes_new; ++j)
      new_orig_rep[j] = orig_rep_of_cur[cd.new_to_old[j]];
    orig_rep_of_cur = std::move(new_orig_rep);

    auto next = make_compact_graph(cur_ptr, cd);
    cur_ptr = next.get();
    level_graphs.push_back(std::move(next));
  }

  py::array_t<int> result(orig_num_nodes);
  int *res_ptr = static_cast<int *>(result.request().ptr);
  for (int u = 0; u < orig_num_nodes; ++u)
    res_ptr[u] = orig_rep_of_cur[cur_node_of_orig[u]];
  return result;
}

// ── Python binding ────────────────────────────────────────────────────────────
PYBIND11_MODULE(hcgc_module, m) {
  m.def("create_graph_hcgc", &create_graph_hcgc,
        "HCGC: Heterogeneous CGC — faithful adaptation of CGC for heterogeneous\n"
        "graphs via mediator paths.\n"
        "\n"
        "Merge criterion (pure marginal Dirichlet energy, no extra hyperparameters):\n"
        "  cost      = w_eff * dist_sq   (= w_um * w_vm * ||μ_u - μ_v||²)\n"
        "  threshold = feat_var_scale * feat_var[t_src]\n"
        "\n"
        "feat_var is recomputed from coalition centroids after every outer rebuild,\n"
        "so the threshold tightens naturally as coalitions grow — self-regulating\n"
        "without fixed compression hyperparameters.\n",
        py::arg("src_nodes"), py::arg("dst_nodes"), py::arg("weights"),
        py::arg("all_features"), py::arg("type_boundaries"),
        py::arg("feature_dims"),
        py::arg("num_levels")            = 1,
        py::arg("inner_passes")          = 2,
        py::arg("max_outer")             = 10,
        py::arg("feat_var_scale")        = 1.0f,
        py::arg("max_candidates")        = 0,
        py::arg("max_hub_degree")        = 0,
        py::arg("hub_degree_caps")       = py::array_t<int>(),
        py::arg("auto_hub_caps")         = true,
        py::arg("skip_reassignment")          = false,
        py::arg("window_size")                = 20,
        py::arg("hub_anchor_percentile")      = 0.0f,
        py::arg("feat_var_scale_per_type")    = py::array_t<float>(),
        py::arg("target_comp_ratio")          = 0.0f);
}
