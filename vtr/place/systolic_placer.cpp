#include "systolic_placer.h"

#include "globals.h"
#include "placer_state.h"
#include "compressed_grid.h"
#include "physical_types.h"
#include "vtr_log.h"
#include "vtr_memory.h"

#include "place_delay_model.h"
#include "timing/PlacerCriticalities.h"
#include "timing/place_timing_update.h"

#include <vector>
#include <string>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <utility>
#include <algorithm>
#include <unordered_map>
#include <omp.h>

namespace {

long   envl(const char* k, long d) { const char* v = getenv(k); return v ? atol(v) : d; }
double envd(const char* k, double d) { const char* v = getenv(k); return v ? atof(v) : d; }

struct Knobs {
    std::string mode = "metro";     // metro | threshold | fixed
    int    swps = 10;
    long   updts = 100000;          // safety cap
    long   temp0 = 65535;
    int    creep = 95;
    int    stall_lim = 12;
    int    threads = 0;
    bool   use_crit = true;         // weight edges by timing criticality
    double cscale = 8.0;            // criticality weight scale
    int    crit_exp = 2;            // criticality exponent (2 = empirical sweet spot)
    long   base = 1;                // base (WL) weight per physical connection
    int    cadence = 0;           // 0 = STA once at start only (criticality ranking is placement-stable,
                                  //     so a single up-front STA matches refreshing every update, and STA
                                  //     is the runtime cost); N>0 = also refresh every N updates.
    bool   crit_mult = false;      // also scale the timing term by multiplicity
    double lambda = 0.3;           // >=0 enables normalized convex blend l*(m/M)+(1-l)*(maxcrit/maxcrit_design)^exp; -1 = legacy base+cscale
    long   wmax = 1000;            // rescale target: max integer edge weight after blend
    double melt = 20.0;            // initial temp = melt * mean|dC| (scale-invariant hot start)
    bool   fanorm = true;          // fanout-normalize WL weight: each connection *= 1/(pins-1)
    long   seed = 0;               // per-run RNG offset; 0 = canonical (bit-identical to hardware)
};

Knobs read_knobs() {
    Knobs k;
    if (const char* v = getenv("SYSTOLIC_MODE")) k.mode = v;
    k.swps      = (int)envl("SYSTOLIC_SWPS", k.swps);
    k.updts     = envl("SYSTOLIC_UPDTS", k.updts);
    k.temp0     = envl("SYSTOLIC_TEMP", k.temp0);
    k.creep     = (int)envl("SYSTOLIC_CREEP", k.creep);
    k.stall_lim = (int)envl("SYSTOLIC_STALL", k.stall_lim);
    k.threads   = (int)envl("SYSTOLIC_THREADS", k.threads);
    k.use_crit  = envl("SYSTOLIC_CRIT", 1) != 0;
    k.cscale    = envd("SYSTOLIC_CSCALE", k.cscale);
    k.crit_exp  = (int)envl("SYSTOLIC_CRIT_EXP", k.crit_exp);
    k.base      = envl("SYSTOLIC_BASE", k.base);
    k.cadence   = (int)envl("SYSTOLIC_CADENCE", k.cadence);
    k.crit_mult = envl("SYSTOLIC_CRITMULT", 0) != 0;
    k.lambda    = envd("SYSTOLIC_LAMBDA", k.lambda);
    k.wmax      = envl("SYSTOLIC_WMAX", k.wmax);
    k.melt      = envd("SYSTOLIC_MELT", k.melt);
    k.fanorm    = envl("SYSTOLIC_FANORM", 1) != 0;
    k.seed      = envl("SYSTOLIC_SEED", k.seed);
    return k;
}

inline bool accept_metro(int64_t d, uint32_t r16, double T) {
    return (double)(r16 & 0xFFFF) < 65536.0 * std::exp(-(double)d / T);
}

// Per-slot mobile state, int32, packed AoS (one cache line per slot): the block id,
// total weight K, and weighted-neighbor sums Sx/Sy. These travel together on a swap.
struct SlotState { int32_t blk; int32_t K; int32_t Sx; int32_t Sy; };

} // namespace

void run_systolic(PlacerState& placer_state,
                  const PlaceDelayModel* delay_model,
                  PlacerCriticalities* criticalities,
                  PlacerSetupSlacks* setup_slacks,
                  NetPinTimingInvalidator* pin_timing_invalidator,
                  SetupTimingInfo* timing_info,
                  t_placer_costs* costs,
                  const PlaceCritParams& crit_params) {
    const Knobs K = read_knobs();
    if (K.threads > 0) omp_set_num_threads(K.threads);
    const bool use_timing = K.use_crit && criticalities != nullptr && delay_model != nullptr;
    double t_start = omp_get_wtime();

    auto& cluster_ctx = g_vpr_ctx.clustering();
    auto& device_ctx = g_vpr_ctx.device();
    const auto& clb = cluster_ctx.clb_nlist;
    auto& blk_loc_registry = placer_state.mutable_blk_loc_registry();
    const auto& grid_blocks = blk_loc_registry.grid_blocks();

    const int layer = 0;
    const size_t Nblk = clb.blocks().size();

    // ---- per-block position (all blocks) + bookkeeping -------------------------
    std::vector<int> pos_x(Nblk, 0), pos_y(Nblk, 0), sub_of(Nblk, 0);
    std::vector<char> is_fixed(Nblk, 0);
    for (auto b : clb.blocks()) {
        const auto& loc = placer_state.block_locs()[b].loc;
        pos_x[size_t(b)] = loc.x; pos_y[size_t(b)] = loc.y;
        sub_of[size_t(b)] = loc.sub_tile;
        is_fixed[size_t(b)] = placer_state.block_locs()[b].is_fixed ? 1 : 0;
    }

    // ---- compressed grids ------------------------------------------------------
    auto& place_ctx = g_vpr_ctx.mutable_placement();
    bool created_cgrids = false;
    if (place_ctx.compressed_block_grids.empty()) {
        place_ctx.compressed_block_grids = create_compressed_block_grids();
        created_cgrids = true;
    }
    const auto& cgrids = place_ctx.compressed_block_grids;

    // ---- slots per movable physical type + red-black pairs ---------------------
    std::vector<int> slot_x, slot_y;
    std::vector<SlotState> st;           // AoS: {blk, K, Sx, Sy} per slot (int32, one cache line)
    std::vector<char> slot_locked;
    std::vector<int> slot_of(Nblk, -1);
    std::vector<int> pm[4], ps[4];

    for (const auto& phys : device_ctx.physical_tile_types) {
        if (phys.name == std::string("io")) continue;
        if (phys.index < 0 || (size_t)phys.index >= cgrids.size()) continue;
        const auto& cg = cgrids[phys.index];
        if (cg.compressed_to_grid_x.empty() || cg.compressed_to_grid_y.empty()) continue;
        const int W = (int)cg.compressed_to_grid_x[0].size();
        const int H = (int)cg.compressed_to_grid_y[0].size();
        if (W == 0 || H == 0) continue;

        const int base = (int)slot_x.size();
        slot_x.resize(base + (size_t)W * H);
        slot_y.resize(base + (size_t)W * H);
        st.resize(base + (size_t)W * H);
        slot_locked.resize(base + (size_t)W * H, 0);
        for (int cy = 0; cy < H; cy++)
            for (int cx = 0; cx < W; cx++) {
                const int x = cg.compressed_to_grid_x[0][cx];
                const int y = cg.compressed_to_grid_y[0][cy];
                const int s = base + cy * W + cx;
                slot_x[s] = x; slot_y[s] = y;
                ClusterBlockId occ = grid_blocks.block_at_location(t_pl_loc{x, y, 0, layer});
                st[s] = SlotState{occ ? (int32_t)size_t(occ) : -1, 0, 0, 0};
                if (occ) { slot_of[size_t(occ)] = s; if (is_fixed[size_t(occ)]) slot_locked[s] = 1; }
            }
        auto gs = [&](int cx, int cy) { return base + cy * W + cx; };
        for (int cy = 0; cy < H; cy++)
            for (int cx0 = 0; cx0 < 2; cx0++)
                for (int cx = cx0; cx + 1 < W; cx += 2) {
                    int a = gs(cx, cy), b = gs(cx + 1, cy);
                    if (slot_locked[a] || slot_locked[b]) continue;
                    pm[cx0 == 0 ? 0 : 2].push_back(a); ps[cx0 == 0 ? 0 : 2].push_back(b);
                }
        // cy-outer / cx-inner: consecutive vertical pairs stride by 1 slot, so the swap
        // streams st row-major (cx-outer would be stride-W column-major = cache-hostile).
        // Red-black pairs are disjoint within a phase, so this reorder is bit-identical.
        for (int cy0 = 0; cy0 < 2; cy0++)
            for (int cy = cy0; cy + 1 < H; cy += 2)
                for (int cx = 0; cx < W; cx++) {
                    int a = gs(cx, cy), b = gs(cx, cy + 1);
                    if (slot_locked[a] || slot_locked[b]) continue;
                    pm[cy0 == 0 ? 1 : 3].push_back(a); ps[cy0 == 0 ? 1 : 3].push_back(b);
                }
    }
    const size_t Nslot = slot_x.size();

    // ---- edges: collapsed by block-pair, multiplicity + pin list ---------------
    // One edge per unordered block pair. multiplicity = # physical driver->sink
    // pin connections; the timing weight uses the MAX criticality over the pair's
    // pins (they share one placement distance, so the worst pin is what limits it).
    std::unordered_map<uint64_t, int> edge_id;
    std::vector<int> edge_a, edge_b, edge_mult;
    std::vector<double> edge_wlw;   // WL weight = sum over connections of 1/(pins-1) (fanout-normalized)
    std::vector<std::vector<std::pair<ClusterNetId, int>>> tmp_pins; // per edge (timing only)
    auto add_conn = [&](ClusterBlockId d, ClusterBlockId s, ClusterNetId net, int ipin, double fw) {
        int a = (int)size_t(d), b = (int)size_t(s);
        if (a > b) std::swap(a, b);
        uint64_t key = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
        auto it = edge_id.find(key);
        int id;
        if (it == edge_id.end()) {
            id = (int)edge_a.size(); edge_id[key] = id;
            edge_a.push_back(a); edge_b.push_back(b); edge_mult.push_back(0); edge_wlw.push_back(0.0);
            if (use_timing) tmp_pins.emplace_back();
        } else id = it->second;
        edge_mult[id]++; edge_wlw[id] += fw;
        if (use_timing) tmp_pins[id].emplace_back(net, ipin);
    };
    for (auto net : clb.nets()) {
        if (clb.net_is_ignored(net)) continue;   // skip global (clock/reset) nets
        int F = (int)clb.net_sinks(net).size();                 // #sinks = pins-1
        double fw = (K.fanorm && F > 1) ? 1.0 / (double)F : 1.0; // 1/(pins-1) star normalization
        ClusterBlockId d = clb.pin_block(clb.net_driver(net));
        int ipin = 1;
        for (auto sp : clb.net_sinks(net)) {
            ClusterBlockId s = clb.pin_block(sp);
            if (d && s && d != s) add_conn(d, s, net, ipin, fw);
            ipin++;
        }
    }
    const size_t Nedge = edge_a.size();
    std::vector<int64_t> edge_w(Nedge);
    for (size_t e = 0; e < Nedge; e++) edge_w[e] = K.base * edge_mult[e]; // WL baseline
    const bool use_lambda = use_timing && K.lambda >= 0.0;   // normalized convex blend
    double WLWmax = 1e-9;
    for (size_t e = 0; e < Nedge; e++) if (edge_wlw[e] > WLWmax) WLWmax = edge_wlw[e];
    std::vector<double> edge_mc, edge_fw;                    // scratch for the blend
    if (use_lambda) { edge_mc.assign(Nedge, 0.0); edge_fw.assign(Nedge, 0.0); }

    // flatten pin lists for the max-criticality reweight (timing only)
    std::vector<int64_t> pin_off(Nedge + 1, 0);
    std::vector<ClusterNetId> pin_net; std::vector<int> pin_ipin;
    if (use_timing) {
        for (size_t e = 0; e < Nedge; e++) pin_off[e + 1] = pin_off[e] + (int64_t)tmp_pins[e].size();
        pin_net.resize(pin_off[Nedge]); pin_ipin.resize(pin_off[Nedge]);
        for (size_t e = 0; e < Nedge; e++) {
            int64_t o = pin_off[e];
            for (auto& p : tmp_pins[e]) { pin_net[o] = p.first; pin_ipin[o] = p.second; o++; }
        }
        tmp_pins.clear(); tmp_pins.shrink_to_fit();
    }

    // per-block adjacency CSR: (neighbor, edge-id)
    std::vector<int64_t> adj_off(Nblk + 1, 0);
    for (size_t e = 0; e < Nedge; e++) { adj_off[edge_a[e] + 1]++; adj_off[edge_b[e] + 1]++; }
    for (size_t b = 0; b < Nblk; b++) adj_off[b + 1] += adj_off[b];
    std::vector<int> adj_nbr(adj_off[Nblk]), adj_edge(adj_off[Nblk]);
    { std::vector<int64_t> cur(adj_off.begin(), adj_off.end() - 1);
      for (size_t e = 0; e < Nedge; e++) {
          int a = edge_a[e], b = edge_b[e];
          adj_nbr[cur[a]] = b; adj_edge[cur[a]] = (int)e; cur[a]++;
          adj_nbr[cur[b]] = a; adj_edge[cur[b]] = (int)e; cur[b]++;
      } }
    // Static weights folded inline into adjacency (streamed with adj_nbr in the gather),
    // refilled from edge_w whenever weights change (see reweight). Enabled by STA-once:
    // with cadence==0 this scatter runs exactly once, so the gather never touches edge_w.
    std::vector<int32_t> adj_w(adj_off[Nblk]);
    auto fold_weights = [&]() {
        const int64_t nent = adj_off[Nblk];
        for (int64_t j = 0; j < nent; j++) adj_w[j] = (int32_t)edge_w[adj_edge[j]];
    };

    // ---- per-slot cached state + RNG ------------------------------------------
    std::vector<uint32_t> rng(Nslot);
    // seed offsets the per-slot RNG init; seed 0 reproduces the canonical (hardware-identical) run
    const uint32_t seed_mix = 0x9E3779B9u * (uint32_t)K.seed;
    for (size_t s = 0; s < Nslot; s++) rng[s] = (uint32_t)((((uint32_t)s + seed_mix) * 2654435761u) | 1u) & 0xFFFF;

    // push current slot placement into VPR's block_locs + grid (for STA)
    auto sync_to_vpr = [&]() {
        auto& mbl = placer_state.mutable_block_locs();
        for (size_t s = 0; s < Nslot; s++) {
            if (st[s].blk < 0) continue;
            ClusterBlockId b = ClusterBlockId(st[s].blk);
            if (!is_fixed[size_t(b)]) mbl[b].loc = t_pl_loc{slot_x[s], slot_y[s], sub_of[size_t(b)], layer};
        }
        blk_loc_registry.mutable_grid_blocks().load_from_block_locs(placer_state.block_locs());
    };
    // recompute edge weights from current criticalities (max over the pair's pins)
    auto reweight = [&]() {
        if (!use_lambda) {
            #pragma omp parallel for schedule(static)
            for (long e = 0; e < (long)Nedge; e++) {
                double mx = 0.0;
                for (int64_t j = pin_off[e]; j < pin_off[e + 1]; j++) {
                    double cr = criticalities->criticality(pin_net[j], pin_ipin[j]);
                    if (cr > mx) mx = cr;
                }
                double t = (K.crit_exp == 1) ? mx : std::pow(mx, (double)K.crit_exp);
                int64_t tw = (int64_t)(K.cscale * t + 0.5);
                if (K.crit_mult) tw *= edge_mult[e];
                edge_w[e] = K.base * edge_mult[e] + tw; // WL (mult) + timing (max)
            }
            return;
        }
        // Normalized convex blend: w = l*(m/M) + (1-l)*(maxcrit/maxcrit_design)^exp,
        // then rescale all weights so the largest is `wmax` (keeps int resolution;
        // both terms are in [0,1] so raw weights would collapse to 0/1).
        double mcd = 0.0;
        #pragma omp parallel for schedule(static) reduction(max:mcd)
        for (long e = 0; e < (long)Nedge; e++) {
            double mx = 0.0;
            for (int64_t j = pin_off[e]; j < pin_off[e + 1]; j++) {
                double cr = criticalities->criticality(pin_net[j], pin_ipin[j]);
                if (cr > mx) mx = cr;
            }
            edge_mc[e] = mx;
            if (mx > mcd) mcd = mx;
        }
        if (mcd <= 0.0) mcd = 1.0;
        double fwmax = 0.0;
        #pragma omp parallel for schedule(static) reduction(max:fwmax)
        for (long e = 0; e < (long)Nedge; e++) {
            double c = edge_mc[e] / mcd;
            double t = (K.crit_exp == 1) ? c : std::pow(c, (double)K.crit_exp);
            double fw = K.lambda * (edge_wlw[e] / WLWmax) + (1.0 - K.lambda) * t;
            edge_fw[e] = fw;
            if (fw > fwmax) fwmax = fw;
        }
        if (fwmax <= 0.0) fwmax = 1.0;
        double scale = (double)K.wmax / fwmax;
        #pragma omp parallel for schedule(static)
        for (long e = 0; e < (long)Nedge; e++) {
            int64_t w = (int64_t)(edge_fw[e] * scale + 0.5);
            edge_w[e] = w < 1 ? 1 : w;
        }
    };
    fold_weights();   // populate inline adj_w from the WL baseline (refreshed after each reweight)

    // ---- Metropolis accept LUT -------------------------------------------------
    // The accept threshold 65536*exp(-dC/T) depends only on the dimensionless ratio
    // x = dC/T, so this table is temperature-independent: built once, reused every
    // update. exp(-x) < 1/65536 for x > ~11.1, so cap at x=16 (idx>=LUT_N => reject).
    // Store integer thresholds so the accept test stays a plain (rng16 < tbl[idx]) compare.
    const int LUT_SCALE = 256;
    const int LUT_N = 16 * LUT_SCALE;                 // x in [0,16), step 1/256  -> 16 KB
    std::vector<uint32_t> exp_lut(LUT_N);
    for (int i = 0; i < LUT_N; i++)
        exp_lut[i] = (uint32_t)(65536.0 * std::exp(-((double)i + 0.5) / LUT_SCALE) + 0.5);
    const bool use_lut = (getenv("SYSTOLIC_NOLUT") == nullptr);  // A/B vs exact exp

    // ---- annealing loop --------------------------------------------------------
    const bool fixed_sched = (K.mode == "fixed");
    const bool metro = (K.mode == "metro");
    long temp = K.temp0, dec = 0, thr = 0;
    if (fixed_sched) { long steps = K.updts * K.swps - 1; dec = steps > 0 ? temp / steps : 0; thr = steps > 0 ? temp - dec * (steps - 1) : 0; }
    int phase = 0;
    int64_t best_cost = -1;
    int stall = 0, used_updates = 0, n_sta = 0;
    const double COST_EPS = 1e-3, MELT_A = 0.80;
    const long TEMP_FLOOR = metro ? 1 : 100;
    double t_sync = 0, t_sta = 0, t_rw = 0, t_place = 0;   // runtime breakdown
    double t_gather = 0, t_swap = 0;                        // place sub-split (gather vs swaps)

    // Shared state for the persistent parallel region (declared once, outside).
    int64_t cost = 0; long naccept = 0, nbadatt = 0, nbadacc = 0; int ovf = 0;
    double invTscaled = 0.0, _p0 = 0.0, _ps = 0.0;
    bool done = false;
    // Persistent OpenMP team: fork ONCE for the whole anneal (not per phase). Parallel
    // work uses `omp for` (implicit barrier per region); serial work — STA, reweight,
    // melt, the adaptive schedule — runs in `omp single` (also a barrier). Red-black
    // disjointness keeps swaps lock-free; integer reductions make results bit-identical
    // across thread counts. STA runs in `omp single` while the other threads idle at the
    // barrier — set OMP_WAIT_POLICY=passive so they yield to Tatum's TBB during that call.
    #pragma omp parallel
    {
    for (long u = 0; u < K.updts && !done; u++) {
        #pragma omp single
        {
        if (use_timing && (u == 0 || (K.cadence > 0 && u % K.cadence == 0))) {
            double _ta = omp_get_wtime(); sync_to_vpr();
            double _tb = omp_get_wtime(); t_sync += _tb - _ta;
            perform_full_timing_update(crit_params, delay_model, criticalities, setup_slacks,
                                       pin_timing_invalidator, timing_info, costs, placer_state);
            double _tc = omp_get_wtime(); t_sta += _tc - _tb;
            reweight();
            fold_weights();   // refresh inline adj_w with the new criticality-weighted edge_w
            t_rw += omp_get_wtime() - _tc;
            n_sta++;
            if (u == 0 && getenv("SYSTOLIC_DIAG")) {
                // fanout distribution + mean/critical criticality per fanout bucket
                const int NB = 12;
                long nnets[NB] = {0}; double csum[NB] = {0}; long csink[NB] = {0}, chi[NB] = {0};
                long tot_sinks = 0, tot_hi = 0;
                auto bkt = [](int F) { int b = 0; while ((1 << (b + 1)) <= F && b < NB - 1) b++; return b; };
                for (auto net : clb.nets()) {
                    if (clb.net_is_ignored(net)) continue;
                    int F = (int)clb.net_sinks(net).size();
                    if (F < 1) continue;
                    int b = bkt(F); nnets[b]++;
                    int ip = 1;
                    for (auto sp : clb.net_sinks(net)) { (void)sp;
                        double cr = criticalities->criticality(net, ip);
                        csum[b] += cr; csink[b]++; tot_sinks++;
                        if (cr > 0.5) { chi[b]++; tot_hi++; }
                        ip++;
                    }
                }
                VTR_LOG("[systolic-diag] fanout bucket | #nets | #sinks | mean_crit | #crit>0.5 | %%of_all_crit\n");
                for (int b = 0; b < NB; b++) if (nnets[b])
                    VTR_LOG("  F>=%-5d | %6ld | %8ld | %.3f | %8ld | %.1f%%\n",
                            1 << b, nnets[b], csink[b], csink[b] ? csum[b] / csink[b] : 0.0,
                            chi[b], tot_hi ? 100.0 * chi[b] / tot_hi : 0.0);
                VTR_LOG("[systolic-diag] total sinks=%ld, crit>0.5=%ld\n", tot_sinks, tot_hi);
            }
        }
        _p0 = omp_get_wtime(); cost = 0; ovf = 0;
        } // end single: STA + reweight + timing

        #pragma omp for schedule(static)
        for (long s = 0; s < (long)Nslot; s++)
            if (st[s].blk >= 0) { pos_x[st[s].blk] = slot_x[s]; pos_y[st[s].blk] = slot_y[s]; }

        #pragma omp for schedule(static) reduction(+:cost)
        for (long s = 0; s < (long)Nslot; s++) {
            int b = st[s].blk;
            if (b < 0) { st[s].K = st[s].Sx = st[s].Sy = 0; continue; }
            int64_t k = 0, sx = 0, sy = 0;
            for (int64_t j = adj_off[size_t(b)]; j < adj_off[size_t(b) + 1]; j++) {
                int n = adj_nbr[j]; int64_t w = adj_w[j];   // weight folded inline (streamed)
                k += w; sx += w * pos_x[n]; sy += w * pos_y[n];
            }
            if (k != (int32_t)k || sx != (int32_t)sx || sy != (int32_t)sy) ovf = 1;  // benign race, guard only
            st[s].K = (int32_t)k; st[s].Sx = (int32_t)sx; st[s].Sy = (int32_t)sy;
            int64_t x = slot_x[s], y = slot_y[s];
            cost += k * (x * x + y * y) - (x * sx + y * sy);
        }
        #pragma omp single
        {
        if (ovf) VTR_LOG("[systolic] WARNING: int32 slot-state overflow at update %ld — widen SlotState or lower WMAX\n", u);
        t_gather += omp_get_wtime() - _p0;
        if (use_lambda && u == 0) {
            // scale-invariant hot start: temp = melt * mean|dC| over all candidate pairs
            double sd = 0.0; long cnt = 0;
            for (int p = 0; p < 4; p++) {
                const bool horiz = (p == 0 || p == 2);
                const auto& A = pm[p]; const auto& B = ps[p];
                for (size_t i = 0; i < A.size(); i++) {
                    int sa = A[i], sb = B[i];
                    int64_t ca = horiz ? slot_x[sa] : slot_y[sa];
                    int64_t cb = horiz ? slot_x[sb] : slot_y[sb];
                    int64_t Sa = horiz ? st[sa].Sx : st[sa].Sy;
                    int64_t Sb = horiz ? st[sb].Sx : st[sb].Sy;
                    int64_t da = (int64_t)st[sa].K*(cb*cb-ca*ca) - 2*Sa*(cb-ca);
                    int64_t db = (int64_t)st[sb].K*(ca*ca-cb*cb) - 2*Sb*(ca-cb);
                    int64_t d = da + db;
                    sd += (d < 0 ? -d : d); cnt++;
                }
            }
            double mean = cnt ? sd / (double)cnt : 1.0;
            temp = (long)(K.melt * mean); if (temp < 1) temp = 1;
            if (fixed_sched) { long steps = K.updts*K.swps - 1; dec = steps>0?temp/steps:0; thr = steps>0?temp-dec*(steps-1):0; }
        }
        naccept = 0; nbadatt = 0; nbadacc = 0;
        invTscaled = (temp > 0) ? (double)LUT_SCALE / (double)temp : 0.0;   // temp const within a metro update
        _ps = omp_get_wtime();
        } // end single: gather timing + melt + reset

        for (int sw = 0; sw < K.swps; sw++) {
            const int p = phase;
            const bool horiz = (p == 0 || p == 2);
            const auto& A = pm[p]; const auto& B = ps[p];
            const long np = (long)A.size();
            #pragma omp for schedule(static) reduction(+:naccept,nbadatt,nbadacc)
            for (long i = 0; i < np; i++) {
                int sa = A[i], sb = B[i];
                int64_t ca = horiz ? slot_x[sa] : slot_y[sa];
                int64_t cb = horiz ? slot_x[sb] : slot_y[sb];
                int64_t Sa = horiz ? st[sa].Sx : st[sa].Sy;
                int64_t Sb = horiz ? st[sb].Sx : st[sb].Sy;
                int64_t da = (int64_t)st[sa].K * (cb * cb - ca * ca) - 2 * Sa * (cb - ca);
                int64_t db = (int64_t)st[sb].K * (ca * ca - cb * cb) - 2 * Sb * (ca - cb);
                int64_t d = da + db;
                bool acc;
                if (d < 0) acc = true;
                else if (fixed_sched) acc = ((rng[sa] & 0xFFFF) < (uint32_t)temp);
                else if (metro) {
                    if (use_lut) {
                        double idxf = (double)d * invTscaled;   // = (dC/T) * LUT_SCALE
                        acc = (idxf < (double)LUT_N) && ((rng[sa] & 0xFFFF) < exp_lut[(int)idxf]);
                    } else {
                        acc = accept_metro(d, rng[sa], (double)temp);
                    }
                }
                else                  acc = ((rng[sa] & 0xFFFF) < (uint32_t)temp);
                if (d > 0) { nbadatt++; if (acc) nbadacc++; }
                if (acc) {
                    std::swap(st[sa], st[sb]);   // blk,K,Sx,Sy travel together (one AoS record)
                    if (st[sa].blk >= 0) slot_of[st[sa].blk] = sa;
                    if (st[sb].blk >= 0) slot_of[st[sb].blk] = sb;
                    naccept++;
                }
            }
            #pragma omp for schedule(static)
            for (long s = 0; s < (long)Nslot; s++) {
                uint32_t v = rng[s];
                uint32_t fb = ((v >> 15) ^ (v >> 14) ^ (v >> 12) ^ (v >> 3)) & 1u;
                rng[s] = ((v << 1) | fb) & 0xFFFF;
            }
            #pragma omp single
            { phase = (phase + 1) % 4; if (fixed_sched) temp = (temp > thr) ? temp - dec : 0; }
        }

        #pragma omp single
        {
            t_swap += omp_get_wtime() - _ps;
            t_place += omp_get_wtime() - _p0;
            used_updates = (int)u + 1;
            if (fixed_sched) {
                if (best_cost < 0 || cost < best_cost) best_cost = cost;
            } else {
                double pbad = nbadatt > 0 ? (double)nbadacc / (double)nbadatt : 0.0;
                if      (pbad > 0.96) temp = temp / 2;
                else if (pbad > MELT_A) temp = (temp * 9) / 10;
                else                    temp = (temp * K.creep) / 100;
                if (pbad < MELT_A) {
                    if (best_cost < 0 || (double)cost < (double)best_cost - COST_EPS * (double)best_cost) { best_cost = cost; stall = 0; }
                    else stall++;
                }
                if (stall >= K.stall_lim || temp <= TEMP_FLOOR) done = true;
            }
        }
        (void)naccept;
    }
    } // end persistent parallel region

    // ---- write final locations back into VPR placement state --------------------
    sync_to_vpr();
    if (created_cgrids) vtr::release_memory(place_ctx.compressed_block_grids);

    double t_total = omp_get_wtime() - t_start;
    double t_other = t_total - t_place - t_sta - t_sync - t_rw;   // setup (adjacency) + writeback
    VTR_LOG("[systolic] mode=%s crit=%d slots=%zu blocks=%zu edges=%zu updates=%d STA=%d cost=%ld  %.4f s\n",
            K.mode.c_str(), (int)use_timing, Nslot, Nblk, Nedge, used_updates, n_sta, (long)best_cost, t_total);
    VTR_LOG("[systolic] time: place %.4fs (%.0f%%) | STA %.4fs (%.0f%%) | sync %.4fs (%.0f%%) | reweight %.4fs (%.0f%%) | setup+writeback %.4fs (%.0f%%)\n",
            t_place, 100*t_place/t_total, t_sta, 100*t_sta/t_total, t_sync, 100*t_sync/t_total,
            t_rw, 100*t_rw/t_total, t_other, 100*t_other/t_total);
    if (getenv("SYSTOLIC_DIAG"))
        VTR_LOG("[systolic] place split: gather %.4fs (%.0f%%) | swaps %.4fs (%.0f%%)\n",
                t_gather, 100*t_gather/t_place, t_swap, 100*t_swap/t_place);
}
