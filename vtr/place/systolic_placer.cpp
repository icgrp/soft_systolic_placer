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
#include "NetPinTimingInvalidator.h"

#include <vector>
#include <string>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <utility>
#include <algorithm>
#include <functional>
#include <unordered_map>
#include <omp.h>

namespace {

long   envl(const char* k, long d) { const char* v = getenv(k); return v ? atol(v) : d; }
double envd(const char* k, double d) { const char* v = getenv(k); return v ? atof(v) : d; }

struct Knobs {
    std::string mode = "metro";     // metro | fixed
    int    swps = 10;
    long   updts = 100000;          // safety cap
    long   temp0 = 65535;
    int    creep = 95;
    int    stall_lim = 12;
    int    hold = 1;               // metro: hold temp for this many gathers/updates before one cooling step
                                   // (equilibrate at each temperature; pbad accumulated over the window)
    int    threads = 0;
    bool   use_crit = true;         // weight edges by timing criticality
    int    crit_exp = 2;            // criticality exponent (2 = empirical sweet spot)
    long   base = 1;                // base (WL) weight per physical connection
    int    cadence = 0;           // 0 = STA once at start only (criticality ranking is placement-stable,
                                  //     so a single up-front STA matches refreshing every update, and STA
                                  //     is the runtime cost); N>0 = also refresh every N updates.
    double lambda = 0.3;           // normalized convex blend l*(m/M)+(1-l)*(maxcrit/maxcrit_design)^exp, in [0,1]
    long   wmax = 1000;            // rescale target: max integer edge weight after blend
    double melt = 20.0;            // initial temp = melt * mean|dC| (scale-invariant hot start)
    bool   fanorm = true;          // fanout-normalize WL weight: each connection *= 1/(pins-1)
    // (tested criticality-gated fanorm: apply 1/F only to low-crit edges. Failed on 2/3 Koios designs
    //  because the STA flags the routed-critical high-fanout nets as near-zero crit, so the gate
    //  down-weights them anyway. fanorm-off — protect ALL high-fanout nets — is more robust. Reverted.)
    long   seed = 0;               // per-run RNG offset; 0 = canonical (bit-identical to hardware)
    // Idea #10: adaptive swps-per-update (staleness-aware step size). `swps` sets the STARTING
    // ceiling; when enabled, it only ever shrinks, checked at the same cadence as the existing
    // pbad-based temp cooling (every K.hold updates), based on measured PE-hop displacement --
    // never on design identity, so a design that never cools (e.g. a persistent near-zero-cost-
    // gradient block population) simply never triggers a shrink instead of needing an exception.
    bool   adaptive_swps = false;   // SYSTOLIC_ADAPTIVE_SWPS
    int    swps_min = 1;            // SYSTOLIC_SWPS_MIN: floor
    double swps_shrink_thr = 0.25;  // SYSTOLIC_SWPS_SHRINK_THR: shrink when median PE-hop displacement
                                    // (among blocks that moved) is below this fraction of the current ceiling
    double swps_shrink_mult = 0.5;  // SYSTOLIC_SWPS_SHRINK_MULT: multiplicative shrink factor
    double swps_sample_err = 0.05;  // SYSTOLIC_ADAPTIVE_SWPS_SAMPLE_ERR: target standard error (worst
                                    // case, p=0.5) for the displacement check's sample-proportion estimate.
                                    // Sample size is derived from this, not fixed: n0 = 0.25/err^2 is the
                                    // infinite-population size, then finite-population-corrected by Nblk
                                    // (n = n0/(1+(n0-1)/Nblk)) -- for every design in this project's
                                    // benchmark sets Nblk >> n0, so the correction barely moves the number
                                    // (e.g. Nblk=560, the smallest design tested: n0=100 -> n=85), but it's
                                    // the statistically correct form rather than a guessed constant.
};

Knobs read_knobs() {
    Knobs k;
    if (const char* v = getenv("SYSTOLIC_MODE")) k.mode = v;
    k.swps      = (int)envl("SYSTOLIC_SWPS", k.swps);
    k.updts     = envl("SYSTOLIC_UPDTS", k.updts);
    k.temp0     = envl("SYSTOLIC_TEMP", k.temp0);
    k.creep     = (int)envl("SYSTOLIC_CREEP", k.creep);
    k.stall_lim = (int)envl("SYSTOLIC_STALL", k.stall_lim);
    k.hold      = (int)envl("SYSTOLIC_HOLD", k.hold);
    k.threads   = (int)envl("SYSTOLIC_THREADS", k.threads);
    k.use_crit  = envl("SYSTOLIC_CRIT", 1) != 0;
    k.crit_exp  = (int)envl("SYSTOLIC_CRIT_EXP", k.crit_exp);
    k.base      = envl("SYSTOLIC_BASE", k.base);
    k.cadence   = (int)envl("SYSTOLIC_CADENCE", k.cadence);
    k.lambda    = envd("SYSTOLIC_LAMBDA", k.lambda);
    k.wmax      = envl("SYSTOLIC_WMAX", k.wmax);
    k.melt      = envd("SYSTOLIC_MELT", k.melt);
    k.fanorm    = envl("SYSTOLIC_FANORM", 1) != 0;
    k.seed      = envl("SYSTOLIC_SEED", k.seed);
    k.adaptive_swps    = envl("SYSTOLIC_ADAPTIVE_SWPS", 0) != 0;
    k.swps_min         = (int)envl("SYSTOLIC_SWPS_MIN", k.swps_min);
    k.swps_shrink_thr  = envd("SYSTOLIC_SWPS_SHRINK_THR", k.swps_shrink_thr);
    k.swps_shrink_mult = envd("SYSTOLIC_SWPS_SHRINK_MULT", k.swps_shrink_mult);
    k.swps_sample_err  = envd("SYSTOLIC_ADAPTIVE_SWPS_SAMPLE_ERR", k.swps_sample_err);
    return k;
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
    // Mutable crit_params so we can RAMP the VPR criticality exponent over the anneal (VTR-style, 1->8),
    // applied at each cadence STA refresh. SYSTOLIC_VPREXP_RAMP=1 enables; FIRST/LAST set the endpoints.
    PlaceCritParams cp = crit_params;
    const bool   vpre_ramp  = std::getenv("SYSTOLIC_VPREXP_RAMP") && atoi(std::getenv("SYSTOLIC_VPREXP_RAMP"));
    const double vpre_first = std::getenv("SYSTOLIC_VPREXP_FIRST") ? atof(std::getenv("SYSTOLIC_VPREXP_FIRST")) : 1.0;
    const double vpre_last  = std::getenv("SYSTOLIC_VPREXP_LAST")  ? atof(std::getenv("SYSTOLIC_VPREXP_LAST"))  : 8.0;
    // Per-connection criticality-gated fanorm (SYSTOLIC_CRITGATE): a connection keeps FULL WL weight if it is
    // critical, else it's fanout-normalized (1/(pins-1)). Fixes the asymmetry where fanorm zeros the WL
    // pull on critical WIDE-net connections (they'd otherwise get only ~half a narrow net's pull). No
    // threshold: each edge's WL weight blends continuously between fanorm'd and full by its OWN raw
    // criticality c=edge_mc[e] (already on a fixed [0,1] scale) -- c^critgate_smooth_exp of full weight,
    // the rest fanorm'd. (Earlier discrete-threshold/percentile/Otsu/biggest-gap gate variants were
    // tried and superseded by this continuous blend; see TODO_IDEAS.md idea #2c.) The gate needs RAW
    // crit (wide nets are crushed at high exp), so VPR crit_exponent is pinned to 1 and the ramp exponent
    // is applied to the timing term in reweight() instead. Default off -> byte-identical.
    const bool   critgate = std::getenv("SYSTOLIC_CRITGATE") && atoi(std::getenv("SYSTOLIC_CRITGATE"));
    // MEASURED first attempt reused ramp_exp (which climbs to 8) as the blend's own sharpening
    // exponent -- at exp=8, pow(c,8) crushes any mid-range c toward 0 (c=0.5 -> 0.004), reintroducing
    // the exact "aggressive exponent starves moderately-critical connections" problem crit-gate was
    // invented to fix in the first place. Decouple: a separate, gentle, FIXED exponent for the WL
    // blend (default 1.0 = plain linear interpolation by raw normalized criticality).
    const double critgate_smooth_exp = std::getenv("SYSTOLIC_CRITGATE_SMOOTH_EXP") ? atof(std::getenv("SYSTOLIC_CRITGATE_SMOOTH_EXP")) : 1.0;
    // Progress-tied STA (SYSTOLIC_PSTA): fixed update-count cadence is brittle on short/lean (low-hold)
    // runs -- if total updates < cadence, only the u=0 STA ever fires and the anneal is steered by
    // stale pre-anneal criticalities the whole time. Instead guarantee a STA+reweight at ~50% log-temp
    // progress (mid-anneal) and one more right when convergence is first detected (late/near-end),
    // reusing stall_lim itself as the post-refresh polish window. Off by default (byte-identical).
    const bool   psta     = std::getenv("SYSTOLIC_PSTA") && atoi(std::getenv("SYSTOLIC_PSTA"));
    const double psta_mid = std::getenv("SYSTOLIC_PSTA_MID") ? atof(std::getenv("SYSTOLIC_PSTA_MID")) : 0.5;
    double ramp_exp = vpre_first;   // timing-term exponent (updated each STA when critgate on); read by reweight()
    long temp0_ramp = K.temp0;   // reference temp for ramp progress (updated after melt at u==0)
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
    // pos_cx/pos_cy: block's position in its own type's compressed sub-array (PE-hop
    // space), refreshed alongside pos_x/pos_y at every gather -- see slot_cx/slot_cy.
    std::vector<int> pos_cx(Nblk, 0), pos_cy(Nblk, 0);
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
    // slot_cx/slot_cy: this slot's (cx,cy) index within its own type's compressed
    // sub-array -- the coordinate system the physical systolic PE array's neighbor
    // topology actually sees. Distinct from slot_x/slot_y (real target-fabric coords):
    // for a sparse type (BRAM/DSP columns) one PE-hop in (cx,cy) can span many fabric
    // units, while for a dense type (CLB, no compression gaps) the two coincide. A
    // block never leaves its own type's slot range, so raw (cx,cy) (no base offset
    // needed) is always comparable within one block's history.
    std::vector<int> slot_cx, slot_cy;
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
        slot_cx.resize(base + (size_t)W * H);
        slot_cy.resize(base + (size_t)W * H);
        st.resize(base + (size_t)W * H);
        slot_locked.resize(base + (size_t)W * H, 0);
        for (int cy = 0; cy < H; cy++)
            for (int cx = 0; cx < W; cx++) {
                const int x = cg.compressed_to_grid_x[0][cx];
                const int y = cg.compressed_to_grid_y[0][cy];
                const int s = base + cy * W + cx;
                slot_x[s] = x; slot_y[s] = y;
                slot_cx[s] = cx; slot_cy[s] = cy;
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
        // (tested a softer 1/F^fanexp, fanexp in [0,1]: fanorm OFF always won, monotonically — reverted.)
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
    double WLWmax = 1e-9;
    for (size_t e = 0; e < Nedge; e++) if (edge_wlw[e] > WLWmax) WLWmax = edge_wlw[e];
    // WL baseline weights: fanout-normalized (edge_wlw rescaled to [1,wmax]) when fanorm is on,
    // else raw multiplicity. This makes inverse-fanout apply to the WL-only path too, not just
    // the timing blend. When timing is on, reweight() overwrites edge_w before the first gather.
    if (K.fanorm) {
        double scale = (double)K.wmax / WLWmax;
        for (size_t e = 0; e < Nedge; e++) { int64_t w = (int64_t)(edge_wlw[e] * scale + 0.5); edge_w[e] = w < 1 ? 1 : w; }
    } else {
        for (size_t e = 0; e < Nedge; e++) edge_w[e] = K.base * edge_mult[e];
    }
    std::vector<double> edge_mc, edge_fw;                    // scratch for the blend
    if (use_timing) { edge_mc.assign(Nedge, 0.0); edge_fw.assign(Nedge, 0.0); }

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
        // Normalized convex blend: w = l*(m/M) + (1-l)*(maxcrit/maxcrit_design)^exp,
        // then rescale all weights so the largest is `wmax` (keeps int resolution;
        // both terms are in [0,1] so raw weights would collapse to 0/1).
        // No need to normalize by the design's own max criticality: VPR's "relaxed criticality"
        // formula (calc_relaxed_criticality, timing_util.cpp) shifts slack by the worst slack in
        // each domain, so the worst-slack pin's criticality is EXACTLY 1.0 by construction --
        // confirmed empirically (mcd=1.000000 on every STA call, every design tested). edge_mc[e] is
        // already on a fixed, guaranteed [0,1] scale.
        #pragma omp parallel for schedule(static)
        for (long e = 0; e < (long)Nedge; e++) {
            double mx = 0.0;
            for (int64_t j = pin_off[e]; j < pin_off[e + 1]; j++) {
                double cr = criticalities->criticality(pin_net[j], pin_ipin[j]);
                if (cr > mx) mx = cr;
            }
            edge_mc[e] = mx;
        }
        // crit-gate WL weight: when on, blend continuously between fanorm'd and full by this edge's
        // own normalized criticality c (blend = c^critgate_smooth_exp); when off, plain fanorm'd weight.
        auto gated_wl = [&](double c, long e) {
            if (!critgate) return edge_wlw[e];
            double blend = std::pow(c, critgate_smooth_exp);
            return blend * (double)edge_mult[e] + (1.0 - blend) * edge_wlw[e];
        };
        double wl_max = 1e-9;
        #pragma omp parallel for schedule(static) reduction(max:wl_max)
        for (long e = 0; e < (long)Nedge; e++) {
            double wl = gated_wl(edge_mc[e], e);
            if (wl > wl_max) wl_max = wl;
        }
        double fwmax = 0.0;
        #pragma omp parallel for schedule(static) reduction(max:fwmax)
        for (long e = 0; e < (long)Nedge; e++) {
            double c = edge_mc[e];   // already on a fixed [0,1] scale -- see note above
            double t = critgate ? std::pow(c, ramp_exp)
                                : ((K.crit_exp == 1) ? c : std::pow(c, (double)K.crit_exp));
            double wl = gated_wl(c, e);
            double fw = K.lambda * (wl / wl_max) + (1.0 - K.lambda) * t;
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

    // ---- annealing loop --------------------------------------------------------
    const bool fixed_sched = (K.mode == "fixed");
    const bool metro = (K.mode == "metro");
    long temp = K.temp0, dec = 0, thr = 0;
    if (fixed_sched) { long steps = K.updts * K.swps - 1; dec = steps > 0 ? temp / steps : 0; thr = steps > 0 ? temp - dec * (steps - 1) : 0; }
    int phase = 0;
    int64_t best_cost = -1;
    int stall = 0, used_updates = 0, n_sta = 0, hold_ctr = 0;
    int swps_cur = K.swps;   // idea #10 adaptive_swps: mutable, shrinks over time; K.swps is the starting ceiling
    // Evenly-strided sample of block ids, built once -- checking the whole design every hold-window
    // (every ~K.hold updates) cost as much as it saved on designs that rarely or never shrink (measured:
    // net SLOWER on conv_layer_hls/deepfreeze.style3, since the O(Nblk log Nblk) check ran every
    // hold-window regardless of payoff). Sample size is derived from K.swps_sample_err (target standard
    // error of the shrink decision's underlying sample-proportion estimate, worst-case p=0.5), with a
    // finite-population correction by Nblk -- see the knob comment for the formula and why it barely
    // moves for any design in this project's benchmark sets.
    std::vector<int> swps_sample_blocks;
    if (K.adaptive_swps) {
        double n0 = 0.25 / (K.swps_sample_err * K.swps_sample_err);
        double n_fpc = n0 / (1.0 + (n0 - 1.0) / (double)Nblk);
        int sample_n = std::min((int)Nblk, (int)std::lround(n_fpc));
        if (sample_n < 1) sample_n = std::min((int)Nblk, 1);
        swps_sample_blocks.reserve(sample_n);
        if (sample_n > 0) {
            double stride = (double)Nblk / sample_n;
            for (int i = 0; i < sample_n; i++) {
                int b = (int)(i * stride);
                if (b >= (int)Nblk) b = (int)Nblk - 1;
                swps_sample_blocks.push_back(b);
            }
        }
    }
    // Median PE-hop displacement since the last gather, among sampled blocks that moved (same
    // statistic as the SYSTOLIC_SWPS_DIAG distance diagnostic's "pe median_moved", just over a fixed
    // sample instead of all Nblk) -- the direct staleness signal the adaptive_swps controller shrinks
    // against. O(sample size); only called at hold-window boundaries.
    auto median_pe_displacement = [&]() -> double {
        std::vector<int> nz; nz.reserve(swps_sample_blocks.size());
        for (int b : swps_sample_blocks) {
            int s = slot_of[b];
            if (s < 0) continue;
            int d = std::abs(slot_cx[s] - pos_cx[b]) + std::abs(slot_cy[s] - pos_cy[b]);
            if (d > 0) nz.push_back(d);
        }
        if (nz.empty()) return 0.0;
        std::nth_element(nz.begin(), nz.begin() + nz.size() / 2, nz.end());
        return (double)nz[nz.size() / 2];
    };
    bool psta_mid_done = false, psta_late_pending = false, psta_late_used = false;
    const double COST_EPS = 1e-3, MELT_A = 0.80;
    const long TEMP_FLOOR = metro ? 1 : 100;
    double t_sync = 0, t_sta = 0, t_rw = 0, t_place = 0;   // runtime breakdown
    double t_gather = 0, t_swap = 0;                        // place sub-split (gather vs swaps)

    // Shared state for the persistent parallel region (declared once, outside).
    int64_t cost = 0; long naccept = 0, nbadatt = 0, nbadacc = 0; int ovf = 0;
    // Staleness diagnostic (idea #10, SYSTOLIC_SWPS_DIAG): swps phases within one update all reuse the
    // SAME gather snapshot (K/Sx/Sy), so later phases' swap decisions can be made on stale neighbor
    // positions. Track the SUM of accepted cost deltas during a update's swap loop (pred_delta_sum) and
    // compare it, at the NEXT gather, against the true recomputed cost change. Any gap between them is
    // exactly how much staleness distorted that update's decisions -- no proxy, directly measured.
    int64_t pred_delta_sum = 0, last_gather_cost = -1;
    long long nmoves = 0;   // TEMP: exact count of pairwise move evaluations
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
        // Shared log-temp progress signal in [0,1], used by both the VPR-exponent ramp and the
        // progress-tied STA mid-point trigger (0 at u==0 / before the post-melt temp0_ramp is set).
        double anneal_prog = 0.0;
        if (u > 0 && temp0_ramp > 1) {
            double denom = std::log((double)temp0_ramp);
            anneal_prog = denom > 0 ? (denom - std::log((double)(temp > 1 ? temp : 1))) / denom : 1.0;
            anneal_prog = anneal_prog < 0 ? 0.0 : (anneal_prog > 1 ? 1.0 : anneal_prog);
        }
        bool psta_mid_hit = psta && !psta_mid_done && anneal_prog >= psta_mid;
        if (psta_mid_hit) psta_mid_done = true;
        bool cadence_hit = (K.cadence > 0 && u % K.cadence == 0);
        bool was_late = psta_late_pending;
        if (use_timing && (u == 0 || cadence_hit || psta_mid_hit || was_late)) {
            psta_late_pending = false;
            if (getenv("SYSTOLIC_STA_DIAG")) {
                const char* why = u == 0 ? "u0" : was_late ? "late" : psta_mid_hit ? "mid" : "cadence";
                VTR_LOG("[systolic-sta-diag] n_sta=%d u=%ld why=%s temp=%ld stall=%d prog=%.3f\n",
                        n_sta, u, why, temp, stall, anneal_prog);
            }
            double _ta = omp_get_wtime(); sync_to_vpr();
            double _tb = omp_get_wtime(); t_sync += _tb - _ta;
            // For a true cadence refresh (u>0), recompute connection delays from the CURRENT placement
            // and invalidate all sink pins, so perform_full_timing_update's incremental STA actually
            // re-propagates (our swaps bypass VPR's move machinery, so nothing is marked changed).
            // Without this, cadence>0 is a no-op (frozen initial-placement criticalities). Gated on u>0
            // so cadence=0 (default) stays byte-identical to VPR's initial timing state.
            if (u > 0) {
                comp_td_connection_delays(delay_model, placer_state);
                if (pin_timing_invalidator)
                    for (auto net_id : clb.nets())
                        for (auto pin_id : clb.net_sinks(net_id))
                            pin_timing_invalidator->invalidate_connection(pin_id);
            }
            if (vpre_ramp) {   // VTR-style: ramp criticality exponent first->last as temp cools
                ramp_exp = vpre_first + anneal_prog * (vpre_last - vpre_first);
                if (!critgate) cp.crit_exponent = (float)ramp_exp;   // uniform exponent via VPR (default path)
            }
            if (critgate) {   // gate needs RAW crit: VPR returns raw, exponent applied to timing in reweight()
                if (!vpre_ramp) ramp_exp = vpre_last;
                cp.crit_exponent = 1.0f;
            }
            perform_full_timing_update(cp, delay_model, criticalities, setup_slacks,
                                       pin_timing_invalidator, timing_info, costs, placer_state);
            double _tc = omp_get_wtime(); t_sta += _tc - _tb;
            reweight();
            fold_weights();   // refresh inline adj_w with the new criticality-weighted edge_w
            t_rw += omp_get_wtime() - _tc;
            n_sta++;
            if ((u == 0 || was_late) && getenv("SYSTOLIC_DIAG")) {
                if (was_late) VTR_LOG("[systolic-diag] (snapshot at LATE STA, u=%ld)\n", u);
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
            if (st[s].blk >= 0) {
                pos_x[st[s].blk] = slot_x[s]; pos_y[st[s].blk] = slot_y[s];
                pos_cx[st[s].blk] = slot_cx[s]; pos_cy[st[s].blk] = slot_cy[s];
            }

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
        if (u > 0 && getenv("SYSTOLIC_SWPS_DIAG")) {
            int64_t predicted = last_gather_cost + pred_delta_sum;
            int64_t residual = cost - predicted;
            double pbad_prev = nbadatt > 0 ? (double)nbadacc / (double)nbadatt : 0.0;
            VTR_LOG("[systolic-swps-diag] u=%ld true_cost=%ld predicted=%ld residual=%ld "
                    "rel=%.6f naccept=%ld pbad=%.3f temp=%ld\n",
                    u, (long)cost, (long)predicted, (long)residual,
                    cost != 0 ? (double)residual / (double)cost : 0.0, naccept, pbad_prev, temp);
        }
        last_gather_cost = cost;
        pred_delta_sum = 0;
        t_gather += omp_get_wtime() - _p0;
        if (metro && u == 0) {
            // scale-invariant hot start: temp = melt * mean|dC| over all candidate pairs.
            // Gated on `metro` (not timing) so WL-only metro is also scale-invariant, not stuck
            // at the fixed temp0 (which can be mis-scaled vs a large design's WL cost).
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
        naccept = 0;
        if (hold_ctr == 0) { nbadatt = 0; nbadacc = 0; }   // accumulate bad-counts across the hold window
        if (u == 0) temp0_ramp = (temp > 1 ? temp : 2);    // reference temp for the exponent ramp
        invTscaled = (temp > 0) ? (double)LUT_SCALE / (double)temp : 0.0;   // temp const within a metro update
        _ps = omp_get_wtime();
        } // end single: gather timing + melt + reset

        for (int sw = 0; sw < swps_cur; sw++) {
            const int p = phase;
            const bool horiz = (p == 0 || p == 2);
            const auto& A = pm[p]; const auto& B = ps[p];
            const long np = (long)A.size();
            #pragma omp for schedule(static) reduction(+:naccept,nbadatt,nbadacc,pred_delta_sum)
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
                    double idxf = (double)d * invTscaled;   // = (dC/T) * LUT_SCALE
                    acc = (idxf < (double)LUT_N) && ((rng[sa] & 0xFFFF) < exp_lut[(int)idxf]);
                }
                else                  acc = ((rng[sa] & 0xFFFF) < (uint32_t)temp);
                if (d > 0) { nbadatt++; if (acc) nbadacc++; }
                if (acc) {
                    std::swap(st[sa], st[sb]);   // blk,K,Sx,Sy travel together (one AoS record)
                    if (st[sa].blk >= 0) slot_of[st[sa].blk] = sa;
                    if (st[sb].blk >= 0) slot_of[st[sb].blk] = sb;
                    naccept++;
                    pred_delta_sum += d;   // staleness diagnostic (#10) -- see note at declaration
                }
            }
            #pragma omp for schedule(static)
            for (long s = 0; s < (long)Nslot; s++) {
                uint32_t v = rng[s];
                uint32_t fb = ((v >> 15) ^ (v >> 14) ^ (v >> 12) ^ (v >> 3)) & 1u;
                rng[s] = ((v << 1) | fb) & 0xFFFF;
            }
            #pragma omp single
            { phase = (phase + 1) % 4; nmoves += np;
              if (fixed_sched) temp = (temp > thr) ? temp - dec : 0;
            }
        }

        #pragma omp single
        {
            if (getenv("SYSTOLIC_SWPS_DIAG")) {
                // Net displacement per block THIS update, in two coordinate systems: PE-hops within
                // the block's own compressed sub-array (slot_cx/pos_cx -- what the physical systolic
                // array's neighbor topology sees) and real target-fabric units (slot_x/pos_x -- what
                // the WL cost function sees). These diverge for sparse types: e.g. a BRAM sub-array's
                // compressed grid packs only BRAM-column locations, so one PE-hop there can span many
                // fabric columns, whereas for a dense type (CLB, no compression gaps) the two coincide.
                // "How often to run updates" is a PE-hop-staleness question (that's the actual hardware
                // neighbor structure the K/Sx/Sy caches travel across), so PE-hops is the primary
                // metric; fabric units are reported alongside for the associated WL-cost context.
                // pos_x/pos_y/pos_cx/pos_cy hold each block's position as of the last gather (start of
                // this update); slot_of[] was kept live during the swap loop, so slot_{x,y,cx,cy}[slot_of[b]]
                // is its position now, at the end of this update's phases.
                std::vector<int> dists_pe, dists_fab; dists_pe.reserve(Nblk); dists_fab.reserve(Nblk);
                long moved = 0;
                for (size_t b = 0; b < Nblk; b++) {
                    int s = slot_of[b];
                    if (s < 0) continue;
                    int dpe  = std::abs(slot_cx[s] - pos_cx[b]) + std::abs(slot_cy[s] - pos_cy[b]);
                    int dfab = std::abs(slot_x[s]  - pos_x[b])  + std::abs(slot_y[s]  - pos_y[b]);
                    dists_pe.push_back(dpe); dists_fab.push_back(dfab);
                    if (dpe > 0) moved++;
                }
                auto summarize = [](const std::vector<int>& dists, double& mean_all, double& mean_moved,
                                     double& median_moved, int& max_moved) {
                    mean_all = 0; for (int d : dists) mean_all += d;
                    mean_all = dists.empty() ? 0.0 : mean_all / dists.size();
                    std::vector<int> nz; for (int d : dists) if (d > 0) nz.push_back(d);
                    median_moved = 0; mean_moved = 0; max_moved = 0;
                    if (!nz.empty()) {
                        std::nth_element(nz.begin(), nz.begin() + nz.size() / 2, nz.end());
                        median_moved = nz[nz.size() / 2];
                        for (int d : nz) { mean_moved += d; if (d > max_moved) max_moved = d; }
                        mean_moved /= nz.size();
                    }
                };
                if (!dists_pe.empty()) {
                    double pe_mean_all, pe_mean_moved, pe_median_moved, fab_mean_all, fab_mean_moved, fab_median_moved;
                    int pe_max_moved, fab_max_moved;
                    summarize(dists_pe, pe_mean_all, pe_mean_moved, pe_median_moved, pe_max_moved);
                    summarize(dists_fab, fab_mean_all, fab_mean_moved, fab_median_moved, fab_max_moved);
                    VTR_LOG("[systolic-dist-diag] u=%ld moved=%ld/%zu(%.1f%%) "
                            "pe: mean_all=%.4f mean_moved=%.3f median_moved=%.1f max_moved=%d | "
                            "fab: mean_all=%.4f mean_moved=%.3f median_moved=%.1f max_moved=%d\n",
                            u, moved, Nblk, 100.0 * moved / (double)Nblk,
                            pe_mean_all, pe_mean_moved, pe_median_moved, pe_max_moved,
                            fab_mean_all, fab_mean_moved, fab_median_moved, fab_max_moved);
                }
            }
            t_swap += omp_get_wtime() - _ps;
            t_place += omp_get_wtime() - _p0;
            used_updates = (int)u + 1;
            if (fixed_sched) {
                if (best_cost < 0 || cost < best_cost) best_cost = cost;
            } else if (++hold_ctr >= K.hold) {   // hold temp for K gathers before cooling (equilibrate; pbad over the window)
                double pbad = nbadatt > 0 ? (double)nbadacc / (double)nbadatt : 0.0;
                if (K.adaptive_swps && swps_cur > K.swps_min) {
                    // pbad is NOT used here -- it can decouple from true displacement (e.g. a design
                    // with a persistent near-zero-cost-gradient block population keeps pbad near 0
                    // forever, since those swaps are never "bad", while the blocks keep moving the
                    // full swps distance every update). Measure displacement directly instead.
                    double med = median_pe_displacement();
                    if (med < K.swps_shrink_thr * (double)swps_cur) {
                        int nxt = (int)((double)swps_cur * K.swps_shrink_mult);
                        if (nxt < K.swps_min) nxt = K.swps_min;
                        if (nxt < swps_cur) {
                            if (getenv("SYSTOLIC_SWPS_DIAG"))
                                VTR_LOG("[systolic-adaptive-swps] u=%ld shrink swps %d -> %d (median_pe=%.1f thr=%.2f)\n",
                                        u, swps_cur, nxt, med, K.swps_shrink_thr * (double)swps_cur);
                            swps_cur = nxt;
                        }
                    }
                }
                if      (pbad > 0.96) temp = temp / 2;
                else if (pbad > MELT_A) temp = (temp * 9) / 10;
                else                    temp = (temp * K.creep) / 100;
                if (pbad < MELT_A) {
                    if (best_cost < 0 || (double)cost < (double)best_cost - COST_EPS * (double)best_cost) { best_cost = cost; stall = 0; }
                    else stall++;
                }
                if (stall >= K.stall_lim || temp <= TEMP_FLOOR) {
                    if (psta && !psta_late_used) {
                        // Guarantee one late STA+reweight on the near-converged placement before
                        // really stopping, so the final polish moves use fresh criticalities instead
                        // of whatever was current when the anneal happened to plateau. Reuses stall
                        // itself as the polish window (up to stall_lim more hold-cycles) -- no new knob.
                        psta_late_used = true;
                        psta_late_pending = true;
                        stall = 0;
                        if (getenv("SYSTOLIC_STA_DIAG"))
                            VTR_LOG("[systolic-sta-diag] LATE-REQUESTED u=%ld temp=%ld used_updates=%d\n",
                                    u, temp, used_updates);
                    } else {
                        if (getenv("SYSTOLIC_STA_DIAG"))
                            VTR_LOG("[systolic-sta-diag] DONE u=%ld temp=%ld stall=%d used_updates=%d n_sta=%d\n",
                                    u, temp, stall, used_updates, n_sta);
                        done = true;
                    }
                }
                hold_ctr = 0;
            }
        }
        (void)naccept;
    }
    } // end persistent parallel region

    // ---- write final locations back into VPR placement state --------------------
    sync_to_vpr();
    // Refresh connection delays from the FINAL placement and invalidate all sink pins, so VPR's
    // post-placement timing analysis (the reported "Placement estimated CPD/Fmax") reflects the actual
    // final placement instead of the stale initial-placement delays. Placement itself is unchanged.
    if (use_timing && pin_timing_invalidator) {
        comp_td_connection_delays(delay_model, placer_state);
        for (auto net_id : clb.nets())
            for (auto pin_id : clb.net_sinks(net_id))
                pin_timing_invalidator->invalidate_connection(pin_id);
    }
    if (created_cgrids) vtr::release_memory(place_ctx.compressed_block_grids);

    double t_total = omp_get_wtime() - t_start;
    double t_other = t_total - t_place - t_sta - t_sync - t_rw;   // setup (adjacency) + writeback
    VTR_LOG("[systolic] TEMP move evals = %lld\n", nmoves);
    VTR_LOG("[systolic] mode=%s crit=%d slots=%zu blocks=%zu edges=%zu updates=%d STA=%d cost=%ld  %.4f s\n",
            K.mode.c_str(), (int)use_timing, Nslot, Nblk, Nedge, used_updates, n_sta, (long)best_cost, t_total);
    if (K.adaptive_swps)
        VTR_LOG("[systolic] adaptive_swps: start=%d final=%d\n", K.swps, swps_cur);
    VTR_LOG("[systolic] time: place %.4fs (%.0f%%) | STA %.4fs (%.0f%%) | sync %.4fs (%.0f%%) | reweight %.4fs (%.0f%%) | setup+writeback %.4fs (%.0f%%)\n",
            t_place, 100*t_place/t_total, t_sta, 100*t_sta/t_total, t_sync, 100*t_sync/t_total,
            t_rw, 100*t_rw/t_total, t_other, 100*t_other/t_total);
    if (getenv("SYSTOLIC_DIAG"))
        VTR_LOG("[systolic] place split: gather %.4fs (%.0f%%) | swaps %.4fs (%.0f%%)\n",
                t_gather, 100*t_gather/t_place, t_swap, 100*t_swap/t_place);
}
