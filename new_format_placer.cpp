// C++/OpenMP software placer for the systolic page placer.
//
// Faithful port of the optimized numba backend in scripts/gpusim.py (which is
// itself bit-identical to scripts/pysim.py). Implements:
//   - Move 3: per-slot state [cell, K, Sx, Sy] packed AoS, int32, travels with
//     the cell on an accepted swap; slot_of rebuilt once per update.
//   - squared-cost delta computed on the fly (x^2), only the active axis per phase.
//   - column-band threading via ONE persistent OpenMP parallel region with an
//     `omp for` per phase (implicit barrier per phase). Threads are not re-forked
//     between phases; red-black guarantees intra-phase disjointness so no locks.
//
// Reproduces pysim bit-for-bit (LFSR, red-black pairing, phase counter, integer
// temperature schedule), so its trace diffs clean with scripts/diff_trace.py.
//
// Build:  g++ -O3 -march=native -fopenmp -std=c++17 cpp/placer.cpp -o cpp/placer
// Run:    cpp/placer grid_info netlist io_placement placer_init out_dir \
//                    [--updts U] [--swps S] [--temp T] [--threads N] [--no-trace]
//
// Threads: the swap is memory-bound, so pin ONE thread per physical core (avoid
//          SMT siblings sharing a core) for stable, best runtime. Do it with env
//          vars -- no rebuild needed:
//              OMP_PLACES=cores OMP_PROC_BIND=spread OMP_NUM_THREADS=N ./placer ...
//          OMP_PLACES=cores makes each "place" a full physical core (the runtime
//          handles the SMT/topology mapping). OMP_PROC_BIND=spread distributes
//          threads across CCDs/sockets for more aggregate L3 + bandwidth; try
//          'close' instead for more shared-L3 locality. Keep N <= #physical cores.
//          Confirm the binding with OMP_DISPLAY_AFFINITY=true.


#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cmath>
#include <omp.h>


using std::string; using std::vector; using std::map; using std::unordered_map;


// ---------------------------------------------------------------- parsing utils
static vector<string> split(const string &s, char d) {
    vector<string> out; string cur; std::stringstream ss(s);
    while (std::getline(ss, cur, d)) out.push_back(cur);
    return out;
}
static vector<string> ws_split(const string &s) {
    vector<string> out; std::stringstream ss(s); string t;
    while (ss >> t) out.push_back(t);
    return out;
}
static bool all_digits(const string &s) {
    if (s.empty()) return false;
    for (char c : s) if (c < '0' || c > '9') return false;
    return true;
}
static string first_token(const string &s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == string::npos) return "";
    size_t e = s.find_first_of(" \t\r\n", b);
    return s.substr(b, e == string::npos ? string::npos : e - b);
}


int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: placer grid netlist io init outdir [--updts U --swps S --temp T --threads N --no-trace]\n"); return 1; }
    string f_grid = argv[1], f_net = argv[2], f_io = argv[3], f_init = argv[4], outdir = argv[5];
    int UP = 2, SW = 1, THREADS = 0; long TEMP = 0; bool no_trace = false;
    string weight_mode = "default";
    for (int i = 6; i < argc; i++) {
        string a = argv[i];
        if (a == "--updts") UP = atoi(argv[++i]);
        else if (a == "--swps") SW = atoi(argv[++i]);
        else if (a == "--temp") TEMP = atol(argv[++i]);
        else if (a == "--threads") THREADS = atoi(argv[++i]);
        else if (a == "--weight_mode") weight_mode = argv[++i];
        else if (a == "--no-trace") no_trace = true;
    }
    if (THREADS > 0) omp_set_num_threads(THREADS);


    // ---------------------------------------------------------------- grid_info
    long total_width = 0, total_height = 0;
    map<string,int> Wt, Ht;                          // per type width/height
    struct GRow { string ty; int cx, cy, x, y; };
    vector<GRow> grows;
    {
        std::ifstream f(f_grid); string line;
        std::getline(f, line); { auto t = ws_split(line); total_width = atol(t[0].c_str()); total_height = atol(t[2].c_str()); }
        std::getline(f, line);                        // header
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            auto t = ws_split(line); if (t.size() < 5) continue;
            GRow g{t[0], atoi(t[1].c_str()), atoi(t[2].c_str()), atoi(t[3].c_str()), atoi(t[4].c_str())};
            grows.push_back(g);
            Wt[g.ty] = std::max(Wt.count(g.ty) ? Wt[g.ty] : 0, g.cx + 1);
            Ht[g.ty] = std::max(Ht.count(g.ty) ? Ht[g.ty] : 0, g.cy + 1);
        }
    }
    // sorted type list (matches Python sorted(grid.grid_types)); map iteration is sorted
    vector<string> types; for (auto &kv : Wt) types.push_back(kv.first);
    map<string,long> slot_base, cell_base, Nt;
    long S = 0, C = 0;
    for (auto &ty : types) { long n = (long)Wt[ty] * Ht[ty]; slot_base[ty] = S; cell_base[ty] = C; Nt[ty] = n; S += n; C += n; }
    auto gslot = [&](const string &ty, int x, int y) { return slot_base[ty] + (long)y * Wt[ty] + x; };


    // ---------------------------------------------------------------- placer_init
    map<string, vector<int>> init_id, init_seed;     // per type, indexed by cy*W+cx
    for (auto &ty : types) { init_id[ty].assign(Nt[ty], 0); init_seed[ty].assign(Nt[ty], 0); }
    {
        std::ifstream f(f_init); string line; std::getline(f, line);   // header
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            auto t = ws_split(line); if (t.size() < 5) continue;
            string ty = t[0]; int cx = atoi(t[1].c_str()), cy = atoi(t[2].c_str());
            int seed = atoi(t[3].c_str()), id = atoi(t[4].c_str());
            long idx = (long)cy * Wt[ty] + cx;
            init_seed[ty][idx] = seed; init_id[ty][idx] = id;
        }
    }


    // ---------------------------------------------------------------- io_placement
    vector<string> io_names; unordered_map<string,int> io_index; vector<int> io_x, io_y;
    {
        std::ifstream f(f_io); string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            auto t = ws_split(line); if (t.size() < 3) continue;
            io_index[t[0]] = (int)io_names.size(); io_names.push_back(t[0]);
            io_x.push_back(atoi(t[1].c_str())); io_y.push_back(atoi(t[2].c_str()));
        }
    }
    int n_io = (int)io_names.size();


    // ---------------------------------------------------------------- netlist
    struct NInfo { string ty; int sid; };
    unordered_map<string, NInfo> nameInfo;           // name -> (type, systolic id)
    unordered_map<string, string> nameVtr;           // name -> vtr block id (string)
    map<string, vector<string>> idToName;            // per type, sid -> name
    unordered_map<string, unordered_map<string,long>> nameWeights;  // name -> nbr -> weight
    unordered_map<string, vector<std::pair<string,int>>> rawConnections;
    string net_name, net_hash;
    {
        std::ifstream f(f_net); string line;
        std::getline(f, line); { auto t = ws_split(line); net_name = t[1]; net_hash = t[3]; }
        std::getline(f, line);                        // throwaway
        map<string,int> typeCount;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            auto fields = split(line, ',');
            auto head = ws_split(fields[0]);          // "name ty vtrid"
            if (head.size() < 3) continue;
            string name = head[0], ty = head[1], vtr = head[2];
            int sid = typeCount[ty]++;
            nameInfo[name] = {ty, sid}; nameVtr[name] = vtr;
            if ((int)idToName[ty].size() <= sid) idToName[ty].resize(sid + 1);
            idToName[ty][sid] = name;
            for (size_t i = 1; i < fields.size(); i++) {
                string tok = fields[i]; if (tok.empty()) continue;
                if (weight_mode == "default") {
                    string conn_name = first_token(tok);
                    if (!conn_name.empty()) nameWeights[name][conn_name] += 1;
                } else {
                    auto conn = ws_split(tok);
                    if (conn.size() >= 2) {
                        rawConnections[name].push_back({conn[0], atoi(conn[1].c_str())});
                    } else {
                        auto p = tok.rfind(':');
                        if (p != string::npos && all_digits(tok.substr(p+1))) rawConnections[name].push_back({tok.substr(0,p), atoi(tok.substr(p+1).c_str())});
                        else rawConnections[name].push_back({tok, 1});
                    }
                }
            }
        }
        if (weight_mode != "default") {
            auto mode = split(weight_mode, '_');
            if (mode.size() != 3) { fprintf(stderr, "Invalid weight_mode: %s\n", weight_mode.c_str()); return 1; }
            int A = atoi(mode[0].c_str()), Cpow = atoi(mode[1].c_str());
            double L = atof(mode[2].c_str());
            if (L < 0.0 || L > 1.0) { fprintf(stderr, "gated-convex L must be in [0, 1], got %g\n", L); return 1; }
            unordered_map<string, unordered_map<string,int>> mult, maxDepth;
            int globalMaxDepth = 0, globalMaxMultiplicity = 0;
            for (auto &kv : rawConnections) {
                for (auto &conn : kv.second) {
                    int m = ++mult[kv.first][conn.first];
                    int &d = maxDepth[kv.first][conn.first];
                    d = std::max(d, conn.second);
                    globalMaxMultiplicity = std::max(globalMaxMultiplicity, m);
                    globalMaxDepth = std::max(globalMaxDepth, d);
                }
            }
            double globalD = std::max(1, globalMaxDepth);
            double globalM = std::max(1, globalMaxMultiplicity);
            double maxScore = 0.0;
            for (auto &kv : mult) {
                for (auto &conn : kv.second) {
                    double multiplicityScore = (double)conn.second / globalM;
                    double depthScore = std::pow((double)maxDepth[kv.first][conn.first] / globalD, Cpow);
                    double score = ((1.0 - L) * multiplicityScore) + (L * multiplicityScore * depthScore);
                    maxScore = std::max(maxScore, score);
                }
            }
            double scoreScale = maxScore > 0.0 ? maxScore : 1.0;
            for (auto &kv : mult) {
                auto &w = nameWeights[kv.first];
                for (auto &conn : kv.second) {
                    double multiplicityScore = (double)conn.second / globalM;
                    double depthScore = std::pow((double)maxDepth[kv.first][conn.first] / globalD, Cpow);
                    double score = ((1.0 - L) * multiplicityScore) + (L * multiplicityScore * depthScore);
                    w[conn.first] = (long)(A * (score / scoreScale));
                }
            }
        }
    }


    // ---------------------------------------------------------------- flatten
    long S_all = S + n_io, C_all = C + n_io;
    vector<int32_t> slot_x(S_all, 0), slot_y(S_all, 0);
    vector<int32_t> st((size_t)S * 4, 0);            // [cell, K, Sx, Sy] per real slot
    vector<int32_t> slot_of(C_all, 0), rng(S, 0), Kc(C, 0);
    // real slots + initial placement
    for (auto &ty : types) {
        int W = Wt[ty], H = Ht[ty];
        for (auto &g : grows) if (g.ty == ty) {
            long s = gslot(ty, g.cx, g.cy);
            slot_x[s] = g.x; slot_y[s] = g.y;
        }
        for (int x = 0; x < W; x++) for (int y = 0; y < H; y++) {
            long s = gslot(ty, x, y); long idx = (long)y * W + x;
            long gc = cell_base[ty] + init_id[ty][idx];
            st[s*4+0] = (int32_t)gc; slot_of[gc] = (int32_t)s; rng[s] = init_seed[ty][idx] & 0xFFFF;
        }
    }
    // io pseudo slots/cells
    for (int k = 0; k < n_io; k++) {
        long gs = S + k, gc = C + k;
        slot_x[gs] = io_x[k]; slot_y[gs] = io_y[k]; slot_of[gc] = (int32_t)gs;
    }
    // resolve a connection name to a global cell id (or -1 to skip)
    auto resolve = [&](const string &nm) -> long {
        auto it = nameInfo.find(nm);
        if (it != nameInfo.end()) {
            auto cb = cell_base.find(it->second.ty);
            if (cb != cell_base.end()) return cb->second + it->second.sid;
        }
        auto io = io_index.find(nm);
        if (io != io_index.end()) return C + io->second;
        return -1;
    };
    // CSR adjacency for real cells
    vector<int64_t> adj_off(C + 1, 0);
    vector<int32_t> adj_nbr, adj_w;
    for (auto &ty : types) {
        for (long local = 0; local < Nt[ty]; local++) {
            long gc = cell_base[ty] + local;
            if (local < (long)idToName[ty].size() && !idToName[ty][local].empty()) {
                const string &name = idToName[ty][local];
                for (auto &kv : nameWeights[name]) {
                    long g_nbr = resolve(kv.first); if (g_nbr < 0) continue;
                    Kc[gc] += (int32_t)kv.second;
                    adj_nbr.push_back((int32_t)g_nbr); adj_w.push_back((int32_t)kv.second);
                }
            }
            adj_off[gc + 1] = (int64_t)adj_nbr.size();
        }
    }
    // init packed K (Sx/Sy filled by first update)
    for (long s = 0; s < S; s++) st[s*4+1] = Kc[st[s*4+0]];


    // red-black pairs per phase (matches pysim phase_0..3)
    vector<int32_t> pm, ps; long phase_off[5] = {0,0,0,0,0};
    for (int phase = 0; phase < 4; phase++) {
        for (auto &ty : types) {
            int W = Wt[ty], H = Ht[ty];
            if (phase == 0 || phase == 2) {
                int start = (phase == 0) ? 0 : 1;
                for (int cy = 0; cy < H; cy++) for (int cx = start; cx < W; cx += 2) {
                    if (cx == W - 1) continue;
                    pm.push_back((int32_t)gslot(ty, cx, cy)); ps.push_back((int32_t)gslot(ty, cx + 1, cy));
                }
            } else {
                int start = (phase == 1) ? 1 : 0;
                // cy-outer / cx-inner: consecutive pairs stride by 1 slot, so the swap
                // streams st row-major (was cx-outer -> stride W, column-major = cache-hostile).
                // Red-black keeps pairs disjoint, so reordering them is bit-identical.
                for (int cy = start; cy < H; cy += 2) {
                    if (cy == H - 1) continue;
                    for (int cx = 0; cx < W; cx++) {
                        pm.push_back((int32_t)gslot(ty, cx, cy)); ps.push_back((int32_t)gslot(ty, cx, cy + 1));
                    }
                }
            }
        }
        phase_off[phase + 1] = (long)pm.size();
    }
    long Csz = C, Ssz = S;                            // for capture in the parallel region
    vector<int64_t> Sxc(C, 0), Syc(C, 0);


    // ---------------------------------------------------------------- schedule
    long temp = TEMP;
    long steps = (long)UP * SW - 1;
    long dec = steps > 0 ? temp / steps : 0;
    long thr = steps > 0 ? temp - dec * (steps - 1) : 0;
    int phase = 0;


    std::filesystem::create_directories(outdir);
    string trace_path = outdir + "/behavioral_trace.csv";
    FILE *tf = nullptr;


    // ---------------------------------------------------------------- trace writer (serial)
    auto write_trace = [&](const char *state) {
        for (auto &ty : types) {
            int W = Wt[ty], H = Ht[ty];
            for (int x = 0; x < W; x++) for (int y = 0; y < H; y++) {
                long s = gslot(ty, x, y); long gc = st[s*4+0]; long local = gc - cell_base[ty];
                fprintf(tf, "%s,%s,%d,%d,%ld,%d,%d,%d,%d,0,0,0,%ld\n",
                        state, ty.c_str(), x, y, local, slot_x[s], slot_y[s], st[s*4+2], st[s*4+3], temp);
            }
        }
        fprintf(tf, "null,null,0,0,0,0,0,0,0,0,0,0,0\n");
    };


    if (!no_trace) {
        tf = fopen(trace_path.c_str(), "w");
        fprintf(tf, "state,pe_type,pe_x,pe_y,blk_id,blk_x,blk_y,px,py,temp_blk_id,temp_x,temp_y,temperature\n");
    }


    // ---------------------------------------------------------------- placement (persistent threads)
    double _t0 = omp_get_wtime();
    #pragma omp parallel
    {
        for (int u = 0; u < UP; u++) {
            // update: rebuild slot_of, gather (per cell), scatter (to slots)
            #pragma omp for schedule(static)
            for (long s = 0; s < Ssz; s++) slot_of[st[s*4+0]] = (int32_t)s;
            #pragma omp for schedule(static)
            for (long c = 0; c < Csz; c++) {
                int64_t ax = 0, ay = 0;
                for (int64_t j = adj_off[c]; j < adj_off[c+1]; j++) {
                    int32_t t = slot_of[adj_nbr[j]];
                    ax += (int64_t)adj_w[j] * slot_x[t];
                    ay += (int64_t)adj_w[j] * slot_y[t];
                }
                Sxc[c] = ax; Syc[c] = ay;
            }
            #pragma omp for schedule(static)
            for (long s = 0; s < Ssz; s++) { long c = st[s*4+0]; st[s*4+2] = (int32_t)Sxc[c]; st[s*4+3] = (int32_t)Syc[c]; }


            if (!no_trace) {
                #pragma omp single
                write_trace("post_sum");             // implicit barrier after single
            }


            for (int sw = 0; sw < SW; sw++) {
                long lo = phase_off[phase], hi = phase_off[phase+1];
                if (phase == 0 || phase == 2) {      // horizontal: x-axis
                    #pragma omp for schedule(static)
                    for (long i = lo; i < hi; i++) {
                        int32_t sa = pm[i], sb = ps[i];
                        int64_t xa = slot_x[sa], xb = slot_x[sb];
                        int64_t da = (int64_t)st[sa*4+1]*(xb*xb-xa*xa) - 2*(int64_t)st[sa*4+2]*(xb-xa);
                        int64_t db = (int64_t)st[sb*4+1]*(xa*xa-xb*xb) - 2*(int64_t)st[sb*4+2]*(xa-xb);
                        if (da + db < 0 || rng[sa] < temp)
                            for (int k = 0; k < 4; k++) { int32_t t = st[sa*4+k]; st[sa*4+k] = st[sb*4+k]; st[sb*4+k] = t; }
                    }
                } else {                             // vertical: y-axis
                    #pragma omp for schedule(static)
                    for (long i = lo; i < hi; i++) {
                        int32_t sa = pm[i], sb = ps[i];
                        int64_t ya = slot_y[sa], yb = slot_y[sb];
                        int64_t da = (int64_t)st[sa*4+1]*(yb*yb-ya*ya) - 2*(int64_t)st[sa*4+3]*(yb-ya);
                        int64_t db = (int64_t)st[sb*4+1]*(ya*ya-yb*yb) - 2*(int64_t)st[sb*4+3]*(ya-yb);
                        if (da + db < 0 || rng[sa] < temp)
                            for (int k = 0; k < 4; k++) { int32_t t = st[sa*4+k]; st[sa*4+k] = st[sb*4+k]; st[sb*4+k] = t; }
                    }
                }
                // advance every slot's LFSR once per phase
                #pragma omp for schedule(static)
                for (long s = 0; s < Ssz; s++) {
                    int32_t v = rng[s];
                    int32_t fb = ((v>>15) ^ (v>>14) ^ (v>>12) ^ (v>>3)) & 1;
                    rng[s] = ((v<<1) | fb) & 0xFFFF;
                }
                #pragma omp single
                { phase = (phase + 1) % 4; temp = (temp > thr) ? temp - dec : 0; }  // barrier after single
            }


            if (!no_trace) {
                #pragma omp single
                write_trace("post_swap");
            }
        }
    }


    fprintf(stderr, "[place %.4f s, %d threads, %ld cells]\n", omp_get_wtime() - _t0, omp_get_max_threads(), C);
    if (!no_trace) { fclose(tf); fprintf(stderr, "[completed] -> %s\n", trace_path.c_str()); }
    else {
        // placement-engine mode: write final_systolic.place (VTR format)
        string pp = outdir + "/final_systolic.place";
        FILE *pf = fopen(pp.c_str(), "w");
        fprintf(pf, "Netlist_File: %s Netlist_ID: %s\n", net_name.c_str(), net_hash.c_str());
        fprintf(pf, "Array size: %ld x %ld logic blocks\n\n", total_width, total_height);
        fprintf(pf, "#block name x       y       subblk  layer   block number\n");
        fprintf(pf, "#---------- --      --      ------  -----   ------------\n");
        for (auto &ty : types) {
            int W = Wt[ty], H = Ht[ty];
            for (int x = 0; x < W; x++) for (int y = 0; y < H; y++) {
                long s = gslot(ty, x, y); long local = st[s*4+0] - cell_base[ty];
                if (local >= 0 && local < (long)idToName[ty].size() && !idToName[ty][local].empty()) {
                    const string &nm = idToName[ty][local];
                    fprintf(pf, "%s %-8d%-8d%-8s%-8s#%-8s\n", nm.c_str(), slot_x[s], slot_y[s], "0", "0", nameVtr[nm].c_str());
                }
            }
        }
        fclose(pf); fprintf(stderr, "[completed] -> %s\n", pp.c_str());
    }
    return 0;
}
