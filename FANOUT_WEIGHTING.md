# Fanout weighting in the systolic placer

How net fanout is handled in the squared-wirelength objective, why unnormalized
fanout over-weights high-fanout nets, the `1/(pins-1)` normalization (with a
derivation and references), and the measured effect.

---

## 1. How nets become weighted two-pin connections

Our objective is decoupled **squared-Manhattan wirelength**:
`cost = Σ_slots [K·(x²+y²) − (x·Sx + y·Sy)] = Σ_connections w·(Δx²+Δy²)`.
It only understands **two-pin connections**, so a multi-pin net (a hyperedge) must
be decomposed. We use the **star model**: the net driver is connected to each of
its `F` sinks → `F` two-pin connections (`F` = fanout = #sinks = pins−1). We then
collapse connections by block-pair (multiplicity), and for the timing objective
each connection also carries the criticality of its sink pin.

## 2. The over-weighting problem

With unit weights, a fanout-`F` net contributes `F` squared terms to the cost — it
grows **linearly with fanout**. But the real *routed* wirelength of a high-fanout
net grows much slower (≈ its bounding box; routing shares wires). Two standard
tools correct for this, and our unnormalized model used neither:

- **VPR** multiplies each net's half-perimeter by a `crossing_count(fanout)` factor
  that grows *sub-linearly*, shrinking per-connection influence as fanout rises.
- **Analytic/quadratic placers** normalize the two-pin weight by `1/(p−1)` (star)
  or `2/p` (clique) so a net's total contribution is ~fanout-independent.

Consequence: high-fanout nets **dominate** the objective, so the annealer
over-invests in clustering them at the expense of the many small nets — inflating
total wirelength relative to VPR's HPWL.

## 3. Measurement (why this matters here)

`--place_algorithm systolic` with env `SYSTOLIC_DIAG=1` dumps the fanout
distribution and criticality-by-fanout (from real STA). On our benchmarks:

- **High-fanout nets dominate the connection count.** On LU32PEEng ~60% of all
  sinks come from fanout≥8 nets (hetero_placer ~33%), including a 2296-sink net on
  hetero and 512–1024-fanout nets on LU32. These are exactly what the squared-star
  model over-weights.
- **Critical connections are overwhelmingly *low*-fanout.** ~86% of hetero's
  crit>0.5 sinks are on fanout≤3 nets; ~72% for LU32. (hetero has only 37 critical
  connections total, out of 117k — its whole CPD is set by a handful.)

So down-weighting high-fanout nets attacks the WL over-weighting **without**
touching the (low-fanout) critical connections — it should help WL and leave
timing alone.

## 4. The fix: `1/(pins−1)` fanout normalization

Each driver→sink connection's WL weight is scaled by `1/(pins−1) = 1/F`. A net's
total WL weight is then `F · (1/F) = 1`, i.e. **every net contributes the same
total weight regardless of fanout**. Enabled by default (`SYSTOLIC_FANORM=1`).
This applies to the WL term of the blend; the criticality (timing) term is
unaffected, so critical connections keep their pull.

## 5. Derivation of the `1/(p−1)` factor

Take a net with `p` pins at positions `x_1..x_p`.

**Clique vs star, and their equivalence.** Two ways to turn the hyperedge into
quadratic two-pin springs:
- *Clique*: connect all pairs → `p(p−1)/2` springs, cost `Σ_{i<j}(x_i−x_j)²`.
- *Star*: add a center node (or use the driver) → `p` (or `p−1`) springs,
  cost `Σ_i (x_i − x_c)²`, minimized at the centroid `x_c = x̄`.

Using the identity `Σ_{i<j}(x_i−x_j)² = p · Σ_i (x_i − x̄)²`, the unit-weight
clique cost equals `p ×` the (minimized) star cost. So with edge weights `γ`
(clique) and `p·γ` (star) the two models produce identical forces/optima — they
are interchangeable (Sigl–Doll–Johannes, DAC 1991).

**Why normalize, and why `1/(p−1)`.** With unit weights a `p`-pin net's cost
scales with `p`, so large nets dominate and (in analytic solvers) create dense,
ill-conditioned rows. Normalize by giving each two-pin connection weight
`W/(p−1)` for a net of weight `W`:
- **Star (our case):** the driver has `p−1` sink connections, so the net's total
  weight is `(p−1)·W/(p−1) = W` — **constant, independent of fanout.** That is
  exactly the property we want.
- **Clique:** each pin has `p−1` incident edges, so with `W/(p−1)` **every pin
  "sees" total incident weight `W`** to the net regardless of net size — bounding
  the linear-system row sums.

This `1/(p−1)` normalization is used explicitly in the **Bound2Bound net model of
Kraftwerk2** (its two-pin weight is `2/((p−1)·|distance|)`), and the `2/p` variant
appears in standard textbook treatments. For our star objective, `1/(p−1) = 1/F`
per driver→sink connection is the natural choice, and it is what we implement.

**Relation to routed WL.** Constant-per-net (`1/(p−1)`) is the analytic-placement
convention. VPR's `crossing_count` grows *slightly* with fanout (a net's routed
length does increase with fanout, just sub-linearly), so a possible refinement is
to use VPR-style `crossing_count(F)/F` instead of `1/F`. `1/(p−1)` is the standard
first-order fix and already recovers most of the benefit (Section 7).

## 6. References

- G. Sigl, K. Doll, F. M. Johannes, *"Analytical Placement: A Linear or a Quadratic
  Objective Function?"*, DAC 1991 — canonical clique-vs-star net-model analysis and
  the weighting equivalence.
- P. Spindler, U. Schlichtmann, F. M. Johannes, *"Kraftwerk2 — A Fast Force-Directed
  Quadratic Placement Approach Using an Accurate Net Model"*, IEEE TCAD 2008 — the
  Bound2Bound net model, where the `1/(p−1)` normalization appears explicitly:
  [scispace](https://scispace.com/papers/kraftwerk2-a-fast-force-directed-quadratic-placement-1a80tcixba).
- Chris Chu, *"Placement"* (EDA textbook chapter) — clique/star/hybrid net models
  and their weighting:
  [PDF](http://cc.ee.ntu.edu.tw/~ywchang/Courses/PD_Source/EDA_placement.pdf).
- M. Pantelias, *"Quadratic VLSI Placement"* (course notes) — net-model decomposition
  overview: [PDF](https://www.csd.uoc.gr/~hy583/presentations_fall_2005/pantelias.pdf).
- M.-C. Kim et al., *"SimPL: An Effective Placement Algorithm"*, ICCAD 2010 — modern
  quadratic placement using these net models:
  [PDF](https://web.eecs.umich.edu/~imarkov/pubs/conf/iccad10-simpl.pdf).
- A. B. Kahng, J. Lienig, I. L. Markov, J. Hu, *"VLSI Physical Design: From Graph
  Partitioning to Timing Closure"*, Springer — textbook derivation of clique (`2/p`)
  and star net models.
- V. Betz, J. Rose, A. Marquardt, *"Architecture and CAD for Deep-Submicron FPGAs"*,
  1999 — VPR's bounding-box cost with the fanout `crossing_count` correction.

## 7. Measured effect (fanorm off → on, routed)

| design | WL off → on | CPD off → on |
|---|---|---|
| sha (small, low fanout skew) | 15,733 → 15,857 (+0.8%, noise) | 14.31 → 13.76 |
| hetero_placer | 772,644 → **741,679 (−4.0%)** | 10.58 → 10.71 |
| LU32PEEng | 972,425 → **904,330 (−7.0%)** | 80.99 → 79.76 |

Fanout normalization cuts wirelength 4–7% on the big designs (the ones with heavy
fanout tails), CPD-neutral-to-slightly-better, and is within noise on the small
design. It closed hetero's WL gap to VPR-bbox from +7.8% to +3.5%. Default: **on**.

## 7b. Tested and rejected: directed (asymmetric) weighting

A natural idea is to weight each connection by `1/(pins−1)` from the **driver's** side
(so a net doesn't over-pull its driver) but by `1` from the **sink's** side (so a sink
feels each of its nets equally). This is representable in the O(1) swap — directed weights
only change how each cell's cached `K`/`Sx` are built; the swap still reads cached values.
(It does break the symmetric factor-2 identity that makes the swap delta an *exact* cost
change, so it becomes a heuristic, but it stays O(1).)

Measured, it is **worse**, mostly on WL:
arm_core +5.9% WL / +7.8% CPD, LU32PEEng +11.7% WL / −1.0% CPD, hetero +7.2% WL / +0.8% CPD,
or1200 −0.7% WL / +5.9% CPD. Reason: unit sink weight makes a fanout-`p` net's total pull
grow ~`p` again → high-fanout nets over-cluster → WL degrades. This is the "net-total
domination" horn of the per-pin-vs-per-net tension in §5 — symmetric `1/(p−1)` (constant
per-net total) controls it and wins. So we keep symmetric; the directed option was
implemented, measured, and **removed** (kept here only as a record).

## 8. Knobs (in `vpr/src/place/systolic_placer.cpp`)

- `SYSTOLIC_FANORM` (default 1): 1 = fanout-normalize WL weight by `1/(pins−1)`;
  0 = raw multiplicity (legacy).
- `SYSTOLIC_DIAG` (default 0): dump the fanout distribution + criticality-by-fanout.
- Related weighting knobs: `SYSTOLIC_LAMBDA` (WL/timing blend), `SYSTOLIC_CRIT_EXP`,
  `SYSTOLIC_WMAX`, `SYSTOLIC_CADENCE`. See the header of `systolic_placer.cpp`.
