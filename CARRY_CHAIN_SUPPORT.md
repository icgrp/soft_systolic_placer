# Design: carry-chain (placement-macro) support for the systolic placer

Status: **design only, not implemented.** Captures why we'd want it, why the current
placer can't do it, why it is nonetheless feasible without losing the O(1) swap or the
multithreading, a concrete design sketch, and honest expectations.

---

## 1. Motivation

The VTR arch we use (`heterogeneous_k10.xml`) is explicitly **"No Carry Chains"**. With
no dedicated carry logic, synthesis maps arithmetic (adders, ALUs) to **ripple chains of
LUTs** — e.g. a 32-bit add becomes ~32 LUT levels. Measured consequences:

- The designs where the systolic placer *beats* VPR-timing are the **deep** ones
  (LU32PEEng critical-path logic depth ≈ 123, mcml ≈ 103), and their critical paths run
  through arithmetic (`replace_alu…`). They are deep *because* arithmetic is LUT-mapped.
- Our Fmax advantage correlates with **logic-dominated** critical paths; we lag on
  **routing-dominated** ones (routing-fraction-of-critical-path vs our CPD gap: r = +0.68;
  logic depth only r = −0.46, a proxy). See `VTR_COMPARISON_RESULTS.md` / the short-path
  analysis.
- On a **real** FPGA (carry chains present), that arithmetic collapses to a few fast-carry
  levels → those critical paths become short and routing-dominated → our advantage would
  likely shrink. To target realistic arches at all, the placer must handle carry chains.

A carry chain is a **placement macro**: a rigid group of blocks that must keep fixed
relative positions (a contiguous column, `cout`→`cin`). VPR already models these
(`t_pl_macro`) and moves them as units.

## 2. Why the current placer can't (and why fixed-first is wrong)

- The red-black swap moves **individual blocks independently** between slots, so it would
  shred any carry chain. The current no-carry arch simply avoids the issue.
- **Placing macros first and locking them is a bad idea**: carry chains are tightly
  entangled with the surrounding fabric — you can't know where to cluster or spread them
  without knowing where all the other cells go. Macros must be **co-optimized** with cells.

## 3. Why it is feasible (the key insights)

1. **O(1) is preserved.** A macro move's cost delta is just the **sum of the per-block
   O(1) decoupled deltas** (`K/Sx/Sy`). It scales with **chain length** (a small constant,
   ~3–8), *not* with fanout/connections. Moving multiple blocks ≠ losing O(1).
   - Subtlety: for a *rigid* move the intra-chain connections don't change (distance
     preserved), so they should net to zero. With stale `Sx` they'd add a tiny error —
     either exclude intra-macro edges from the delta, or let the next gather correct it
     (same staleness we already tolerate).
2. **Co-optimization, not fixed-first.** Macros move *during* the anneal alongside cells.
3. **Multithreading survives via band ownership.** If each thread owns a band and a chain
   fits inside a band, the macro is entirely within one thread's domain → no cross-thread
   conflict.

## 4. Design sketch

### Macro representation
- A macro = ordered member blocks with fixed relative offsets (vertical column for carry).
- Store: per macro, member list + offsets + anchor slot; per block, its macro id (or none).
- A macro occupies a contiguous run of compatible slots.

### Cost evaluation (O(chain length))
- To move a macro from anchor `A` to `A'`, each member moves to the corresponding new slot;
  `delta = Σ_members da(member)` using the cached decoupled `K/Sx/Sy`.
- Exclude intra-macro edges (rigid → unchanged) or rely on gather correction.

### Move types (red-black analog)
- **Singletons**: existing adjacent-slot red-black swaps, unchanged.
- **Macros**: shift a chain by one position (e.g. one column over), rotating the displaced
  cells/macro in the target region into the vacated slots; cost = sum of the affected
  blocks' per-block deltas (still O(#affected)). Equal-shape macro↔macro swaps also work.
- **Legality**: the target must be a contiguous run of compatible slots (correct block
  type; a valid column for the chain) — a stronger check than the single-cell type test.

### Multithreading: band decomposition + band-granularity red-black
- Threads own bands; macros interior to a band move with no conflict.
- **Cross-band / boundary macros are the hard part.** Handle with red-black *at the band
  level*:
  - Phase A: move only macros fully interior to each band (parallel, safe).
  - Phase B: **shift the band boundaries** (offset the partition) so macros that were at a
    boundary become interior and get their turn.
  - Or process boundary-straddling macros in a short synchronized serial pass.
- Band orientation: vertical carry chains fit vertical bands for vertical moves; horizontal
  exploration (trying another column) crosses bands and is covered by the boundary-shifting
  phases.

## 5. Remaining challenges / open questions
- Cross-band macro migration (the band-level coloring above) — the main new complexity.
- Legality/contiguity checks for macro targets (contiguous column of correct type).
- Move-proposal mix: how often to attempt macro moves vs cell swaps, and displacement range.
- Interaction with criticality weighting: carry-chain nets are often critical, so the
  existing `criticality^exp` weights apply naturally — likely a plus.
- Intra-macro edge handling in the decoupled cost (exclude vs stale-correct).
- Initial macro placement: start from VPR's initial legal placement (it already groups and
  places macros); reuse `t_pl_macro` groupings when integrated in VPR.

## 6. Expected outcome (honest)
- **Coverage, not (necessarily) advantage.** This lets the systolic placer *run* on
  carry-chain arches — removing the "carry-free only" limitation and letting us target
  realistic FPGAs.
- On such arches, arithmetic collapses → short, routing-dominated critical paths → by the
  r=+0.68 finding we expect to be **competitive-to-lagging** with VPR-timing on Fmax, while
  keeping the large placement-speed advantage. So build this for generality; set Fmax
  expectations accordingly.

## 7. Hardware (accelerator) implications
- The FPGA accelerator would need macro-aware moves (move a group together); the
  chain-length-scaled cost is cheap in hardware. The band decomposition maps to the
  systolic array's spatial tiling, and the cross-band handling is the array-tiling problem.
  Future hardware work, but nothing here is fundamentally hostile to the accelerator.

## 8. Relationship to prior negative results
Unlike the linear-timing term and directed weighting (both tested and rejected as net-
negative), carry-chain support is a **coverage/correctness** feature with a clear feasible
design — the open question is engineering effort and the Fmax expectation above, not whether
it can be made to work within the O(1) + multithreaded framework.
