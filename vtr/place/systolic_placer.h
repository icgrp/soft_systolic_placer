#pragma once
/**
 * @file systolic_placer.h
 * @brief Relaxed parallel simulated-annealing placer (the "systolic" placer).
 *
 * A custom placement algorithm selected via --place_algorithm systolic. Instead
 * of VPR's per-move annealer it runs a red-black checkerboard of pair swaps over
 * a grid of slots, evaluating an O(1) decoupled squared-Manhattan wirelength cost
 * weighted (optionally) by timing criticality. It reuses VPR's block-location,
 * delay-model and timing/criticality data structures, recomputing criticalities
 * from real STA on a cadence during the anneal, then writes the result back into
 * VPR's placement state so the normal finalization / routing flow proceeds.
 *
 * Behaviour knobs are read from environment variables (research code):
 *   SYSTOLIC_MODE    metro|threshold|fixed    (default metro)
 *   SYSTOLIC_SWPS    swap phases per update    (default 10)
 *   SYSTOLIC_UPDTS   max updates (safety cap)  (default 100000)
 *   SYSTOLIC_TEMP    initial temperature       (default 65535)
 *   SYSTOLIC_CREEP   productive-band cool %    (default 95)
 *   SYSTOLIC_STALL   plateau window (updates)  (default 12)
 *   SYSTOLIC_THREADS OpenMP threads            (default: OMP default)
 *   SYSTOLIC_CRIT    1=timing-weighted, 0=WL-only (default 1)
 *   SYSTOLIC_CSCALE  criticality weight scale  (default 8)
 *   SYSTOLIC_CRIT_EXP criticality exponent     (default 1)
 *   SYSTOLIC_BASE    base (WL) weight per edge (default 1)
 *   SYSTOLIC_CADENCE 0 = STA once at start only (default); N>0 = also refresh every N updates
 */

class PlacerState;
class PlaceDelayModel;
class PlacerCriticalities;
class PlacerSetupSlacks;
class NetPinTimingInvalidator;
class SetupTimingInfo;
struct t_placer_costs;
struct PlaceCritParams;

/// Run the systolic placer, mutating block locations in @p placer_state.
/// Timing objects may be null when timing analysis is disabled (WL-only).
void run_systolic(PlacerState& placer_state,
                  const PlaceDelayModel* delay_model,
                  PlacerCriticalities* criticalities,
                  PlacerSetupSlacks* setup_slacks,
                  NetPinTimingInvalidator* pin_timing_invalidator,
                  SetupTimingInfo* timing_info,
                  t_placer_costs* costs,
                  const PlaceCritParams& crit_params);
