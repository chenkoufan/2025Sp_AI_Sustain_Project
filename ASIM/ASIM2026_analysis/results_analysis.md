# Comprehensive control-strategy results

## Analysis scope

The analysis contains 225 weather–layout cases: three sky conditions, 15 occupancy counts, and five seat distributions per count. The same seat distributions are compared across all three strategies, so strategy differences are paired by weather and layout. Values are instantaneous lighting power (W), not time-integrated energy. The five layouts are design samples; min–max bands quantify seating-pattern sensitivity and are not confidence intervals.

## Draft Results text

Under clear conditions, scenario-mean power was 75.5 W for zonal PIR ceiling-only control, 16.3 W for rule-based hybrid control, and 17.4 W for optimized hybrid control. Relative to zonal PIR, the optimized strategy reduced mean power by 76.9% (58.1 W). Relative to the rule-based hybrid strategy, it produced a 6.7% increase (1.1 W in magnitude).
Under overcast conditions, scenario-mean power was 244.5 W for zonal PIR ceiling-only control, 120.0 W for rule-based hybrid control, and 93.0 W for optimized hybrid control. Relative to zonal PIR, the optimized strategy reduced mean power by 62.0% (151.5 W). Relative to the rule-based hybrid strategy, it produced a 22.5% reduction (27.0 W in magnitude).
Under night conditions, scenario-mean power was 443.5 W for zonal PIR ceiling-only control, 314.8 W for rule-based hybrid control, and 236.1 W for optimized hybrid control. Relative to zonal PIR, the optimized strategy reduced mean power by 46.8% (207.4 W). Relative to the rule-based hybrid strategy, it produced a 25.0% reduction (78.7 W in magnitude).

### Occupancy and layout effects

Mean optimized power increased overall from one to 15 occupants as follows: Clear: 3.0–31.2 W; Overcast: 38.8–125.1 W; Night: 110.1–300.4 W. The non-monotonic steps and shaded min–max bands in the occupancy plots show that seat location, zone activation, and daylight availability affect power in addition to occupant count.

### Seating-pattern sensitivity

- **Clear:** Zonal PIR ceiling-only: mean within-count range 44.9 W (maximum 79.2 W at 1 occupant); Rule-based hybrid: mean within-count range 12.5 W (maximum 19.5 W at 10 occupants); Optimized hybrid: mean within-count range 14.6 W (maximum 21.0 W at 10 occupants).
- **Overcast:** Zonal PIR ceiling-only: mean within-count range 59.8 W (maximum 171.6 W at 2 occupants); Rule-based hybrid: mean within-count range 38.9 W (maximum 76.5 W at 2 occupants); Optimized hybrid: mean within-count range 24.8 W (maximum 76.5 W at 2 occupants).
- **Night:** Zonal PIR ceiling-only: mean within-count range 60.7 W (maximum 171.6 W at 2 occupants); Rule-based hybrid: mean within-count range 35.2 W (maximum 105.6 W at 2 occupants); Optimized hybrid: mean within-count range 31.4 W (maximum 85.2 W at 3 occupants).

### Power allocation

- **Clear:** optimized control allocated 0.0 W to ceiling lighting and 17.4 W to task lighting (0.0% ceiling). The rule-based hybrid allocation was 0.0 W ceiling and 16.3 W task.
- **Overcast:** optimized control allocated 61.1 W to ceiling lighting and 31.9 W to task lighting (65.7% ceiling). The rule-based hybrid allocation was 86.8 W ceiling and 33.3 W task.
- **Night:** optimized control allocated 194.5 W to ceiling lighting and 41.6 W to task lighting (82.4% ceiling). The rule-based hybrid allocation was 278.8 W ceiling and 36.0 W task.

## Interpretation requiring verification

The optimized solution was lower than the rule-based hybrid in 7/75 Clear cases, equal in 28/75, and higher in 40/75. Consequently, the Clear rule-based hybrid has a lower scenario-average power than the current GA output. Before describing the GA as globally energy-optimal, both strategies should be rechecked against identical illuminance and uniformity constraints. If the rule baseline is feasible under those same constraints, the result points to optimizer convergence, objective weighting, or search-space settings rather than an advantage of optimization under Clear sky.

## Figure captions

**Power by occupancy.** Mean lighting power for zonal PIR ceiling-only, rule-based hybrid, and optimized hybrid control under (a) Clear, (b) Overcast, and (c) Night conditions. Lines show the mean across five seat distributions at each occupancy count; shaded bands show the observed minimum–maximum range (n = 5 layouts per count). Panel y-axis ranges differ to retain visibility across sky conditions.

**Allocation and savings.** (a) Scenario-mean ceiling and task-light power for each strategy and sky condition, averaged equally across 75 occupancy-layout cases per condition. The x-axis identifies the PIR, rule-based hybrid, and optimized (Opt.) strategies; fill and hatching identify lighting components. (b) Percentage reduction in optimized-hybrid mean power relative to the two rule-based baselines. Negative values denote higher, rather than lower, optimized power.

**Savings by occupancy.** Density-resolved optimized-hybrid power reduction relative to (a) zonal PIR ceiling-only and (b) rule-based hybrid control. Each point is calculated from the mean power of five matched seat distributions at that occupancy count; the horizontal line marks no difference.

## Alt text

The occupancy figure contains three line-chart panels. Power generally rises with occupancy and with decreasing daylight. Optimized hybrid remains below zonal PIR in all three conditions, but the Clear rule-based hybrid line often lies below the optimized line. Shaded ranges are widest at low-to-intermediate occupancy where seat placement changes which zones are active.

## Reporting notes

- Report these quantities as **power (W)**. Energy requires an operating duration and should be reported in Wh or kWh.
- Scenario means weight every occupancy count and layout equally; they are comparative test-set summaries, not predictions of annual building energy.
- No inferential significance tests were applied because the five layouts are simulation design samples rather than independent measurements from a target population.
