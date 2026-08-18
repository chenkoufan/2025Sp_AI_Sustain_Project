# Deep analysis of matched lighting-control results

## Design boundary

All comparisons are paired by sky condition, occupancy count, and seating layout. There are 75 deterministic simulation cases per sky condition (15 occupancy counts × five layouts). The five layouts are design samples, not independent observations from a population; therefore the analysis reports descriptive effect sizes and full case counts, not inferential p values or confidence intervals. Power is reported in W, not energy.

## Main findings

1. No control strategy is uniformly best across daylight conditions and comparator choice.
2. GA-derived control strongly reduces power relative to ceiling-only zonal PIR in almost all cases, but its advantage over the rule-based hybrid is conditional on reduced daylight.
3. Under Night, the GA trades a small increase in task-light power for a much larger decrease in ceiling-light power.
4. Seating layout explains a material fraction of power variation under Clear and Overcast, so occupancy count alone is an incomplete control input.

## Paired performance

- **Clear:** mean paired GA saving was 58.06 W versus zonal PIR (median 68.40 W) and -1.10 W versus the rule-based hybrid (median -1.50 W).
- **Overcast:** mean paired GA saving was 151.45 W versus zonal PIR (median 176.10 W) and 27.02 W versus the rule-based hybrid (median 36.60 W).
- **Night:** mean paired GA saving was 207.42 W versus zonal PIR (median 232.20 W) and 78.68 W versus the rule-based hybrid (median 95.10 W).

## Mechanism

- **Clear:** GA minus rule hybrid = +0.00 W ceiling, +1.10 W task, and +1.10 W total.
- **Overcast:** GA minus rule hybrid = -25.70 W ceiling, -1.32 W task, and -27.02 W total.
- **Night:** GA minus rule hybrid = -84.30 W ceiling, +5.62 W task, and -78.68 W total.

## Density bands

### Relative to Zonal PIR ceiling-only
- Clear: Low (1–5): +45.1 W (+84.6%); Medium (6–10): +59.6 W (+77.8%); High (11–15): +69.5 W (+71.9%)
- Overcast: Low (1–5): +98.3 W (+60.6%); Medium (6–10): +179.1 W (+64.9%); High (11–15): +177.0 W (+60.0%)
- Night: Low (1–5): +147.0 W (+45.9%); Medium (6–10): +248.0 W (+50.0%); High (11–15): +227.2 W (+44.2%)
### Relative to Rule-based hybrid
- Clear: Low (1–5): -0.8 W (-11.4%); Medium (6–10): -1.5 W (-9.7%); High (11–15): -1.0 W (-3.7%)
- Overcast: Low (1–5): +10.5 W (+14.1%); Medium (6–10): +31.9 W (+24.7%); High (11–15): +38.7 W (+24.7%)
- Night: Low (1–5): +47.0 W (+21.3%); Medium (6–10): +100.2 W (+28.8%); High (11–15): +88.8 W (+23.7%)

## Spatial sensitivity

- **Clear:** Zonal PIR ceiling-only 44.9 W / 48.8%; Rule-based hybrid 12.5 W / 26.0%; Optimized hybrid 14.6 W / 29.8% (mean within-density range / share of total variation).
- **Overcast:** Zonal PIR ceiling-only 59.8 W / 14.4%; Rule-based hybrid 38.9 W / 15.9%; Optimized hybrid 24.8 W / 23.8% (mean within-density range / share of total variation).
- **Night:** Zonal PIR ceiling-only 60.7 W / 10.9%; Rule-based hybrid 35.2 W / 9.8%; Optimized hybrid 31.4 W / 8.5% (mean within-density range / share of total variation).

## Interpretation boundary

The workbooks do not contain the final illuminance and uniformity outputs for each retained solution. Consequently, these tables establish comparative power and allocation behavior but do not independently verify that every baseline and optimized case satisfies identical visual-comfort constraints. The isolated cases in which GA exceeds zonal PIR, and the Clear near-parity with the rule hybrid, should be checked against the original comfort outputs and repeated optimizer runs before global-optimality claims are made.
