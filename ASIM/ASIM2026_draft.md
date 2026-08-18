# BALANCING BACKGROUND AND TASK LIGHTING UNDER DYNAMIC OCCUPANCY: A SIMULATION-BASED OPTIMIZATION STUDY

> Working draft for ASIM 2026. Numerical claims marked **[verified from dataset]** were checked against the three local workbooks on 10 August 2026. Items marked **[TO VERIFY]** must be confirmed before submission.

## ABSTRACT

Traditional lighting design is commonly based on full-load occupancy, whereas actual workplace occupancy varies in both density and spatial distribution. This mismatch can lead to unnecessary illumination and energy use in underutilized spaces. Background lighting provides ambient illumination at the room scale, while task lighting addresses localized needs at individual workstations. Coordinating these two lighting layers may therefore improve the response to changing occupancy conditions. This study proposes a Balanced Light Control Framework to investigate the minimum lighting power achievable while maintaining prescribed visual-comfort requirements.

A three-dimensional model of an office laboratory in Singapore was evaluated under three lighting conditions and 15 occupancy levels, ranging from one to 15 occupants. Five random seating distributions were generated at each occupancy level and applied consistently across the three lighting conditions, producing 225 optimization scenarios. **[verified from dataset]** For each scenario, the output levels of three ceiling-light zones and 24 individual task lights were treated as decision variables. A penalty-based objective function combined total lighting power with penalties for unmet illuminance and uniformity requirements defined by SS 531. **[TO VERIFY: exact clauses, thresholds, and standard edition]** A genetic algorithm implemented through the Galapagos component was used to identify a minimum-power lighting configuration for each scenario. **[TO VERIFY: algorithm settings and stopping criteria]**

The optimized configurations required an average power of 17.62 W under clear conditions, 93.01 W under overcast conditions, and 236.10 W at night. **[verified from dataset]** Ceiling lighting contributed approximately 1.0%, 65.7%, and 82.4% of total optimized power under the three conditions, respectively. **[verified from dataset]** These results show that the energy-minimizing balance between background and task lighting changes substantially with daylight availability. They also indicate that spatial seating distribution, rather than occupant count alone, can materially affect the required lighting power. **[descriptive interpretation; quantify in Results]** Comparisons with conventional control baselines are required to quantify the framework's relative saving potential. **[TO ADD]**

## KEYWORDS

Occupancy-centric lighting, task lighting, lighting optimization, visual comfort, daylight availability

## 1. INTRODUCTION

### 1.1 Background and practical problem

**[DRAFT NEXT]** Explain the mismatch between design occupancy and dynamically occupied offices, and why density plus spatial distribution matter for lighting demand.

### 1.2 Background lighting and personalized task lighting

**[DRAFT NEXT]** Introduce the different spatial functions of centralized ceiling lighting and workstation-level task lights.

### 1.3 Research gap

**[DRAFT NEXT]** Establish that prior work commonly treats occupancy-based ceiling-light control or personalized lighting separately, leaving their coordinated minimum-power operation insufficiently characterized.

### 1.4 Objective and contributions

This study investigates how background and task lighting can be jointly configured under different daylight and occupancy conditions to minimize lighting power while satisfying visual-comfort constraints. Its intended contributions are:

1. a joint control representation for three ceiling-light zones and 24 individual task lights;
2. a penalty-based optimization formulation incorporating illuminance and uniformity requirements;
3. a controlled comparison across 225 matched lighting–occupancy–seating scenarios; and
4. a case-specific estimate of the optimized division of power between background and task lighting.

## 2. METHODOLOGY

### 2.1 Case-study office and simulation model

**[TO EXTRACT AND VERIFY]** Office geometry, location, orientation, window properties, material reflectance, workplane definition, lighting layout, luminaire specifications, and simulation engine/settings.

**Proposed Figure 1.** Case-study office, workstation numbering, three ceiling-light zones, and task-light locations.

### 2.2 Experimental design

The experiment considered three lighting conditions: Clear, Overcast, and Night. **[TO VERIFY: whether these labels represent sky models, timestamps, or measured/simulated daylight states]** Occupancy count ranged from one to 15. For each count, five random seating distributions were generated from the 24 available workstations. The same 75 occupancy–seating scenarios were evaluated under all three lighting conditions, resulting in 225 optimized cases. **[verified from dataset]**

**Proposed Table 1.** Experimental factors, levels, and number of scenarios.

### 2.3 Decision variables and lighting power

The decision vector comprises the discrete output levels of three ceiling-light zones and 24 task lights. Based on the exported results, total lighting power is calculated as

\[
P = 13.2\sum_{j=1}^{3}L_{c,j} + 1.5\sum_{i=1}^{24}L_{t,i},
\]

where \(L_{c,j}\) and \(L_{t,i}\) denote the discrete ceiling- and task-light levels, respectively. **[verified against all 225 dataset rows; TO VERIFY physical basis and units per level]**

### 2.4 Visual-comfort constraints and objective function

**[TO ADD]** Define occupied workplane illuminance, surrounding-area illuminance, uniformity, thresholds, penalty weights, and the complete objective function. Distinguish hard requirements from soft penalty terms.

### 2.5 Genetic-algorithm optimization

**[TO ADD]** Population size, initialization, crossover/mutation settings, number of generations, convergence or stopping criterion, repeat runs, treatment of infeasible solutions, and software versions.

### 2.6 Comparison strategies

**[TO ADD]** Define at least:

- ceiling-light-only baseline;
- rule-based combined ceiling/task-light control;
- optional occupancy-sensor/PIR-style zonal baseline if it can be implemented consistently.

The baseline assumptions must be fixed before computing percentage savings.

## 3. RESULTS

### 3.1 Optimized power across lighting conditions

Across all 75 occupancy–seating cases in each lighting condition, mean optimized power was 17.62 W for Clear, 93.01 W for Overcast, and 236.10 W for Night. The corresponding ranges were 0–46.2 W, 0–135.0 W, and 110.1–303.9 W. **[verified from dataset]**

**Proposed Figure 2.** Optimized power versus occupancy count, showing the five seating distributions and the mean for each lighting condition.

### 3.2 Allocation between ceiling and task lighting

Ceiling lighting accounted for approximately 1.0% of optimized power under Clear conditions, 65.7% under Overcast conditions, and 82.4% at Night. All ceiling zones were switched off in 74 of 75 Clear cases, five of 75 Overcast cases, and none of the Night cases. **[verified from dataset]**

**Proposed Figure 3.** Mean ceiling- and task-light power by occupancy count and lighting condition.

### 3.3 Effect of seating distribution

The five seating distributions at a given occupancy level sometimes produced substantially different optimized power requirements. For example, at an occupancy count of three under Overcast conditions, optimized power ranged from 9 to 84 W. **[verified from dataset]** This variation motivates treating spatial occupancy distribution as an input rather than relying only on occupant count.

**[TO ADD]** A systematic variability metric across all occupancy levels, such as within-count range, standard deviation, or coefficient of variation.

### 3.4 Comparison with baseline strategies

**[TO ADD AFTER BASELINE CALCULATION]** Report absolute and percentage differences using matched scenarios. Avoid energy terminology unless a scenario duration is defined; otherwise report power.

## 4. DISCUSSION

### 4.1 Main findings

**[DRAFT AFTER RESULTS FIGURES]** Interpret the daylight-dependent transition from task-light-dominant operation to ceiling-light-dominant operation.

### 4.2 Implications for occupancy-centric lighting control

**[DRAFT AFTER BASELINE DEFINITION]** Explain why sensing spatial distribution may offer information beyond total occupant count and how the optimization results can serve as a case-specific performance benchmark.

### 4.3 Limitations

- simulation-based case study in one office geometry;
- three selected lighting/daylight conditions rather than a full annual simulation;
- occupancy range limited to 15 of 24 workstations;
- five random distributions per occupancy count;
- visual-comfort criteria represented through modelled illuminance and uniformity;
- optimized configurations are theoretical results and do not include sensor error, occupant preference, control delay, commissioning constraints, or hardware limitations;
- generalizability of penalty weights and genetic-algorithm solutions remains to be evaluated.

## 5. CONCLUSION AND IMPLICATIONS

**[DRAFT LAST]** State only conclusions supported by the completed baseline comparison and sensitivity checks.

## ACKNOWLEDGEMENTS

**[TO CONFIRM]**

## REFERENCES

**[TO BUILD FROM VERIFIED SOURCES IN ASIM AUTHOR–DATE STYLE]**

## OPEN QUESTIONS / EVIDENCE NEEDED

1. Exact definition of Clear, Overcast, and Night, including date/time and sky model.
2. SS 531 edition, illuminance and uniformity thresholds, and exact calculation method.
3. Complete objective function and penalty weights.
4. Galapagos settings and whether each scenario was optimized once or repeatedly.
5. Physical meaning of each task- and ceiling-light level.
6. Definition and raw results for the comparison baselines.
7. Whether `power` is instantaneous electrical power in watts.
8. Confirmed title, authors, affiliations, funding, acknowledgements, and disclosure statements.
