"""Apply a full-manuscript terminology, logic, and concision pass."""

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import os
import shutil
import zipfile

from lxml import etree


ROOT = Path(__file__).resolve().parent
DOCX = ROOT / "FullPaperTemplateASIM2026 - KK.docx"
BACKUP = ROOT / "FullPaperTemplateASIM2026 - KK.before_coherence_polish.docx"
TEMP = ROOT / "FullPaperTemplateASIM2026 - KK.coherence.tmp.docx"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W14_NS = "http://schemas.microsoft.com/office/word/2010/wordml"
W = f"{{{W_NS}}}"
NS = {"w": W_NS}
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"


REPLACEMENTS = {
    "BALANCING BACKGROUND AND TASK LIGHTING": (
        "COORDINATING CEILING AND TASK LIGHTING UNDER VARIABLE OCCUPANCY AND DAYLIGHT: A SIMULATION-BASED OPTIMIZATION STUDY"
    ),
    "Traditional building design is typically based on full-load occupancy": (
        "Office lighting controls often respond to aggregate occupancy, although lighting demand also depends on daylight and where occupants sit. "
        "We evaluated a coordinated lighting control framework that assigns discrete output levels to three ceiling-light zones and 24 workstation task lights under visual-comfort constraints. "
        "A Singapore office laboratory was simulated under Clear, Overcast, and Night conditions for 1–15 occupants. Five matched seating layouts at each occupancy count yielded 225 scenarios. "
        "We compared zonal passive infrared (PIR) ceiling-only control, rule-based hybrid control, and genetic-algorithm (GA)-optimized hybrid control. "
        "Relative to zonal PIR, GA-optimized control reduced mean lighting power by 76.9%, 62.0%, and 46.8% under Clear, Overcast, and Night, respectively. "
        "Relative to rule-based hybrid control, it reduced mean power by 22.5% under Overcast and 25.0% at Night, but increased it by 6.7% under Clear. "
        "GA used less power in 60/75 Overcast and 69/75 Night cases, but only 7/75 Clear cases. "
        "At Night, task-light power increased by 5.62 W while ceiling-light power decreased by 84.30 W, yielding a net reduction of 78.68 W. "
        "Seating layout also produced substantial variation at fixed occupancy counts. Hybrid lighting therefore delivered the primary power reduction, while GA optimization added value mainly under limited daylight and multi-workstation occupancy."
    ),
    "Occupancy-centric lighting optimization": (
        "Occupancy-responsive lighting control, Hybrid lighting, Genetic algorithm, Daylight-responsive control, Visual comfort"
    ),
    "Lighting systems in offices are often designed and commissioned": (
        "Office lighting is commonly designed for full occupancy, while actual use varies across time and workstations. "
        "Schedule-based or count-based controls can therefore illuminate unoccupied areas or workstations already served by daylight. "
        "Room-level data confirm substantial occupancy variability (Tekler et al. 2022). Effective control must respond not only to how many people are present, but also to where they sit."
    ),
    "Centralized ceiling lighting and personalized task lighting": (
        "Ceiling lighting and task lighting operate at different spatial scales. Ceiling luminaires support general illumination and uniformity, whereas task lights deliver light at individual workplanes and allow local adjustment (De Korte et al. 2015). "
        "High-resolution daylight and occupancy sensing can restrict electric lighting to occupied areas (Lowcay et al. 2020). Task-light studies likewise show the value of localized control (Papinutto et al. 2021). "
        "Yet these systems are usually studied separately. The unresolved question is how to divide lighting power between ceiling and task layers as daylight, occupancy count, and seating layout change."
    ),
    "This study asks: how should ceiling- and task-light levels be coordinated": (
        "This study asks how ceiling- and task-light levels should be coordinated under variable daylight and occupancy while maintaining visual-comfort requirements. "
        "We evaluated a coordinated lighting control framework in a simulated open-plan office. The framework represents three ceiling-light zones and 24 task lights. "
        "A penalty-based genetic algorithm (GA) enforces illuminance and uniformity constraints while minimizing lighting power. "
        "Three strategies were compared across 225 matched sky–occupancy–layout scenarios: zonal PIR ceiling-only control, rule-based hybrid control, and GA-optimized hybrid control. "
        "This design separates the benefit of hybrid lighting from the additional benefit of optimization. It also identifies how both benefits vary with sky condition, occupancy count, and seating layout. "
        "The study is a matched simulation analysis rather than a real-time deployment."
    ),
    "METHODOLOGY": "METHODS",
    "The complete workflow is summarized in Figure 1.": (
        "Figure 1 summarizes the optimization workflow. Each scenario combined the three-dimensional office model, one sky condition, and a seating layout with 1–15 occupants (Figure 1a). "
        "The upper limit was selected because occupancy exceeded 15 people during only 0.3% of recorded time in the reference dataset (Tekler et al. 2022). "
        "Each scenario was paired with a 27-variable control vector. The vector specified discrete output levels for three ceiling-light zones and 24 task lights (Figure 1b)."
    ),
    "For each candidate vector, ClimateStudio calculated": (
        "For each candidate vector, ClimateStudio calculated illuminance and uniformity for occupied task areas and their surroundings. The same vector was converted to total lighting power (Figure 1c). "
        "These outputs defined the visual-comfort constraints and optimization objective. The GA updated the control vector until it retained the lowest-fitness feasible solution (Figure 1d,e). "
        "Each retained solution was then compared with zonal PIR ceiling-only control and rule-based hybrid control for the same sky condition, occupancy count, and seating layout (Figure 1f)."
    ),
    "A three-dimensional model was constructed for a single open-plan office laboratory": (
        "The case study represented one open-plan office laboratory within a larger Singapore building. Figure 2 locates the simulated room within the host model; only this room was evaluated. "
        "The room contains 24 fixed workstations, each with an independently adjustable task light. Ceiling lighting is divided into three zones serving seats 0–7, 8–15, and 16–23. "
        "ClimateStudio calculated illuminance and uniformity for occupied task areas and their surroundings."
    ),
    "Figure 1. Workflow of the simulation-based lighting-control study": "Figure 1. Workflow of the coordinated lighting-control study",
    "Figure 3. Room-scale simulation model": "Figure 3. Room-scale simulation model, seating layout, ceiling-light zones, and task-light arrangement",
    "As shown in Table 1, the experiment considered three representative conditions": (
        "Table 1 summarizes the scenario matrix. Simulations used three sky conditions labelled Clear, Overcast, and Night. "
        "Daylight conditions were referenced to Changi International Airport, Singapore, at 13:00 on 21 March. The Overcast case used ‘Sky: [Changi.Intl.AP,SG,SGP CIE_Overcast 3.21.13.00]’. "
        "Clear used the corresponding clear-sky setting, while Night excluded daylight. Occupancy count ranged from 1 to 15. "
        "Five unique seating layouts were randomly selected at each occupancy count. The same 75 occupancy–layout cases were evaluated under all three sky conditions, yielding 225 matched scenarios."
    ),
    "The control vector comprises discrete output levels": (
        "The control vector specified discrete output levels for three ceiling-light zones and 24 task lights. Task lights at unoccupied workstations remained off. "
        "Each task-light level added 1.5 W, and each ceiling-zone level added 13.2 W. Total lighting power was calculated as:"
    ),
    "This expression reproduced the reported power": (
        "This expression reproduced the reported lighting power for all 225 retained solutions. Because the scenarios had no assigned duration, results are reported as lighting power (W), not energy (kWh)."
    ),
    "Two deterministic controls were evaluated using the same 225 occupied-seat lists": (
        "Two deterministic baselines used the same 225 occupied-seat lists. Under zonal PIR ceiling-only control, task lights remained off and empty ceiling-light zones were switched off. "
        "For each occupied zone, the ceiling-light level equalled the highest seat-specific requirement among its occupants. "
        "Rule-based hybrid control applied the same zone-level maximum to ceiling lighting and assigned a prescribed task-light level to each occupied workstation. Unoccupied task lights remained off. "
        "Both baselines used exact seat locations; rule-based hybrid control therefore assumed workstation-level occupancy triggering."
    ),
    "For zonal PIR under Night": (
        "Under Night, zonal PIR assigned ceiling level 13 to the two outer seat columns and level 11 to the two inner columns. "
        "Under Overcast, seat requirements were 0/3:10, 1/2:9, 4–6:8, 7:9, 8/11:8, 9/10:7, 12/15/16/19:5, 13/14/17/18:4, and 20–23:0. "
        "Under Clear, they were 0–3:6, 4–5:1, 6–7:2, 8–10:1, 11:2, and 12–23:0. "
        "Rule-based hybrid control used the ceiling/task pair (8,3) for every occupied seat under Night. Under Overcast, the pairs were 0–3:(5,4), 4–7:(4,4), 8–11:(3,3), 12–19:(0,3), and 20–23:(0,0). "
        "Under Clear, they were 0–3:(0,4), 4–11:(0,1), 12–15:(0,0), 16–19:(0,2), and 20–23:(0,0). "
        "These levels met the stated visual requirements. Baseline power used the same 13.2 W ceiling-zone and 1.5 W task-light coefficients."
    ),
    "Visual-comfort requirements were based on SS 531:2006": (
        "Visual-comfort constraints followed SS 531:2006. Occupied task areas required at least 500 lux and uniformity Emin/Eavg ≥ 0.70. "
        "Surrounding areas required at least 300 lux and uniformity ≥ 0.50. Normalized violations were added to lighting power through a quadratic penalty function:"
    ),
    "Here, v represents the normalized violation": (
        "Here, v is the normalized violation of an illuminance or uniformity threshold, and λ is its penalty coefficient. Penalties prioritized visual feasibility before lighting power. "
        "[AUTHOR INPUT REQUIRED: insert penalty coefficients and the sensitivity test used to reject infeasible low-power solutions.]"
    ),
    "Figure 4 documents a representative seven-occupant case": (
        "Figure 4 presents a representative seven-occupant case. It links the Galapagos optimization interface to the corresponding seating layout and simulated illuminance field."
    ),
    "Figure 4. Representative seven-occupant optimization": (
        "Figure 4. Representative seven-occupant optimization: (a) Galapagos interface; (b) corresponding seating layout and simulated illuminance field"
    ),
    "Galapagos in Grasshopper was operated in minimization mode": (
        "Galapagos in Grasshopper operated in minimization mode. For each scenario, it evolved combinations of the 27 discrete control variables. "
        "ClimateStudio and associated scripts evaluated illuminance, uniformity, and lighting power for each candidate. The retained solution was the lowest-fitness configuration satisfying the encoded constraints. "
        "[AUTHOR INPUT REQUIRED: insert population size, generation limit, convergence criterion, random seed, repeated-run count, and software versions.]"
    ),
    "Overall power demand and two-stage benefit": "Overall lighting power and control-stage effects",
    "Figure 5 compares the three strategies as occupancy increased": (
        "Figure 5 shows how lighting power changed with occupancy count. Lines show the mean of five matched seating layouts; bands show their minimum–maximum range. "
        "Under Clear, mean power for zonal PIR, rule-based hybrid, and GA-optimized hybrid control was 75.50, 16.34, and 17.44 W. The corresponding means were 244.46, 120.03, and 93.01 W under Overcast. "
        "At Night, they were 443.52, 314.78, and 236.10 W. Relative to zonal PIR, GA-optimized control reduced mean power by 76.9%, 62.0%, and 46.8%, respectively. "
        "Rule-based hybrid control accounted for reductions of 78.4%, 50.9%, and 29.0%. Relative to that stronger baseline, GA increased Clear power by 6.7% but reduced Overcast and Night power by 22.5% and 25.0%. "
        "Hybrid lighting therefore supplied the main Clear-condition reduction, while GA added power savings under reduced daylight."
    ),
    "Figure 5. Lighting power versus occupancy": (
        "Figure 5. Lighting power versus occupancy count for zonal PIR ceiling-only control, rule-based hybrid control, and GA-optimized hybrid control under (a) Clear, (b) Overcast, and (c) Night. "
        "Lines show the mean across five matched seating layouts; shaded bands show the observed minimum–maximum range (n = 5 layouts). Panel y-axis ranges differ."
    ),
    "Conditional GA advantage and power-allocation mechanism": "Conditional GA performance and power-allocation mechanism",
    "Paired comparisons clarified the condition dependence of optimization": (
        "Paired comparisons sharpened the mean results (Figure 6). GA used less power than zonal PIR in 217/225 scenarios, including all 75 Night cases. "
        "Against rule-based hybrid control, the pattern changed. Under Clear, GA was lower in 7/75 cases, equal in 28, and higher in 40. "
        "Under Overcast, GA was lower in 60/75 cases, including 59/60 cases with 4–15 occupants. The remaining case in this range was equal. "
        "At Night, all five one-occupant cases were equal. For 2–15 occupants, GA was lower in 69/70 cases and equal in one. "
        "The additional GA reduction was therefore consistent only when daylight was limited and several workstations were occupied."
    ),
    "Component-level differences identified the source": (
        "Component-level differences showed how GA achieved the additional reduction (Figure 7a). Under Clear, GA changed ceiling, task, and total power by 0.00, +1.10, and +1.10 W relative to rule-based hybrid control. "
        "Under Overcast, the corresponding changes were −25.70, −1.32, and −27.02 W. At Night, they were −84.30, +5.62, and −78.68 W. "
        "The Night result combined a small increase in localized task lighting with a much larger reduction in ceiling lighting."
    ),
    "Figure 6. Paired strategy evidence across sky condition and occupancy density": (
        "Figure 6. Paired strategy evidence across sky condition and occupancy count. (a,b) Mean paired power saving of GA-optimized hybrid control relative to zonal PIR ceiling-only control and rule-based hybrid control, respectively. "
        "Each cell averages five matched seating layouts; positive values indicate lower GA power. (c) Scenario-level outcomes and mean paired differences across 75 matched cases per comparison."
    ),
    "Figure 7. Source of incremental GA performance": (
        "Figure 7. Source of incremental GA performance and sensitivity to seating layout. (a) Mean component change calculated as GA minus rule-based hybrid control; negative values indicate lower GA power. "
        "(b) Mean within-count minimum–maximum power range across five seating layouts. Percentages show the within-count share of total variation; ranges are descriptive, not confidence intervals."
    ),
    "Occupancy-density and seating-layout sensitivity": "Occupancy-count and seating-layout sensitivity",
    "GA-optimized mean power increased overall from one to 15 occupants": (
        "GA-optimized mean power generally increased with occupancy count. From one to 15 occupants, it rose from 3.0 to 31.2 W under Clear, 38.8 to 125.1 W under Overcast, and 110.1 to 300.4 W at Night. "
        "The trajectories were not monotonic because seating layouts activated different ceiling-light zones and received different daylight. "
        "For GA, the mean within-count range was 14.6 W under Clear, 24.8 W under Overcast, and 31.4 W at Night (Figure 7b). These ranges represented 29.8%, 23.8%, and 8.5% of total variation. "
        "The corresponding ranges were 44.9, 59.8, and 60.7 W for zonal PIR, and 12.5, 38.9, and 35.2 W for rule-based hybrid control. "
        "Occupancy count alone therefore did not determine lighting power."
    ),
    "The two-stage comparison distinguishes the value": (
        "The results separate two control decisions that are often conflated: combining ceiling and task lighting, and optimizing their coordination. "
        "Hybrid lighting delivered most of the reduction when daylight already supplied the background level. GA became valuable as ceiling-light demand increased and several occupied workstations had to share it. "
        "This distinction explains why the same optimization method performed differently across sky conditions."
    ),
    "The Clear result is an important counterexample": (
        "Clear defined the practical limit of GA optimization in this study. Both hybrid strategies kept ceiling lighting off, leaving little power to redistribute. "
        "Rule-based hybrid control was lower in 40/75 Clear cases, while 28 cases were equal. The 1.10 W mean difference also shows that the retained Galapagos solutions are not a global performance bound. "
        "Claims of optimality should therefore wait for convergence checks and repeated runs."
    ),
    "The results also reveal an interaction among daylight": (
        "Sky condition, occupancy count, seating layout, and control architecture formed a clear operating hierarchy. Sky condition set the optimization margin, while occupancy count and seating layout determined the coordination problem. "
        "GA became consistently advantageous from four occupants under Overcast and from two occupants at Night. No occupancy count produced a positive mean GA saving under Clear. "
        "A practical controller could therefore use sky condition and seating layout as gate variables. Rule-based hybrid control would serve daylight-rich conditions, while GA would serve reduced daylight or multi-zone occupancy. "
        "This approach requires spatial occupancy sensing because zone-level PIR provides less information than the seat-level inputs used here."
    ),
    "Several limitations bound interpretation": (
        "The evidence covers one office geometry, three sky conditions, and five designed seating layouts per occupancy count. It reports lighting power rather than annual energy and excludes glare, personal preference, sensor error, control latency, and hardware dynamics. "
        "Before deployment, the retained solutions require direct comfort verification, repeated GA runs, annual simulation, and hardware testing."
    ),
    "This study established a matched simulation framework": (
        "Across 225 matched scenarios, coordinated hybrid lighting substantially reduced lighting power relative to zonal PIR ceiling-only control. "
        "GA-optimized hybrid control reduced mean power by 76.9% under Clear, 62.0% under Overcast, and 46.8% at Night. Yet rule-based hybrid control already produced most of this reduction. "
        "Relative to the rule-based hybrid, GA reduced mean power by 22.5% under Overcast and 25.0% at Night, but increased it by 6.7% under Clear."
    ),
    "The most important finding is therefore conditional rather than universal": (
        "The value of optimization followed a four-factor hierarchy. Sky condition defined the available optimization margin, occupancy count defined the coordination load, seating layout defined spatial demand, and control architecture determined how that demand could be served. "
        "GA became consistently advantageous from four occupants under Overcast and from two occupants at Night. At Night, 5.62 W of additional task lighting replaced 84.30 W of ceiling lighting, reducing total power by 78.68 W. "
        "Variation among seating layouts further showed that occupancy count alone is an incomplete control input."
    ),
    "These findings support a conditional control strategy": (
        "These findings support a conditional controller rather than one strategy for all conditions. Rule-based hybrid control is sufficient when daylight is abundant; GA is most useful under limited daylight or spatially distributed occupancy. "
        "The conclusion applies to simulated lighting power in the studied office. Annual energy, comfort, convergence, sensing, and hardware performance still require validation. "
        "Within this scope, the framework identifies when the computational cost of optimization is justified."
    ),
}

TABLE_REPLACEMENTS = {
    "Lighting condition": "Sky condition",
    "Seating distribution": "Seating layout",
    "Five patterns per count": "Five layouts per count",
    "Matched factorial matrix": "Matched scenario matrix",
}

LOWCAY_REFERENCE = (
    "Lowcay D, Gunay HB, and O’Brien W. 2020. Simulating energy savings potential with high-resolution daylight and occupancy sensing in open-plan offices. "
    "Journal of Building Performance Simulation 13(5): 606–619. https://doi.org/10.1080/19401493.2020.1807604."
)


def paragraph_text(p):
    return "".join(t.text or "" for t in p.xpath(".//w:t", namespaces=NS))


def replace_paragraph_text(p, new_text):
    p_pr = p.find(f"{W}pPr")
    first_run = p.find(f"{W}r")
    first_r_pr = first_run.find(f"{W}rPr") if first_run is not None else None
    for child in list(p):
        if child is not p_pr:
            p.remove(child)
    run = etree.SubElement(p, f"{W}r")
    if first_r_pr is not None:
        run.append(deepcopy(first_r_pr))
    t = etree.SubElement(run, f"{W}t")
    t.set(XML_SPACE, "preserve")
    t.text = new_text


def update_document_xml(raw_xml):
    root = etree.fromstring(raw_xml)
    paragraphs = root.xpath("//w:body//w:p", namespaces=NS)
    matched = set()

    for p in paragraphs:
        old = paragraph_text(p)
        if not old:
            continue
        for prefix, new_text in REPLACEMENTS.items():
            if old.startswith(prefix):
                if prefix in matched:
                    raise RuntimeError(f"Duplicate paragraph match: {prefix}")
                replace_paragraph_text(p, new_text)
                matched.add(prefix)
                break
        else:
            if old in TABLE_REPLACEMENTS:
                replace_paragraph_text(p, TABLE_REPLACEMENTS[old])

    missing = set(REPLACEMENTS) - matched
    if missing:
        raise RuntimeError("Missing paragraph prefixes:\n" + "\n".join(sorted(missing)))

    # Add the missing reference after De Korte et al. and before Papinutto et al.
    refs = [p for p in paragraphs if paragraph_text(p).startswith("De Korte EM")]
    if len(refs) != 1:
        raise RuntimeError("Could not locate the De Korte reference")
    lowcay = deepcopy(refs[0])
    lowcay.set(f"{{{W14_NS}}}paraId", "61F20001")
    lowcay.set(f"{{{W14_NS}}}textId", "61F20011")
    replace_paragraph_text(lowcay, LOWCAY_REFERENCE)
    refs[0].addnext(lowcay)

    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")


def update_core_properties(raw_xml):
    root = etree.fromstring(raw_xml)
    modified = root.find("{http://purl.org/dc/terms/}modified")
    if modified is not None:
        modified.text = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")


def build():
    if not DOCX.exists():
        raise FileNotFoundError(DOCX)
    if not BACKUP.exists():
        shutil.copy2(DOCX, BACKUP)

    with zipfile.ZipFile(DOCX, "r") as zin:
        infos = zin.infolist()
        payload = {info.filename: zin.read(info.filename) for info in infos}

    payload["word/document.xml"] = update_document_xml(payload["word/document.xml"])
    if "docProps/core.xml" in payload:
        payload["docProps/core.xml"] = update_core_properties(payload["docProps/core.xml"])

    if TEMP.exists():
        TEMP.unlink()
    with zipfile.ZipFile(TEMP, "w") as zout:
        for info in infos:
            zout.writestr(info, payload[info.filename])
    with zipfile.ZipFile(TEMP, "r") as check:
        if check.testzip():
            raise RuntimeError("ZIP integrity check failed")
    os.replace(TEMP, DOCX)
    print(f"updated={DOCX}")
    print(f"backup={BACKUP}")
    print(f"paragraphs_revised={len(REPLACEMENTS)}")


if __name__ == "__main__":
    build()
