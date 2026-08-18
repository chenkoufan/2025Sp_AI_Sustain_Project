"""Update the ASIM manuscript with the corrected and deeper results analysis.

The script edits the OOXML package directly so the existing ASIM template,
legacy VML figures, styles, headers, and pagination settings are retained.
"""

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import os
import shutil
import zipfile

from lxml import etree


ROOT = Path(__file__).resolve().parent
DOCX = ROOT / "FullPaperTemplateASIM2026 - KK.docx"
BACKUP = ROOT / "FullPaperTemplateASIM2026 - KK.before_deep_results.docx"
TEMP = ROOT / "FullPaperTemplateASIM2026 - KK.deep_results.tmp.docx"
FIG5 = ROOT / "ASIM2026_figures" / "results_power_by_occupancy.png"
FIG6 = ROOT / "ASIM2026_figures" / "paper_results_conditional_mechanism.png"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = f"{{{W_NS}}}"
NS = {"w": W_NS}
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"


REPLACEMENTS = {
    "Traditional building design is typically based on full-load occupancy": (
        "Traditional building design is typically based on full-load occupancy, although workplace occupancy varies in both density and spatial distribution, causing unnecessary illumination in underused areas. "
        "This study proposes a Balanced Light Control Framework coordinating three ceiling-light zones and 24 workstation task lights subject to visual-comfort requirements. "
        "A Singapore office laboratory was simulated under Clear, Overcast, and Night conditions for 1–15 occupants; five matched seating distributions at each count produced 225 scenarios. "
        "A penalty-based genetic algorithm (GA) was compared with zonal PIR ceiling-only control and occupancy-triggered rule-based hybrid control. "
        "Mean GA-optimized power was 17.44, 93.01, and 236.10 W under Clear, Overcast, and Night, respectively, corresponding to reductions of 76.9%, 62.0%, and 46.8% relative to zonal PIR. "
        "Relative to rule-based hybrid control, GA reduced mean power by 22.5% under Overcast and 25.0% at Night, but increased it by 6.7% under Clear. "
        "GA was lower in 60/75 Overcast and 69/75 Night cases, whereas rule-based hybrid control was lower in 40/75 Clear cases. "
        "At Night, a 5.62 W increase in task-light power was offset by an 84.30 W reduction in ceiling-light power, indicating that localized task lighting can replace larger background-light loads. "
        "Seating layout also produced substantial within-count variation. These results show that hybridization provides the primary power reduction, whereas optimization adds value mainly when daylight is limited and several occupied workstations must be coordinated."
    ),
    "This study asks: how should ceiling and task-light levels be coordinated": (
        "This study asks: how should ceiling- and task-light levels be coordinated under changing daylight and occupancy patterns to minimize lighting power while satisfying visual-comfort constraints? "
        "A Balanced Light Control Framework is evaluated in a simulated open-plan office. The contribution is threefold: (1) a joint representation of three ceiling-light zones and 24 task lights; "
        "(2) a penalty-based optimization formulation incorporating illuminance and uniformity; and (3) matched comparisons of zonal PIR ceiling-only control, occupancy-triggered rule-based hybrid control, and GA-optimized hybrid control across 225 lighting–occupancy–seating scenarios. "
        "The analysis separates the architectural benefit of hybrid lighting from the incremental benefit of optimization and tests how both change with daylight, occupancy density, and occupied-seat location. "
        "The emphasis is a case-specific simulation comparison rather than a deployed real-time controller or proof of global optimality."
    ),
    "The complete workflow is summarized in Figure 1.": (
        "The complete workflow is summarized in Figure 1. Environmental inputs comprised the three-dimensional office model, one of three sky conditions, and a randomly generated occupancy distribution containing 1–15 occupants (Figure 1(a)); this upper bound was selected because occupancy exceeded 15 people during only 0.3% of recorded time in the reference dataset (Tekler et al. 2022). "
        "These inputs were paired with a 27-variable lighting-control vector comprising the discrete output levels of three ceiling-light zones and 24 workstation task lights (Figure 1(b))."
    ),
    "As shown in Table 1, the experiment considered three representative conditions": (
        "As shown in Table 1, the experiment considered three representative conditions labelled Clear, Overcast, and Night. Daylight simulations were referenced to Changi International Airport, Singapore, at 13:00 on 21 March. "
        "The Overcast case used the ClimateStudio identifier ‘Sky: [Changi.Intl.AP,SG,SGP CIE_Overcast 3.21.13.00]’; Clear used the corresponding clear-sky setting, whereas Night excluded daylight. "
        "Occupancy count ranged from 1 to 15. At each count, five unique seating distributions were randomly selected from the 24 workstations. "
        "The same 75 occupancy–seating patterns were evaluated under all three conditions, enabling matched comparisons and yielding 225 optimization scenarios."
    ),
    "For zonal PIR under Night": (
        "For zonal PIR under Night, the two outer seat columns in every four-seat row required ceiling level 13 and the two middle columns level 11. "
        "Under Overcast, seat-level requirements were 0/3:10, 1/2:9, 4–6:8, 7:9, 8/11:8, 9/10:7, 12/15/16/19:5, 13/14/17/18:4, and 20–23:0. "
        "Under Clear, they were 0–3:6, 4–5:1, 6–7:2, 8–10:1, 11:2, and 12–23:0. For rule-based hybrid control, each pair (ceiling, task) was (8,3) at every occupied seat under Night. "
        "Under Overcast the pairs were 0–3:(5,4), 4–7:(4,4), 8–11:(3,3), 12–19:(0,3), and 20–23:(0,0). "
        "Under Clear they were 0–3:(0,4), 4–11:(0,1), 12–15:(0,0), 16–19:(0,2), and 20–23:(0,0). "
        "These fixed levels were established for the stated visual requirements. Baseline power was then calculated with the same 13.2 W ceiling-zone and 1.5 W task-light level coefficients."
    ),
    "Comparative power demand across strategies": "Overall power demand and two-stage benefit",
    "Figure 5 compares the three strategies as occupancy increased.": (
        "Figure 5 compares the three strategies as occupancy increased. Each line is the mean of five matched seat distributions and each band is their observed minimum–maximum range. "
        "Mean power for zonal PIR, rule-based hybrid, and GA-optimized hybrid control was 75.50, 16.34, and 17.44 W under Clear; 244.46, 120.03, and 93.01 W under Overcast; and 443.52, 314.78, and 236.10 W at Night. "
        "Relative to zonal PIR, GA reduced mean power by 76.9%, 62.0%, and 46.8%, respectively. Separating the benefit into two stages, replacing zonal PIR with rule-based hybrid control saved 78.4%, 50.9%, and 29.0%; replacing the rule-based hybrid with GA then increased Clear power by 6.7% but reduced Overcast and Night power by a further 22.5% and 25.0%. "
        "Thus, hybridization supplied most of the Clear-sky benefit, whereas optimization contributed materially under reduced daylight."
    ),
    "Figure 5. Lighting power versus occupancy": (
        "Figure 5. Lighting power versus occupancy for zonal PIR ceiling-only, occupancy-triggered rule-based hybrid, and GA-optimized hybrid control under (a) Clear, (b) Overcast, and (c) Night conditions. "
        "Lines show the mean across five matched seat distributions at each occupancy count; shaded bands show the observed minimum–maximum range (n = 5 layouts). Panel y-axis ranges differ to preserve visibility."
    ),
    "Power allocation and paired strategy outcomes": "Conditional GA advantage and power-allocation mechanism",
    "The daylight-dependent difference was primarily associated with ceiling-light demand": (
        "The advantage of GA over the rule-based hybrid depended jointly on sky condition and occupancy density (Figure 6a,b). Under Clear, mean paired saving was non-positive at every occupancy count; GA was lower in 7/75 cases, equal in 28, and higher in 40. "
        "Under Overcast, GA was lower in 60/75 cases, including 59 of the 60 cases with 4–15 occupants; one case was equal and none was higher in this density range. "
        "At Night, all five one-occupant cases were equal, while 69 of the 70 cases with 2–15 occupants were lower and one was equal. "
        "The incremental optimization benefit therefore emerged consistently when daylight was limited and multiple occupied workstations required coordinated lighting."
    ),
    "Paired case comparisons reinforce the condition dependence.": (
        "Component-level differences identify the source of this benefit (Figure 6c). Relative to rule-based hybrid control, GA changed mean ceiling, task, and total power by 0.00, +1.10, and +1.10 W under Clear; −25.70, −1.32, and −27.02 W under Overcast; and −84.30, +5.62, and −78.68 W at Night. "
        "The Night reduction therefore resulted from a small increase in localized task-light power being outweighed by a much larger decrease in background ceiling-light power."
    ),
    "Figure 6. Strategy-level power allocation and comparative savings.": (
        "Figure 6. Conditional performance and source of the GA–rule-based difference. (a) Mean paired power saving of GA-optimized hybrid control relative to rule-based hybrid control at each occupancy count; positive values indicate lower GA power (five matched layouts per cell). "
        "(b) Case-level outcomes across 75 matched scenarios per sky condition. (c) Mean change in ceiling, task, and total power, calculated as GA minus rule-based hybrid; negative values indicate lower GA power."
    ),
    "Occupancy and seating-pattern sensitivity": "Occupancy-density and seating-layout sensitivity",
    "GA-derived mean power increased overall from one to 15 occupants": (
        "GA-optimized mean power increased overall from one to 15 occupants, from 3.0 to 31.2 W under Clear, 38.8 to 125.1 W under Overcast, and 110.1 to 300.4 W at Night. "
        "The trajectories were not strictly monotonic because different layouts activated different ceiling zones and received different daylight. For GA, the mean within-count range across five layouts was 14.6 W under Clear, 24.8 W under Overcast, and 31.4 W at Night, representing 29.8%, 23.8%, and 8.5% of total power variation, respectively. "
        "The corresponding mean ranges for zonal PIR were 44.9, 59.8, and 60.7 W, and those for rule-based hybrid control were 12.5, 38.9, and 35.2 W. "
        "Occupancy count therefore did not uniquely determine lighting power; occupied-seat location remained an important state variable, particularly when daylight was available."
    ),
    "The comparison shows that the value of coordinated task and background lighting": (
        "The two-stage comparison distinguishes the value of the hybrid lighting architecture from the value of numerical optimization. Moving from zonal PIR ceiling-only control to rule-based hybrid control reduced mean power by 78.4% under Clear, 50.9% under Overcast, and 29.0% at Night. "
        "GA provided no additional saving under Clear, but reduced the rule-based hybrid mean by a further 22.5% under Overcast and 25.0% at Night. "
        "This pattern suggests that local task lighting captures most available savings when daylight already supplies the background level, whereas optimization becomes valuable when several occupied workstations must share a non-zero ceiling-light demand."
    ),
    "The Clear result is an important counterexample": (
        "The Clear result is an important counterexample to the assumption that a genetic algorithm must outperform a fixed rule. Rule-based hybrid control was lower in 40/75 Clear cases, 28 cases were equal, and GA was lower in only seven. "
        "If both strategies satisfy identical illuminance and uniformity constraints, the 1.10 W mean difference indicates that the retained Galapagos solutions should not be interpreted as strict global optima. "
        "Convergence, penalty weighting, discrete search resolution, and repeated-run variability should therefore be examined. The results are accordingly described as GA-optimized feasible solutions rather than a proven global performance bound."
    ),
    "The variation among seating patterns also has a direct sensing implication.": (
        "The results also reveal an interaction among daylight, density, seating layout, and control architecture. GA became consistently advantageous over the rule-based hybrid from four occupants under Overcast and from two occupants at Night, while no occupancy density produced a positive mean saving under Clear. "
        "Within-count variation further shows that a controller based only on total occupancy would assign identical settings to spatially different cases despite different daylight exposure and zone activation. "
        "A practical controller could therefore use sky condition and spatial occupancy as gate variables: apply the simpler rule-based hybrid when daylight is abundant, and invoke optimization when daylight is limited or occupied workstations span several zones. "
        "The idealized baselines used known occupied-seat IDs; real zone-level PIR sensing would provide less information and may require more conservative settings."
    ),
    "Several limitations bound interpretation.": (
        "Several limitations bound interpretation. The study represents one office geometry and three selected lighting conditions rather than annual daylight variation. Occupancy was limited to 15 of 24 workstations, with five designed layouts per count; their minimum–maximum ranges are descriptive and are not statistical confidence intervals. "
        "The comfort model considered simulated illuminance and uniformity but not individual preference, glare, sensor uncertainty, control delay, or hardware dynamics. Scenario means weight every count and layout equally and should not be interpreted as annual energy use. "
        "The three cases in which GA exceeded a zero-power zonal PIR baseline require a direct comfort-feasibility check, and all retained settings should be cross-checked against the original illuminance outputs. Repeated GA runs and implementation-level validation are also needed for a reproducibility audit."
    ),
    "This study evaluated coordinated control of three ceiling-light zones": (
        "This study evaluated coordinated control of three ceiling-light zones and 24 task lights across 225 matched lighting–occupancy–seating scenarios. "
        "Compared with zonal PIR ceiling-only control, GA-optimized hybrid control reduced mean power by 76.9% under Clear, 62.0% under Overcast, and 46.8% at Night. "
        "Compared with occupancy-triggered rule-based hybrid control, GA reduced mean power by a further 22.5% under Overcast and 25.0% at Night, principally by lowering ceiling-light demand, but used 6.7% more power under Clear. "
        "The results therefore separate two contributions: hybrid lighting provides the primary reduction, while optimization adds value mainly under weak daylight and multi-workstation coordination. "
        "Variation among layouts at the same occupancy count further supports spatially resolved sensing rather than count-only control. The framework provides a reproducible case-study platform for defining when optimization is worthwhile, while annual simulations, comfort verification, repeated optimization, and implementation testing remain necessary before operational energy savings are claimed."
    ),
}


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
    text = etree.SubElement(run, f"{W}t")
    text.set(XML_SPACE, "preserve")
    text.text = new_text


def update_document_xml(raw_xml):
    root = etree.fromstring(raw_xml)
    paragraphs = root.xpath("//w:body//w:p", namespaces=NS)
    matched = set()
    for p in paragraphs:
        old = paragraph_text(p)
        if not old:
            continue
        for prefix, replacement in REPLACEMENTS.items():
            if old.startswith(prefix):
                if prefix in matched:
                    raise RuntimeError(f"Duplicate paragraph match for: {prefix}")
                replace_paragraph_text(p, replacement)
                matched.add(prefix)
                break
    missing = set(REPLACEMENTS) - matched
    if missing:
        raise RuntimeError("Missing paragraph prefixes:\n" + "\n".join(sorted(missing)))
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")


def update_core_properties(raw_xml):
    root = etree.fromstring(raw_xml)
    modified = root.find("{http://purl.org/dc/terms/}modified")
    if modified is not None:
        modified.text = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")


def build():
    for path in (DOCX, FIG5, FIG6):
        if not path.exists():
            raise FileNotFoundError(path)
    if not BACKUP.exists():
        shutil.copy2(DOCX, BACKUP)

    with zipfile.ZipFile(DOCX, "r") as zin:
        infos = zin.infolist()
        payload = {info.filename: zin.read(info.filename) for info in infos}

    payload["word/document.xml"] = update_document_xml(payload["word/document.xml"])
    payload["word/media/image8.png"] = FIG5.read_bytes()
    payload["word/media/image9.png"] = FIG6.read_bytes()
    if "docProps/core.xml" in payload:
        payload["docProps/core.xml"] = update_core_properties(payload["docProps/core.xml"])

    if TEMP.exists():
        TEMP.unlink()
    with zipfile.ZipFile(TEMP, "w") as zout:
        for info in infos:
            zout.writestr(info, payload[info.filename])

    with zipfile.ZipFile(TEMP, "r") as check:
        bad = check.testzip()
        if bad:
            raise RuntimeError(f"ZIP integrity failure: {bad}")

    os.replace(TEMP, DOCX)
    print(f"updated={DOCX}")
    print(f"backup={BACKUP}")
    print(f"paragraphs_replaced={len(REPLACEMENTS)}")


if __name__ == "__main__":
    build()
