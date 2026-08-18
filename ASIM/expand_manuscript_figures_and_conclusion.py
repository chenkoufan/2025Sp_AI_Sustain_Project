"""Restore full-size result figures and expand the manuscript conclusion."""

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import os
import re
import shutil
import zipfile

from lxml import etree


ROOT = Path(__file__).resolve().parent
DOCX = ROOT / "FullPaperTemplateASIM2026 - KK.docx"
BACKUP = ROOT / "FullPaperTemplateASIM2026 - KK.before_fullsize_figures.docx"
TEMP = ROOT / "FullPaperTemplateASIM2026 - KK.fullsize.tmp.docx"
FIG6 = ROOT / "ASIM2026_figures" / "deep_paired_strategy_evidence.png"
FIG7 = ROOT / "ASIM2026_figures" / "deep_mechanism_and_layout.png"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W14_NS = "http://schemas.microsoft.com/office/word/2010/wordml"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
V_NS = "urn:schemas-microsoft-com:vml"
O_NS = "urn:schemas-microsoft-com:office:office"
W = f"{{{W_NS}}}"
NS = {"w": W_NS, "w14": W14_NS, "r": R_NS, "v": V_NS, "o": O_NS}
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"


RESULTS_CONDITIONAL = (
    "Paired comparisons clarified the condition dependence of optimization (Figure 6). Relative to zonal PIR, GA used less power in 217 of 225 matched scenarios: "
    "70/75 under Clear, 72/75 under Overcast, and all 75 Night cases. The corresponding mean paired savings were 58.06, 151.45, and 207.42 W. "
    "Relative to the stronger rule-based hybrid, however, Clear showed non-positive mean saving at every occupancy count; GA was lower in 7/75 cases, equal in 28, and higher in 40. "
    "Under Overcast, GA was lower in 60/75 cases, including 59 of the 60 cases with 4–15 occupants; one case was equal and none was higher in this density range. "
    "At Night, all five one-occupant cases were equal, while 69 of the 70 cases with 2–15 occupants were lower and one was equal. "
    "The incremental GA benefit therefore emerged consistently when daylight was limited and multiple occupied workstations required coordinated lighting."
)

RESULTS_MECHANISM = (
    "Component-level differences identified the source of the incremental benefit (Figure 7a). Relative to rule-based hybrid control, GA changed mean ceiling, task, and total power by "
    "0.00, +1.10, and +1.10 W under Clear; −25.70, −1.32, and −27.02 W under Overcast; and −84.30, +5.62, and −78.68 W at Night. "
    "The Night reduction therefore did not result from uniformly dimming every luminaire. Instead, a small increase in localized task-light power was outweighed by a much larger decrease in background ceiling-light power, "
    "showing how coordinated hybrid control can replace room-scale illumination with targeted light at occupied workstations."
)

RESULTS_LAYOUT = (
    "GA-optimized mean power increased overall from one to 15 occupants, from 3.0 to 31.2 W under Clear, 38.8 to 125.1 W under Overcast, and 110.1 to 300.4 W at Night. "
    "The trajectories were not strictly monotonic because different layouts activated different ceiling zones and received different daylight. For GA, the mean within-count range across five layouts was 14.6 W under Clear, 24.8 W under Overcast, and 31.4 W at Night, "
    "representing 29.8%, 23.8%, and 8.5% of total power variation, respectively (Figure 7b). The corresponding mean ranges for zonal PIR were 44.9, 59.8, and 60.7 W, and those for rule-based hybrid control were 12.5, 38.9, and 35.2 W. "
    "Occupancy count therefore did not uniquely determine lighting power; occupied-seat location remained an important state variable, particularly when daylight was available."
)

FIG6_CAPTION = (
    "Figure 6. Paired strategy evidence across sky condition and occupancy density. (a,b) Mean paired power saving of GA-optimized hybrid control relative to zonal PIR ceiling-only and rule-based hybrid control, respectively. "
    "Each cell averages five matched seating layouts; positive values indicate lower GA power. (c) Scenario-level dominance and mean paired difference across 75 matched cases per comparison."
)

FIG7_CAPTION = (
    "Figure 7. Source of incremental GA performance and sensitivity to seating layout. (a) Mean component change calculated as GA minus rule-based hybrid control; negative values indicate lower GA power. "
    "(b) Mean within-density minimum–maximum power range across five seating layouts. Percentages above bars show the within-density share of total variation; these descriptive ranges are not confidence intervals."
)

CONCLUSION_1 = (
    "This study established a matched simulation framework for separating the benefit of a hybrid ceiling–task lighting architecture from the incremental benefit of genetic-algorithm optimization. "
    "Across 225 lighting–occupancy–seating scenarios, GA-optimized hybrid control reduced mean power relative to zonal PIR ceiling-only control by 76.9% under Clear, 62.0% under Overcast, and 46.8% at Night. "
    "However, the first transition—from zonal PIR to occupancy-triggered rule-based hybrid control—already reduced mean power by 78.4%, 50.9%, and 29.0%, respectively. "
    "The second transition—from rule-based hybrid to GA control—then increased Clear power by 6.7% but reduced Overcast and Night power by a further 22.5% and 25.0%. "
    "This decomposition shows that hybridization produces the primary power reduction, while optimization contributes an additional benefit only within particular operating conditions."
)

CONCLUSION_2 = (
    "The most important finding is therefore conditional rather than universal. Under Clear, no occupancy density produced a positive mean GA saving relative to the rule-based hybrid, indicating that a simple daylight-specific rule was already sufficient for the sampled scenarios. "
    "Under Overcast, the GA advantage became consistent from four occupants, and at Night it appeared from two occupants. The Night mechanism was especially clear: task-light power increased by 5.62 W, but ceiling-light power decreased by 84.30 W, yielding a net reduction of 78.68 W. "
    "At the same time, substantial power ranges among layouts with identical occupancy counts showed that occupant number alone cannot represent lighting demand. Taken together, the results establish a four-factor hierarchy: sky condition defines the available optimization margin, occupancy density defines the coordination load, seating layout determines the spatial demand, and control architecture determines whether that spatial information can be exploited."
)

CONCLUSION_3 = (
    "These findings support a conditional control strategy rather than a single controller for all situations. Rule-based hybrid control is the more parsimonious option when daylight is abundant, whereas GA optimization is most valuable when daylight is limited, several workstations are occupied, or occupants are distributed across multiple lighting zones. "
    "This conclusion remains bounded by the present simulation design: one office geometry, three selected sky conditions, five designed layouts per density, and power values without an operating-duration model. The current work also does not include hardware response, sensor uncertainty, glare or individual preference, and the retained solutions require direct cross-checking against the original illuminance outputs and repeated GA runs. "
    "Within these boundaries, the framework provides both a reproducible comparison platform and a practical basis for deciding when the computational cost of optimization is justified."
)


def text_of(p):
    return "".join(t.text or "" for t in p.xpath(".//w:t", namespaces=NS))


def replace_text(p, new_text):
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


def find_paragraph(root, prefix):
    matches = [p for p in root.xpath("//w:body//w:p", namespaces=NS) if text_of(p).startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one paragraph starting with {prefix!r}, found {len(matches)}")
    return matches[0]


def remove_rendered_page_breaks(element):
    for node in element.xpath(".//w:lastRenderedPageBreak", namespaces=NS):
        node.getparent().remove(node)


def set_shape_size_and_alt(image_paragraph, width_pt, height_pt, alt, shape_id=None, rel_id=None):
    shape = image_paragraph.xpath(".//v:shape", namespaces=NS)[0]
    shape.set("style", f"width:{width_pt:.2f}pt;height:{height_pt:.2f}pt;visibility:visible")
    shape.set("alt", alt)
    if shape_id:
        shape.set("id", shape_id)
    if rel_id:
        image_data = image_paragraph.xpath(".//v:imagedata", namespaces=NS)[0]
        image_data.set(f"{{{R_NS}}}id", rel_id)


def next_relationship_id(rels_root):
    used = {int(m.group(1)) for e in rels_root for m in [re.fullmatch(r"rId(\d+)", e.get("Id", ""))] if m}
    value = 1
    while value in used:
        value += 1
    return f"rId{value}"


def update_document_xml(raw_xml, fig7_rel_id):
    root = etree.fromstring(raw_xml)

    p_conditional = find_paragraph(root, "The advantage of GA over the rule-based hybrid")
    p_mechanism = find_paragraph(root, "Component-level differences identify the source")
    p_fig6_caption = find_paragraph(root, "Figure 6. Conditional performance")
    p_layout = find_paragraph(root, "GA-optimized mean power increased overall")
    p_conclusion = find_paragraph(root, "This study evaluated coordinated control of three ceiling-light zones")

    replace_text(p_conditional, RESULTS_CONDITIONAL)
    replace_text(p_mechanism, RESULTS_MECHANISM)
    replace_text(p_fig6_caption, FIG6_CAPTION)
    replace_text(p_layout, RESULTS_LAYOUT)
    replace_text(p_conclusion, CONCLUSION_1)

    # Restore Figure 6 to its full aspect ratio (174 mm wide).
    fig6_image = p_fig6_caption.getprevious()
    if not fig6_image.xpath(".//v:imagedata", namespaces=NS):
        raise RuntimeError("Figure 6 image paragraph not found before its caption")
    set_shape_size_and_alt(
        fig6_image,
        width_pt=492.75,
        height_pt=297.87,
        alt="Paired power-saving heatmaps and scenario-level outcomes across Clear, Overcast, and Night conditions.",
    )

    # Add Figure 7 at full width by cloning the established figure/caption formatting.
    fig7_image = deepcopy(fig6_image)
    remove_rendered_page_breaks(fig7_image)
    fig7_image.set(f"{{{W14_NS}}}paraId", "71F10001")
    fig7_image.set(f"{{{W14_NS}}}textId", "71F10011")
    pict = fig7_image.xpath(".//w:pict", namespaces=NS)[0]
    pict.set(f"{{{W14_NS}}}anchorId", "71F10021")
    set_shape_size_and_alt(
        fig7_image,
        width_pt=492.75,
        height_pt=225.65,
        alt="Power-allocation mechanism and seating-layout sensitivity across the three sky conditions.",
        shape_id="_x0000_i1034",
        rel_id=fig7_rel_id,
    )

    fig7_caption = deepcopy(p_fig6_caption)
    remove_rendered_page_breaks(fig7_caption)
    fig7_caption.set(f"{{{W14_NS}}}paraId", "71F10002")
    fig7_caption.set(f"{{{W14_NS}}}textId", "71F10012")
    replace_text(fig7_caption, FIG7_CAPTION)

    p_fig6_caption.addnext(fig7_caption)
    p_fig6_caption.addnext(fig7_image)

    # Expand the conclusion to three paragraphs while retaining the original style.
    conclusion_2 = deepcopy(p_conclusion)
    conclusion_2.set(f"{{{W14_NS}}}paraId", "71F10003")
    conclusion_2.set(f"{{{W14_NS}}}textId", "71F10013")
    remove_rendered_page_breaks(conclusion_2)
    replace_text(conclusion_2, CONCLUSION_2)

    conclusion_3 = deepcopy(p_conclusion)
    conclusion_3.set(f"{{{W14_NS}}}paraId", "71F10004")
    conclusion_3.set(f"{{{W14_NS}}}textId", "71F10014")
    remove_rendered_page_breaks(conclusion_3)
    replace_text(conclusion_3, CONCLUSION_3)

    p_conclusion.addnext(conclusion_3)
    p_conclusion.addnext(conclusion_2)

    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")


def update_relationships(raw_xml):
    root = etree.fromstring(raw_xml)
    rel_id = next_relationship_id(root)
    rel = etree.SubElement(root, f"{{{REL_NS}}}Relationship")
    rel.set("Id", rel_id)
    rel.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image")
    rel.set("Target", "media/image10.png")
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes"), rel_id


def update_core_properties(raw_xml):
    root = etree.fromstring(raw_xml)
    modified = root.find("{http://purl.org/dc/terms/}modified")
    if modified is not None:
        modified.text = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")


def build():
    for path in (DOCX, FIG6, FIG7):
        if not path.exists():
            raise FileNotFoundError(path)
    if not BACKUP.exists():
        shutil.copy2(DOCX, BACKUP)

    with zipfile.ZipFile(DOCX, "r") as zin:
        infos = zin.infolist()
        payload = {info.filename: zin.read(info.filename) for info in infos}

    rels_name = "word/_rels/document.xml.rels"
    payload[rels_name], fig7_rel_id = update_relationships(payload[rels_name])
    payload["word/document.xml"] = update_document_xml(payload["word/document.xml"], fig7_rel_id)
    payload["word/media/image9.png"] = FIG6.read_bytes()
    payload["word/media/image10.png"] = FIG7.read_bytes()
    if "docProps/core.xml" in payload:
        payload["docProps/core.xml"] = update_core_properties(payload["docProps/core.xml"])

    if TEMP.exists():
        TEMP.unlink()
    with zipfile.ZipFile(TEMP, "w") as zout:
        for info in infos:
            zout.writestr(info, payload[info.filename])
        # New media entry uses standard deflate compression.
        zout.writestr("word/media/image10.png", payload["word/media/image10.png"], compress_type=zipfile.ZIP_DEFLATED)

    with zipfile.ZipFile(TEMP, "r") as check:
        bad = check.testzip()
        if bad:
            raise RuntimeError(f"ZIP integrity failure: {bad}")

    os.replace(TEMP, DOCX)
    print(f"updated={DOCX}")
    print(f"backup={BACKUP}")
    print(f"figure7_relationship={fig7_rel_id}")


if __name__ == "__main__":
    build()
