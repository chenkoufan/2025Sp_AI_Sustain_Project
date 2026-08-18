from copy import deepcopy
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import re
import shutil
import tempfile

from lxml import etree


DOCX = Path("FullPaperTemplateASIM2026 - KK_v4.docx")
FIGURE = Path("ASIM2026_figures/optimized_paired_strategy_evidence.png")
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
V = "urn:schemas-microsoft-com:vml"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {"w": W, "v": V, "r": R}


REPLACEMENTS = {
    "Relative to rule-based hybrid control, it reduced mean power by 22.5% under Overcast and 25.0% at Night, but was 6.7% higher under Clear.":
        "Relative to rule-based hybrid control, it reduced mean power by 22.5% under Overcast and 25.0% at Night, but was 6.7% higher under Clear. Across matched cases, optimized hybrid control was lower in 60/75 Overcast and 69/75 Night scenarios, but only 7/75 Clear scenarios.",
    "The lower optimized total under Overcast and Night therefore came mainly from reduced ceiling lighting.":
        "The lower optimized total under Overcast and Night therefore came mainly from reduced ceiling lighting. At Night, 5.62 W of additional task lighting replaced 84.30 W of ceiling lighting, reducing total power by 78.68 W.",
    "Figure 6(b) compares mean reductions relative to the rule-based controls. Optimized hybrid control used 6.7% more power than rule-based hybrid control under Clear, but 22.5% less under Overcast and 25.0% less at Night. It used less power in 7/75, 60/75, and 69/75 matched cases, respectively. The result shows that a simple task-light rule was effective under Clear, whereas the larger ceiling–task search space offered additional reductions under Overcast and Night.":
        "Paired comparisons showed how consistently these mean differences occurred (Figure 7). Against zonal PIR, optimized hybrid control used less power in 217/225 scenarios. Against rule-based hybrid control, the lower/equal/higher counts were 7/28/40 under Clear, 60/12/3 under Overcast, and 69/6/0 at Night. Under Overcast, 59/60 cases with 4–15 occupants were lower and one was equal. At Night, all five one-occupant cases were equal; among 2–15 occupants, 69/70 were lower and one was equal. The additional reduction was therefore concentrated under Overcast and Night, particularly above the lowest occupancy counts.",
    "Comparison with rule-based hybrid control separates the benefit of coordinating two lighting layers from the additional benefit of optimization. Rule-based hybrid control was slightly lower under Clear, where both strategies kept ceiling lighting off. Optimized hybrid control was lower under Overcast and Night, where ceiling and task levels had to be allocated jointly. This result supports sky-condition-responsive settings rather than one control rule for every daylight condition.":
        "Comparison with rule-based hybrid control separates the benefit of coordinating two lighting layers from the additional benefit of optimized settings. Rule-based hybrid control was slightly lower under Clear, where both strategies kept ceiling lighting off. Optimized hybrid control became consistently favorable from four occupants under Overcast and from two occupants at Night. Reduced daylight and multi-workstation occupancy created more opportunities to substitute localized task lighting for shared ceiling lighting. The preferred control setting should therefore respond to sky condition and seating layout.",
    "Daylight changed both the required ceiling–task balance and the value of searching beyond fixed rules. Rule-based hybrid control was 6.7% lower under Clear, while optimized hybrid control was 22.5% lower under Overcast and 25.0% lower at Night. Seating layout also affected power at the same occupancy count. For the studied office, effective control should therefore account for sky condition and seating layout, not occupancy count alone. These findings concern simulated lighting power; annual energy and implementation performance require separate evaluation.":
        "Daylight changed both the required ceiling–task balance and the benefit beyond fixed rules. Rule-based hybrid control was 6.7% lower under Clear, while optimized hybrid control was 22.5% lower under Overcast and 25.0% lower at Night. The optimized strategy became consistently favorable from four occupants under Overcast and from two occupants at Night. At Night, 5.62 W of additional task lighting replaced 84.30 W of ceiling lighting, reducing total power by 78.68 W. Seating layout also affected power at the same occupancy count. Effective control should therefore account for sky condition and seating layout, not occupancy count alone. These findings concern simulated lighting power; annual energy and implementation performance require separate evaluation.",
}

FIGURE7_CAPTION = (
    "Figure 7. Paired strategy evidence across sky condition and occupancy count. "
    "(a,b) Mean paired power saving of optimized hybrid control relative to zonal PIR ceiling-only control "
    "and rule-based hybrid control. Each cell averages five matched seating layouts; positive values indicate "
    "lower optimized power. (c) Scenario-level outcomes and mean paired differences across 75 matched cases per comparison."
)


def paragraph_text(paragraph):
    return "".join(paragraph.xpath(".//w:t/text()", namespaces=NS)).strip()


def set_paragraph_text(paragraph, text):
    nodes = paragraph.xpath(".//w:t", namespaces=NS)
    if not nodes:
        run = etree.SubElement(paragraph, f"{{{W}}}r")
        nodes = [etree.SubElement(run, f"{{{W}}}t")]
    nodes[0].text = text
    for node in nodes[1:]:
        node.text = ""


with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    with ZipFile(DOCX, "r") as archive:
        archive.extractall(tmp_path)

    document_path = tmp_path / "word" / "document.xml"
    relationships_path = tmp_path / "word" / "_rels" / "document.xml.rels"
    parser = etree.XMLParser(remove_blank_text=False)
    document_tree = etree.parse(str(document_path), parser)
    root = document_tree.getroot()

    found = set()
    figure6_caption = None
    figure6_image = None
    for paragraph in root.xpath(".//w:p", namespaces=NS):
        current = paragraph_text(paragraph)
        revised = current
        for old, new in REPLACEMENTS.items():
            if old in revised:
                revised = revised.replace(old, new)
                found.add(old)
        if revised != current:
            set_paragraph_text(paragraph, revised)
        if current.startswith("Figure 6. Power allocation and comparative reduction"):
            figure6_caption = paragraph

    missing = set(REPLACEMENTS) - found
    if missing:
        raise RuntimeError("Expected text not found:\n" + "\n".join(sorted(missing)))
    if figure6_caption is None:
        raise RuntimeError("Figure 6 caption not found")

    previous = figure6_caption.getprevious()
    while previous is not None:
        if previous.xpath(".//v:imagedata", namespaces=NS):
            figure6_image = previous
            break
        previous = previous.getprevious()
    if figure6_image is None:
        raise RuntimeError("Figure 6 image paragraph not found")

    rel_tree = etree.parse(str(relationships_path), parser)
    rel_root = rel_tree.getroot()
    existing_ids = [rel.get("Id") for rel in rel_root]
    numbers = [int(match.group(1)) for rid in existing_ids if (match := re.fullmatch(r"rId(\d+)", rid))]
    new_rid = f"rId{max(numbers) + 1}"
    relationship = etree.SubElement(rel_root, f"{{{PKG_REL}}}Relationship")
    relationship.set("Id", new_rid)
    relationship.set("Type", f"{R}/image")
    relationship.set("Target", "media/image10.png")

    image_paragraph = deepcopy(figure6_image)
    image_data = image_paragraph.xpath(".//v:imagedata", namespaces=NS)[0]
    image_data.set(f"{{{R}}}id", new_rid)
    shape = image_paragraph.xpath(".//v:shape", namespaces=NS)[0]
    shape.set("style", "width:492.75pt;height:297.9pt;visibility:visible")
    shape_ids = []
    for candidate in root.xpath(".//v:shape", namespaces=NS):
        match = re.search(r"(\d+)$", candidate.get("id", ""))
        if match:
            shape_ids.append(int(match.group(1)))
    shape.set("id", f"_x0000_i{max(shape_ids) + 1 if shape_ids else 2000}")

    caption_paragraph = deepcopy(figure6_caption)
    set_paragraph_text(caption_paragraph, FIGURE7_CAPTION)
    figure6_caption.addnext(image_paragraph)
    image_paragraph.addnext(caption_paragraph)

    document_tree.write(str(document_path), encoding="UTF-8", xml_declaration=True, standalone=True)
    rel_tree.write(str(relationships_path), encoding="UTF-8", xml_declaration=True, standalone=True)
    shutil.copyfile(FIGURE, tmp_path / "word" / "media" / "image10.png")

    rebuilt = DOCX.with_suffix(".updated.docx")
    with ZipFile(rebuilt, "w", ZIP_DEFLATED) as archive:
        for path in tmp_path.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(tmp_path))
    shutil.move(rebuilt, DOCX)

print(f"Updated {len(found)} manuscript passages and inserted Figure 7 in {DOCX}")
