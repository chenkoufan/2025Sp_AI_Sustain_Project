from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import shutil
import tempfile

from lxml import etree


DOCX = Path("FullPaperTemplateASIM2026 - KK_v3.docx")
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}

REPLACEMENTS = {
    "Coordinating ceiling and task lighting therefore offered substantial power-reduction potential, while the achievable reduction depended on daylight and spatial occupancy.":
        "Coordinating ceiling and task lighting therefore offered substantial power-reduction potential, while the achievable reduction depended on daylight and seating layout.",
    "Under Clear, both hybrid strategies used no ceiling lighting; optimized and rule-based hybrid control required 17.44 and 16.34 W of task lighting.":
        "Under Clear, both hybrid strategies used no ceiling lighting; optimized hybrid control and rule-based hybrid control required 17.44 and 16.34 W of task lighting.",
    "The three-weather comparison shows that coordinated ceiling and task lighting can reduce power across a wide range of daylight availability.":
        "Comparison across the three sky conditions shows that coordinated ceiling and task lighting can reduce power under different levels of daylight availability.",
    "The fixed rule was slightly lower under Clear, where both strategies kept ceiling lighting off.":
        "Rule-based hybrid control was slightly lower under Clear, where both strategies kept ceiling lighting off.",
    "This result supports weather-responsive settings rather than one control rule for every daylight condition.":
        "This result supports sky-condition-responsive settings rather than one control rule for every daylight condition.",
    "For the studied office, effective control therefore depends on sky condition and occupant location, not occupancy count alone.":
        "For the studied office, effective control should therefore account for sky condition and seating layout, not occupancy count alone.",
}


def paragraph_text(paragraph):
    return "".join(paragraph.xpath(".//w:t/text()", namespaces=NS))


def set_paragraph_text(paragraph, text):
    nodes = paragraph.xpath(".//w:t", namespaces=NS)
    if not nodes:
        raise RuntimeError("Paragraph has no text node")
    nodes[0].text = text
    for node in nodes[1:]:
        node.text = ""


with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    with ZipFile(DOCX, "r") as archive:
        archive.extractall(tmp_path)

    xml_path = tmp_path / "word" / "document.xml"
    tree = etree.parse(str(xml_path), etree.XMLParser(remove_blank_text=False))
    found = set()
    for paragraph in tree.xpath(".//w:p", namespaces=NS):
        current = paragraph_text(paragraph)
        revised = current
        for old, new in REPLACEMENTS.items():
            if old in revised:
                revised = revised.replace(old, new)
                found.add(old)
        if revised != current:
            set_paragraph_text(paragraph, revised)

    missing = set(REPLACEMENTS) - found
    if missing:
        raise RuntimeError("Expected sentences not found:\n" + "\n".join(sorted(missing)))

    tree.write(str(xml_path), encoding="UTF-8", xml_declaration=True, standalone=True)

    rebuilt = DOCX.with_suffix(".polished.docx")
    with ZipFile(rebuilt, "w", ZIP_DEFLATED) as archive:
        for path in tmp_path.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(tmp_path))
    shutil.move(rebuilt, DOCX)

print(f"Applied {len(found)} final terminology and flow refinements to {DOCX}")
