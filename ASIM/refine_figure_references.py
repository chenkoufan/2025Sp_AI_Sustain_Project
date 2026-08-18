from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import shutil
import tempfile
import xml.etree.ElementTree as ET


DOCX = Path("FullPaperTemplateASIM2026 - KK.docx")
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
ET.register_namespace("w", NS["w"])

REPLACEMENTS = {
    "Each scenario combined the three-dimensional office model, one sky condition, and a seating layout with 1–15 occupants as shown in Figure 1(a).":
        "As shown in Figure 1(a), each scenario combined the three-dimensional office model, one sky condition, and a seating layout with 1–15 occupants.",
    "The vector specified discrete output levels for three ceiling-light zones and 24 task lights as shown in Figure 1(b).":
        "Figure 1(b) shows the resulting 27-variable vector for three ceiling-light zones and 24 task lights.",
    "The same vector was converted to total lighting power as shown in Figure 1(c).":
        "Figure 1(c) links the control vector to illuminance, uniformity, and total lighting power.",
    "The GA updated the control vector until it retained the lowest-fitness feasible solution through the loop shown in Figure 1(d,e).":
        "The GA iterated selection, crossover, mutation, and elitism through the loop shown in Figure 1(d,e) until the lowest-fitness feasible solution was retained.",
    "Each retained solution was then compared with zonal PIR ceiling-only control and rule-based hybrid control for the same sky condition, occupancy count, and seating layout as summarized in Figure 1(f).":
        "Figure 1(f) compares each retained solution with the two rule-based controls under the same sky condition, occupancy count, and seating layout.",
    "Component-level differences showed how GA achieved the additional reduction as shown in Figure 7(a).":
        "Figure 7(a) shows how component-level differences produced the additional GA reduction.",
    "For GA, the mean within-count range was 14.6 W under Clear, 24.8 W under Overcast, and 31.4 W at Night as shown in Figure 7(b).":
        "Figure 7(b) summarizes the mean within-count ranges for GA: 14.6 W under Clear, 24.8 W under Overcast, and 31.4 W at Night.",
}


def paragraph_text(p):
    return "".join(t.text or "" for t in p.findall(".//w:t", NS))


def replace_paragraph_text(p, new_text):
    texts = p.findall(".//w:t", NS)
    if not texts:
        return
    texts[0].text = new_text
    for node in texts[1:]:
        node.text = ""


with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    with ZipFile(DOCX, "r") as zin:
        zin.extractall(tmp_path)

    xml_path = tmp_path / "word" / "document.xml"
    tree = ET.parse(xml_path)
    root = tree.getroot()

    found = set()
    for p in root.findall(".//w:p", NS):
        current = paragraph_text(p)
        revised = current
        for old, new in REPLACEMENTS.items():
            if old in revised:
                revised = revised.replace(old, new)
                found.add(old)
        if revised != current:
            replace_paragraph_text(p, revised)

    missing = set(REPLACEMENTS) - found
    if missing:
        raise RuntimeError("Expected paragraphs not found:\n" + "\n".join(sorted(missing)))

    tree.write(xml_path, encoding="UTF-8", xml_declaration=True)

    rebuilt = DOCX.with_suffix(".tmp.docx")
    with ZipFile(rebuilt, "w", ZIP_DEFLATED) as zout:
        for path in tmp_path.rglob("*"):
            if path.is_file():
                zout.write(path, path.relative_to(tmp_path))
    shutil.move(rebuilt, DOCX)

print(f"Updated {len(found)} figure-reference sentences in {DOCX}")
