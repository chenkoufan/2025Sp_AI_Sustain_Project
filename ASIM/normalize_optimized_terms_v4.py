from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import shutil
import tempfile

from lxml import etree


DOCX = Path("FullPaperTemplateASIM2026 - KK_v4.docx")
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}
REPLACEMENTS = {
    "The lower optimized total under Overcast and Night":
        "The lower total under optimized hybrid control under Overcast and Night",
    "Optimized hybrid power generally increased with occupancy count":
        "Lighting power under optimized hybrid control generally increased with occupancy count",
    "The optimized strategy became consistently favorable":
        "Optimized hybrid control became consistently favorable",
}


def text(paragraph):
    return "".join(paragraph.xpath(".//w:t/text()", namespaces=NS))


def set_text(paragraph, value):
    nodes = paragraph.xpath(".//w:t", namespaces=NS)
    nodes[0].text = value
    for node in nodes[1:]:
        node.text = ""


with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    with ZipFile(DOCX) as archive:
        archive.extractall(tmp_path)
    xml_path = tmp_path / "word" / "document.xml"
    tree = etree.parse(str(xml_path), etree.XMLParser(remove_blank_text=False))
    found = set()
    for paragraph in tree.xpath(".//w:p", namespaces=NS):
        current = text(paragraph)
        revised = current
        for old, new in REPLACEMENTS.items():
            if old in revised:
                revised = revised.replace(old, new)
                found.add(old)
        if revised != current:
            set_text(paragraph, revised)
    if found != set(REPLACEMENTS):
        raise RuntimeError(f"Missing replacements: {set(REPLACEMENTS) - found}")
    tree.write(str(xml_path), encoding="UTF-8", xml_declaration=True, standalone=True)
    rebuilt = DOCX.with_suffix(".normalized.docx")
    with ZipFile(rebuilt, "w", ZIP_DEFLATED) as archive:
        for path in tmp_path.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(tmp_path))
    shutil.move(rebuilt, DOCX)

print(f"Normalized {len(found)} optimized-hybrid terms in {DOCX}")
