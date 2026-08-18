from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import shutil
import tempfile

from lxml import etree


DOCX = Path("FullPaperTemplateASIM2026 - KK_v4.docx")
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}
REPLACEMENTS = {
    (
        "Under Overcast, seat requirements were 0/3:10, 1/2:9, 4–6:8, 7:9, 8/11:8, "
        "9/10:7, 12/15/16/19:5, 13/14/17/18:4, and 20–23:0."
    ): (
        "Under Overcast, seat requirements were 0/3:10, 1/2:9, 4–6:8, 7:9, and 8/11:8. "
        "The remaining requirements were 9/10:7, 12/15/16/19:5, 13/14/17/18:4, and 20–23:0."
    ),
    (
        "Under Overcast, optimized hybrid control used 61.07 W of ceiling lighting and 31.94 W "
        "of task lighting, compared with 86.77 and 33.26 W for rule-based hybrid control."
    ): (
        "Under Overcast, optimized hybrid control used 61.07 W of ceiling lighting and 31.94 W "
        "of task lighting. Rule-based hybrid control used 86.77 and 33.26 W, respectively."
    ),
    (
        "From one to 15 occupants, it rose from 3.0 to 31.2 W under Clear, 38.8 to 125.1 W under "
        "Overcast, and 110.1 to 300.4 W at Night."
    ): (
        "From one to 15 occupants, it rose from 3.0 to 31.2 W under Clear and from 38.8 to 125.1 W "
        "under Overcast. At Night, it rose from 110.1 to 300.4 W."
    ),
}


def paragraph_text(paragraph):
    return "".join(paragraph.xpath(".//w:t/text()", namespaces=NS))


def replace_paragraph_text(paragraph, value):
    nodes = paragraph.xpath(".//w:t", namespaces=NS)
    if not nodes:
        return
    nodes[0].text = value
    for node in nodes[1:]:
        node.text = ""


with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    with ZipFile(DOCX) as archive:
        archive.extractall(tmp_path)

    xml_path = tmp_path / "word" / "document.xml"
    tree = etree.parse(str(xml_path), etree.XMLParser(remove_blank_text=False))
    matches = set()
    for paragraph in tree.xpath(".//w:p", namespaces=NS):
        current = paragraph_text(paragraph)
        revised = current
        for old, new in REPLACEMENTS.items():
            if old in revised:
                revised = revised.replace(old, new)
                matches.add(old)
        if revised != current:
            replace_paragraph_text(paragraph, revised)

    if matches != set(REPLACEMENTS):
        raise RuntimeError(f"Missing replacements: {set(REPLACEMENTS) - matches}")

    tree.write(str(xml_path), encoding="UTF-8", xml_declaration=True, standalone=True)
    rebuilt = DOCX.with_suffix(".sentence_fix.docx")
    with ZipFile(rebuilt, "w", ZIP_DEFLATED) as archive:
        for path in tmp_path.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(tmp_path))
    shutil.move(rebuilt, DOCX)

print(f"Split {len(matches)} long sentences in {DOCX}")
