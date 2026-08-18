from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import re
import shutil
import tempfile


CURRENT = Path("FullPaperTemplateASIM2026 - KK.docx")
REFERENCE = Path("FullPaperTemplateASIM2026 - KK.before_coherence_polish.docx")
XML_NAME = "word/document.xml"


def read_xml(docx):
    with ZipFile(docx, "r") as archive:
        return archive.read(XML_NAME).decode("utf-8")


def root_tag(xml):
    match = re.search(r"<[^!?][^>]*>", xml)
    if not match:
        raise RuntimeError("document.xml root tag not found")
    return match.group(0), match.span()


current_xml = read_xml(CURRENT)
reference_xml = read_xml(REFERENCE)
current_root, current_span = root_tag(current_xml)
reference_root, _ = root_tag(reference_xml)

declarations = re.findall(r'\s+xmlns(?::[A-Za-z0-9_.-]+)?="[^"]+"', reference_root)
missing = [declaration for declaration in declarations if declaration.split("=", 1)[0].strip() not in current_root]

if missing:
    repaired_root = current_root[:-1] + "".join(missing) + ">"
    current_xml = current_xml[: current_span[0]] + repaired_root + current_xml[current_span[1] :]

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    with ZipFile(CURRENT, "r") as archive:
        archive.extractall(tmp_path)
    (tmp_path / XML_NAME).write_text(current_xml, encoding="utf-8", newline="")

    rebuilt = CURRENT.with_suffix(".namespace-repaired.docx")
    with ZipFile(rebuilt, "w", ZIP_DEFLATED) as archive:
        for path in tmp_path.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(tmp_path))
    shutil.move(rebuilt, CURRENT)

print(f"Restored {len(missing)} namespace declarations in {CURRENT}")
