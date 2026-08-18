"""Repair the four new Word paragraph IDs after the full-size figure insertion."""

from pathlib import Path
import os
import zipfile


ROOT = Path(__file__).resolve().parent
DOCX = ROOT / "FullPaperTemplateASIM2026 - KK.docx"
TEMP = ROOT / "FullPaperTemplateASIM2026 - KK.ids.tmp.docx"


def build():
    with zipfile.ZipFile(DOCX, "r") as zin:
        infos = zin.infolist()
        payload = {info.filename: zin.read(info.filename) for info in infos}

    xml = payload["word/document.xml"]
    for old, new in (
        (b"A7F10001", b"71F10001"),
        (b"A7F10002", b"71F10002"),
        (b"A7F10003", b"71F10003"),
        (b"A7F10004", b"71F10004"),
        (b"A7F10011", b"71F10011"),
        (b"A7F10012", b"71F10012"),
        (b"A7F10013", b"71F10013"),
        (b"A7F10014", b"71F10014"),
        (b"A7F10021", b"71F10021"),
    ):
        xml = xml.replace(old, new)
    payload["word/document.xml"] = xml

    if TEMP.exists():
        TEMP.unlink()
    with zipfile.ZipFile(TEMP, "w") as zout:
        for info in infos:
            zout.writestr(info, payload[info.filename])
    with zipfile.ZipFile(TEMP, "r") as check:
        if check.testzip():
            raise RuntimeError("ZIP integrity check failed")
    os.replace(TEMP, DOCX)
    print(DOCX)


if __name__ == "__main__":
    build()
