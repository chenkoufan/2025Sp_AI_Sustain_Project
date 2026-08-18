"""Read-only terminology and sentence-length audit for the current manuscript."""

from pathlib import Path
import re
import sys
from zipfile import ZipFile
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
DOCX = ROOT / "FullPaperTemplateASIM2026 - KK.docx"
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def read_paragraphs(path):
    with ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    return [
        "".join(t.text or "" for t in paragraph.findall(".//w:t", NS))
        for paragraph in root.findall(".//w:p", NS)
    ]


def count(text, phrase):
    return len(re.findall(re.escape(phrase), text, flags=re.IGNORECASE))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    paragraphs = read_paragraphs(DOCX)
    text = "\n".join(paragraphs)
    groups = {
        "framework": ["Balanced Light Control Framework", "coordinated lighting control framework"],
        "seat": ["seating distribution", "seat distribution", "seating pattern", "seating layout", "occupied-seat location"],
        "power": ["power demand", "electrical power", "lighting power", "energy consumption", "energy saving"],
        "ga": ["GA-derived", "GA-optimized", "genetic algorithm"],
        "pir": ["idealized zonal PIR ceiling-only", "zonal PIR ceiling-only"],
        "rule": ["occupancy-triggered rule-based hybrid", "rule-based hybrid control", "rule-based hybrid"],
        "sky": ["Clear-sky", "Clear sky", "Clear", "Overcast", "Night"],
    }
    for group, variants in groups.items():
        print(group, [(variant, count(text, variant)) for variant in variants])
    print("self_reference", [(v, count(text, v)) for v in ("this study", "this paper", "this work")])

    for i, paragraph in enumerate(paragraphs):
        body = " ".join(paragraph.split())
        if not body or body.isupper() or body.startswith(("Figure ", "Table ")):
            continue
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
        long_lengths = [len(s.split()) for s in sentences if len(s.split()) > 30]
        if long_lengths:
            print("LONG", i, long_lengths, body[:160])

    for i, paragraph in enumerate(paragraphs):
        if "Figure 1(" in paragraph or "Figure 7(" in paragraph:
            print("FIGREF", i, " ".join(paragraph.split()))


if __name__ == "__main__":
    main()
