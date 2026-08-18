"""Apply the final tone and cross-reference adjustments after the main polish."""

from copy import deepcopy
from pathlib import Path
import os
import zipfile

from lxml import etree


ROOT = Path(__file__).resolve().parent
DOCX = ROOT / "FullPaperTemplateASIM2026 - KK.docx"
TEMP = ROOT / "FullPaperTemplateASIM2026 - KK.style.tmp.docx"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = f"{{{W_NS}}}"
NS = {"w": W_NS}
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"


REPLACEMENTS = {
    "Office lighting controls often respond to aggregate occupancy": (
        "Office lighting controls often respond to aggregate occupancy, although lighting demand also depends on daylight and where occupants sit. "
        "We used a genetic algorithm (GA) to optimize a coordinated lighting control framework with three ceiling-light zones and 24 workstation task lights under visual-comfort constraints. "
        "A Singapore office laboratory was simulated under Clear, Overcast, and Night conditions for 1–15 occupants. Five matched seating layouts at each occupancy count yielded 225 scenarios. "
        "We compared zonal passive infrared (PIR) ceiling-only control, rule-based hybrid control, and GA-optimized hybrid control. "
        "Relative to zonal PIR, GA-optimized hybrid control reduced mean lighting power by 76.9%, 62.0%, and 46.8% under Clear, Overcast, and Night, respectively. "
        "Relative to rule-based hybrid control, it reduced mean power by 22.5% under Overcast and 25.0% at Night, but increased it by 6.7% under Clear. "
        "GA used less power in 60/75 Overcast and 69/75 Night cases, but only 7/75 Clear cases. "
        "At Night, task-light power increased by 5.62 W while ceiling-light power decreased by 84.30 W, yielding a net reduction of 78.68 W. "
        "Seating layout also produced substantial variation at fixed occupancy counts. Hybrid lighting therefore delivered the primary power reduction, while GA optimization added value mainly under limited daylight and multi-workstation occupancy."
    ),
    "This study asks how ceiling- and task-light levels should be coordinated": (
        "This study asks how ceiling- and task-light levels should be coordinated under variable daylight and occupancy while maintaining visual-comfort requirements. "
        "We evaluated a coordinated lighting control framework in a simulated open-plan office. The framework represents three ceiling-light zones and 24 task lights. "
        "A penalty-based genetic algorithm (GA) enforces illuminance and uniformity constraints while minimizing lighting power. "
        "Three strategies were compared across 225 matched sky–occupancy–layout scenarios: zonal PIR ceiling-only control, rule-based hybrid control, and GA-optimized hybrid control. "
        "This design separates the benefit of hybrid lighting from the additional benefit of optimization. It also identifies how both benefits vary with sky condition, occupancy count, and seating layout. "
        "The analysis therefore focuses on relative control performance across matched scenarios."
    ),
    "Figure 5 shows how lighting power changed with occupancy count": (
        "Figure 5 shows how lighting power changed with occupancy count. Lines show the mean of five matched seating layouts; bands show their minimum–maximum range. "
        "Under Clear, mean power for zonal PIR, rule-based hybrid, and GA-optimized hybrid control was 75.50, 16.34, and 17.44 W. The corresponding means were 244.46, 120.03, and 93.01 W under Overcast. "
        "At Night, they were 443.52, 314.78, and 236.10 W. Relative to zonal PIR, GA-optimized hybrid control reduced mean power by 76.9%, 62.0%, and 46.8%, respectively. "
        "Rule-based hybrid control accounted for reductions of 78.4%, 50.9%, and 29.0%. Relative to that stronger baseline, GA increased Clear power by 6.7% but reduced Overcast and Night power by 22.5% and 25.0%. "
        "Hybrid lighting therefore supplied most of the reduction under Clear, while GA added savings under reduced daylight."
    ),
    "The results separate two control decisions that are often conflated": (
        "The results separate two control decisions that are often conflated: combining ceiling and task lighting, and optimizing their coordination. "
        "Hybrid lighting delivered most of the reduction when daylight already supplied background illuminance. GA became valuable when nonzero ceiling-light demand had to serve several occupied workstations. "
        "This distinction explains why the same optimization method performed differently across sky conditions."
    ),
    "Clear defined the practical limit of GA optimization": (
        "Clear identified where additional optimization was unnecessary. Both hybrid strategies kept ceiling lighting off, leaving little power to redistribute. "
        "Rule-based hybrid control was lower in 40/75 Clear cases, while 28 cases were equal. The 1.10 W mean gap warrants repeated-run convergence checks before the GA solutions are interpreted as optimal."
    ),
    "Across 225 matched scenarios, coordinated hybrid lighting": (
        "Across 225 matched scenarios, coordinated hybrid lighting substantially reduced lighting power relative to zonal PIR ceiling-only control. "
        "GA-optimized hybrid control reduced mean power by 76.9% under Clear, 62.0% under Overcast, and 46.8% at Night. Yet rule-based hybrid control already produced most of this reduction. "
        "Relative to rule-based hybrid control, GA reduced mean power by 22.5% under Overcast and 25.0% at Night, but increased it by 6.7% under Clear."
    ),
}


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


def update_xml(raw):
    root = etree.fromstring(raw)
    matched = set()
    for p in root.xpath("//w:body//w:p", namespaces=NS):
        old = text_of(p)
        for prefix, new in REPLACEMENTS.items():
            if old.startswith(prefix):
                replace_text(p, new)
                matched.add(prefix)
                break
    missing = set(REPLACEMENTS) - matched
    if missing:
        raise RuntimeError("Missing adjustment targets: " + ", ".join(sorted(missing)))
    xml = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")
    for old, new in (
        (b"Figure 1a", b"Figure 1(a)"),
        (b"Figure 1b", b"Figure 1(b)"),
        (b"Figure 1c", b"Figure 1(c)"),
        (b"Figure 1d,e", b"Figure 1(d,e)"),
        (b"Figure 1f", b"Figure 1(f)"),
        (b"Figure 7a", b"Figure 7(a)"),
        (b"Figure 7b", b"Figure 7(b)"),
    ):
        xml = xml.replace(old, new)
    return xml


def build():
    with zipfile.ZipFile(DOCX, "r") as zin:
        infos = zin.infolist()
        payload = {info.filename: zin.read(info.filename) for info in infos}
    payload["word/document.xml"] = update_xml(payload["word/document.xml"])
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
