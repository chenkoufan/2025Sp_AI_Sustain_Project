const fs = require("fs");
const {
  AlignmentType,
  BorderStyle,
  Document,
  Math,
  MathFraction,
  MathRun,
  MathSubScript,
  MathSubSuperScript,
  MathSuperScript,
  Packer,
  Paragraph,
  Table,
  TableCell,
  TableRow,
  TextRun,
  WidthType,
} = require("docx");

const outputPath = process.argv[2] || "formulation.docx";
const bodyFont = "Times New Roman";
const bodySize = 20; // 10 pt
const contentWidth = 9360;
const noBorders = {
  top: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
  bottom: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
  left: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
  right: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
  insideHorizontal: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
  insideVertical: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
};

const lc = () => new MathSubScript({
  children: [new MathRun("L")],
  subScript: [new MathRun("c,j")],
});

const lt = () => new MathSubScript({
  children: [new MathRun("L")],
  subScript: [new MathRun("t,i")],
});

const equation = new Math({
  children: [
    new MathRun("P = 13.2 "),
    new MathSubSuperScript({
      subScript: [new MathRun("j=1")],
      superScript: [new MathRun("3")],
      children: [new MathRun("∑")],
    }),
    lc(),
    new MathRun(" + 1.5 "),
    new MathSubSuperScript({
      subScript: [new MathRun("i=1")],
      superScript: [new MathRun("24")],
      children: [new MathRun("∑")],
    }),
    lt(),
    new MathRun("."),
  ],
});

const equationTable = (math, number) => new Table({
  width: { size: contentWidth, type: WidthType.DXA },
  columnWidths: [700, 7960, 700],
  borders: noBorders,
  rows: [
    new TableRow({
      children: [
        new TableCell({
          width: { size: 700, type: WidthType.DXA },
          borders: noBorders,
          children: [new Paragraph("")],
        }),
        new TableCell({
          width: { size: 7960, type: WidthType.DXA },
          borders: noBorders,
          children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [math] })],
        }),
        new TableCell({
          width: { size: 700, type: WidthType.DXA },
          borders: noBorders,
          children: [
            new Paragraph({
              alignment: AlignmentType.RIGHT,
              children: [new TextRun({ text: `(${number})`, font: bodyFont, size: bodySize })],
            }),
          ],
        }),
      ],
    }),
  ],
});

const sub = (base, index) => new MathSubScript({
  children: [new MathRun(base)],
  subScript: [new MathRun(index)],
});

const squaredSub = (base, index) => new MathSuperScript({
  children: [sub(base, index)],
  superScript: [new MathRun("2")],
});

const shortfallEquation = new Math({
  children: [
    sub("v", "k"),
    new MathRun(" = max(0, "),
    new MathFraction({
      numerator: [sub("r", "k"), new MathRun(" − "), sub("x", "k")],
      denominator: [sub("r", "k")],
    }),
    new MathRun(")."),
  ],
});

const fitnessEquation = new Math({
  children: [
    new MathRun("F = P + s[200"),
    squaredSub("v", "E1"),
    new MathRun(" + 150"),
    squaredSub("v", "E0"),
    new MathRun(" + 200"),
    squaredSub("v", "U1"),
    new MathRun(" + 150"),
    squaredSub("v", "U2"),
    new MathRun("]."),
  ],
});

const doc = new Document({
  styles: {
    default: {
      document: {
        run: { font: bodyFont, size: bodySize },
        paragraph: { spacing: { line: 240, after: 120 } },
      },
    },
    paragraphStyles: [
      {
        id: "FormulationHeading",
        name: "Formulation Heading",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { font: bodyFont, size: 22, bold: true },
        paragraph: { spacing: { before: 0, after: 120 }, outlineLevel: 0 },
      },
    ],
  },
  sections: [
    {
      properties: {
        page: {
          margin: { top: 1134, right: 1134, bottom: 1134, left: 1134 },
        },
      },
      children: [
        new Paragraph({
          style: "FormulationHeading",
          children: [new TextRun("Lighting power formulation")],
        }),
        new Paragraph({
          children: [
            new TextRun(
              "The control vector specified discrete output levels for three ceiling-light zones and 24 task lights. " +
              "Task lights at unoccupied workstations remained off. Each task-light level added 1.5 W, and each " +
              "ceiling-zone level added 13.2 W. Total lighting power was calculated as:"
            ),
          ],
        }),
        equationTable(equation, 1),
        new Paragraph({
          spacing: { before: 120, after: 120 },
          children: [
            new TextRun("Here, "),
            new TextRun({ text: "P", italics: true }),
            new TextRun(" is the total lighting power in watts; "),
            new TextRun({ text: "L", italics: true }),
            new TextRun({ text: "c,j", subScript: true }),
            new TextRun(" is the discrete output level of ceiling-light zone "),
            new TextRun({ text: "j", italics: true }),
            new TextRun("; and "),
            new TextRun({ text: "L", italics: true }),
            new TextRun({ text: "t,i", subScript: true }),
            new TextRun(" is the discrete output level of task light "),
            new TextRun({ text: "i", italics: true }),
            new TextRun(". The indices "),
            new TextRun({ text: "j", italics: true }),
            new TextRun(" = 1, ..., 3 and "),
            new TextRun({ text: "i", italics: true }),
            new TextRun(" = 1, ..., 24 represent the three ceiling-light zones and 24 workstations, respectively. " +
              "For an unoccupied workstation, "),
            new TextRun({ text: "L", italics: true }),
            new TextRun({ text: "t,i", subScript: true }),
            new TextRun(" = 0."),
          ],
        }),
        new Paragraph({
          children: [
            new TextRun({
              text: "AUTHOR INPUT: Add the allowable discrete ranges for the ceiling-light and task-light levels.",
              bold: true,
              color: "C00000",
            }),
          ],
        }),
        new Paragraph({
          style: "FormulationHeading",
          children: [new TextRun("Penalty-based objective formulation")],
        }),
        new Paragraph({
          children: [
            new TextRun(
              "The GA minimized a scalar fitness function combining total lighting power with quadratic penalties " +
              "for unmet illuminance and uniformity requirements. For each performance metric, the normalized " +
              "shortfall was calculated as:"
            ),
          ],
        }),
        equationTable(shortfallEquation, 2),
        new Paragraph({
          spacing: { before: 120, after: 120 },
          children: [
            new TextRun("where "),
            new TextRun({ text: "x", italics: true }),
            new TextRun({ text: "k", subScript: true }),
            new TextRun(" is the simulated value of metric "),
            new TextRun({ text: "k", italics: true }),
            new TextRun(", "),
            new TextRun({ text: "r", italics: true }),
            new TextRun({ text: "k", subScript: true }),
            new TextRun(" is its required threshold, and "),
            new TextRun({ text: "v", italics: true }),
            new TextRun({ text: "k", subScript: true }),
            new TextRun(" is the normalized shortfall. A requirement that was satisfied produced "),
            new TextRun({ text: "v", italics: true }),
            new TextRun({ text: "k", subScript: true }),
            new TextRun(" = 0; otherwise, the shortfall increased with the relative deficit."),
          ],
        }),
        new Paragraph({
          children: [
            new TextRun(
              "The required thresholds were 750 lx for E1, 500 lx for E0, 0.70 for U1, and 0.50 for U2. " +
              "Using a quadratic penalty exponent and a power-dependent scaling factor s = max(1, P), the fitness was:"
            ),
          ],
        }),
        equationTable(fitnessEquation, 3),
        new Paragraph({
          spacing: { before: 120, after: 120 },
          children: [
            new TextRun("Here, "),
            new TextRun({ text: "F", italics: true }),
            new TextRun(" is the scalar fitness minimized by Galapagos, "),
            new TextRun({ text: "P", italics: true }),
            new TextRun(" is the total lighting power, and "),
            new TextRun({ text: "s", italics: true }),
            new TextRun(" scales the penalty weights relative to the power magnitude. A configuration was feasible " +
              "when all four requirements were satisfied. For feasible configurations, all penalty terms were zero " +
              "and the fitness equalled total lighting power. Infeasible configurations received quadratic penalties " +
              "that increased with the magnitude of each normalized violation."),
          ],
        }),
        new Paragraph({
          children: [
            new TextRun({
              text: "AUTHOR CHECK: The supplied code uses illuminance thresholds of 750 and 500 lx, although its comments state 500 and 300 lx. It also contains illu0 = num(illu2); verify whether this should be illu0 = num(illu0), and confirm which version generated the reported results.",
              bold: true,
              color: "C00000",
            }),
          ],
        }),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(outputPath, buffer);
  console.log(`Created ${outputPath}`);
});
