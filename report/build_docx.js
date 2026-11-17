// Sinh báo cáo Word (.docx) từ AST của docs/BAO_CAO.md.
//   node report/build_docx.js build/bao_cao.ast.json report/thong_tin.json report/figures build/BaoCao_GraphK.docx
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, BorderStyle, ShadingType, ImageRun, LevelFormat, TableOfContents, PageBreak,
  Footer, PageNumber, ExternalHyperlink,
} = require("docx");

const [astPath, infoPath, figDir, outPath] = process.argv.slice(2);
const ast = JSON.parse(fs.readFileSync(astPath, "utf8"));
const info = JSON.parse(fs.readFileSync(infoPath, "utf8"));

const FONT = "Times New Roman";
const MONO = "Courier New";
const CONTENT_W = 9071; // A4 (11906) trừ lề trái 3 cm (1701) và lề phải 2 cm (1134), đơn vị DXA

const plain = (runs) => runs.map((r) => r.text).join("");

function textRuns(runs, base = {}) {
  return runs.map((r) => {
    const opt = { text: r.text, bold: !!r.b || base.bold, italics: !!r.i || base.italics, size: base.size };
    if (r.code) Object.assign(opt, { font: MONO, size: (base.size || 26) - 4 });
    if (r.link) {
      return new ExternalHyperlink({ link: r.link, children: [new TextRun({ ...opt, style: "Hyperlink" })] });
    }
    return new TextRun(opt);
  });
}

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

const NUM = /^(\d+(\.\d+)*\.?)\s+/;
const children = [];
const title = plain((ast.find((b) => b.type === "h" && b.level === 1) || { runs: [{ text: "Báo cáo" }] }).runs);

// Trang bìa
const center = (text, opt = {}) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: opt.after ?? 120 },
  children: [new TextRun({ text, bold: opt.bold, size: opt.size || 28 })] });
children.push(center(info.truong, { bold: true }), center(info.khoa, { bold: true, after: 2400 }),
  center(info.mon_hoc, { after: 480 }), center(title, { bold: true, size: 36, after: 2400 }),
  center(`Giảng viên: ${info.giang_vien}`), center(`Thực hiện: ${info.tac_gia}`), center(`Lớp: ${info.lop}`, { after: 2400 }),
  center(info.dia_diem_ngay), new Paragraph({ children: [new PageBreak()] }),
  new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Mục lục")] }),
  new TableOfContents("Mục lục", { hyperlink: true, headingStyleRange: "1-3" }),
  new Paragraph({ children: [new PageBreak()] }));

const border = { style: BorderStyle.SINGLE, size: 4, color: "8A93A6" };
const borders = { top: border, bottom: border, left: border, right: border };

for (const b of ast) {
  if (b.type === "h") {
    if (b.level === 1) continue;
    const level = { 2: HeadingLevel.HEADING_1, 3: HeadingLevel.HEADING_2, 4: HeadingLevel.HEADING_3 }[b.level];
    children.push(new Paragraph({ heading: level, children: textRuns(b.runs) }));
  } else if (b.type === "p") {
    children.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120 }, children: textRuns(b.runs) }));
  } else if (b.type === "list") {
    for (const it of b.items) {
      children.push(new Paragraph({
        numbering: { reference: it.ordered ? "so" : "cham", level: Math.min(it.level, 2) },
        spacing: { after: 60 }, children: textRuns(it.runs),
      }));
    }
  } else if (b.type === "quote") {
    for (const l of b.lines) {
      children.push(new Paragraph({ indent: { left: 567 }, spacing: { after: 40 }, children: textRuns(l, { italics: true, size: 22 }) }));
    }
  } else if (b.type === "table") {
    const n = b.header.length;
    const first = n > 2 ? Math.round(CONTENT_W * 0.22) : Math.round(CONTENT_W * 0.35);
    const rest = Math.floor((CONTENT_W - first) / (n - 1));
    const widths = [first, ...Array(n - 1).fill(rest)];
    widths[n - 1] += CONTENT_W - widths.reduce((a, c) => a + c, 0);
    const cell = (runs, i, head) => new TableCell({
      borders, width: { size: widths[i], type: WidthType.DXA },
      shading: head ? { fill: "E8ECF4", type: ShadingType.CLEAR, color: "auto" } : undefined,
      margins: { top: 60, bottom: 60, left: 100, right: 100 },
      children: [new Paragraph({ children: textRuns(runs, { bold: head, size: 22 }) })],
    });
    const rows = [new TableRow({ tableHeader: true, children: b.header.map((c, i) => cell(c, i, true)) }),
      ...b.rows.map((r) => new TableRow({ children: widths.map((_, i) => cell(r[i] || [], i, false)) }))];
    children.push(new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths, rows }),
      new Paragraph({ spacing: { after: 120 }, children: [] }));
  } else if (b.type === "figure") {
    const file = path.join(figDir, path.basename(b.path));
    if (!fs.existsSync(file)) {
      children.push(new Paragraph({ children: [new TextRun({ text: `[thiếu hình: ${path.basename(b.path)}]`, color: "C00000" })] }));
      continue;
    }
    const { w, h } = pngSize(file);
    const width = 600; // điểm ảnh ở 96 dpi, vừa bề rộng nội dung
    children.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, children: [
      new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width, height: Math.round((width * h) / w) },
        altText: { title: plain(b.caption), description: plain(b.caption), name: path.basename(b.path) } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: textRuns(b.caption, { italics: true, size: 22 }) }));
  }
}

const heading = (size) => ({ run: { font: FONT, size, bold: true, color: "1F2A44" },
  paragraph: { spacing: { before: 240, after: 120 }, keepNext: true } });
const levels = (fmt) => [0, 1, 2].map((l) => ({
  level: l, format: fmt === "bullet" ? LevelFormat.BULLET : LevelFormat.DECIMAL,
  text: fmt === "bullet" ? ["•", "–", "◦"][l] : `%${l + 1}.`, alignment: AlignmentType.LEFT,
  style: { paragraph: { indent: { left: 567 * (l + 1), hanging: 283 } } },
}));

const doc = new Document({
  creator: info.tac_gia, title,
  styles: {
    default: { document: { run: { font: FONT, size: 26 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, ...heading(32) },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, ...heading(28) },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, ...heading(26) },
    ],
  },
  numbering: { config: [{ reference: "cham", levels: levels("bullet") }, { reference: "so", levels: levels("decimal") }] },
  features: { updateFields: true },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1701, right: 1134 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 22 })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, buf);
  console.log("đã ghi", outPath);
});
