// Sinh bài báo Word (.docx) theo khuôn trình bày IEEE (hội nghị, A4, hai cột) từ AST của docs/BAO_CAO.md.
//   node report/build_docx.js build/bao_cao.ast.json report/thong_tin.json report/figures build/BaoCao_GraphK.docx
// Khuôn: Times New Roman 10 pt; tiêu đề 24 pt; tóm tắt và từ khóa 9 pt đậm; mục cấp 1 đánh số La Mã, chữ hoa nhỏ, căn giữa;
// mục cấp 2 đánh chữ A, B, … in nghiêng; bảng đánh số La Mã, chú thích đặt trên; hình chú thích dưới, 8 pt;
// tài liệu tham khảo 8 pt dạng [n]. Bảng nhiều cột và hình rất ngang được đặt trong đoạn một cột (trải hai cột).
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, BorderStyle, ImageRun, LevelFormat, SectionType, Footer, PageNumber, ExternalHyperlink, TabStopType,
} = require("docx");

const [astPath, infoPath, figDir, outPath] = process.argv.slice(2);
const ast = JSON.parse(fs.readFileSync(astPath, "utf8"));
const info = JSON.parse(fs.readFileSync(infoPath, "utf8"));

const FONT = "Times New Roman";
const MONO = "Courier New";
// A4, lề theo khuôn IEEE A4: trên 19 mm, dưới 43 mm, trái/phải 14,32 mm; khoảng cách cột 4,22 mm (đơn vị DXA)
const PAGE = { width: 11906, height: 16838 };
const MARGIN = { top: 1077, bottom: 2438, left: 812, right: 812 };
const COL_SPACE = 239;
const TEXT_W = PAGE.width - MARGIN.left - MARGIN.right;
const COL_W = Math.floor((TEXT_W - COL_SPACE) / 2);
const PT = (p) => p * 2; // cỡ chữ docx tính theo nửa điểm

const plain = (runs) => runs.map((r) => r.text).join("");
const ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV"];

function textRuns(runs, base = {}) {
  return runs.map((r) => {
    const opt = { text: r.text, bold: !!r.b || base.bold, italics: !!r.i || base.italics, size: base.size, smallCaps: base.smallCaps };
    if (r.code) Object.assign(opt, { font: MONO, size: (base.size || PT(10)) - 2 });
    if (r.link) return new ExternalHyperlink({ link: r.link, children: [new TextRun({ ...opt, style: "Hyperlink" })] });
    return new TextRun(opt);
  });
}

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

// ---- gom nội dung thành các đoạn (section): một cột cho tiêu đề và phần trải hai cột, hai cột cho thân bài
const sections = [];
let current = null;
function open(cols) {
  if (current && current.cols === cols) return;
  current = { cols, children: [] };
  sections.push(current);
}
const push = (...xs) => current.children.push(...xs);

const title = plain((ast.find((b) => b.type === "h" && b.level === 1) || { runs: [{ text: "Báo cáo" }] }).runs);
open(1);
push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [new TextRun({ text: title, size: PT(24) })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: info.tac_gia, size: PT(11) })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: `${info.khoa}, ${info.truong}`, size: PT(10), italics: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: `${info.mon_hoc} – Lớp ${info.lop}`, size: PT(10), italics: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 360 },
    children: [new TextRun({ text: `Giảng viên hướng dẫn: ${info.giang_vien}`, size: PT(10), italics: true })] }));
open(2);

const NUM = /^(\d+(\.\d+)*\.?)\s+/;
let sec = 0, sub = 0, tableNo = 0, figNo = 0;
const border = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const body = (runs, opt = {}) => new Paragraph({ alignment: AlignmentType.JUSTIFIED, indent: { firstLine: opt.indent ?? 202 },
  spacing: { after: 0, line: 240 }, children: textRuns(runs, { size: PT(10) }) });

for (const b of ast) {
  if (b.type === "h") {
    if (b.level === 1) continue;
    const text = plain(b.runs);
    const numbered = NUM.test(text);
    const name = text.replace(NUM, "");
    if (b.level === 2) {
      if (numbered) { sec += 1; sub = 0; }
      push(new Paragraph({ heading: HeadingLevel.HEADING_1, alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: (numbered ? `${ROMAN[sec]}. ` : "") + name, smallCaps: true })] }));
    } else {
      sub += 1;
      push(new Paragraph({ heading: HeadingLevel.HEADING_2,
        children: [new TextRun({ text: `${String.fromCharCode(64 + sub)}. ${name}`, italics: true })] }));
    }
  } else if (b.type === "abstract") {
    push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120 }, children: [
      new TextRun({ text: "Tóm tắt—", bold: true, italics: true, size: PT(9) }), ...textRuns(b.runs, { bold: true, size: PT(9) })] }));
  } else if (b.type === "keywords") {
    push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 200 }, children: [
      new TextRun({ text: "Từ khóa—", bold: true, italics: true, size: PT(9) }), ...textRuns(b.runs, { bold: true, size: PT(9) })] }));
  } else if (b.type === "p") {
    push(body(b.runs));
  } else if (b.type === "list") {
    for (const it of b.items) {
      push(new Paragraph({ numbering: { reference: it.ordered ? "so" : "cham", level: Math.min(it.level, 2) },
        alignment: AlignmentType.JUSTIFIED, spacing: { after: 0 }, children: textRuns(it.runs, { size: PT(10) }) }));
    }
  } else if (b.type === "table") {
    const n = b.header.length;
    const longest = Math.max(...[b.header, ...b.rows].flatMap((r) => r.map(plain)).map((s) => s.length));
    const wide = n >= 5 || longest > 45;
    if (wide) open(1);
    const W = wide ? TEXT_W : COL_W;
    const first = Math.round(W * (n > 2 ? 0.24 : 0.4));
    const rest = Math.floor((W - first) / (n - 1));
    const widths = [first, ...Array(n - 1).fill(rest)];
    widths[n - 1] += W - widths.reduce((a, c) => a + c, 0);
    tableNo += 1;
    push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 160 }, keepNext: true,
      children: [new TextRun({ text: `BẢNG ${ROMAN[tableNo]}`, size: PT(8) })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, keepNext: true,
      children: textRuns(b.caption || [], { size: PT(8), smallCaps: true }) }));
    // kẻ ngang kiểu IEEE: viền trên/dưới bảng và dưới hàng tiêu đề, không kẻ dọc
    const cell = (runs, i, head, last) => new TableCell({
      width: { size: widths[i], type: WidthType.DXA },
      borders: { left: none, right: none, top: head ? border : none, bottom: head || last ? border : none },
      margins: { top: 20, bottom: 20, left: 60, right: 60 },
      children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
        children: textRuns(runs, { bold: head, size: PT(8) }) })],
    });
    const rows = [new TableRow({ tableHeader: true, children: b.header.map((c, i) => cell(c, i, true, false)) }),
      ...b.rows.map((r, k) => new TableRow({ children: widths.map((_, i) => cell(r[i] || [], i, false, k === b.rows.length - 1)) }))];
    push(new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: widths, rows }),
      new Paragraph({ spacing: { after: 160 }, children: [] }));
    if (wide) open(2);
  } else if (b.type === "figure") {
    const file = path.join(figDir, path.basename(b.path));
    figNo += 1;
    const caption = plain(b.caption).replace(/^Hình\s+\d+\.\s*/, "");
    if (!fs.existsSync(file)) {
      push(new Paragraph({ children: [new TextRun({ text: `[thiếu hình: ${path.basename(b.path)}]`, color: "C00000" })] }));
      continue;
    }
    const { w, h } = pngSize(file);
    const wide = w / h >= 2;
    if (wide) open(1);
    const width = Math.round(((wide ? TEXT_W : COL_W) / 1440) * 96); // điểm ảnh ở 96 dpi
    push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 }, keepNext: true, children: [
      new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width, height: Math.round((width * h) / w) },
        altText: { title: caption, description: caption, name: path.basename(b.path) } })] }),
    new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 160 }, children: [
      new TextRun({ text: `Hình ${figNo}. `, size: PT(8) }), new TextRun({ text: caption, size: PT(8) })] }));
    if (wide) open(2);
  } else if (b.type === "refs") {
    push(new Paragraph({ heading: HeadingLevel.HEADING_1, alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "Tài liệu tham khảo", smallCaps: true })] }));
    b.items.forEach((it, i) => push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, indent: { left: 360, hanging: 360 },
      tabStops: [{ type: TabStopType.LEFT, position: 360 }], spacing: { after: 40 },
      children: [new TextRun({ text: `[${i + 1}]\t`, size: PT(8) }), ...textRuns(it, { size: PT(8) })] })));
  } else if (b.type === "quote") {
    for (const l of b.lines) push(new Paragraph({ indent: { left: 360 }, children: textRuns(l, { italics: true, size: PT(9) }) }));
  }
}

const levels = (fmt) => [0, 1, 2].map((l) => ({
  level: l, format: fmt === "bullet" ? LevelFormat.BULLET : LevelFormat.DECIMAL,
  text: fmt === "bullet" ? ["•", "–", "◦"][l] : `%${l + 1})`, alignment: AlignmentType.LEFT,
  style: { paragraph: { indent: { left: 288 * (l + 1), hanging: 202 } } },
}));

const doc = new Document({
  creator: info.tac_gia, title,
  styles: {
    default: { document: { run: { font: FONT, size: PT(10) } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: PT(10) }, paragraph: { spacing: { before: 160, after: 80 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: PT(10), italics: true }, paragraph: { spacing: { before: 120, after: 60 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: "cham", levels: levels("bullet") }, { reference: "so", levels: levels("decimal") }] },
  sections: sections.filter((s) => s.children.length).map((s, i) => ({
    properties: {
      type: i === 0 ? SectionType.NEXT_PAGE : SectionType.CONTINUOUS,
      page: { size: PAGE, margin: MARGIN },
      column: s.cols === 2 ? { count: 2, space: COL_SPACE, equalWidth: true } : { count: 1 },
    },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: PT(9) })] })] }) },
    children: s.children,
  })),
});

Packer.toBuffer(doc).then((buf) => {
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, buf);
  console.log("đã ghi", outPath);
});
