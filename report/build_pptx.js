// Sinh slide seminar Đồ án 8 (.pptx) từ report/data/ket_qua.json, report/noi_dung_slide.json và report/figures/.
//   node report/build_pptx.js report build/Seminar_GraphK.pptx
// Màu ghi trực tiếp bằng mã hex (không dùng màu theme) để chạy được cả trên Colab.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const [reportDir, outPath] = process.argv.slice(2);
const K = JSON.parse(fs.readFileSync(path.join(reportDir, "data", "ket_qua.json"), "utf8"));
const T = JSON.parse(fs.readFileSync(path.join(reportDir, "noi_dung_slide.json"), "utf8"));
const info = JSON.parse(fs.readFileSync(path.join(reportDir, "thong_tin.json"), "utf8"));
const fig = (n) => path.join(reportDir, "figures", n);

const THEME = {
  name: "GraphK", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "1F2A44", lt1: "FFFFFF", dk2: "2E3A59", lt2: "EEF1F6", accent1: "E07A3F", accent2: "4C9A8A",
    accent3: "5B6FB5", accent4: "9AA3B5", accent5: "C8553D", accent6: "F2B134", hlink: "5B6FB5", folHlink: "2E3A59" },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "GKT trên đồ thị tiên quyết tường minh";
pres.author = info.tac_gia;

// ------------------------------------------------------------------ định dạng số
const f = (x, d = 3) => (x == null ? "—" : Number(x).toFixed(d).replace(".", ","));
const n0 = (x) => Number(x).toLocaleString("de-DE"); // 191.874
const pct = (x, d = 1) => (100 * x).toFixed(d).replace(".", ",") + "%";

// ------------------------------------------------------------------ layout
const W = 13.33, H = 7.5, M = 0.6;
pres.defineSlideMaster({
  title: "TIEU_DE_TOI", background: { color: HEX.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 2.2, w: W - 2 * M, h: 1.6, fontSize: 40, bold: true, color: HEX.lt1, fontFace: THEME.headFontFace, valign: "bottom" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 4.0, w: W - 2 * M, h: 1.6, fontSize: 18, color: HEX.accent4, valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CHUONG", background: { color: HEX.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 2.6, w: W - 2 * M, h: 1.4, fontSize: 40, bold: true, color: HEX.lt1, fontFace: THEME.headFontFace }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 4.1, w: W - 2 * M, h: 1.0, fontSize: 18, color: HEX.accent6 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "NOI_DUNG", background: { color: HEX.lt1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.35, w: W - 2 * M, h: 0.9, fontSize: 32, bold: true, color: HEX.dk1, fontFace: THEME.headFontFace, valign: "middle" }, text: "" } },
  ],
  slideNumber: { x: W - 1.0, y: H - 0.5, w: 0.5, h: 0.3, fontSize: 11, color: HEX.accent4 },
});

let sec = "";
const addSlide = (master, section) => {
  if (section && section !== sec) { pres.addSection({ title: section }); sec = section; }
  return pres.addSlide({ masterName: master, sectionTitle: sec });
};
const text = (s, t, o) => s.addText(t, { isTextBox: true, fontFace: THEME.bodyFontFace, color: HEX.dk1, fontSize: 16, valign: "top", margin: 0, ...o });
const bullets = (items, o = {}) => items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 6, ...o } }));
const card = (s, x, y, w, h, fill, name) => s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill }, line: { color: fill }, objectName: name });
const circle = (s, x, y, d, fill, label, name) => {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill }, objectName: name });
  text(s, label, { x, y, w: d, h: d, align: "center", valign: "middle", bold: true, color: HEX.lt1, fontSize: 16 });
};
const stat = (s, x, y, w, value, label, color) => {
  text(s, value, { x, y, w, h: 1.0, fontSize: 44, bold: true, color, fontFace: THEME.headFontFace });
  text(s, label, { x, y: y + 1.0, w, h: 0.7, fontSize: 14, color: HEX.dk2 });
};
const image = (s, file, x, y, w, h, alt) => {
  if (fs.existsSync(file)) s.addImage({ path: file, x, y, w, h, sizing: { type: "contain", w, h }, altText: alt });
  else text(s, `[thiếu hình ${path.basename(file)}]`, { x, y, w, h, color: HEX.accent5 });
};
const chartStyle = {
  catAxisLabelColor: HEX.dk2, valAxisLabelColor: HEX.dk2, catAxisLabelFontFace: "Calibri", valAxisLabelFontFace: "Calibri",
  dataLabelFontFace: "Calibri", titleFontFace: "Calibri", legendFontFace: "Calibri", catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
  dataLabelFontSize: 11, valGridLine: { color: "D9DEE8", size: 0.5 }, catGridLine: { style: "none" }, showValue: true,
  dataLabelFormatCode: "0.000", valAxisLabelFormatCode: "0.00", dataLabelColor: HEX.dk1,
};

const byLabel = Object.fromEntries((K.bang1 || []).map((r) => [r["mô hình"], r]));
const MAIN = ["GKT-Expert", "GKT-Stat", "GKT-Random", "DKT", "AKT"].filter((k) => byLabel[k]);
const D = K.du_lieu, G = D.do_thi_chuyen_gia;
const graphRow = (name) => (K.do_thi || []).find((r) => r.graph === name) || {};

// 1 ---------------------------------------------------------------- bìa
{
  const s = addSlide("TIEU_DE_TOI", "Mở đầu");
  s.addText("Truy vết tri thức dựa trên đồ thị với đồ thị tiên quyết tường minh", { placeholder: "title" });
  s.addText([{ text: "Đồ án 8 — Học máy · GKT trên dữ liệu Junyi Academy", options: { breakLine: true } },
    { text: `${info.tac_gia} · ${info.lop} · GV: ${info.giang_vien}` }], { placeholder: "body" });
  s.addNotes(T.ghi_chu.bia);
}
// 2 ---------------------------------------------------------------- vấn đề và câu hỏi nghiên cứu
{
  const s = addSlide("NOI_DUNG", "Mở đầu");
  s.addText("Đồ thị tiên quyết của chuyên gia có giúp dự đoán tốt hơn?", { placeholder: "title" });
  text(s, T.van_de, { x: M, y: 1.45, w: W - 2 * M, h: 0.9, fontSize: 18, color: HEX.dk2 });
  const rq = [["RQ1", "GKT-Expert có vượt DKT, AKT (không dùng đồ thị)?", HEX.accent1],
    ["RQ2", "Tri thức chuyên gia đóng góp bao nhiêu so với tri thức thống kê, cùng kiến trúc, cùng 979 cạnh?", HEX.accent2],
    ["RQ3", "Nếu đồ thị thống kê thắng: giá trị thật hay rò rỉ gián tiếp? Đánh giá thế nào cho công bằng?", HEX.accent3]];
  rq.forEach(([k, v, c], i) => {
    const y = 2.6 + i * 1.45;
    circle(s, M, y, 1.0, c, k, `rq_${i}`);
    text(s, v, { x: M + 1.3, y: y + 0.12, w: W - 2 * M - 1.3, h: 0.9, fontSize: 18, valign: "middle" });
  });
  s.addNotes(T.ghi_chu.van_de);
}
// 3 ---------------------------------------------------------------- bài toán truy vết tri thức
{
  const s = addSlide("NOI_DUNG", "Mở đầu");
  s.addText("Truy vết tri thức: dự đoán câu trả lời kế tiếp", { placeholder: "title" });
  const seq = [["Phân số", 1], ["Phân số", 0], ["Cộng phân số", 1], ["Rút gọn", 1], ["Nhân phân số", "?"]];
  seq.forEach(([c, a], i) => {
    const x = M + i * 2.45, fill = a === "?" ? HEX.accent6 : a ? HEX.accent2 : HEX.accent5;
    card(s, x, 1.7, 2.15, 1.3, fill, `tuong_tac_${i}`);
    text(s, [{ text: c, options: { bold: true, breakLine: true } }, { text: a === "?" ? "P(đúng) = ?" : a ? "đúng" : "sai" }],
      { x: x + 0.15, y: 1.85, w: 1.85, h: 1.0, color: HEX.lt1, fontSize: 16, align: "center", valign: "middle" });
  });
  text(s, "Mỗi tương tác: (khái niệm q_t, kết quả a_t). Mô hình ước lượng trạng thái tri thức ẩn và P(a_{t+1}=1 | q_{t+1}, lịch sử).",
    { x: M, y: 3.35, w: W - 2 * M, h: 0.6, fontSize: 16, color: HEX.dk2 });
  const fam = [["Tuần tự", "DKT (LSTM)", "quan hệ khái niệm chỉ ngầm trong trạng thái ẩn"],
    ["Chú ý", "AKT, simpleKT", "chú ý theo thời gian, không có cấu trúc tiên quyết"],
    ["Đồ thị", "GKT, SKT, GIKT", "lan truyền trên đồ thị khái niệm — đồ án này"]];
  fam.forEach(([h, m, d], i) => {
    const x = M + i * 4.1;
    card(s, x, 4.25, 3.8, 2.3, i === 2 ? "FBE7DA" : HEX.lt2, `ho_mo_hinh_${i}`);
    text(s, [{ text: h, options: { bold: true, fontSize: 20, breakLine: true } }, { text: m, options: { fontSize: 16, breakLine: true, color: HEX.accent3 } },
      { text: d, options: { fontSize: 14 } }], { x: x + 0.25, y: 4.45, w: 3.3, h: 1.9 });
  });
  s.addNotes(T.ghi_chu.bai_toan);
}
// 4 ---------------------------------------------------------------- dữ liệu
{
  const s = addSlide("NOI_DUNG", "Dữ liệu và đồ thị");
  s.addText("Dữ liệu Junyi Academy sau tiền xử lý", { placeholder: "title" });
  const st = [[String(D.n_nut), "khái niệm (837 dòng, 2 tên lặp)", HEX.accent1], [n0(D.n_nguoi_hoc), "người học (≥ 3 tương tác)", HEX.accent2],
    [(D.n_tuong_tac / 1e6).toFixed(2).replace(".", ",") + " tr", "tương tác", HEX.accent3], [pct(D.ti_le_dung), "trả lời đúng lần đầu", HEX.dk2]];
  st.forEach(([v, l, c], i) => stat(s, M + i * 3.1, 1.6, 2.9, v, l, c));
  text(s, bullets([
    `${n0(D.n_doan)} đoạn chuỗi (≤ 200 bước, trung bình ${f(D.do_dai_tb, 1)}); ${D.n_nut_khong_tuong_tac} khái niệm không có tương tác`,
    "Chia theo người học: tập kiểm tra 38.375 người học; 5 phần ~30.700 người học cho huấn luyện/kiểm định",
    "Báo cáo dùng phần 0: huấn luyện 122.799 người học, kiểm định 30.700, kiểm tra 38.375",
  ]), { x: M, y: 3.9, w: W - 2 * M, h: 2.6, fontSize: 18 });
  s.addNotes(T.ghi_chu.du_lieu);
}
// 5 ---------------------------------------------------------------- đồ thị chuyên gia
{
  const s = addSlide("NOI_DUNG", "Dữ liệu và đồ thị");
  s.addText("Đồ thị tiên quyết chuyên gia không hoàn hảo", { placeholder: "title" });
  const st = [[String(G.n_edges), "cạnh tiên quyết", HEX.accent2], [String(G.n_isolated), "nút cô lập", HEX.accent1],
    [String(G.n_weak_components), "thành phần liên thông yếu", HEX.accent3], ["1", "chu trình (không phải DAG)", HEX.accent5]];
  st.forEach(([v, l, c], i) => stat(s, M + i * 3.1, 1.5, 2.9, v, l, c));
  card(s, M, 3.6, 6.0, 2.9, HEX.lt2, "chu_trinh");
  text(s, [{ text: "Chu trình 3 bài về căn thức", options: { bold: true, fontSize: 18, breakLine: true } },
    { text: "cộng trừ căn thức → nhân chia căn thức → rút gọn căn thức → cộng trừ căn thức", options: { fontSize: 15, breakLine: true } },
    { text: "Giữ nguyên: lan truyền GNN không đòi hỏi DAG.", options: { fontSize: 15, color: HEX.accent3 } }], { x: M + 0.3, y: 3.85, w: 5.4, h: 2.5 });
  card(s, M + 6.4, 3.6, W - 2 * M - 6.4, 2.9, HEX.lt2, "vong_tu_than");
  text(s, [{ text: "Đã loại bỏ", options: { bold: true, fontSize: 18, breakLine: true } },
    { text: "2 vòng tự thân: proportions_1, number_sense_length_l1", options: { bullet: true, fontSize: 15, breakLine: true } },
    { text: "3 cạnh trùng; 2 bài lặp tên (giữ bản mới nhất)", options: { bullet: true, fontSize: 15 } }], { x: M + 6.7, y: 3.85, w: W - 2 * M - 7.0, h: 2.5 });
  s.addNotes(T.ghi_chu.do_thi);
}
// 6 ---------------------------------------------------------------- nghiên cứu liên quan
{
  const s = addSlide("NOI_DUNG", "Nghiên cứu liên quan");
  s.addText("Ba nhóm phương pháp và khoảng trống", { placeholder: "title" });
  const g3 = [["Không dùng đồ thị", "AKT (KDD'20), simpleKT (ICLR'23), pyKT", "quan hệ khái niệm chỉ học ngầm", "kênh lan truyền tường minh theo cạnh tiên quyết", HEX.accent3],
    ["Đồ thị suy từ dữ liệu", "JKT, Bi-CLKT, DGEKT, HHSKT (Q1), GIKT", "có thể chỉ khớp tương quan hành vi → rò rỉ gián tiếp", "đồ thị độc lập dữ liệu + đánh giá trên khái niệm giữ lại", HEX.accent2],
    ["Có quan hệ tiên quyết", "SKT (ICDM'20), PSI-KT (ICLR'24)", "cấu trúc ước lượng từ dữ liệu, chưa tách đóng góp chuyên gia", "cùng kiến trúc, cùng số cạnh: Expert vs Stat vs Random", HEX.accent1]];
  g3.forEach(([h, w, lim, br, c], i) => {
    const x = M + i * 4.1, y = 1.5;
    card(s, x, y, 3.8, 5.1, HEX.lt2, `nhom_${i}`);
    circle(s, x + 0.25, y + 0.25, 0.7, c, String(i + 1), `nhom_so_${i}`);
    text(s, h, { x: x + 1.1, y: y + 0.3, w: 2.5, h: 0.65, bold: true, fontSize: 18, valign: "middle" });
    text(s, w, { x: x + 0.25, y: y + 1.15, w: 3.3, h: 0.9, fontSize: 14, color: HEX.dk2 });
    text(s, [{ text: "Hạn chế", options: { bold: true, breakLine: true, color: HEX.accent5 } }, { text: lim }], { x: x + 0.25, y: y + 2.1, w: 3.3, h: 1.3, fontSize: 14 });
    text(s, [{ text: "Cầu nối sang GKT-Expert", options: { bold: true, breakLine: true, color: HEX.accent2 } }, { text: br }], { x: x + 0.25, y: y + 3.45, w: 3.3, h: 1.5, fontSize: 14 });
  });
  s.addNotes(T.ghi_chu.lien_quan);
}
// 7 ---------------------------------------------------------------- kiến trúc
{
  const s = addSlide("NOI_DUNG", "Phương pháp");
  s.addText("GKT lan truyền hai chiều trên đồ thị có hướng", { placeholder: "title" });
  image(s, fig("hinh1_kien_truc.png"), M, 1.35, 8.6, 4.15, "Sơ đồ kiến trúc GKT lan truyền hai chiều");
  text(s, bullets(["f_fwd: tới khái niệm phụ thuộc; f_bwd: tới khái niệm tiên quyết",
    "Cổng xóa–thêm + GRU, chỉ cập nhật i và lân cận", "Danh sách cạnh nén (CSR): chi phí theo bậc thực tế",
    "Nút cô lập: tự cập nhật (mặc định) hoặc nút gốc ảo"]), { x: M + 8.9, y: 1.5, w: W - 2 * M - 8.9, h: 4.2, fontSize: 15 });
  text(s, "129.701 tham số · ẩn 32 · Adam 10⁻³ · cỡ lô 128 · dừng sớm theo AUC kiểm định", { x: M, y: 5.85, w: W - 2 * M, h: 0.5, fontSize: 14, color: HEX.dk2 });
  s.addNotes(T.ghi_chu.kien_truc);
}
// 8 ---------------------------------------------------------------- ba đồ thị và giao thức chống rò rỉ
{
  const s = addSlide("NOI_DUNG", "Phương pháp");
  s.addText("Ba đồ thị cùng 979 cạnh, một giao thức chống rò rỉ", { placeholder: "title" });
  const cols = [["Chuyên gia", "cạnh tiên quyết Junyi", graphRow("expert"), HEX.accent1],
    ["Thống kê", "AV(A→B) = P(B|A) − P(B), chỉ tập huấn luyện", graphRow("stat"), HEX.accent2],
    ["Ngẫu nhiên", "đối chứng âm, cùng số cạnh", graphRow("random"), HEX.accent4]];
  cols.forEach(([h, d, r, c], i) => {
    const x = M + i * 2.95;
    card(s, x, 1.5, 2.75, 3.0, HEX.lt2, `do_thi_${i}`);
    text(s, h, { x: x + 0.2, y: 1.65, w: 2.35, h: 0.5, bold: true, fontSize: 18, color: c });
    text(s, d, { x: x + 0.2, y: 2.2, w: 2.35, h: 0.9, fontSize: 13 });
    text(s, [{ text: `${r.n_isolated ?? "—"} nút cô lập`, options: { breakLine: true } }, { text: `Jaccard với chuyên gia: ${f(r.jaccard_directed, 3)}` }],
      { x: x + 0.2, y: 3.3, w: 2.35, h: 1.0, fontSize: 13, color: HEX.dk2 });
  });
  card(s, M + 8.95, 1.5, W - 2 * M - 8.95, 3.0, "FBE7DA", "chong_ro_ri");
  text(s, [{ text: "Chống rò rỉ", options: { bold: true, fontSize: 18, breakLine: true } },
    { text: "đồ thị thống kê chỉ dùng người học tập huấn luyện của phần k", options: { bullet: true, breakLine: true } },
    { text: "108 khái niệm giữ lại: không cạnh thống kê nào chạm vào", options: { bullet: true } }], { x: M + 9.15, y: 1.65, w: W - 2 * M - 9.35, h: 2.8, fontSize: 13 });
  text(s, T.do_thi_tach_roi, { x: M, y: 4.8, w: W - 2 * M, h: 1.6, fontSize: 16 });
  s.addNotes(T.ghi_chu.ba_do_thi);
}
// 9 ---------------------------------------------------------------- hình 2
{
  const s = addSlide("NOI_DUNG", "Phương pháp");
  s.addText("Hành vi làm bài không phân biệt được chiều tiên quyết", { placeholder: "title" });
  image(s, fig("hinh2_do_thi.png"), M, 1.35, W - 2 * M, 4.6, "Phân phối bậc và giá trị gia tăng của các đồ thị");
  text(s, T.hinh2, { x: M, y: 6.1, w: W - 2 * M - 0.8, h: 0.8, fontSize: 14, color: HEX.dk2 });
  s.addNotes(T.ghi_chu.hinh2);
}
// 10 --------------------------------------------------------------- kết quả chính
{
  const s = addSlide("NOI_DUNG", "Kết quả");
  s.addText(T.tieu_de_ket_qua, { placeholder: "title" });
  s.addChart(pres.charts.BAR, [{ name: "AUC kiểm tra", labels: MAIN, values: MAIN.map((k) => byLabel[k]["AUC kiểm tra"]) }], {
    x: M, y: 1.35, w: 7.6, h: 5.2, barDir: "col", chartColors: [HEX.accent1, HEX.accent2, HEX.accent4, HEX.accent3, HEX.dk2],
    invertedColors: [HEX.accent1], showLegend: false, showTitle: true, title: "AUC trên tập kiểm tra (phần 0)", titleFontSize: 14, titleColor: HEX.dk1,
    valAxisMinVal: T.truc_auc_min, valAxisMaxVal: T.truc_auc_max, dataLabelPosition: "outEnd", ...chartStyle,
  });
  const rows = [["Mô hình", "AUC", "ACC", "RMSE"].map((h) => ({ text: h, options: { bold: true, fill: { color: HEX.lt2 } } })),
    ...MAIN.map((k) => [k, f(byLabel[k]["AUC kiểm tra"], 4), f(byLabel[k]["ACC kiểm tra"], 4), f(byLabel[k]["RMSE kiểm tra"], 4)])];
  s.addTable(rows, { x: M + 7.9, y: 1.5, w: W - 2 * M - 7.9, colW: [1.55, 0.95, 0.95, 0.88], fontSize: 13, fontFace: THEME.bodyFontFace,
    color: HEX.dk1, border: { type: "solid", pt: 0.5, color: "C9CFDB" } });
  text(s, T.ket_qua_chinh, { x: M + 7.9, y: 4.3, w: W - 2 * M - 7.9, h: 2.3, fontSize: 14 });
  s.addNotes(T.ghi_chu.ket_qua);
}
// 11 --------------------------------------------------------------- RQ3 khái niệm giữ lại
{
  const s = addSlide("NOI_DUNG", "Kết quả");
  s.addText(T.tieu_de_rq3, { placeholder: "title" });
  const b3 = Object.fromEntries((K.bang3 || []).map((r) => [r["mô hình"], r]));
  const lab = MAIN.filter((k) => b3[k]);
  s.addChart(pres.charts.BAR, [
    { name: "khái niệm còn lại", labels: lab, values: lab.map((k) => b3[k]["AUC khái niệm còn lại"]) },
    { name: "khái niệm giữ lại", labels: lab, values: lab.map((k) => b3[k]["AUC khái niệm giữ lại"]) }], {
    x: M, y: 1.35, w: 8.2, h: 5.2, barDir: "col", barGrouping: "clustered", chartColors: [HEX.accent4, HEX.accent1], showLegend: true,
    legendPos: "b", legendFontSize: 12, showTitle: true, title: "AUC theo nhóm khái niệm", titleFontSize: 14, titleColor: HEX.dk1,
    valAxisMinVal: T.truc_auc_min, valAxisMaxVal: T.truc_auc_max, dataLabelPosition: "outEnd", ...chartStyle,
  });
  card(s, M + 8.5, 1.5, W - 2 * M - 8.5, 5.0, HEX.lt2, "nhan_xet_rq3");
  text(s, T.rq3, { x: M + 8.75, y: 1.7, w: W - 2 * M - 9.0, h: 4.6, fontSize: 14 });
  s.addNotes(T.ghi_chu.rq3);
}
// 12 --------------------------------------------------------------- cắt bỏ
{
  const s = addSlide("NOI_DUNG", "Kết quả");
  s.addText("Nghiên cứu cắt bỏ trên GKT-Expert", { placeholder: "title" });
  const ab = K.bang2 || [];
  const rows = [["Biến thể", "Epoch", "AUC kiểm định", "AUC kiểm tra", "AUC giữ lại"].map((h) => ({ text: h, options: { bold: true, fill: { color: HEX.lt2 } } })),
    ...ab.map((r) => [r["mô hình"], String(r.epoch), f(r["AUC kiểm định"], 4), f(r["AUC kiểm tra"], 4), f(r["AUC khái niệm giữ lại"], 4)])];
  s.addTable(rows, { x: M, y: 1.5, w: W - 2 * M, colW: [4.6, 1.2, 2.1, 2.1, 2.13], fontSize: 15, fontFace: THEME.bodyFontFace,
    color: HEX.dk1, border: { type: "solid", pt: 0.5, color: "C9CFDB" } });
  text(s, T.cat_bo, { x: M, y: 4.2, w: W - 2 * M, h: 2.3, fontSize: 16 });
  s.addNotes(T.ghi_chu.cat_bo);
}
// 13 --------------------------------------------------------------- case study 1
const CS = K.case_study || {};
{
  const s = addSlide("NOI_DUNG", "Case study");
  s.addText("Case study: một người học trong tập kiểm tra", { placeholder: "title" });
  image(s, fig("hinh5_case_study.png"), M, 1.3, 8.9, 5.4, "Bản đồ nhiệt mức thành thạo dự đoán theo bước");
  text(s, [{ text: `Người học ${CS.user_id ?? "—"} · ${CS.q ? CS.q.length : "—"} bước`, options: { bold: true, breakLine: true, fontSize: 16 } },
    { text: "Chuỗi tiên quyết trong đoạn:", options: { breakLine: true, fontSize: 14 } },
    ...(CS.chain || []).slice(0, 4).map(([a, b], i, arr) => ({ text: `${(CS.focus_names || [])[CS.focus.indexOf(a)] || a} → ${(CS.focus_names || [])[CS.focus.indexOf(b)] || b}`,
      options: { bullet: true, fontSize: 13, breakLine: i < arr.length - 1 } }))], { x: M + 9.2, y: 1.4, w: W - 2 * M - 9.2, h: 3.0 });
  text(s, "Hàng: khái niệm; cột: bước. Màu: P(đúng) dự đoán; o đúng, x sai.", { x: M + 9.2, y: 4.6, w: W - 2 * M - 9.2, h: 1.0, fontSize: 12, color: HEX.dk2 });
  s.addNotes(T.ghi_chu.case1);
}
// 14 --------------------------------------------------------------- case study 2
{
  const s = addSlide("NOI_DUNG", "Case study");
  s.addText("Điều mô hình đồ thị nhìn thấy mà DKT không có", { placeholder: "title" });
  const acc = CS.accuracy_next || {};
  Object.entries(acc).forEach(([k, v], i) => stat(s, M + i * 4.1, 1.5, 3.8, pct(v, 0), `${k}: dự đoán đúng bước kế tiếp`, [HEX.accent1, HEX.accent2, HEX.accent3][i]));
  text(s, bullets(T.case_study), { x: M, y: 3.5, w: W - 2 * M, h: 3.0, fontSize: 17 });
  s.addNotes(T.ghi_chu.case2);
}
// 15 --------------------------------------------------------------- câu hỏi phản biện
{
  const s = addSlide("NOI_DUNG", "Thảo luận");
  s.addText("Trả lời hai câu hỏi phản biện", { placeholder: "title" });
  [[T.phan_bien[0][0], T.phan_bien[0][1], HEX.accent1], [T.phan_bien[1][0], T.phan_bien[1][1], HEX.accent2]].forEach(([q, a, c], i) => {
    const x = M + i * 6.15;
    card(s, x, 1.5, 5.9, 5.0, HEX.lt2, `phan_bien_${i}`);
    text(s, q, { x: x + 0.3, y: 1.7, w: 5.3, h: 1.3, bold: true, fontSize: 16, color: c });
    text(s, a, { x: x + 0.3, y: 3.05, w: 5.3, h: 3.3, fontSize: 15 });
  });
  s.addNotes(T.ghi_chu.phan_bien);
}
// 16 --------------------------------------------------------------- hạn chế
{
  const s = addSlide("NOI_DUNG", "Thảo luận");
  s.addText("Hạn chế và hướng phát triển", { placeholder: "title" });
  card(s, M, 1.5, 5.9, 5.0, HEX.lt2, "han_che");
  text(s, [{ text: "Hạn chế", options: { bold: true, fontSize: 18, breakLine: true, color: HEX.accent5 } }, ...bullets(T.han_che, { fontSize: 15 })], { x: M + 0.3, y: 1.7, w: 5.3, h: 4.6 });
  card(s, M + 6.15, 1.5, 5.9, 5.0, HEX.lt2, "huong_phat_trien");
  text(s, [{ text: "Hướng phát triển", options: { bold: true, fontSize: 18, breakLine: true, color: HEX.accent2 } }, ...bullets(T.huong_phat_trien, { fontSize: 15 })], { x: M + 6.45, y: 1.7, w: 5.3, h: 4.6 });
  s.addNotes(T.ghi_chu.han_che);
}
// 17 --------------------------------------------------------------- kết luận
{
  const s = addSlide("TIEU_DE_TOI", "Kết luận");
  s.addText("Kết luận", { placeholder: "title" });
  s.addText(T.ket_luan.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < T.ket_luan.length - 1, paraSpaceAfter: 8 } })), { placeholder: "body" });
  s.addNotes(T.ghi_chu.ket_luan);
}

(async () => {
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  await pres.writeFile({ fileName: outPath });
  console.log("đã ghi", outPath);
})();
