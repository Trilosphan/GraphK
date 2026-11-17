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
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.35, w: W - 2 * M, h: 0.9, fontSize: 28, bold: true, color: HEX.dk1, fontFace: THEME.headFontFace, valign: "middle" }, text: "" } },
  ],
  slideNumber: { x: W - 1.0, y: H - 0.5, w: 0.5, h: 0.3, fontSize: 11, color: HEX.accent4 },
});

// Mỗi slide là một hàm SL.<tên>; thứ tự và phân mục theo dàn ý nằm ở DAN_Y cuối tệp.
// CUR_SEC: mục của dàn ý đang dựng (một section PowerPoint); CRUMB: nhãn mục con ghi nhỏ phía trên tiêu đề.
const SL = {};
let CUR_SEC = "", sec = "", CRUMB = null;
const addSlide = (master) => {
  if (CUR_SEC !== sec) { pres.addSection({ title: CUR_SEC }); sec = CUR_SEC; }
  const s = pres.addSlide({ masterName: master, sectionTitle: sec });
  if (master === "NOI_DUNG" && CRUMB) {
    s.addText(CRUMB, { isTextBox: true, x: M, y: 0.12, w: W - 2 * M, h: 0.3, margin: 0, fontFace: THEME.bodyFontFace,
      fontSize: 12, bold: true, color: HEX.accent3, objectName: "nhan_muc" });
  }
  return s;
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
// công thức LaTeX dựng sẵn bởi report/ve_cong_thuc.py; đặt ảnh ở cỡ chữ `pt`, trả về chiều cao (inch)
const CT_DIR = path.join(reportDir, "figures", "cong_thuc");
const CT = fs.existsSync(path.join(CT_DIR, "kich_thuoc.json")) ? JSON.parse(fs.readFileSync(path.join(CT_DIR, "kich_thuoc.json"), "utf8")) : null;
const formula = (s, key, x, y, pt = 16, maxW = null) => {
  if (!CT || !CT.cong_thuc[key]) { text(s, `[thiếu công thức ${key}]`, { x, y, w: 3, h: 0.4, color: HEX.accent5 }); return 0.4; }
  let k = pt / CT.co_chu;
  const { w, h } = CT.cong_thuc[key];
  if (maxW && w * k > maxW) k = maxW / w;
  s.addImage({ path: path.join(CT_DIR, `${key}.png`), x, y, w: w * k, h: h * k, altText: `Công thức ${key}` });
  return h * k;
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

SL.bia = () => {
  CRUMB = null;
  const s = addSlide("TIEU_DE_TOI", "Mở đầu");
  s.addText("Truy vết tri thức dựa trên đồ thị với đồ thị tiên quyết tường minh", { placeholder: "title" });
  s.addText([{ text: "Đồ án 8 — Học máy · GKT trên dữ liệu Junyi Academy", options: { breakLine: true } },
    { text: `Học viên: ${info.tac_gia} · ${info.lop} · Giảng viên: ${info.giang_vien}` }], { placeholder: "body" });
  s.addNotes(T.ghi_chu.bia);
};
SL.cau_hoi = () => {
  CRUMB = "1.3 Phát biểu bài toán";
  const s = addSlide("NOI_DUNG", "Mở đầu");
  s.addText("Phát biểu bài toán và câu hỏi nghiên cứu", { placeholder: "title" });
  text(s, [{ text: "Đầu vào: ", options: { bold: true } },
    { text: `nhật ký (khái niệm, kết quả) của từng người học và đồ thị tiên quyết có hướng G = (V, E), |V| = ${D.n_nut}, |E| = ${G.n_edges}. ` },
    { text: "Đầu ra:", options: { bold: true } }], { x: M, y: 1.4, w: 8.4, h: 0.95, fontSize: 17, color: HEX.dk2 });
  formula(s, "kt", M + 8.6, 1.83, 18, W - 2 * M - 8.6);
  const rq = [["RQ1", "GKT-Expert có vượt DKT, AKT (không dùng đồ thị)?", HEX.accent1],
    ["RQ2", "Tri thức chuyên gia đóng góp bao nhiêu so với tri thức thống kê, cùng kiến trúc, cùng 979 cạnh?", HEX.accent2],
    ["RQ3", "Nếu đồ thị thống kê thắng: giá trị thật hay rò rỉ gián tiếp? Đánh giá thế nào cho công bằng?", HEX.accent3]];
  rq.forEach(([k, v, c], i) => {
    const y = 2.6 + i * 1.45;
    circle(s, M, y, 1.0, c, k, `rq_${i}`);
    text(s, v, { x: M + 1.3, y: y + 0.12, w: W - 2 * M - 1.3, h: 0.9, fontSize: 18, valign: "middle" });
  });
  s.addNotes(T.ghi_chu.van_de);
};
SL.truy_vet = () => {
  CRUMB = "1.2 Truy vết tri thức";
  const s = addSlide("NOI_DUNG", "Mở đầu");
  s.addText("Truy vết tri thức: dự đoán câu trả lời kế tiếp", { placeholder: "title" });
  const seq = [["Phân số", 1], ["Phân số", 0], ["Cộng phân số", 1], ["Rút gọn", 1], ["Nhân phân số", "?"]];
  seq.forEach(([c, a], i) => {
    const x = M + i * 2.45, fill = a === "?" ? HEX.accent6 : a ? HEX.accent2 : HEX.accent5;
    card(s, x, 1.7, 2.15, 1.3, fill, `tuong_tac_${i}`);
    text(s, [{ text: c, options: { bold: true, breakLine: true } }, { text: a === "?" ? "P(đúng) = ?" : a ? "đúng" : "sai" }],
      { x: x + 0.15, y: 1.85, w: 1.85, h: 1.0, color: HEX.lt1, fontSize: 16, align: "center", valign: "middle" });
  });
  text(s, "Mỗi tương tác là cặp (khái niệm, kết quả). Mô hình ước lượng trạng thái tri thức ẩn và dự đoán",
    { x: M, y: 3.3, w: 7.6, h: 0.75, fontSize: 16, color: HEX.dk2 });
  formula(s, "kt", M + 7.8, 3.38, 20);
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
};
SL.du_lieu = () => {
  CRUMB = "4.1 Dữ liệu";
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
};
SL.tien_xu_ly = () => {
  CRUMB = "3.1 Tiền xử lý dữ liệu";
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
};
SL.ba_nhom = () => {
  CRUMB = "2.1–2.3 Ba nhóm phương pháp";
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
};
SL.kien_truc = () => {
  CRUMB = "3.3 Mô hình GKT tùy biến";
  const s = addSlide("NOI_DUNG", "Phương pháp");
  s.addText("GKT lan truyền hai chiều trên đồ thị có hướng", { placeholder: "title" });
  image(s, fig("hinh1_kien_truc.png"), M, 1.35, 8.6, 4.15, "Sơ đồ kiến trúc GKT lan truyền hai chiều");
  text(s, bullets(["f_fwd: tới khái niệm phụ thuộc; f_bwd: tới khái niệm tiên quyết",
    "Cổng xóa–thêm + GRU, chỉ cập nhật i và lân cận", "Danh sách cạnh nén (CSR): chi phí theo bậc thực tế",
    "Nút cô lập: tự cập nhật (mặc định) hoặc nút gốc ảo"]), { x: M + 8.9, y: 1.5, w: W - 2 * M - 8.9, h: 4.2, fontSize: 15 });
  text(s, "129.701 tham số · ẩn 32 · Adam 10⁻³ · cỡ lô 128 · dừng sớm theo AUC kiểm định", { x: M, y: 5.85, w: W - 2 * M, h: 0.5, fontSize: 14, color: HEX.dk2 });
  s.addNotes(T.ghi_chu.kien_truc);
};
SL.ba_do_thi = () => {
  CRUMB = "3.4 Các thiết lập đồ thị";
  const s = addSlide("NOI_DUNG", "Phương pháp");
  s.addText("Ba đồ thị cùng 979 cạnh và giao thức chống rò rỉ", { placeholder: "title" });
  const cols = [["Chuyên gia", "cạnh tiên quyết Junyi", graphRow("expert"), HEX.accent1],
    ["Thống kê", "", graphRow("stat"), HEX.accent2],
    ["Ngẫu nhiên", "đối chứng âm, cùng số cạnh", graphRow("random"), HEX.accent4]];
  cols.forEach(([h, d, r, c], i) => {
    const x = M + i * 2.95;
    card(s, x, 1.5, 2.75, 3.0, HEX.lt2, `do_thi_${i}`);
    text(s, h, { x: x + 0.2, y: 1.65, w: 2.35, h: 0.5, bold: true, fontSize: 18, color: c });
    if (d) text(s, d, { x: x + 0.2, y: 2.2, w: 2.35, h: 0.9, fontSize: 13 });
    else {
      formula(s, "av", x + 0.2, 2.25, 13, 2.35);
      text(s, "giá trị gia tăng, chỉ tập huấn luyện", { x: x + 0.2, y: 2.62, w: 2.35, h: 0.6, fontSize: 13 });
    }
    text(s, [{ text: `${r.n_isolated ?? "—"} nút cô lập`, options: { breakLine: true } }, { text: `Jaccard với chuyên gia: ${f(r.jaccard_directed, 3)}` }],
      { x: x + 0.2, y: 3.3, w: 2.35, h: 1.0, fontSize: 13, color: HEX.dk2 });
  });
  card(s, M + 8.95, 1.5, W - 2 * M - 8.95, 3.0, "FBE7DA", "chong_ro_ri");
  text(s, [{ text: "Chống rò rỉ", options: { bold: true, fontSize: 18, breakLine: true } },
    { text: "đồ thị thống kê chỉ dùng người học tập huấn luyện của phần k", options: { bullet: true, breakLine: true } },
    { text: "108 khái niệm giữ lại: không cạnh thống kê nào chạm vào", options: { bullet: true } }], { x: M + 9.15, y: 1.65, w: W - 2 * M - 9.35, h: 2.8, fontSize: 13 });
  text(s, T.do_thi_tach_roi, { x: M, y: 4.8, w: W - 2 * M, h: 1.6, fontSize: 16 });
  s.addNotes(T.ghi_chu.ba_do_thi);
};
SL.hinh2 = () => {
  CRUMB = "3.4 Các thiết lập đồ thị";
  const s = addSlide("NOI_DUNG", "Phương pháp");
  s.addText("Hành vi không phân biệt được chiều tiên quyết", { placeholder: "title" });
  image(s, fig("hinh2_do_thi.png"), M, 1.35, W - 2 * M, 4.6, "Phân phối bậc và giá trị gia tăng của các đồ thị");
  text(s, T.hinh2, { x: M, y: 6.1, w: W - 2 * M - 0.8, h: 0.8, fontSize: 14, color: HEX.dk2 });
  s.addNotes(T.ghi_chu.hinh2);
};
SL.ket_qua = () => {
  CRUMB = "4.3 Kết quả: RQ1, RQ2";
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
};
SL.giu_lai = () => {
  CRUMB = "4.3 Kết quả: RQ3";
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
};
SL.cat_bo = () => {
  CRUMB = "4.3 Kết quả: nghiên cứu cắt bỏ";
  const s = addSlide("NOI_DUNG", "Kết quả");
  s.addText("Nghiên cứu cắt bỏ trên GKT-Expert", { placeholder: "title" });
  const ab = K.bang2 || [];
  const rows = [["Biến thể", "Epoch", "AUC kiểm định", "AUC kiểm tra", "AUC giữ lại"].map((h) => ({ text: h, options: { bold: true, fill: { color: HEX.lt2 } } })),
    ...ab.map((r) => [r["mô hình"], String(r.epoch), f(r["AUC kiểm định"], 4), f(r["AUC kiểm tra"], 4), f(r["AUC khái niệm giữ lại"], 4)])];
  s.addTable(rows, { x: M, y: 1.5, w: W - 2 * M, colW: [4.6, 1.2, 2.1, 2.1, 2.13], fontSize: 15, fontFace: THEME.bodyFontFace,
    color: HEX.dk1, border: { type: "solid", pt: 0.5, color: "C9CFDB" } });
  text(s, T.cat_bo, { x: M, y: 4.2, w: W - 2 * M, h: 2.3, fontSize: 16 });
  s.addNotes(T.ghi_chu.cat_bo);
};
// 13 --------------------------------------------------------------- case study 1
const CS = K.case_study || {};
const csName = (id) => (CS.focus_names || [])[(CS.focus || []).indexOf(id)] || String(id);
// Chọn bước minh họa: người học trả lời đúng khái niệm a, trong khi khái niệm phụ thuộc b (cạnh a -> b) chưa được làm;
// lấy bước làm P(đúng b) của GKT-Expert tăng nhiều nhất.
function chooseStep() {
  const tr = (CS.traces || {})["GKT-Expert"];
  if (!tr || !CS.q) return null;
  let best = null;
  (CS.chain || []).forEach(([a, b]) => {
    const firstB = CS.q.indexOf(b);
    CS.q.forEach((c, t) => {
      if (c !== a || !CS.r[t] || t === 0 || (firstB !== -1 && firstB <= t)) return;
      const ib = CS.focus.indexOf(b);
      const d = tr[ib][t] - tr[ib][t - 1];
      if (!best || d > best.delta) best = { t, a, b, delta: d };
    });
  });
  return best;
}
const STEP = chooseStep();
SL.case_chuoi = () => {
  if (!CS.q) return;
  CRUMB = "4.3 Case study";
  const s = addSlide("NOI_DUNG", "Case study");
  s.addText("Case study: chuỗi tiên quyết giải tích", { placeholder: "title" });
  // sơ đồ: xếp khái niệm theo tầng (độ sâu trong chuỗi), cạnh có mũi tên = quan hệ tiên quyết chuyên gia
  const ids = CS.focus;
  const depth = Object.fromEntries(ids.map((i) => [i, 0]));
  for (let k = 0; k < ids.length; k++) (CS.chain || []).forEach(([a, b]) => { depth[b] = Math.max(depth[b], depth[a] + 1); });
  const layers = {};
  ids.forEach((i) => (layers[depth[i]] = layers[depth[i]] || []).push(i));
  const nL = Object.keys(layers).length;
  const pos = {};
  const bw = 2.05, bh = 0.62, x0 = M, areaW = 7.6, areaY = 1.55, areaH = 4.6;
  Object.entries(layers).forEach(([d, arr]) => arr.forEach((id, j) => {
    pos[id] = { x: x0 + (Number(d) * (areaW - bw)) / Math.max(nL - 1, 1), y: areaY + ((j + 0.5) * areaH) / arr.length - bh / 2 };
  }));
  (CS.chain || []).forEach(([a, b], k) => {
    const p = pos[a], q = pos[b];
    const x1 = p.x + bw, y1 = p.y + bh / 2, x2 = q.x, y2 = q.y + bh / 2;
    s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.max(Math.abs(x2 - x1), 0.01), h: Math.max(Math.abs(y2 - y1), 0.01),
      flipV: y2 < y1, line: { color: HEX.accent4, width: 1.5, endArrowType: "triangle" }, objectName: `canh_${k}` });
  });
  ids.forEach((id) => {
    const highlight = STEP && (id === STEP.a ? HEX.accent1 : id === STEP.b ? HEX.accent3 : null);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: pos[id].x, y: pos[id].y, w: bw, h: bh, rectRadius: 0.1,
      fill: { color: highlight || HEX.lt2 }, line: { color: highlight || "C9CFDB" }, objectName: `nut_${id}` });
    text(s, csName(id), { x: pos[id].x, y: pos[id].y, w: bw, h: bh, align: "center", valign: "middle", fontSize: 12,
      bold: !!highlight, color: highlight ? HEX.lt1 : HEX.dk1 });
  });
  const correct = CS.r.filter((v) => v).length;
  const RX = M + 8.0, RW = W - 2 * M - 8.0;
  text(s, [
    { text: "Bài toán truy vết tri thức", options: { bold: true, fontSize: 17, breakLine: true } },
    { text: `Người học ${CS.user_id}: ${CS.q.length} tương tác, ${correct} lần đúng. Mỗi bước, mô hình đọc lịch sử và dự đoán`, options: {} },
  ], { x: RX, y: 1.55, w: RW, h: 1.15, fontSize: 14 });
  formula(s, "kt", RX, 2.75, 15, RW);
  text(s, [
    { text: "Đồ thị tiên quyết chuyên gia", options: { bold: true, fontSize: 17, breakLine: true } },
    { text: "Mũi tên u → v: phải nắm u trước khi học v. GKT dùng chính các cạnh này để truyền bằng chứng giữa các khái niệm.", options: { breakLine: true } },
    ...(STEP ? [{ text: " ", options: { breakLine: true } },
      { text: `Bước minh họa (slide sau): bước ${STEP.t}, làm đúng `, options: {} },
      { text: csName(STEP.a), options: { bold: true, color: HEX.accent1 } },
      { text: "; khái niệm phụ thuộc ", options: {} }, { text: csName(STEP.b), options: { bold: true, color: HEX.accent3 } },
      { text: " chưa được làm.", options: {} }] : []),
  ], { x: RX, y: 3.3, w: RW, h: 3.2, fontSize: 14 });
  s.addNotes("Giới thiệu case study như một ví dụ cụ thể của bài toán: chuỗi (khái niệm, kết quả) của một người học thật trong tập kiểm tra, "
    + "và các cạnh tiên quyết chuyên gia giữa những khái niệm người học đã gặp. Khối màu cam là khái niệm được trả lời ở bước minh họa, khối màu xanh là khái niệm phụ thuộc chưa được làm.");
};
SL.case_buoc = () => {
  if (!STEP) return;
  CRUMB = "4.3 Case study";
  const s = addSlide("NOI_DUNG", "Case study");
  s.addText(`Lý thuyết qua một bước: làm đúng ${csName(STEP.a)}`, { placeholder: "title" });
  card(s, M, 1.45, 6.1, 5.2, HEX.lt2, "cong_thuc");
  // mỗi bước: nhãn (chữ) rồi công thức LaTeX
  const steps = [
    ["GKT lan truyền hai chiều", null, HEX.accent1],
    [`1. Tương tác (i = ${csName(STEP.a)}, a_t = 1)`, ["gkt_x"]],
    ["2. Thông điệp tự thân", ["gkt_self"]],
    ["3. Lân cận: xuôi tới phụ thuộc, ngược tới tiên quyết", ["gkt_nb", "gkt_s"]],
    ["4. Cổng xóa–thêm, GRU (chỉ i và lân cận)", ["gkt_ea", "gkt_gru"]],
    ["5. Dự đoán", ["gkt_pred"]],
    ["DKT (đối chứng): một vector ẩn chung", ["dkt_h", "dkt_pred"], HEX.accent3],
  ];
  let yy = 1.6;
  steps.forEach(([label, keys, color]) => {
    const head = !keys || color;
    if (color && yy > 2) yy += 0.08;
    text(s, label, { x: M + 0.25, y: yy, w: 5.6, h: 0.28, fontSize: head ? 15 : 12, bold: !!head, color: color || HEX.dk2 });
    yy += head ? 0.36 : 0.28;
    (keys || []).forEach((k) => { yy += formula(s, k, M + 0.45, yy, 14, 5.4) + 0.05; });
  });
  // bảng trước / sau cho khái niệm được trả lời và các lân cận trong chuỗi
  const nb = [STEP.a, ...new Set((CS.chain || []).flatMap(([a, b]) => (a === STEP.a ? [b] : b === STEP.a ? [a] : [])))];
  const models = Object.keys(CS.traces);
  const H = (text, extra = {}) => ({ text, options: { bold: true, fill: { color: HEX.lt2 }, align: "center", valign: "middle", ...extra } });
  const rel = (id) => (id === STEP.a ? "được trả lời" : (CS.chain || []).some(([a, b]) => a === STEP.a && b === id) ? "phụ thuộc" : "tiên quyết");
  const cell = (v, bold) => ({ text: f(v, 2), options: { bold, align: "center" } });
  // hai hàng tiêu đề: tên mô hình trải hai cột, rồi "trước" / "sau"
  const rows = [
    [H("Khái niệm", { rowspan: 2, align: "left" }), H("Quan hệ", { rowspan: 2, align: "left" }), ...models.map((m) => H(m, { colspan: 2 }))],
    models.flatMap(() => [H("trước", { fontSize: 10 }), H("sau", { fontSize: 10 })]),
    ...nb.map((id) => {
      const i = CS.focus.indexOf(id);
      return [csName(id), rel(id), ...models.flatMap((m) => {
        const tr = CS.traces[m][i];
        return [cell(tr[STEP.t - 1], false), cell(tr[STEP.t], tr[STEP.t] - tr[STEP.t - 1] > 0.05)];
      })];
    })];
  const tableW = W - 2 * M - 6.4;
  const restW = (tableW - 1.75 - 1.1) / (2 * models.length);
  s.addTable(rows, { x: M + 6.4, y: 1.45, w: tableW, colW: [1.75, 1.1, ...Array(2 * models.length).fill(restW)], fontSize: 11,
    fontFace: THEME.bodyFontFace, color: HEX.dk1, border: { type: "solid", pt: 0.5, color: "C9CFDB" } });
  const dE = STEP.delta;
  text(s, [
    { text: "Đọc bảng: ", options: { bold: true } },
    { text: `P(đúng) dự đoán ngay trước và ngay sau bước ${STEP.t}; in đậm khi tăng hơn 0,05. ` },
    { text: `Với GKT-Expert, ${csName(STEP.b)} tăng ${f(dE, 2)} dù người học chưa làm bài này: bằng chứng đi theo cạnh ${csName(STEP.a)} → ${csName(STEP.b)}. ` },
    { text: "DKT cũng có thể thay đổi dự đoán các khái niệm khác, nhưng qua vector ẩn chung chứ không theo cấu trúc tường minh." },
  ], { x: M + 6.4, y: 1.55 + 0.4 * (nb.length + 2) + 0.3, w: W - 2 * M - 6.4, h: 2.6, fontSize: 13 });
  s.addNotes("Giải thích lý thuyết qua một bước cụ thể. Bên trái là công thức một bước cập nhật của GKT hai chiều và của DKT. "
    + "Bên phải là số liệu thật lấy từ trace của mô hình: mức thành thạo dự đoán của khái niệm vừa trả lời và các khái niệm kề nó, ngay trước và ngay sau bước đó.");
};
SL.case_hinh = () => {
  CRUMB = "4.3 Case study";
  const s = addSlide("NOI_DUNG", "Case study");
  s.addText("Case study: một người học trong tập kiểm tra", { placeholder: "title" });
  image(s, fig("hinh5_case_study.png"), M, 1.3, 8.9, 5.4, "Bản đồ nhiệt mức thành thạo dự đoán theo bước");
  text(s, [{ text: `Người học ${CS.user_id ?? "—"} · ${CS.q ? CS.q.length : "—"} bước`, options: { bold: true, breakLine: true, fontSize: 16 } },
    { text: "Chuỗi tiên quyết trong đoạn:", options: { breakLine: true, fontSize: 14 } },
    ...(CS.chain || []).slice(0, 4).map(([a, b], i, arr) => ({ text: `${(CS.focus_names || [])[CS.focus.indexOf(a)] || a} → ${(CS.focus_names || [])[CS.focus.indexOf(b)] || b}`,
      options: { bullet: true, fontSize: 13, breakLine: i < arr.length - 1 } }))], { x: M + 9.2, y: 1.4, w: W - 2 * M - 9.2, h: 3.0 });
  text(s, "Hàng: khái niệm; cột: bước. Màu: P(đúng) dự đoán; o đúng, x sai.", { x: M + 9.2, y: 4.6, w: W - 2 * M - 9.2, h: 1.0, fontSize: 12, color: HEX.dk2 });
  s.addNotes(T.ghi_chu.case1);
};
SL.case_tom = () => {
  CRUMB = "4.3 Case study";
  const s = addSlide("NOI_DUNG", "Case study");
  s.addText("Điều mô hình đồ thị nhìn thấy mà DKT không có", { placeholder: "title" });
  const acc = CS.accuracy_next || {};
  Object.entries(acc).forEach(([k, v], i) => stat(s, M + i * 4.1, 1.5, 3.8, pct(v, 0), `${k}: dự đoán đúng bước kế tiếp`, [HEX.accent1, HEX.accent2, HEX.accent3][i]));
  text(s, bullets(T.case_study), { x: M, y: 3.5, w: W - 2 * M, h: 3.0, fontSize: 17 });
  s.addNotes(T.ghi_chu.case2);
};
SL.phan_bien = () => {
  CRUMB = "4.4 Đối sánh và thảo luận";
  const s = addSlide("NOI_DUNG", "Thảo luận");
  s.addText("Trả lời hai câu hỏi phản biện", { placeholder: "title" });
  [[T.phan_bien[0][0], T.phan_bien[0][1], HEX.accent1], [T.phan_bien[1][0], T.phan_bien[1][1], HEX.accent2]].forEach(([q, a, c], i) => {
    const x = M + i * 6.15;
    card(s, x, 1.5, 5.9, 5.0, HEX.lt2, `phan_bien_${i}`);
    text(s, q, { x: x + 0.3, y: 1.7, w: 5.3, h: 1.3, bold: true, fontSize: 16, color: c });
    text(s, a, { x: x + 0.3, y: 3.05, w: 5.3, h: 3.3, fontSize: 15 });
  });
  s.addNotes(T.ghi_chu.phan_bien);
};
SL.han_che = () => {
  CRUMB = "5. Kết luận và hướng phát triển";
  const s = addSlide("NOI_DUNG", "Thảo luận");
  s.addText("Hạn chế và hướng phát triển", { placeholder: "title" });
  card(s, M, 1.5, 5.9, 5.0, HEX.lt2, "han_che");
  text(s, [{ text: "Hạn chế", options: { bold: true, fontSize: 18, breakLine: true, color: HEX.accent5 } }, ...bullets(T.han_che, { fontSize: 15 })], { x: M + 0.3, y: 1.7, w: 5.3, h: 4.6 });
  card(s, M + 6.15, 1.5, 5.9, 5.0, HEX.lt2, "huong_phat_trien");
  text(s, [{ text: "Hướng phát triển", options: { bold: true, fontSize: 18, breakLine: true, color: HEX.accent2 } }, ...bullets(T.huong_phat_trien, { fontSize: 15 })], { x: M + 6.45, y: 1.7, w: 5.3, h: 4.6 });
  s.addNotes(T.ghi_chu.han_che);
};
SL.ket_luan = () => {
  CRUMB = null;
  const s = addSlide("TIEU_DE_TOI", "Kết luận");
  s.addText("Kết luận", { placeholder: "title" });
  s.addText(T.ket_luan.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < T.ket_luan.length - 1, paraSpaceAfter: 8 } })), { placeholder: "body" });
  s.addNotes(T.ghi_chu.ket_luan);
};

// ------------------------------------------------------------------ slide mới theo dàn ý
const kd = (a, b, tap = "toàn bộ") => (K.kiem_dinh || []).find((r) => r["mô hình 1"] === a && r["mô hình 2"] === b && r["tập"] === tap);
const sign = (x, d = 4) => (x == null ? "—" : (x >= 0 ? "+" : "−") + f(Math.abs(x), d));
const tableOpt = (o) => ({ fontFace: THEME.bodyFontFace, color: HEX.dk1, border: { type: "solid", pt: 0.5, color: "C9CFDB" }, ...o });
const head = (cols) => cols.map((h) => ({ text: h, options: { bold: true, fill: { color: HEX.lt2 } } }));

SL.tom_tat = () => {
  CRUMB = "Tóm tắt";
  const s = addSlide("NOI_DUNG");
  s.addText("Tóm tắt", { placeholder: "title" });
  const cols = [
    ["Bài toán", HEX.accent3, "Đồ thị tiên quyết do chuyên gia Junyi xác nhận có giúp truy vết tri thức không, và có hơn đồ thị suy từ dữ liệu không?"],
    ["Phương pháp", HEX.accent2, "GKT lan truyền hai chiều; cùng kiến trúc trên ba đồ thị 979 cạnh (chuyên gia, thống kê, ngẫu nhiên); giao thức chống rò rỉ với 108 khái niệm giữ lại."],
    ["Thực nghiệm", HEX.accent1, `${n0(D.n_nguoi_hoc)} người học, ${(D.n_tuong_tac / 1e6).toFixed(2).replace(".", ",")} triệu tương tác; tập kiểm tra 38.375 người học; AUC, ACC, RMSE, kiểm định DeLong.`]];
  cols.forEach(([h, c, t], i) => {
    const x = M + i * 4.1;
    card(s, x, 1.45, 3.8, 2.55, HEX.lt2, `tom_tat_${i}`);
    text(s, h, { x: x + 0.25, y: 1.6, w: 3.3, h: 0.45, bold: true, fontSize: 18, color: c });
    text(s, t, { x: x + 0.25, y: 2.1, w: 3.3, h: 1.8, fontSize: 14 });
  });
  const e = kd("GKT-Expert", "GKT-Stat"), r = kd("GKT-Stat", "GKT-Random"), d = kd("GKT-Expert", "DKT");
  const st = [[sign(e && e.delta), "AUC: Expert − Stat (cùng kiến trúc, cùng số cạnh)", HEX.accent1],
    [sign(r && r.delta), "AUC: Stat − Random (đồ thị thống kê ≈ ngẫu nhiên)", HEX.accent2],
    [sign(d && d.delta), "AUC: Expert − DKT (DKT vẫn cao nhất)", HEX.accent3]];
  st.forEach(([v, l, c], i) => stat(s, M + i * 4.1, 4.3, 3.8, v, l, c));
  s.addNotes("Tóm tắt toàn bộ đề tài trong một slide: câu hỏi, cách làm, dữ liệu và ba con số chính trên tập kiểm tra (phần 0).");
};

const MUC = [
  ["1", "Giới thiệu", "AIED · truy vết tri thức · phát biểu bài toán · đóng góp"],
  ["2", "Nghiên cứu liên quan", "ba nhóm phương pháp · tổng hợp và khoảng trống"],
  ["3", "Phương pháp", "tiền xử lý · nút cô lập · GKT tùy biến · thiết lập đồ thị · quy trình Colab"],
  ["4", "Thực nghiệm", "dữ liệu · độ đo · kết quả · case study · đối sánh"],
  ["5", "Kết luận và hướng phát triển", ""],
  ["P", "Phụ lục", "siêu tham số · tài liệu tham khảo"]];
SL.muc_luc = () => {
  CRUMB = null;
  const s = addSlide("NOI_DUNG");
  s.addText("Nội dung trình bày", { placeholder: "title" });
  MUC.forEach(([k, t, d], i) => {
    const y = 1.45 + i * 0.88;
    circle(s, M, y, 0.65, [HEX.accent3, HEX.accent2, HEX.accent1, HEX.accent5, HEX.dk2, HEX.accent4][i], k, `muc_${i}`);
    text(s, [{ text: t, options: { bold: true, fontSize: 20, breakLine: !!d } }, ...(d ? [{ text: d, options: { fontSize: 14, color: HEX.dk2 } }] : [])],
      { x: M + 0.95, y: y - 0.04, w: W - 2 * M - 0.95, h: 0.8, valign: "middle" });
  });
};

const CHUONG = {
  "1. Giới thiệu": "1.1 AIED · 1.2 Truy vết tri thức · 1.3 Phát biểu bài toán · 1.4 Mục tiêu và đóng góp",
  "2. Nghiên cứu liên quan": "2.1 Không dùng đồ thị · 2.2 Đồ thị suy từ dữ liệu · 2.3 Có quan hệ tiên quyết · 2.4 Tổng hợp",
  "3. Phương pháp": "3.1 Tiền xử lý · 3.2 Nút cô lập · 3.3 GKT tùy biến · 3.4 Thiết lập đồ thị · 3.5 Quy trình Colab",
  "4. Thực nghiệm": "4.1 Dữ liệu · 4.2 Độ đo · 4.3 Kết quả và case study · 4.4 Đối sánh, thảo luận",
  "5. Kết luận": "Hạn chế · hướng phát triển · kết luận",
  "Phụ lục": "A. Siêu tham số · Tài liệu tham khảo",
};
SL.chuong = () => {
  const s = addSlide("CHUONG");
  s.addText(CUR_SEC, { placeholder: "title" });
  s.addText(CHUONG[CUR_SEC] || "", { placeholder: "body" });
};

SL.aied = () => {
  CRUMB = "1.1 Trí tuệ nhân tạo trong giáo dục (AIED)";
  const s = addSlide("NOI_DUNG");
  s.addText("Mô hình người học là lõi của hệ thống học tập thích ứng", { placeholder: "title" });
  const c3 = [["Gia sư thông minh (ITS)", "hướng dẫn từng bước, phản hồi theo câu trả lời của người học", HEX.accent3],
    ["Học tập thích ứng", "chọn bài tập kế tiếp và lộ trình học theo năng lực hiện tại", HEX.accent2],
    ["Mô hình người học", "ước lượng mức thành thạo từng khái niệm; truy vết tri thức là bài toán lõi", HEX.accent1]];
  c3.forEach(([h, d, c], i) => {
    const x = M + i * 4.1;
    card(s, x, 1.5, 3.8, 2.4, HEX.lt2, `aied_${i}`);
    circle(s, x + 0.25, 1.7, 0.6, c, String(i + 1), `aied_so_${i}`);
    text(s, h, { x: x + 1.0, y: 1.7, w: 2.6, h: 0.6, bold: true, fontSize: 17, valign: "middle" });
    text(s, d, { x: x + 0.25, y: 2.5, w: 3.3, h: 1.3, fontSize: 15 });
  });
  card(s, M, 4.2, W - 2 * M, 2.3, "FBE7DA", "aied_junyi");
  text(s, [{ text: "Bối cảnh của đồ án", options: { bold: true, fontSize: 18, breakLine: true } },
    ...bullets(["Junyi Academy có bản đồ quan hệ tiên quyết giữa 835 bài tập, đã được đội ngũ sư phạm xác nhận (ví dụ: Đạo hàm trước Tích phân)",
      "DKT, DKVMN (đồ án 3–6) không khai thác cấu trúc này: quan hệ giữa khái niệm chỉ nằm ngầm trong trạng thái ẩn"], { fontSize: 15 })],
    { x: M + 0.3, y: 4.4, w: W - 2 * M - 0.6, h: 2.0 });
  s.addNotes("Đặt bài toán vào bối cảnh AIED: hệ thống gia sư thông minh và học tập thích ứng cần một mô hình người học tốt.");
};

SL.dong_gop = () => {
  CRUMB = "1.4 Mục tiêu và đóng góp";
  const s = addSlide("NOI_DUNG");
  s.addText("Đóng góp của đồ án", { placeholder: "title" });
  const items = [
    ["GKT lan truyền hai chiều", "hàm thông điệp riêng cho chiều xuôi và chiều ngược; danh sách cạnh nén, chi phí theo bậc thực tế"],
    ["Đánh giá tách bạch", "cùng kiến trúc, cùng 979 cạnh: đồ thị chuyên gia, thống kê, ngẫu nhiên; DKT, AKT làm cơ sở"],
    ["Giao thức chống rò rỉ", "đồ thị thống kê chỉ từ tập huấn luyện; 108 khái niệm giữ lại không có cạnh thống kê"],
    ["Phân tích dữ liệu", "đồ thị chuyên gia có chu trình và vòng tự thân; hành vi không phân biệt được chiều tiên quyết"],
    ["Quy trình tái lập", "6 notebook Colab, điểm kiểm tra trên Drive, kiểm thử đơn vị chống rò rỉ theo thời gian"]];
  items.forEach(([h, d], i) => {
    const y = 1.45 + i * 1.02;
    circle(s, M, y, 0.7, [HEX.accent1, HEX.accent2, HEX.accent3, HEX.accent5, HEX.dk2][i], String(i + 1), `dong_gop_${i}`);
    text(s, [{ text: h, options: { bold: true, fontSize: 18, breakLine: true } }, { text: d, options: { fontSize: 15, color: HEX.dk2 } }],
      { x: M + 1.0, y: y - 0.05, w: W - 2 * M - 1.0, h: 0.9, valign: "middle" });
  });
  s.addNotes("Năm đóng góp, mỗi đóng góp gắn với một phần thực nghiệm phía sau.");
};

const NC = [
  ["1", "AKT", "2020", "KDD (A*)", "không dùng", "—"], ["1", "simpleKT", "2023", "ICLR (A*)", "không dùng", "—"],
  ["2", "JKT", "2021", "Inf. Sci. (Q1)", "bài tập–bài tập, khái niệm–khái niệm", "không"],
  ["2", "Bi-CLKT", "2022", "KBS (Q1)", "bài tập–bài tập, học tương phản", "không"],
  ["2", "DGEKT", "2024", "ACM TOIS (Q1)", "siêu đồ thị + chuyển tiếp", "không"],
  ["2", "HHSKT", "2023", "ESWA (Q1)", "không thuần nhất", "không"],
  ["2", "GIKT", "2020", "ECML-PKDD (A)", "câu hỏi–kỹ năng", "không"],
  ["3", "SKT", "2020", "ICDM (A*)", "tiên quyết + tương tự", "một phần"],
  ["3", "PSI-KT", "2024", "ICLR (A*)", "tiên quyết suy từ dữ liệu", "không"]];
SL.tong_hop_nc = () => {
  CRUMB = "2.4 Tổng hợp";
  const s = addSlide("NOI_DUNG");
  s.addText("Tổng hợp và khoảng trống nghiên cứu", { placeholder: "title" });
  const rows = [head(["Nhóm", "Công trình", "Năm", "Nơi công bố", "Loại đồ thị", "Đồ thị chuyên gia"]),
    ...NC.map((r) => r.map((c) => ({ text: c }))),
    ["—", "GKT-Expert (đồ án)", "2026", "—", "tiên quyết chuyên gia, hai chiều", "có"].map((c) => ({ text: c, options: { bold: true, color: HEX.accent1 } }))];
  s.addTable(rows, tableOpt({ x: M, y: 1.4, w: 8.6, colW: [0.7, 1.7, 0.7, 1.6, 2.6, 1.3], fontSize: 12 }));
  card(s, M + 8.9, 1.4, W - 2 * M - 8.9, 5.1, "FBE7DA", "khoang_trong");
  text(s, [{ text: "Khoảng trống", options: { bold: true, fontSize: 18, breakLine: true } },
    { text: "Chưa có đánh giá tách bạch đồ thị chuyên gia với đồ thị thống kê trong cùng một kiến trúc GKT, cùng số cạnh, kèm giao thức chống rò rỉ trên khái niệm giữ lại.", options: { fontSize: 15 } }],
    { x: M + 9.15, y: 1.6, w: W - 2 * M - 9.4, h: 4.7 });
  s.addNotes("Bảng tổng hợp mục 2.4 của dàn ý: các công trình từ 2020, nơi công bố và loại đồ thị. Không công trình nào dùng đồ thị tiên quyết chuyên gia một cách tách bạch.");
};

SL.nut_co_lap = () => {
  CRUMB = "3.2 Xử lý nút cô lập";
  const s = addSlide("NOI_DUNG");
  s.addText(`${G.n_isolated} nút cô lập: hai cách xử lý`, { placeholder: "title" });
  const c2 = [["(a) Tự cập nhật — mặc định", HEX.accent2, ["nút cô lập chỉ nhận thông điệp tự thân f_self", "tương đương thêm vòng tự thân, như DKT trên từng nút", "không thêm tham số"]],
    ["(b) Nút gốc ảo", HEX.accent3, ["thêm một nút nối tới mọi khái niệm có bậc vào bằng 0", "biểu diễn năng lực chung của người học", "so sánh bằng nghiên cứu cắt bỏ (mục 4.3)"]]];
  c2.forEach(([h, c, b], i) => {
    const x = M + i * 6.15;
    card(s, x, 1.5, 5.9, 3.4, HEX.lt2, `co_lap_${i}`);
    text(s, h, { x: x + 0.3, y: 1.7, w: 5.3, h: 0.5, bold: true, fontSize: 19, color: c });
    text(s, bullets(b, { fontSize: 16 }), { x: x + 0.3, y: 2.35, w: 5.3, h: 2.4 });
  });
  card(s, M, 5.15, W - 2 * M, 1.35, "FBE7DA", "chuan_hoa");
  text(s, [{ text: "Chuẩn hóa an toàn: ", options: { bold: true } },
    { text: "trọng số lân cận bằng nghịch đảo số lân cận cùng chiều; nút bậc 0 không có lân cận nên không sinh phép chia cho 0 (không NaN)." }],
    { x: M + 0.3, y: 5.3, w: W - 2 * M - 0.6, h: 1.1, fontSize: 16, valign: "middle" });
  s.addNotes("Đề bài yêu cầu tự quyết định cách xử lý nút cô lập. Đồ án chọn tự cập nhật làm mặc định và kiểm chứng phương án nút gốc ảo bằng nghiên cứu cắt bỏ.");
};

SL.quy_trinh = () => {
  CRUMB = "3.5 Quy trình thực nghiệm trên Google Colab";
  const s = addSlide("NOI_DUNG");
  s.addText("Quy trình tái lập trên Google Colab (GPU T4)", { placeholder: "title" });
  const nb = [["01", "Tải dữ liệu", "Junyi, kiểm tra SHA256"], ["02", "Tiền xử lý", "chuỗi, node_id, đồ thị"], ["03", "Đồ thị", "thống kê, ngẫu nhiên, giữ lại"],
    ["04", "Huấn luyện", "GKT ×3, DKT, AKT, cắt bỏ"], ["05", "Phân tích", "bảng, DeLong, case study"], ["06", "Xuất báo cáo", "PDF IEEE, Word, slide"]];
  nb.forEach(([k, h, d], i) => {
    const x = M + i * 2.04;
    s.addShape(pres.shapes.CHEVRON, { x, y: 1.6, w: 1.95, h: 1.0, fill: { color: [HEX.accent3, HEX.accent3, HEX.accent2, HEX.accent1, HEX.accent2, HEX.dk2][i] },
      line: { color: HEX.lt1 }, objectName: `notebook_${k}` });
    text(s, k, { x: x + 0.35, y: 1.6, w: 1.3, h: 1.0, align: "center", valign: "middle", bold: true, fontSize: 20, color: HEX.lt1 });
    text(s, [{ text: h, options: { bold: true, breakLine: true } }, { text: d, options: { fontSize: 12, color: HEX.dk2 } }],
      { x, y: 2.75, w: 1.95, h: 0.9, fontSize: 14, align: "center" });
  });
  text(s, bullets(["Mã dùng chung trong src/graphk/; notebook chỉ gọi hàm, sinh từ scripts/tao_notebook.py",
    "Điểm kiểm tra lưu lên Google Drive sau mỗi epoch; phiên bị ngắt thì chạy lại ô để học tiếp",
    "Adam 10⁻³, cắt chuẩn gradient 1,0, tối đa 10 epoch, dừng sớm sau 3 epoch không cải thiện AUC kiểm định",
    "41 ca kiểm thử đơn vị, gồm kiểm thử dự đoán ở bước t không phụ thuộc kết quả từ bước t+1"], { fontSize: 16 }),
    { x: M, y: 4.0, w: W - 2 * M, h: 2.6 });
  s.addNotes("Mục 3.5 của dàn ý: sáu notebook chạy tuần tự trên Colab, kết quả lưu trên Drive. Mã nguồn tại github.com/Trilosphan/GraphK.");
};

SL.do_do = () => {
  CRUMB = "4.2 Độ đo đánh giá";
  const s = addSlide("NOI_DUNG");
  s.addText("Độ đo và kiểm định", { placeholder: "title" });
  const c3 = [["Độ đo chính", HEX.accent1, ["AUC-ROC của dự đoán bước kế tiếp", "trên 38.375 người học chưa thấy khi huấn luyện"]],
    ["Độ đo phụ", HEX.accent2, ["ACC (ngưỡng 0,5)", "RMSE"]],
    ["Phân tích tách bạch", HEX.accent3, ["AUC trên 108 khái niệm giữ lại và khái niệm còn lại", "AUC theo nhóm: cô lập, bậc thấp, bậc cao", "thời gian huấn luyện"]]];
  c3.forEach(([h, c, b], i) => {
    const x = M + i * 4.1;
    card(s, x, 1.5, 3.8, 2.9, HEX.lt2, `do_do_${i}`);
    text(s, h, { x: x + 0.25, y: 1.65, w: 3.3, h: 0.5, bold: true, fontSize: 18, color: c });
    text(s, bullets(b, { fontSize: 15 }), { x: x + 0.25, y: 2.2, w: 3.3, h: 2.1 });
  });
  card(s, M, 4.65, W - 2 * M, 1.85, "FBE7DA", "delong");
  text(s, [{ text: "Kiểm định DeLong ghép cặp", options: { bold: true, fontSize: 17, breakLine: true } },
    { text: "So hiệu AUC của hai mô hình trên cùng các dự đoán (cùng thứ tự tất định). Kiểm định giả định dự đoán độc lập, bỏ qua tương quan trong cùng người học; với hàng triệu dự đoán, p rất nhỏ ngay cả khi hiệu nhỏ, nên đọc kèm độ lớn hiệu.", options: { fontSize: 15 } }],
    { x: M + 0.3, y: 4.8, w: W - 2 * M - 0.6, h: 1.6 });
  s.addNotes("Mục 4.2 của dàn ý. Dàn ý ban đầu dự kiến trung bình ± độ lệch chuẩn qua 5 phần; hiện mới có phần 0 nên dùng kiểm định DeLong ghép cặp.");
};

SL.nhom_kn = () => {
  CRUMB = "4.3 Kết quả: nhóm khái niệm và kiểm định";
  const s = addSlide("NOI_DUNG");
  s.addText("Lợi thế của đồ thị chuyên gia nằm ở nút có cạnh", { placeholder: "title" });
  const b4 = K.bang4 || [];
  const mods = MAIN.filter((m) => b4.length && b4[0][m] != null);
  s.addTable([head(["Nhóm khái niệm", ...mods]), ...b4.map((r) => [r["nhóm"], ...mods.map((m) => f(r[m], 4))])],
    tableOpt({ x: M, y: 1.45, w: 6.6, colW: [2.2, ...Array(mods.length).fill(4.4 / mods.length)], fontSize: 12 }));
  const tests = (K.kiem_dinh || []).filter((r) => r["tập"] === "toàn bộ" || r["tập"] === "khái niệm giữ lại");
  s.addTable([head(["Tập", "So sánh", "Hiệu AUC", "p"]),
    ...tests.map((r) => [r["tập"] === "toàn bộ" ? "toàn bộ" : "giữ lại", `${r["mô hình 1"]} − ${r["mô hình 2"]}`, sign(r.delta),
      r.p_value < 1e-4 ? "< 10⁻⁴" : f(r.p_value, 3)])],
    tableOpt({ x: M + 6.9, y: 1.45, w: W - 2 * M - 6.9, colW: [0.95, 2.55, 1.0, 0.73], fontSize: 11 }));
  text(s, "Trên khái niệm cô lập trong đồ thị chuyên gia, ba biến thể GKT gần như bằng nhau: nút cô lập không nhận thông điệp lân cận, nên lợi thế chỉ xuất hiện ở nút có cạnh, đúng cơ chế lan truyền.",
    { x: M, y: 5.6, w: W - 2 * M, h: 0.9, fontSize: 15, color: HEX.dk2 });
  s.addNotes("Bảng V và Bảng VI của báo cáo. Hiệu AUC khoảng 0,003 có p rất nhỏ nhưng cần đọc kèm độ lớn hiệu.");
};

SL.sieu_tham_so = () => {
  CRUMB = "Phụ lục A. Siêu tham số";
  const s = addSlide("NOI_DUNG");
  s.addText("Siêu tham số", { placeholder: "title" });
  s.addTable([head(["Mô hình", "Tham số chính", "Cỡ lô", "Tối ưu"]),
    ["GKT (ba đồ thị)", "ẩn 32, nhúng 32, dropout 0,1, chuẩn hóa theo hàng, tự cập nhật nút cô lập; 129.701 tham số", "128", "Adam 10⁻³, cắt gradient 1,0, tối đa 10 epoch, dừng sớm sau 3 epoch"],
    ["DKT", "nhúng 64, LSTM 64, dropout 0,1; 194.435 tham số", "128", "như trên"],
    ["AKT", "d_model 256, 4 khối, 8 head, dropout 0,2 (cài đặt pyKT); 4.568.417 tham số", "64", "như trên"],
    ["Đồ thị thống kê", "ngưỡng thành thạo 0,5; hỗ trợ tối thiểu 50; ưu tiên 1 cạnh vào mỗi nút; 979 cạnh", "—", "—"]],
    tableOpt({ x: M, y: 1.5, w: W - 2 * M, colW: [2.2, 5.6, 1.0, 3.33], fontSize: 14 }));
};

SL.tai_lieu = () => {
  CRUMB = "Tài liệu tham khảo";
  const s = addSlide("NOI_DUNG");
  s.addText("Tài liệu tham khảo chính", { placeholder: "title" });
  const refs = T.tai_lieu_slide || [];
  const half = Math.ceil(refs.length / 2);
  [refs.slice(0, half), refs.slice(half)].forEach((col, j) => text(s, col.map((r, i) => ({ text: r, options: { breakLine: i < col.length - 1, paraSpaceAfter: 5 } })),
    { x: M + j * 6.15, y: 1.45, w: 5.9, h: 5.2, fontSize: 12 }));
  s.addNotes("Danh mục đầy đủ (25 tài liệu, định dạng IEEE) ở cuối bài báo.");
};

// ------------------------------------------------------------------ thứ tự theo dàn ý (docs/DAN_Y.md)
const DAN_Y = [
  ["Mở đầu", ["bia", "tom_tat", "muc_luc"]],
  ["1. Giới thiệu", ["chuong", "aied", "truy_vet", "cau_hoi", "dong_gop"]],
  ["2. Nghiên cứu liên quan", ["chuong", "ba_nhom", "tong_hop_nc"]],
  ["3. Phương pháp", ["chuong", "tien_xu_ly", "nut_co_lap", "kien_truc", "ba_do_thi", "hinh2", "quy_trinh"]],
  ["4. Thực nghiệm", ["chuong", "du_lieu", "do_do", "ket_qua", "giu_lai", "nhom_kn", "cat_bo", "case_chuoi", "case_buoc", "case_hinh", "case_tom", "phan_bien"]],
  ["5. Kết luận", ["chuong", "han_che", "ket_luan"]],
  ["Phụ lục", ["chuong", "sieu_tham_so", "tai_lieu"]],
];
DAN_Y.forEach(([section, slides]) => { CUR_SEC = section; slides.forEach((k) => { CRUMB = null; SL[k](); }); });

(async () => {
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  await pres.writeFile({ fileName: outPath });
  console.log("đã ghi", outPath);
})();
