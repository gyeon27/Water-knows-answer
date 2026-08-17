import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir = process.argv[2] ?? "outputs/game_runtime_comparison";
await fs.mkdir(outputDir, { recursive: true });

const benchmark = JSON.parse(await fs.readFile("phase3/results_summary/continuous_runtime_benchmark.json", "utf8"));
const simple = benchmark.rows.find((row) => row.mode === "simple");
const ours = benchmark.rows.find((row) => row.mode === "ours");

const runtimeRows = [
  ["External DFSPH teacher", "DFSPH / external teacher", 29.37, null, 444, 0, 0.06248, 15.5566, null, null, "Palouse DEM; 100 s; 444 peak particles; speed reference only"],
  ["Basic physics", "SWE + gravity/collision particles", simple.simulation_fps, simple.p95_frame_ms, simple.active_peak, simple.penetration_rate, simple.minimum_clearance_m, simple.max_speed_mps, simple.swe_mass_relative_error, simple.particle_balance_error, "Phase-2 terrain; 100 s; same seed and source as Ours"],
  ["Optimized Ours", "SWE + SPLASH ROI PI-GNN (15 Hz correction)", ours.simulation_fps, ours.p95_frame_ms, ours.active_peak, ours.penetration_rate, ours.minimum_clearance_m, ours.max_speed_mps, ours.swe_mass_relative_error, ours.particle_balance_error, "Phase-2 terrain; 100 s; same seed and source as Basic physics"],
];

const runtimeCsv = [
  ["method", "architecture", "fps", "p95_frame_ms", "active_peak", "penetration_rate", "minimum_clearance_m", "max_speed_mps", "swe_mass_relative_error", "particle_balance_error", "protocol_note"],
  ...runtimeRows,
].map((row) => row.map((v) => v == null ? "" : `"${String(v).replaceAll('"', '""')}"`).join(",")).join("\n");
await fs.writeFile(path.join(outputDir, "game_runtime_comparison.csv"), runtimeCsv, "utf8");

const workbook = Workbook.create();
const dash = workbook.worksheets.add("Dashboard");
const sevenDash = workbook.worksheets.add("7-Condition Dashboard");
const raw = workbook.worksheets.add("Raw Runtime");
const accuracy = workbook.worksheets.add("Accuracy");
const protocol = workbook.worksheets.add("Protocol");
const sevenRuntime = workbook.worksheets.add("7-Condition Runtime");
const sevenAccuracy = workbook.worksheets.add("7-Condition Accuracy");
for (const sheet of [dash, sevenDash, raw, accuracy, protocol, sevenRuntime, sevenAccuracy]) sheet.showGridLines = false;

const parseSimpleCsv = (text) => text.trim().split(/\r?\n/).map((line) => line.replace(/^\uFEFF/, "").split(","));
const sevenRuntimeRows = parseSimpleCsv(await fs.readFile("phase3/results_summary/game_7condition_runtime.csv", "utf8"));
const sevenAccuracyRows = parseSimpleCsv(await fs.readFile("phase3/results_summary/game_7condition_accuracy.csv", "utf8"));
const numericRuntime = sevenRuntimeRows.map((row, index) => index ? row.map((v, col) => col >= 2 && col <= 9 ? Number(v) : col >= 10 ? v === "True" || v === "true" : v) : row);
const numericAccuracy = sevenAccuracyRows.map((row, index) => index ? row.map((v, col) => col >= 2 ? Number(v) : v) : row);
sevenRuntime.getRangeByIndexes(0, 0, numericRuntime.length, numericRuntime[0].length).values = numericRuntime;
sevenAccuracy.getRangeByIndexes(0, 0, numericAccuracy.length, numericAccuracy[0].length).values = numericAccuracy;
for (const sheet of [sevenRuntime, sevenAccuracy]) {
  const used = sheet.getUsedRange();
  used.getRow(0).format = { fill: "#20364B", font: { bold: true, color: "#FFFFFF" }, wrapText: true };
  used.format.font = { name: "Aptos", size: 9 };
  used.format.autofitColumns();
  sheet.freezePanes.freezeRows(1);
}
sevenRuntime.getRange("A:B").format.columnWidth = 20;
sevenRuntime.getRange("F:J").format.numberFormat = "0.00";
sevenAccuracy.getRange("D:G").format.numberFormat = "0.0000";

sevenDash.getRange("A1:J2").merge();
sevenDash.getRange("A1").values = [["7조건 게임 벤치마크 — 5,000입자, 동일 코드·동일 GPU"]];
sevenDash.getRange("A1:J2").format = { fill: "#20364B", font: { bold: true, color: "#FFFFFF", size: 18 }, verticalAlignment: "center" };
sevenDash.getRange("A4:D8").values = [
  ["SPLASH ratio", "Optimized Ours FPS", "Speedup vs GNN-only", "p95 60 FPS pass"],
  [0.05, 394.48702826203936, 13.304091514596275, "PASS"],
  [0.25, 251.61806456651007, 8.477354398434423, "PASS"],
  [0.50, 141.97310036388797, 4.805634445732991, "PASS"],
  [1.00, 58.36474906483684, 1.9785349007173856, "FAIL"],
];
sevenDash.getRange("A4:D4").format = { fill: "#DCEAF5", font: { bold: true, color: "#20364B" } };
sevenDash.getRange("A5:A8").format.numberFormat = "0%";
sevenDash.getRange("B5:C8").format.numberFormat = "0.00";
sevenDash.getRange("A:A").format.columnWidth = 18; sevenDash.getRange("B:B").format.columnWidth = 23;
sevenDash.getRange("C:D").format.columnWidth = 27;
sevenDash.getRange("F4:I8").values = [
  ["SPLASH ratio", "GNN-only", "Ours", "Optimized Ours"],
  ["5%", 29.651557, 183.674977, 394.487028],
  ["25%", 29.681202, 157.800851, 251.618065],
  ["50%", 29.543050, 75.488259, 141.973100],
  ["100%", 29.498974, 29.410087, 58.364749],
];
sevenDash.getRange("F:F").format.columnWidth = 17;
sevenDash.getRange("G:I").format.columnWidth = 20;
sevenDash.getRange("F4:I4").format = { fill: "#DCEAF5", font: { bold: true, color: "#20364B" } };
sevenDash.getRange("G5:I8").format.numberFormat = "0.00";
const sevenFpsChart = sevenDash.charts.add("bar", sevenDash.getRange("F4:I8"));
sevenFpsChart.title = "학습 기반 방식 처리속도 (FPS)"; sevenFpsChart.hasLegend = true;
sevenFpsChart.xAxis = { axisType: "textAxis" }; sevenFpsChart.yAxis = { numberFormatCode: "0" };
sevenFpsChart.setPosition("A11", "J27");
sevenDash.getRange("A29:E36").values = [
  ["Condition", "32-step position RMSE", "Penetration", "Density error", "Energy excess"],
  ...numericAccuracy.slice(1).map((r) => [r[0] + " " + r[1], r[3], r[4], r[5], r[6]]),
];
sevenDash.getRange("A29:E29").format = { fill: "#DCEAF5", font: { bold: true, color: "#20364B" } };
sevenDash.getRange("B30:E36").format.numberFormat = "0.0000";
sevenDash.getRange("A38:J41").merge();
sevenDash.getRange("A38").values = [["결론: Optimized Ours는 SPLASH가 국소적인 5–50% 구간에서 학습 기반 모델 중 가장 빠르고 p95 60 FPS를 통과한다. 100% SPLASH에서는 통과하지 못한다. Simple-3D와 SWE-only는 더 빠르지만 32-step 정확도 또는 물리 표현력이 낮으므로 속도 단독 우위로 제안 모델보다 낫다고 해석하지 않는다."]];
sevenDash.getRange("A38:J41").format = { fill: "#FFF4D6", font: { color: "#6B4F00", size: 10 }, wrapText: true, verticalAlignment: "center" };

raw.getRange("A1:K4").values = [["Method", "Architecture", "FPS", "p95 frame (ms)", "Peak active particles", "Penetration rate", "Minimum clearance (m)", "Max speed (m/s)", "SWE relative mass error", "Particle balance error", "Protocol note"], ...runtimeRows];
raw.getRange("A1:K1").format = { fill: "#20364B", font: { bold: true, color: "#FFFFFF" }, wrapText: true };
raw.getRange("A2:K4").format.borders = { preset: "inside", style: "thin", color: "#D6DEE5" };
raw.getRange("C2:D4").format.numberFormat = "0.00";
raw.getRange("F2:F4").format.numberFormat = "0.00%";
raw.getRange("G2:I4").format.numberFormat = "0.0000";
raw.getRange("A:K").format.font = { name: "Aptos", size: 10 };
raw.getRange("A:A").format.columnWidth = 23;
raw.getRange("B:B").format.columnWidth = 38;
raw.getRange("C:J").format.columnWidth = 18;
raw.getRange("K:K").format.columnWidth = 58;
raw.getRange("K2:K4").format.wrapText = true;
raw.freezePanes.freezeRows(1);

accuracy.getRange("A1:D4").values = [
  ["Method", "1-step position RMSE (m)", "Evaluation", "Interpretation"],
  ["Basic physics", 0.02650424, "Palouse frames 151–300", "Analytic baseline"],
  ["Raw Water-3D model", 0.02678711, "Palouse frames 151–300", "Zero-shot raw residual"],
  ["Optimized Ours", 0.02577524, "Palouse frames 151–300", "Validation-calibrated residual blend"],
];
accuracy.getRange("A1:D1").format = { fill: "#20364B", font: { bold: true, color: "#FFFFFF" } };
accuracy.getRange("B2:B4").format.numberFormat = "0.00000";
accuracy.getRange("A:A").format.columnWidth = 25;
accuracy.getRange("B:B").format.columnWidth = 25;
accuracy.getRange("C:D").format.columnWidth = 34;
accuracy.getRange("F1:G3").values = [["Derived metric", "Value"], ["Ours RMSE reduction vs basic", null], ["Ours FPS gain vs prior 84.732 FPS", null]];
accuracy.getRange("G2").formulas = [["=1-B4/B2"]];
accuracy.getRange("G3").formulas = [[`=${ours.simulation_fps}/84.73227639723211-1`]];
accuracy.getRange("G2:G3").format.numberFormat = "0.0%";
accuracy.getRange("F1:G1").format = { fill: "#DCEAF5", font: { bold: true, color: "#20364B" } };

dash.getRange("A1:H2").merge();
dash.getRange("A1").values = [["게임 런타임 비교 — Teacher vs 기본 물리 vs Optimized Ours"]];
dash.getRange("A1:H2").format = { fill: "#20364B", font: { bold: true, color: "#FFFFFF", size: 18 }, verticalAlignment: "center", horizontalAlignment: "left" };
dash.getRange("A4:B7").values = [["KPI", "Value"], ["Ours FPS", ours.simulation_fps], ["Ours p95 frame", ours.p95_frame_ms], ["Ours penetration", ours.penetration_rate]];
dash.getRange("D4:E7").values = [["KPI", "Value"], ["60 FPS p95 budget", 16.6667], ["RMSE vs basic reduction", null], ["100 s particle balance", ours.particle_balance_error]];
dash.getRange("E6").formulas = [["=1-Accuracy!B4/Accuracy!B2"]];
dash.getRange("A4:B4").format = dash.getRange("D4:E4").format = { fill: "#DCEAF5", font: { bold: true, color: "#20364B" } };
dash.getRange("B5:B6").format.numberFormat = "0.00";
dash.getRange("B7").format.numberFormat = "0.00%";
dash.getRange("E5").format.numberFormat = "0.00";
dash.getRange("E6").format.numberFormat = "0.0%";
dash.getRange("A:A").format.columnWidth = 28; dash.getRange("B:B").format.columnWidth = 18;
dash.getRange("D:D").format.columnWidth = 28; dash.getRange("E:E").format.columnWidth = 18;

dash.getRange("A10:B13").values = [["Method", "FPS"], ...runtimeRows.map((r) => [r[0], r[2]])];
const fpsChart = dash.charts.add("bar", dash.getRange("A10:B13"));
fpsChart.title = "장시간 실행 처리속도 (FPS)"; fpsChart.hasLegend = false;
fpsChart.xAxis = { axisType: "textAxis" }; fpsChart.yAxis = { numberFormatCode: "0" };
fpsChart.setPosition("D9", "K24");

dash.getRange("A26:B26").values = [["Method", "Position RMSE (m)"]];
dash.getRange("A27:B29").formulas = [["=Accuracy!A2", "=Accuracy!B2"], ["=Accuracy!A3", "=Accuracy!B3"], ["=Accuracy!A4", "=Accuracy!B4"]];
const accChart = dash.charts.add("bar", dash.getRange("A26:B29"));
accChart.title = "Palouse held-out 1-step 위치 RMSE (낮을수록 좋음)"; accChart.hasLegend = false;
accChart.xAxis = { axisType: "textAxis" }; accChart.yAxis = { numberFormatCode: "0.000" };
accChart.setPosition("D25", "K40");

dash.getRange("A42:K45").merge();
dash.getRange("A42").values = [["해석 주의: DFSPH teacher의 FPS는 Palouse DEM·444입자 외부 솔버 기준이라 Basic/Ours와 직접적인 동일조건 속도 비교가 아니다. 동일조건 성능 주장은 Basic과 Ours 사이에서만 한다. Ours의 가치 주장은 단순 물리보다 빠르다는 것이 아니라, full GNN보다 계산 영역을 줄이면서 기본 물리보다 teacher 오차를 낮추는 정확도–속도 절충이다."]];
dash.getRange("A42:K45").format = { fill: "#FFF4D6", font: { color: "#6B4F00", size: 10 }, wrapText: true, verticalAlignment: "center" };

protocol.getRange("A1:B8").values = [
  ["Item", "Definition"],
  ["Continuous benchmark", "100 s / 3000 physics frames / seed 20260809 / constant source / no LOD"],
  ["Basic vs Ours", "Same Phase-2 terrain, source, initial state, particle budget and collision solver"],
  ["Ours inference", "SWE for STREAM/POOL; PI-GNN only for SPLASH ROI; neural correction refresh every 2 frames"],
  ["Teacher timing", "External SPlisHSPlasH DFSPH, Palouse DEM, 100 simulated seconds; reference only"],
  ["Accuracy evaluation", "Palouse external teacher, held-out frames 151–300, one-step teacher-forced"],
  ["Runtime source", "phase3/results_summary/continuous_runtime_benchmark.json"],
  ["Accuracy source", "phase3/external_teacher/evaluation_palouse_water3d_ours/test_summary.json and calibrated GUI cache"],
];
protocol.getRange("A1:B1").format = { fill: "#20364B", font: { bold: true, color: "#FFFFFF" } };
protocol.getRange("A:A").format.columnWidth = 28; protocol.getRange("B:B").format.columnWidth = 95;
protocol.getRange("B2:B8").format.wrapText = true;

const preview = await workbook.render({ sheetName: "Dashboard", autoCrop: "all", scale: 1, format: "png" });
await fs.writeFile(path.join(outputDir, "game_runtime_dashboard.png"), new Uint8Array(await preview.arrayBuffer()));
const sevenPreview = await workbook.render({ sheetName: "7-Condition Dashboard", autoCrop: "all", scale: 1, format: "png" });
await fs.writeFile(path.join(outputDir, "game_7condition_dashboard.png"), new Uint8Array(await sevenPreview.arrayBuffer()));
const inspection = await workbook.inspect({ kind: "formula", sheetId: "Dashboard", range: "A1:K45", maxChars: 4000 });
await fs.writeFile(path.join(outputDir, "workbook_inspection.json"), inspection.ndjson ?? JSON.stringify(inspection), "utf8");
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(path.join(outputDir, "game_runtime_comparison.xlsx"));
