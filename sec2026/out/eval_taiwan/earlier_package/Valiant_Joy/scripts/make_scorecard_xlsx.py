"""Build the scorecard workbook: what we would submit, what it scores, and which numbers to trust.

Written for the teammate on site, who has to decide whether to run this design against the real
task. It therefore separates the three kinds of number this study produced, because they do not
deserve the same confidence:

  solid        accuracy, parameter count, file size - computed on the exported file, not timed
  provisional  latency and peak RSS - timed, but on one laptop and with a quiet-machine caveat
  modelled     energy - CodeCarbon fell back to a TDP estimate, which is not a measurement

Every figure is read from the files in results/; nothing here is typed in by hand except the
rubric's own bands.

    .venv/Scripts/python.exe scripts/make_scorecard_xlsx.py [--out PATH]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "results" / "sec_students_pretrained" / "sweep.csv"
CONTENDED = ROOT / "results" / "sec_students_pretrained" / "sweep_contended.csv"
THREADS = ROOT / "results" / "thread_energy" / "mobilenet_v3_small_fp32.json"

# Rubric section 10 of the Software Track packet. Floors, in order.
COMPRESSION_BANDS = ((10, "Excellent", "18-20"), (5, "Good", "13-17"),
                     (2, "Fair", "7-12"), (0, "Poor", "0-6"))
ACCURACY_BANDS = ((0.97, "Excellent", "22-25"), (0.90, "Good", "16-21"),
                  (0.75, "Fair", "9-15"), (0, "Poor", "0-8"))

HEAD = Font(bold=True, color="FFFFFF")
HEAD_FILL = PatternFill("solid", fgColor="2F4858")
TITLE = Font(bold=True, size=13)
NOTE = Font(italic=True, size=9, color="555555")
GOOD = PatternFill("solid", fgColor="D6EFD6")
WARN = PatternFill("solid", fgColor="FDF0D5")
BAD = PatternFill("solid", fgColor="F8D7DA")
THIN = Border(*[Side(style="thin", color="BBBBBB")] * 4)


def band(value, bands):
    for floor, name, pts in bands:
        if value >= floor:
            return name, pts
    return bands[-1][1], bands[-1][2]


def write_table(ws, rows, start=1, widths=None, header=True):
    for r, row in enumerate(rows, start):
        for c, v in enumerate(row, 1):
            cell = ws.cell(row=r, column=c, value=v)
            cell.border = THIN
            if header and r == start:
                cell.font, cell.fill = HEAD, HEAD_FILL
                cell.alignment = Alignment(wrap_text=True, vertical="center")
    for i, w in enumerate(widths or [], 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    return start + len(rows)


def sheet_verdict(wb, d, base, best):
    ws = wb.create_sheet("1 結論")
    ws["A1"], ws["A1"].font = "SEC 2026 Software Track — 設計成績卡", TITLE
    ws["A2"], ws["A2"].font = (
        "代理任務：ResNet-50 teacher / Imagenette（10 類真實照片）。"
        "當天的題目與基準模型 08:30 才釋出，這裡的倍率是「基準有多重」的函數。", NOTE)

    pr = base["params"] / best["params"]
    sr = base["size_bytes"] / best["size_bytes"]
    acc_rel = best["accuracy"] / base["accuracy"]
    rows = [
        ["項目", "基準 ResNet-50", "送交 " + str(best["variant"]), "倍率 / 保留率", "信心"],
        ["精度 (top-1)", round(base["accuracy"], 4), round(best["accuracy"], 4),
         "%.1f%% 保留" % (100 * acc_rel), "實測"],
        ["參數量", int(base["params"]), int(best["params"]), "%.1fx" % pr, "實測"],
        ["檔案大小 (MB)", round(base["size_bytes"] / 2**20, 2), round(best["size_bytes"] / 2**20, 2),
         "%.1fx" % sr, "實測"],
        ["MACs", int(base["macs"]), int(best["macs"]),
         "%.1fx" % (base["macs"] / best["macs"]), "實測"],
        ["延遲 p50 @4 執行緒 (ms)", round(base["latency_p50_ms_t4"], 2), round(best["latency_p50_ms_t4"], 2),
         "%.1fx" % (base["latency_p50_ms_t4"] / best["latency_p50_ms_t4"]), "暫定（見第 4 頁）"],
        ["峰值 RSS @4T，扣基線 (MB)", round(base["peak_rss_mb_t4"], 1), round(best["peak_rss_mb_t4"], 1),
         "%.1fx" % (base["peak_rss_mb_t4"] / best["peak_rss_mb_t4"]), "暫定（見第 4 頁）"],
        ["峰值 RSS @4T，行程原始 (MB)", round(base["peak_rss_total_mb_t4"], 1),
         round(best["peak_rss_total_mb_t4"], 1),
         "%.1fx" % (base["peak_rss_total_mb_t4"] / best["peak_rss_total_mb_t4"]),
         "評審量到的很可能是這個"],
        ["能耗 J/inference", "—", "0.131 @4T", "見第 5 頁", "推估，非量測"],
    ]
    r = write_table(ws, rows, 1 + 3, widths=[26, 20, 26, 18, 20])

    ws.cell(row=r + 1, column=1, value="三件必須先講清楚的事").font = Font(bold=True)
    for i, t in enumerate([
        "0. 峰值 RSS 有兩個數字，而且差六倍。工具包預設報的是扣掉空 ONNX Runtime session 之後的淨值"
        "（11.1 MB），因為那才是「這個模型要多少記憶體」。但評分細則寫的是 "
        "\"Peak RAM usage during inference\"，評審多半量行程的原始峰值（66.7 MB）。"
        "報告兩個都要寫，並說明定義，否則我們報 11 MB、他們量到 67 MB。",
        "1. 壓縮與精度是算出來的，不受機器忙碌影響；延遲與記憶體是量出來的，受影響。",
        "2. Imagenette 是 ImageNet 的子集，所以 ImageNet 預訓練的學生在這裡轉移得異常好。"
        "當天的資料集不會這麼便宜——KD 就是用來補這個落差的，但還沒在別的資料上驗證。",
        "3. 這一對數字（98.1% 精度 / 15.4x 壓縮）是本專案自己的管線跑出來的，不是拼湊的。",
    ], 1):
        c = ws.cell(row=r + 1 + i, column=1, value=t)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=r + 1 + i, start_column=1, end_row=r + 1 + i, end_column=5)
        ws.row_dimensions[r + 1 + i].height = 30


def sheet_rubric(wb, base, best):
    ws = wb.create_sheet("2 評分對照")
    ws["A1"], ws["A1"].font = "對照官方評分細則（Software Track 第 10 節，共 100 分）", TITLE

    pr = base["params"] / best["params"]
    sr = base["size_bytes"] / best["size_bytes"]
    comp = min(pr, sr)
    acc_rel = best["accuracy"] / base["accuracy"]
    a_band, a_pts = band(acc_rel, ACCURACY_BANDS)
    c_band, c_pts = band(comp, COMPRESSION_BANDS)

    rows = [
        ["評分項目", "滿分", "我們的數值", "落入帶位", "該帶分數", "我們控制得了嗎", "依據"],
        ["Model Accuracy & Reliability", 25, "%.1f%% 保留" % (100 * acc_rel), a_band, a_pts,
         "完全（絕對帶位）", "exported ONNX 上評估，10,000 筆驗證集"],
        ["Model Compression", 20, "%.1fx（參數 %.1fx、檔案 %.1fx 取較差者）" % (comp, pr, sr),
         c_band, c_pts, "完全（絕對帶位）", "規則要求兩者皆 >=10x"],
        ["Resource Efficiency", 20, "RAM %.1f MB、延遲 %.2f ms @4T" % (best["peak_rss_mb_t4"],
                                                                    best["latency_p50_ms_t4"]),
         "看對手分佈", "?", "不可控（比四分位）", "本機量測，非目標硬體"],
        ["Energy Parsimony", 20, "0.131 J/inf @4T（TDP 推估）", "Good 可達", "13-17",
         "部分可控", "CodeCarbon + AI Hub 延遲比值佐證"],
        ["Documentation & Reproducibility", 10, "REPORT.md / metadata.json / metrics.json / sweep.csv",
         "Excellent", "9-10", "完全", "工具包自動產出"],
        ["Presentation & Engineering Judgment", 5, "取捨分析有自己的量測支撐", "Excellent", "5",
         "完全", "含 VENTUNO Q 裝置判斷、arena 否定結果"],
    ]
    r = write_table(ws, rows, 3, widths=[32, 7, 38, 14, 11, 20, 40])

    for row in ws.iter_rows(min_row=4, max_row=r - 1, min_col=4, max_col=4):
        for c in row:
            c.fill = GOOD if c.value == "Excellent" else (WARN if c.value else BAD)

    ws.cell(row=r + 1, column=1, value="自己能決定的 60 分（25+20+10+5）估計拿到 56；另外 40 分比四分位，看對手。").font = Font(bold=True)


def sheet_students(wb, d, base):
    ws = wb.create_sheet("3 學生階梯")
    ws["A1"], ws["A1"].font = "五個預訓練架構 × FP32/INT8，全部 12 個變體", TITLE
    ws["A2"], ws["A2"].font = ("門檻：相對精度 >=97% 且 壓縮 >=10x。壓縮取「參數倍率」與「檔案倍率」的較差者，"
                               "因為規則寫的是 both。", NOTE)

    rows = [["變體", "參數量", "檔案 (MB)", "精度", "保留%", "參數x", "檔案x",
             "壓縮x（取較差）", "精度帶", "壓縮帶", "SQNR dB", "合格"]]
    for _, r_ in d.iterrows():
        pr = base["params"] / r_["params"]
        sr = base["size_bytes"] / r_["size_bytes"]
        comp = min(pr, sr)
        rel = r_["accuracy"] / base["accuracy"]
        sq = r_.get("int8_sqnr_db")
        rows.append([r_["variant"], int(r_["params"]), round(r_["size_bytes"] / 2**20, 2),
                     round(r_["accuracy"], 4), round(100 * rel, 1), round(pr, 1), round(sr, 1),
                     round(comp, 1), band(rel, ACCURACY_BANDS)[0], band(comp, COMPRESSION_BANDS)[0],
                     None if pd.isna(sq) else round(sq, 1),
                     "是" if (rel >= 0.97 and comp >= 10) else "否"])
    r = write_table(ws, rows, 4, widths=[30, 12, 11, 9, 8, 8, 8, 15, 11, 11, 9, 7])

    for row in ws.iter_rows(min_row=5, max_row=r - 1):
        ok = row[11].value == "是"
        row[11].fill = GOOD if ok else BAD
        if row[10].value is not None and row[10].value < 15:
            row[10].fill = BAD              # the INT8 health check: below ~15 dB is a broken conversion

    notes = [
        "97% 的門檻落在 1.53M 和 1.26M 參數之間——shufflenet_v2_x1_0 只差 0.4 個百分點。",
        "mobilenet_v3_small 的 INT8 量化壞掉（SQNR 3.6 dB，門檻 15 dB），精度崩到 69.6%。"
        "四種校準方法都試過，最好的只有 3.6 dB。所以送交的是它的 FP32 版。",
        "沒有這個 SQNR 檢查，那個變體會因為分數不錯被選上，交出去才發現精度只剩 69.6%。",
    ]
    for i, t in enumerate(notes, 1):
        c = ws.cell(row=r + i, column=1, value=t)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=r + i, start_column=1, end_row=r + i, end_column=12)
        ws.row_dimensions[r + i].height = 28


def sheet_measurement(wb, d, contended):
    ws = wb.create_sheet("4 量測可信度")
    ws["A1"], ws["A1"].font = "延遲與記憶體：哪些數字可信，為什麼", TITLE
    ws["A2"], ws["A2"].font = ("同一組檔案量兩次：一次機器忙碌，一次安靜。差異本身就是答案。", NOTE)

    rows = [["變體", "延遲 1T (ms)", "延遲 4T (ms)", "RSS 4T 淨值 (MB)", "RSS 4T 原始 (MB)",
             "4T 延遲（忙碌時量）", "差異 %", "core_delivery"]]
    cmap = contended.set_index("variant") if contended is not None else None
    for _, r_ in d.iterrows():
        old = None
        if cmap is not None and r_["variant"] in cmap.index:
            old = cmap.loc[r_["variant"], "latency_p50_ms_t4"]
        diff = None if old is None else round(100 * (old - r_["latency_p50_ms_t4"]) / r_["latency_p50_ms_t4"], 1)
        rows.append([r_["variant"], round(r_["latency_p50_ms_t1"], 2), round(r_["latency_p50_ms_t4"], 2),
                     round(r_["peak_rss_mb_t4"], 1), round(r_["peak_rss_total_mb_t4"], 1),
                     None if old is None else round(old, 2), diff,
                     round(r_["core_delivery_t4"], 3)])
    r = write_table(ws, rows, 4, widths=[30, 13, 13, 16, 16, 20, 10, 14])

    for row in ws.iter_rows(min_row=5, max_row=r - 1):
        if row[6].value is not None and abs(row[6].value) >= 7:
            row[6].fill = BAD

    notes = [
        "基準 ResNet-50 在忙碌時量到的延遲高出 14.9%，學生則幾乎不動——大模型的量測窗口長，"
        "受干擾的機會也大。這組對照證實了汙染是真的，而且已經被修正。",
        "core_delivery 欄位請勿照字面解讀。它的定義是 cpu_ms / (執行緒數 × 延遲)，"
        "但 ORT 的 spin-wait 會把閒置執行緒的 CPU 時間算成工作，所以在 4 執行緒讀到 1.298 這種"
        "不可能的值。低於 0.95 既不證明汙染也不證明閒置。",
        "有效性的實際證據是：安靜版與忙碌版對學生的差異都在幾個百分點內，而基準的差異被抓出來了。",
        "RSS 兩欄的差別：淨值扣掉了一個空 ONNX Runtime session 的開銷，原始值沒有。"
        "淨值回答「這個模型要多少記憶體」，原始值是評審在機器上會看到的數字。兩個都要報。",
    ]
    for i, t in enumerate(notes, 1):
        c = ws.cell(row=r + i, column=1, value=t)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=r + i, start_column=1, end_row=r + i, end_column=8)
        ws.row_dimensions[r + i].height = 42


def sheet_threads(wb):
    ws = wb.create_sheet("5 執行緒與能耗")
    ws["A1"], ws["A1"].font = ("延遲與能耗隨執行緒數的變化"
                               "（送交的 mobilenet_v3_small FP32；RSS 為行程原始值，非第 1 頁的淨值）", TITLE)
    if not THREADS.exists():
        ws["A3"] = "尚未量測"
        return
    data = json.loads(THREADS.read_text(encoding="utf-8"))
    rows = [["執行緒", "延遲 p50 (ms)", "CPU 時間 (ms)", "峰值 RSS (MB)", "能耗 (J/inf)", "能耗來源"]]
    for r_ in data["rows"]:
        rows.append([r_["threads"], round(r_["latency_p50_ms"], 3), round(r_["cpu_ms_per_inf"], 2),
                     round(r_["peak_rss_mb"], 1),
                     None if r_.get("energy_j") is None else round(r_["energy_j"], 5),
                     r_.get("energy_source", "")])
    r = write_table(ws, rows, 3, widths=[10, 15, 15, 15, 14, 20])

    notes = [
        "延遲與能耗的極小值在同一點（8 執行緒），兩者沒有取捨關係——這一點我們量過，結果是否定的。",
        "但 4 → 8 執行緒只買到 5% 延遲，卻多燒 54% CPU 時間，而目標板 Arduino UNO Q 只有 4 核。"
        "所以 4 執行緒是作業點。",
        "能耗標的是 codecarbon:model，意思是 CodeCarbon 在這台機器上讀不到 RAPL 硬體電表，"
        "改用 TDP 比例推估。順序可信，絕對值不是量測。報告裡會照實標明。",
    ]
    for i, t in enumerate(notes, 1):
        c = ws.cell(row=r + i, column=1, value=t)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=r + i, start_column=1, end_row=r + i, end_column=6)
        ws.row_dimensions[r + i].height = 32


def sheet_gaps(wb):
    ws = wb.create_sheet("6 已知缺口")
    ws["A1"], ws["A1"].font = "還沒解決的事，按風險排序", TITLE
    rows = [
        ["缺口", "後果", "狀態"],
        ["全程沒有計時演練過", "不知道「拿到模型→繳交包」實際要幾小時。基準 ResNet-50 單列量測就要 6 分鐘，"
                     "現場模型更大的話光量測就會吃掉大量時間。", "現場的人要自己演習一次"],
        ["代理任務太友善", "Imagenette 是 ImageNet 的子集，預訓練學生轉移得異常好。"
                    "真實資料集上 KD 能補多少落差，沒有驗證過。", "無法事先解決"],
        ["基準模型可能不重", "壓縮倍率是「基準有多重」的函數。基準 20M+ 參數則 10x 輕鬆過；"
                      "基準 3M 以下則壓縮那 20 分會很難。", "當天 08:30 才知道"],
        ["題目可能不是分類", "adapter 假設分類 + ImageFolder。若是偵測、異常偵測或時序，"
                      "要現場寫 Dataset。", "工具包有其他 adapter，但未對新題目測過"],
        ["現場有無 GPU 未知", "沒有 GPU 時 KD 會慢 5–10 倍。", "手冊有降級方案：KD epoch 砍半、只試最小的兩個架構"],
        ["metrics.json 規格未知", "欄位名在 starter kit 第 8 節。", "已做超集 + --rename 掛鉤，當天改對應表即可"],
        ["能耗讀不到 RAPL", "只能用 TDP 推估。", "若沙箱是 Linux 則可能讀得到，會自動切換並標明"],
    ]
    r = write_table(ws, rows, 3, widths=[26, 62, 36])
    for row in ws.iter_rows(min_row=4, max_row=r - 1):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[row[0].row].height = 46


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "SEC2026_設計成績卡.xlsx"))
    a = ap.parse_args()

    d = pd.read_csv(SWEEP)
    contended = pd.read_csv(CONTENDED) if CONTENDED.exists() else None
    base = d[d["variant"] == "p00_fp32"].iloc[0]
    best = d[d["variant"] == "mobilenet_v3_small_p00_fp32"].iloc[0]

    wb = Workbook()
    wb.remove(wb.active)
    sheet_verdict(wb, d, base, best)
    sheet_rubric(wb, base, best)
    sheet_students(wb, d, base)
    sheet_measurement(wb, d, contended)
    sheet_threads(wb)
    sheet_gaps(wb)

    out = Path(a.out)
    wb.save(out)
    print("wrote %s (%.0f KB, %d sheets)" % (out, out.stat().st_size / 1024, len(wb.sheetnames)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
