# 比賽當天作戰手冊

SEC 2026 Minimalist Edge AI Challenge，Software Track。一天，約 6 小時實際工作時間。
題目和基準模型 **08:30 才拿到**。這份文件假設你已經照 `HANDOVER.md` 第 4 節把環境裝好、測試是綠的。

> 這份是當天拿在手上照著做的。背景與理由在 `HANDOVER.md`，不要在現場讀那一份。

---

## 0. 評分規則（決定優先序的唯一依據）

| 項目 | 分 | 怎麼拿 | 我們控制得了嗎 |
|---|---|---|---|
| Model Accuracy & Reliability | 25 | **保留 ≥97% 基準精度** = Excellent | ✅ 完全 |
| Model Compression | 20 | **參數量和檔案大小都 ≥10×** = Excellent | ✅ 完全 |
| Resource Efficiency | 20 | 峰值 RAM 與延遲，比**所有隊伍的四分位** | ❌ 看對手 |
| Energy Parsimony | 20 | J/inference，比四分位；**但「相對基準有明確下降＋AI Hub 佐證」就有 13–17 分** | 部分 |
| Documentation & Reproducibility | 10 | 評審只靠繳交物就能重建 | ✅ 完全 |
| Presentation & Engineering Judgment | 5 | 用自己的量測解釋取捨 | ✅ 完全 |

**60 分完全由我們決定，40 分看對手。** 先把 60 分拿滿，不要為了多壓幾倍而跌破 97%。

**壓縮超過 10× 不會多拿分**（該項封頂），只在 Resource／Energy 的四分位上還有用。

---

## 1. 時間表

| 時間 | 做什麼 | 死線 |
|---|---|---|
| 08:00–08:30 | 報到、環境健檢（§2） | |
| 08:30–09:00 | Kickoff，拿到 starter kit | |
| 09:00–09:30 | 讀題、接資料、跑通基準（§3） | **09:30 前基準要有數字** |
| 09:30–11:00 | 學生階梯蒸餾（§4） | |
| 11:00–11:30 | 挑選、量化、打包（§5） | |
| 11:30–12:00 | **AI Hub 送件**（§6，每個 job 約 5 分鐘，先送） | |
| 12:00–12:30 | 午餐（AI Hub 在跑） | |
| 12:30–14:00 | 能耗量測（§7）＋ 第二輪優化 | |
| 14:00–15:00 | metrics.json、報告（§8） | **15:00 前要有可交的版本** |
| 15:00–17:00 | 最終 benchmark、補強、繳交 | |

**15:00 一定要有一份可以交的東西。** 之後的時間是用來變好，不是用來完成。

---

## 2. 報到後 30 分鐘：環境健檢

```bash
.venv\Scripts\python -m pytest -q          # 必須 173 passed
.venv\Scripts\python -c "import torch, onnxruntime as ort; print(torch.__version__, torch.cuda.is_available(), ort.__version__)"
.venv\Scripts\python -c "from core import aihub; print(len(aihub.list_devices()), 'devices')"
```

**記下來寫進報告**：CPU 型號、核心數、有沒有 AVX-VNNI、ONNX Runtime 版本、Python 版本、有沒有 GPU、有沒有網路。

> ORT 版本會改變 INT8 的加速比（實測過 1.23.2 → 1.24.4 讓同一個檔案在四執行緒慢 31%）。**不要引用別台機器的數字。**

---

## 3. 拿到題目後的第一件事（09:00–09:30）

### 3.1 量三個數字，其他都先不管

```bash
.venv\Scripts\python -c "
import torch, sys
m = torch.load(sys.argv[1], weights_only=False)
print('參數量 %.2f M' % (sum(p.numel() for p in m.parameters())/1e6))
" <他們給的檢查點>
```

| 要知道的 | 為什麼 |
|---|---|
| 基準的**參數量** | 決定 10× 的目標落在哪。基準 20M+ → 換架構就過；基準 <3M → 10× 很難，要提早說 |
| 基準的**精度** | ×0.97 就是你的硬底線 |
| **幾類、什麼資料** | 決定 adapter 怎麼接 |

### 3.2 接資料

資料若是 ImageFolder 格式（每類一個資料夾），改 `adapter_args` 就好：

```yaml
adapter_args:
  dataset: imagefolder
  data_root: <他們的資料路徑>
  img_size: <他們的尺寸>
  arch: <基準的架構名>
  pretrained: true
```

不是 ImageFolder 就寫一個 `Dataset` 塞進 `adapters/classification.py` 的 `_datasets()`。**不要重寫 adapter。**

### 3.3 跑通基準

```bash
.venv\Scripts\python scripts\compress.py --config configs\<你的設定>.yaml --quick
```

`--quick` 幾分鐘內跑完，目的是**確認整條路通**，不是要數字。通了再跑正式的。

---

## 4. 學生階梯（09:30–11:00）——這一段決定 Accuracy 和 Compression

### 核心規則：換架構，不要縮寬度

torchvision 只有 `mobilenet_v2` **全寬**有預訓練權重。寫 `width_mult: 0.5` 會靜默變成從零訓練，8 個 epoch 練不起來——上一輪所有學生只有 65–77% 就是這個原因。現在會警告，看到就改。

設定檔照 `configs/experiments/sec_students_pretrained.yaml` 改：

```yaml
sweep:
  students:
    - {arch: mobilenet_v2, pretrained: true}        # 2.24 M
    - {arch: mobilenet_v3_small, pretrained: true}  # 1.53 M
    - {arch: shufflenet_v2_x1_0, pretrained: true}  # 1.26 M
    - {arch: mnasnet0_5, pretrained: true}          # ~0.95 M
    - {arch: shufflenet_v2_x0_5, pretrained: true}  # 0.35 M
distill:
  epochs: 8          # 時間不夠就降到 5，不要降到 3 以下
```

**選哪一個**：在 ≥97% 的前提下挑參數量最小的那個。工具包會自己擋掉不合格的（`max_accuracy_drop_rel: 0.03`）。

**如果全部都過不了 97%**：先加 KD epoch（最有效），再往大的架構退。寧可 10.5× 拿滿精度分，也不要 66× 掉一個精度帶。

---

## 5. 挑選與打包（11:00–11:30）

```bash
.venv\Scripts\python scripts\compress.py --config configs\<你的設定>.yaml
```

產出在 `deliverables/<run>/`：`best.onnx`、`best_fp32.onnx`、`REPORT.md`、`metadata.json`、`candidates.csv`、`sweep.csv`。

**檢查三件事**：
1. `REPORT.md` 的「parameters」那一列有沒有 ≥10×（**不是只看 size**）
2. 精度那一列的「% retained」有沒有 ≥97%
3. `int8_sqnr_db` 有沒有 >15 dB（低於就是量化壞了）

---

## 6. AI Hub（11:30，先送，別等）

每個 job 約 5 分鐘，大部分是排隊。**午餐前送出去。**

```bash
.venv\Scripts\python scripts\aihub_profile.py --config configs\<你的設定>.yaml
```

它會把 `aihub_latency_ms`、`aihub_peak_memory_mb` 寫回 `sweep.csv`，報告會自動帶上。

**裝置選擇**：預設走 `Dragonwing RB3 Gen 2 Vision Kit`。
**不要選 `Arduino VENTUNO Q`**——名字最像，但它是 Dragonwing IQ8／40 TOPS，比賽目標 UNO Q 是 QRB2210／4× Cortex-A53，差一整個世代。理由已經寫在設定檔的 `aihub.device_rationale`，會自動印進報告。

---

## 7. 能耗（12:30–14:00）——這一段要安靜的機器

設定檔加上：

```yaml
measure:
  energy:
    provider: codecarbon
    min_seconds: 30
```

然後重量（不重訓）：

```bash
.venv\Scripts\python scripts\remeasure.py --config configs\<你的設定>.yaml
```

**量測期間整機 CPU 要 ≤15%**，否則數字作廢。跑完看 `core_delivery` 欄位，低於 0.95 就重來。

**Energy 要拿到 Good（13–17 分）需要兩件事**，兩件都做得到：
1. 相對基準有明確下降 ← 交一個小 10 倍的模型就自動成立
2. 本地與 AI Hub 「大致一致」← **AI Hub 不報能耗**，所以用比值對齊：本地能耗比 ≈ AI Hub 延遲比（因為能耗 ≈ 功率 × 延遲）。這句話要寫進報告。

`energy_source` 會標 `codecarbon:rapl`（真量測）或 `codecarbon:model`（TDP 推估）。**不要把推估說成量測**，Poor 帶的條件是「數字缺失或無法重現」，誠實標註是防守。

---

## 8. 繳交物（14:00–15:00）

官方要求：**量化後的 ONNX ＋ `metrics.json` ＋ 1–2 頁報告 ＋ 可重現的全部程式碼**。

```bash
.venv\Scripts\python scripts\make_metrics.py --deliverable deliverables\<run>
```

`metrics.json` 的欄位名規格在 starter kit 的第 8 節（預先釋出的文件沒有）。拿到後寫一個對應表再跑一次：

```bash
.venv\Scripts\python scripts\make_metrics.py --deliverable deliverables\<run> --rename official.json
```

`official.json` 長這樣：`{"joules_per_inference": "<他們的欄位名>", ...}`

**報告**：`REPORT.md` 已經有官方要的三樣（壓縮流程、AI Hub 裝置與理由、精度對效率的取捨）。轉成 PDF，壓到 2 頁內。

---

## 9. 出事的時候

| 症狀 | 處置 |
|---|---|
| 學生精度上不去 | 先確認有沒有 `no pretrained weights` 的警告；再加 KD epoch；最後才換大架構 |
| 量化後精度崩 | 看 `int8_sqnr_db`。<15 dB 就改 `calibrate_method: percentile`；還不行就交 `best_fp32.onnx`（壓縮比掉 4 倍但精度保住） |
| 匯出失敗 | `opset` 降到 13 試試 |
| AI Hub 連不上 | 跳過，Energy 從 Good 掉到 Fair（約 −5 分），不要在這上面燒時間 |
| 沒有 GPU | KD epoch 砍半，batch 砍半，架構只試最小的兩個 |
| 完全來不及 | **交 FP32 學生**：參數量 10× 照樣成立，只是檔案少了 4 倍。壓縮仍可能在 Excellent |

---

## 10. 不要做的事

- 不要為了多壓幾倍跌破 97% 精度（確定 −6 分換不確定的 +4 分）
- 不要在量測時跑別的東西
- 不要相信別台機器量的延遲
- 不要編任何一個數字——沒量到就寫 not measured，規則對「不可重現」的懲罰比對「缺失」重
- 不要在 15:00 之後還在改模型
