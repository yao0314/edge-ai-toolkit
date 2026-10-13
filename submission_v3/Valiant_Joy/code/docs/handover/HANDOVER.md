# SEC 2026 交接文件

寫於 2026-09-22（台灣時間），寫給兩種讀者：在會場的隊友，以及隊友電腦上新開的 Claude Code。
新電腦上沒有之前的對話紀錄，也沒有 Claude 的記憶；一切以這個資料夾為準。

閱讀順序：這份 → `README.md`（工具包完整說明，英文）→ 需要時再看 `docs/handover/lessons/`（踩過的坑，一篇一個教訓，見該資料夾的 `INDEX.md`）。

---

## 1. 一句話現況

工具包（`edge-ai-toolkit`）已經能在 ONNX Runtime CPU 上走完整條流程：剪枝、INT8 量化、量測、評分、挑出候選、打包成可繳交的資料夾。四種任務都有 adapter：影像分類、異常偵測、物件偵測、多目標追蹤。比賽細節大多沒公布，所以所有未知項都做成 YAML 參數，沒有寫死。

## 2. 比賽基本資料

- **SEC 2026 Minimalist Edge AI Challenge，Software Optimization Track**；2026-10-13 至 10-15，美國加州 Santa Clara University。
- 官方寫的是「virtual development environment」、「No hardware required」：軟體組大概率碰不到實體板子（UNO Q 是硬體組的）。
- 官方列的四個目標：model accuracy、model compression、resource efficiency、energy parsimony。**官方稱它們是「要優化的目標」，沒有寫「依此評分」**；唯一明寫的評選方式是向評審 pitch。
- 官方舉的任務例子：即時多目標追蹤、環境異常偵測（原文是 "such as"，只是舉例）。
- **未知**：評分權重與公式、會拿到什麼（PyTorch／ONNX／TFLite 模型？資料？）、要繳什麼格式、能耗怎麼量、時限、沙箱的 CPU／RAM／GPU、有沒有網路、能不能帶自己的程式。
- 給主辦方的 8 個問題已擬好，放在使用者電腦的 `SEC/organizer_questions.txt`（不在工具包裡）：訓練資料、精度門檻、評分與權重、pitch、能耗、沙箱規格與網路、模型格式、時限。「能不能帶自己的程式」沒問到。**有沒有寄出、有沒有回覆，請向使用者確認**；也可以在官方 Discord 問。

詳細的已知／未知與出處：`docs/handover/lessons/sec2026-competition-unknowns.md`。注意那篇寫於 9/14，裡面說「只有偵測、沒有追蹤指標」已經過時：之後補上了 `adapters/tracking.py` 和 `adapters/mot_metrics.py`（MOTA、IDF1），設定檔 `configs/tracking_smoke.yaml`。`README.md` 的比賽事實表把四個目標寫成「Judging criteria (confirmed)」、把 UNO Q 寫成目標硬體，那是較早的寫法，以本節為準。

## 3. 人、時差、分工

- **使用者**（楊元蓁）人在台灣，遠端指揮；**隊友**在會場。
- 時差：加州 10 月是夏令時間（UTC-7），台灣是 UTC+8，**台灣比加州快 15 小時**。加州早上 9 點＝台灣隔天凌晨 0 點（例：加州 10/13 09:00＝台灣 10/14 00:00）；加州下午 5 點＝台灣隔天早上 8 點。也就是加州整個白天，都是台灣的半夜到清晨。
- 新電腦上的 Claude Code 沒有記憶：靠專案根目錄的 `CLAUDE.md`（開專案時自動讀取）和這個資料夾。
- Claude 帳號是個人使用的。建議在隊友電腦上用使用者自己的帳號登入 Claude Code（`/login`），或由隊友用他自己的帳號操作，不要共用同一個帳號的登入。

## 4. 新電腦從零開始：要裝什麼、怎麼跑

### 4.1 要裝的東西（Windows，隊友的筆電）

| 軟體 | 為什麼要裝 | 怎麼裝 |
|---|---|---|
| **Python 3.11**（使用者筆電用 3.11.9） | 工具包的執行環境。3.10 以下不行；3.12 也驗證過 | PowerShell：`winget install -e --id Python.Python.3.11`，或到 python.org 下載 3.11 的 Windows 安裝檔（安裝時勾「Add python.exe to PATH」） |
| **Git for Windows** | Claude Code 在 Windows 上需要它 | `winget install -e --id Git.Git` |
| **Claude Code** | 用 AI 寫程式、操作這個專案 | 二選一：裝 Claude 桌面 App，用「Code」分頁打開工具包資料夾；或在 PowerShell 執行 `irm https://claude.ai/install.ps1 \| iex`，之後在工具包資料夾裡打 `claude`。用自己的帳號 `/login` |
| 解壓縮工具 | 解開工具包 | Windows 內建即可 |

不需要裝 CUDA、NVIDIA 驅動或 Visual Studio：Windows 上 PyPI 的 PyTorch 本來就是 CPU 版，其他套件在使用者筆電上也都是直接裝好的，不需要編譯工具。裝完 Python 和 Git 後，**關掉再重開 PowerShell**，新的指令才會生效。

### 4.2 安裝工具包（一次，約 10–20 分鐘，要網路）

1. 把工具包解壓到**路徑短、沒有中文和空格、不在 OneDrive 同步範圍內**的地方，例如 `C:\SEC\edge-ai-toolkit`（放在 OneDrive 裡，同步程式可能鎖住檔案或把 `.venv` 也上傳）。
2. 在 PowerShell 執行（不用啟用虛擬環境，直接叫 `.venv` 裡的 python，避開 PowerShell 的腳本執行限制）：

```powershell
cd C:\SEC\edge-ai-toolkit
py -3.11 -m venv .venv                                   # 建立虛擬環境
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt  # 下載約 1–2 GB（估計），CPU 版 PyTorch
.venv\Scripts\python -m pytest -q                        # 應該 167 passed；使用者筆電約 2.5 分鐘
```

3. 之後每個指令都用 `.venv\Scripts\python` 開頭，例如 `.venv\Scripts\python scripts\measure.py --task classification --quick`。
4. 只有要處理 TFLite 或 Qualcomm AI Hub 時，才另外裝 `requirements-optional.txt`（含 TensorFlow，很大）。

**常見問題**
- `py` 找不到：Python 沒裝好或沒重開 PowerShell；改用 python.org 的安裝檔重裝。
- `import torch` 或 `import onnxruntime` 出現 DLL 錯誤：裝 Microsoft Visual C++ Redistributable（x64）後重開。
- 測試不是 167 passed：先別開始工作，把錯誤訊息整段貼給 Claude。

### 4.3 Linux（主辦方沙箱若是 Linux）

```bash
python3.11 -m venv .venv          # 或 python3.12
source .venv/bin/activate
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m pytest -q -rs           # -rs：列出被跳過的測試，確認不是靜靜地沒跑
```

先裝 CPU 版 PyTorch 是因為 Linux 上直接 `pip install -r requirements.txt` 會多拉約 1 GB 用不到的 CUDA。Alpine（musl）和 glibc 2.28 以下的系統不支援（README「Platform compatibility」）。

### 4.4 常用指令

完整說明在 `README.md` 的 Quick start。Windows 上把 `python` 換成 `.venv\Scripts\python`。

```bash
python scripts/measure.py --task classification --quick           # 基準量測，CPU 上 < 5 分鐘
python scripts/sweep.py --config configs/classification.yaml --quick
python scripts/compress.py --task classification                  # 一條龍：掃描 → 挑選 → 打包交付
python scripts/remeasure.py --config configs/classification.yaml  # 換機器後只重量、不重訓
python scripts/make_report_xlsx.py                                # 全部結果的 Excel 報表
```

- 沙箱若**沒有網路**：離線套件包只在使用者筆電的 `vendor/`（Linux x86_64，Python 3.11／3.12 各一份，每份約 470 MB），要用就先複製過去；或趁有網路時用 `python scripts/make_offline_bundle.py --target linux-x86_64 --python 311` 重做（這一步需要網路）。在沙箱上：`pip install --no-index --find-links <bundle> -r <bundle>/requirements.txt`。
- 第一次跑會下載資料集到 `data/`、預訓練權重到 `weights/`，之後可離線。**現場網路不保證，資料集和權重請出發前先下載好。**
- **輸出一律 ONNX**（使用者的決定）。主辦方要別的格式時，拿交付資料夾裡的 `best_fp32.onnx` 去轉，再用目標格式自己的工具量化；不要轉 INT8 的 QDQ 檔。**要交 TFLite 的話**：ONNX 轉 TFLite 的工具（onnx2tf）還沒放進 `requirements*.txt`，工具包的 `target_runtime: tflite` 也還沒實作，所以轉檔要在有網路的機器上另外裝工具來做。

### 現場三種情境（README「Model format compatibility」）

- **A：拿到 PyTorch 模型＋資料**：複製一個現成 adapter，把 `load_data`／`evaluate` 換成主辦方的資料和指標，`--model` 指向他們的權重。剪枝＋量化＋掃描全部可用。
- **B：只拿到 ONNX／TFLite**：`sweep.py --model x.onnx`，剪枝會標成 skipped，量化和量測照跑。還沒寫好 adapter 前，`measure.py --model x --input-shape ...` 可以先量資源（精度留空）。**如果只給驗證集**：影像分類 adapter 的資料夾模式要 `train/` 和 `val/`，校準用 `train/`；把驗證集切成「校準用」和「評分用」兩份，不要用同一批資料校準又評分。也要確認前處理（尺寸、裁切、正規化）和主辦方模型一致：adapter 預設是 ImageNet 的平均值／標準差加 Resize／CenterCrop。
- **C：只拿到任務和資料**：直接用或改內建的 adapter（classification／anomaly／detection／tracking）。
- 沒有 PyTorch 權重就不能剪枝。**第一件事是問主辦方給不給 PyTorch 權重**，這決定策略。

## 5. 目前成果（`deliverables/`，ONNX Runtime 1.30.0，使用者的筆電）

每個資料夾裡有 `REPORT.md`、`best.onnx`、`best_fp32.onnx`、候選表。挑選規則：精度下降不超過 `score.max_accuracy_drop` 的變體裡總分最高者（異常偵測、兩個分類是 0.02；兩個物件偵測是 0.01）。

| 任務 | 資料夾 | 選中的變體 | 精度 | 檔案大小 | 延遲（1 執行緒 → 4 執行緒）|
|---|---|---|---|---|---|
| 異常偵測（ECG5000） | `anomaly` | `p50_fp32`（剪 50%） | AUC 0.9817 → 0.9698 | 3.78× 小 | 2.02× → 1.98× 快 |
| 影像分類 MobileNetV2 | `classification_unstructured` | `p00_int8` | top-1 0.9234 → 0.9220 | 3.40× 小 | 1.62× → 1.18× 快 |
| 影像分類 MobileViT-XXS | `classification_mobilevit` | `p00_int8_dyn_mm`（動態 INT8） | top-1 0.9270 → 0.9272 | 1.49× 小 | 1.06× → 1.12× 快 |
| 物件偵測 YOLOv8n 320 px | `detection` | `320_p00_int8` | mAP50-95 0.3057 → 0.2988 | 3.58× 小 | 2.10× → 1.35× 快 |
| 物件偵測 YOLOv8n 640 px | `detection_640` | `640_p00_int8` | mAP50-95 0.3687 → 0.3600 | 3.49× 小 | 2.18× → 1.56× 快 |

- 給評審看的數字只用 `deliverables/*/REPORT.md`；`README.md` 的 Example results 表是較早的執行，數字會有出入（例如 MobileViT 的 FP32 top-1 在那裡是 0.9165）。
- 能耗一律「not measured」：官方沒公布量法前，不編數字。

## 6. 研究得到、比賽可直接用的結論

第 1–6 點出自使用者投稿 TAAI 2026 的論文，第 7–10 點出自工具包自己的實驗紀錄（`results/paper/FINDINGS.md`、`README.md`、`core/quantize.py`）。兩者都是在使用者的筆電（Intel i7-13620H，有 AVX-VNNI）上用同一套工具包量的。論文資料在 https://github.com/Pric0123/taai2026-int8-cpu-artifact （若已改成私人則看不到）。

1. **卷積模型用靜態 INT8**（QDQ，uint8 激活、int8 權重）：一執行緒快 1.63–3.21 倍；四執行緒時 ResNet-50、MobileNetV2、YOLOv8n 保有 59–77% 的加速，但最小的自編碼器只剩 1.07 倍（在雜訊範圍內）。
2. **卷積模型絕對不要用動態 INT8**：量化器的動態模式預設會連卷積一起量化，結果在一執行緒時比 FP32 慢 12.6–16.8 倍。
3. **注意力／混合模型用動態 INT8**（只量化 MatMul／Gemm 的權重）：MobileViT-XXS 的完整靜態 INT8 比 FP32 慢、精度掉 4.62 點；動態 INT8 速度和 FP32 差不到 8%、精度不掉，主要收穫是檔案變小。
4. **激活用 uint8（U8S8），不要用量化器文件預設的 S8S8**：在使用者筆電上，S8S8 讓試過的兩個模型都比 FP32 還慢。工具包預設已是 U8S8（`core/quantize.py`）。
5. **ONNX Runtime 版本會改變加速比**：檔案完全不變、只換版本（1.23.2 → 1.24.4），MobileViT 的靜態 INT8 在四執行緒慢了 31%，YOLOv8n 的靜態 INT8 在四執行緒卻快了 28%。**到了沙箱先記下它的 ONNX Runtime 版本，用 `remeasure.py` 在沙箱上重量**，不要直接引用這裡的數字。工具包鎖定 1.30.0。
6. **卷積模型的靜態 INT8，一執行緒的加速比會高估多執行緒**（論文中 31 組全部如此；動態 INT8 不一定，例如 MobileViT 是 1.06 倍 → 1.12 倍）：以實際部署的執行緒數為準。沙箱核心數未知；UNO Q 是 4 核，但它屬於硬體組。
7. **剪枝值不值得看模型，恢復訓練的設定很關鍵**：在 MobileNetV2／CIFAR-10 上，用預設 `prune.recovery_lr_mult` 1.0 剪 25% 會超出精度預算；改成全域剪枝（`global_pruning: true`）、恢復訓練 20 個 epoch、`recovery_lr_mult` 5 之後，「剪枝＋量化」比「只量化」多壓一截（4.35 倍對 3.40 倍），代價是 2.0 點精度預算用掉 1.57 點（lr_mult 5–20 都比 1.0 好）。起點設定檔是 `configs/experiments/classification_prune_global.yaml`（它只設了全域剪枝；20 個 epoch 和 lr_mult 5 要另外設）。在 YOLOv8n 上，把學習率提高 20 倍則完全沒有改善（README「Pruning recovery」一節）。異常偵測的自編碼器在預設設定下剪 50% 就划算（見第 5 節）。細節：`lessons/pruning-earns-nothing-within-budget.md`（標題是舊結論，內文已更正）。
8. **校準方法用 `calibrate_method: auto`**：minmax 在離群值多的模型上會崩（剪 25% 的 ResNet-50：minmax 0.468，percentile 0.894）。
9. **偵測器輸出尾端保持 FP32**（像素座標和 0–1 分數混在同一個張量，共用一個量化尺度會毀掉 mAP）。
10. **INT8 健康檢查**：每個變體都會報 `int8_sqnr_db`，低於約 15 dB 代表量化壞了。

## 7. 量測紀律（數字要能信，就得照做）

- 量測時機器要安靜：整機 CPU ≤ 15%、插著電、一次只跑一個量測。Windows 看 `Win32_PerfFormattedData_PerfOS_Processor` 的 `_Total`；Linux 用 `mpstat 1 5` 或 `top` 看整機閒置率（閒置 ≥ 85%）。**不要**看 Windows 的 `LoadPercentage`，也不要看工具包的 `other_load_pct`，兩者都不可靠。
- 每次量測後看 CSV 的 `core_delivery` 欄（實際拿到的核心比例）：工具包在低於 0.90 時會警告；論文把低於 0.95 的 session 視為可疑，要一併報告。
- 用中位數、多次獨立 session、隨機順序；每個數字都要能追溯到某個 CSV。
- ONNX 檔案裡的圖不等於實際執行的圖：ONNX Runtime 會在推論前刪掉約九成的 Q/DQ 節點。要看執行圖，用 `optimized_model_filepath` 存下來看。
- 能耗：沒有官方量法前，`energy_j` 留空。沙箱若提供量測工具，接到 `core/energy.py` 的 `command` provider。
- 硬體的說法要先用 CPUID 驗證再寫進簡報或報告。

## 8. 隊友電腦（MSI Bravo 15）注意事項

照片上的規格：AMD Ryzen 7 4800H（8 核 16 執行緒）、16 GB RAM、Radeon RX 5500M、硬碟剩約 43 GB。

- **應該沒有 AVX-VNNI**（Zen 2 架構；這是依型號推斷，拿到電腦先用 CPUID 確認）：INT8 會走 AVX2 路徑，加速比預期會比第 5、6 節低（沒有實測過）。若 INT8 精度或 `int8_sqnr_db` 明顯變差，可試 `quantize.reduce_range: true`（給沒有 VNNI 的舊 CPU 用的防溢位設定，預設關閉）。比賽若在主辦方沙箱跑，以沙箱為準。
- **沒有 NVIDIA GPU**：PyTorch 只能用 CPU 訓練，會慢。分類任務的 `--quick` 在使用者筆電上只用 CPU 實測 52 秒（在這顆 CPU 上會再慢一些）；完整掃描預計要好幾個小時（沒有在純 CPU 上實測過）。
- **8 個核心都一樣**：沒有大小核的問題；工具包仍會把每個執行緒綁在不同的實體核心上（避開同一核心的兩個邏輯執行緒），不用改。
- **硬碟**：CPU 版 PyTorch 的環境估計約 1–2 GB（沒實測），資料集約 2–3 GB（COCO 訓練子集 `coco_train5000` 加上它的標註檔約 1.27 GB，只有剪枝偵測器時才需要，可省）。不要把使用者筆電的 `.venv`（6.6 GB，GPU 版）搬過去，在隊友電腦上重新 `pip install`。

## 9. 給 Claude 的工作規則（`CLAUDE.md` 也有，這裡是原因）

- 用白話繁體中文回答使用者，簡短，先給建議再說理由；術語要解釋；需要決定時問是非題。
- 使用者很在意 token：不要一輪又一輪地派 agent 審查；重要結論做一次獨立檢查就好，除非改動了數字。
- 對抗性檢查（試著推翻某個結論）一定由另一個獨立 agent 設計，不能由提出結論的人自己出題：`lessons/adversarial-tests-need-a-separate-agent.md`。
- agent 不可搜尋整顆硬碟、不可留下背景程序；每個 agent 用自己的暫存資料夾，只刪自己的；結束後檢查有沒有殘留的 `python`／`find` 程序：`lessons/workflow-agents-leave-orphan-processes.md`、`lessons/scratchpad-is-not-storage.md`、`lessons/clean-up-orphan-files.md`。
- **CPU 過載或過熱就停止自己的程序**；如果是使用者自己的工作造成的，只通知、不動它。
- 下載檔案、對外發布（GitHub、網站）、寄信之前，都要先問使用者。
- 不要寫死任何比賽未知項，全部走 YAML；不要編造任何數字。
- 在 Windows 綠燈不代表 Linux 也綠燈：`lessons/run-the-suite-on-linux.md`。

## 10. 出發前檢查清單

- [ ] 確認主辦方有沒有回覆那 8 個問題（見第 2 節）；「能不能帶自己的程式」可在官方 Discord 問。
- [ ] 把**完整的工具包資料夾**給隊友（不是 GitHub 上的論文資料包，那個少了測試、`compress.py` 和報表腳本）。可在使用者筆電上用 `python scripts/make_dist.py` 打包（約 50 MB，不含 `.venv`、資料集、權重）；`CLAUDE.md` 放根目錄，這份文件放 `docs/handover/`。
- [ ] 隊友電腦：照第 4.1、4.2 節裝好 Python 3.11、Git、Claude Code，建 `.venv` 並安裝套件，跑測試（應該 167 passed）。
- [ ] 先下載資料集和預訓練權重：`python scripts/fetch_coco_val.py --n 500`（640 px 偵測再加 `--n 2000`）；`scripts/fetch_imagenette.py` 只有 ResNet-50 壓力測試用得到；CIFAR-10、ECG5000、coco128 和預訓練權重會在第一次跑時自動下載。之後跑 `python scripts/check_data.py data/coco_val500`（它只檢查 YOLO 格式的偵測資料集）。
- [ ] **追蹤任務的資料 `data/mot_smoke`（約 4 MB）不會自動下載，也沒有腳本能重做**：從使用者筆電把 `data/mot_smoke` 複製過去，或準備真正的 MOT 格式資料。
- [ ] 每個任務用 `--quick` 各跑一次，確認整條流程在隊友電腦上通。
- [ ] 若沙箱可能是 Linux 且沒網路：把離線套件包一起帶去（`python scripts/make_dist.py --with-vendor linux-x86_64-py311`，多約 470 MB），見第 4 節。
- [ ] 簡報素材：Pareto 圖（`results/*/pareto_*.png`）、`deliverables/*/REPORT.md`、Excel 報表（`python scripts/make_report_xlsx.py`）。
- [ ] 到現場第一件事：記下沙箱的 CPU 型號、有無 AVX-VNNI、核心數、ONNX Runtime 版本、Python 版本、有沒有網路。
