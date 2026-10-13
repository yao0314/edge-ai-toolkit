# 現場要裝什麼

目標：從一台乾淨的筆電到「`pytest` 175 passed」。約 20–40 分鐘，**取決於網速**。

---

## 一、先確認的三件事（30 秒，但決定後面怎麼走）

```bash
python --version          # 要 3.11.x
nvidia-smi                # 有沒有 GPU，有的話記下 VRAM
ping -n 2 pypi.org        # 有沒有網路
```

| 結果 | 影響 |
|---|---|
| 不是 3.11 | 版本都是對 3.11.9 釘的，其他版本沒測過 |
| 沒有 GPU | 蒸餾會慢 5–10 倍，KD epoch 要砍半，架構只試最小的兩個 |
| **沒有網路** | 見第四節，**這是最大的風險** |

---

## 二、要裝的東西

### 2.1 系統層級（只有一樣）

**Python 3.11**。沒有別的。不需要 CUDA toolkit（PyTorch 的 wheel 自帶）、不需要 Visual Studio、不需要 conda。

```powershell
winget install Python.Python.3.11
```

### 2.2 Python 套件

```bash
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -r requirements-optional.txt
```

`requirements.txt` 有 **25 個釘住版本的套件**。主要的幾個與用途：

| 套件 | 用途 | 下載量 |
|---|---|---|
| `torch` 2.14.0 ＋ `torchvision` | 訓練、蒸餾、剪枝 | **~200 MB（CPU）／~2.5 GB（CUDA）** |
| `onnx` ＋ `onnxruntime` 1.30.0 | 匯出與 INT8 量化、所有延遲量測 | ~60 MB |
| `torch-pruning` 1.6.1 | 結構化剪枝 | 小 |
| `timm` 1.0.29 | MobileViT／DeiT／ViT 架構 | 小 |
| `ultralytics` 8.4.147 ＋ `lap` ＋ `pycocotools` | 偵測任務（分類用不到，但 import 會檢查） | ~50 MB |
| `pandas` `numpy` `scipy` `matplotlib` `scikit-learn` `openpyxl` | 表格、Pareto 圖、Excel 報表 | ~80 MB |
| `psutil` `pyyaml` `tqdm` `pytest` | 量測、設定、測試 | 小 |

`requirements-optional.txt` 的 5 個裡，**比賽只需要前兩個**：

| 套件 | 要不要 |
|---|---|
| `qai-hub` 0.55.0 | **要**。Resource 和 Energy 兩個各 20 分都要它佐證 |
| `codecarbon` | **要**。比賽文件點名的能耗工具 |
| `ai-edge-litert` / `tensorflow-cpu` / `tf2onnx` | **不要**。只有交 TFLite 才用得到，而且 `tensorflow-cpu` 自己就 250 MB |

> 不要裝 `tensorflow-cpu`，除非主辦方明確要求 TFLite 格式。

**總下載量：CPU 版約 400–500 MB，CUDA 版約 2.7 GB。**

### 2.3 Qualcomm AI Hub 的 token（要帳號，不是裝軟體）

在 https://aihub.qualcomm.com 註冊（免費），Account Settings 產 API token，然後：

```bash
.venv\Scripts\qai-hub.exe configure --api_token <你的TOKEN>
```

寫進 `~/.qai_hub/client.ini`。**這一步要網路，而且只能本人做。**

驗證：

```bash
.venv\Scripts\python -c "from core import aihub; print(len(aihub.list_devices()), 'devices')"
```

應該印出 `76 devices` 左右。

---

## 三、驗收

```bash
.venv\Scripts\python -m pytest -q
```

**必須是 `175 passed`。** 不是的話先修這個，其他都別做。

常見失敗：

| 症狀 | 原因 |
|---|---|
| `ModuleNotFoundError` | `requirements-optional.txt` 沒裝 |
| 大量 skip | 正常，用 `-rs` 看跳過的理由 |
| `lap` 編譯失敗 | 它需要編譯器；分類任務用不到，可以 `pip install --no-deps` 跳過 ultralytics |

---

## 四、沒有網路的話（**最大風險**）

### 4.1 會壞掉的東西

| 缺什麼 | 後果 |
|---|---|
| pip 套件 | 完全跑不起來 |
| **torchvision 的 ImageNet 預訓練權重** | **學生從零訓練，精度掉到約 70%，離 97% 的門檻差 20 個百分點。整個壓縮策略歸零。** |
| AI Hub | Energy 從 Good 掉到 Fair，約 −5 分 |

第二項最致命，而且最容易被忽略——它不是套件，是執行時才下載的檔案。

### 4.2 已經解決的部分

**打包檔 `edge-ai-toolkit.zip` 裡已經含 7 個預訓練檢查點**（`weights/torch/hub/checkpoints/`，188 MB）：

```
mobilenet_v2  mobilenet_v3_small  shufflenetv2_x0.5  shufflenetv2_x1  mnasnet0.5
resnet18  resnet50
```

前五個就是學生階梯用的架構。**解壓之後它們就在正確的位置，torchvision 會直接讀取，不會再去下載。**

### 4.3 還沒解決的部分：pip 套件

專案裡有離線 wheel 包（`vendor/`，Linux x86_64，Python 3.11／3.12 各一份，每份約 470 MB），但**預設沒有打進 zip**。要帶就：

```bash
python scripts/make_dist.py --with-vendor linux-x86_64-py311
```

現場安裝：

```bash
pip install --no-index --find-links <bundle> -r <bundle>/requirements.txt
```

**注意：vendor 裡只有 Linux 的 wheel。如果現場是 Windows 而且沒網路，這條路走不通**——必須在出發前就把 `.venv` 整個帶過去，或在有網路的地方先裝好。

---

## 五、出發前檢查清單

- [ ] Python 3.11 裝好
- [ ] `pip install -r requirements.txt -r requirements-optional.txt` 跑完
- [ ] **不要**裝 `tensorflow-cpu`
- [ ] `pytest -q` → 175 passed
- [ ] AI Hub token 設好，`list_devices()` 回傳 76 左右
- [ ] 解壓後確認 `weights/torch/hub/checkpoints/` 有 7 個 `.pth`
- [ ] 跑一次 `python scripts/compress.py --task classification --quick`，確認整條路通
- [ ] 照 `COMPETITION_DAY.md` 演習一次，計時

最後一項最重要。手冊裡的時間表是估計值，沒有人實際走過。
