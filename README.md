# 商品标题文本分类 Demo

基于 `bert-base-chinese` 微调的中文商品标题分类项目，模拟电商场景下的 **HS 编码归类**问题：输入一条中文商品标题，输出商品类目及置信度。

按「数据 → 训练 → 推理 → 接口」四层单向拆分，每层只依赖上一层。

---

## ⚠️ 重要声明：数据是仿真的，指标无参考价值

**请先读完这一段再往下看。**

本项目的数据集**不是真实业务数据**，而是由 `data_process.py` 按固定随机种子**在本地随机生成的 2000 条仿真标题**。类目词与标题之间的对应关系（例如"骁龙处理器"只出现在手机类）是**人为设计**的，类间区分度被刻意拉到很高。

由此带来两个必须知道的后果：

1. **模型在验证集上的准确率会高到接近满分，这个数字不代表任何真实效果。** 换到真实电商数据上，指标必然大幅下降。
2. **本仓库中出现的任何准确率、置信度数字，都不可作为模型效果的证据对外引用。**

本项目真正的价值在于**工程流程跑通**（四模块分层、数据清洗、训练、推理、HTTP 服务），而不在于模型效果。把它当模板看，不要当效果看。

---

## 项目结构

```
.
├── data_process.py      # ① 数据层：生成仿真数据 + 文本清洗
├── train.py             # ② 训练层：微调 bert-base-chinese
├── predict.py           # ③ 推理层：封装 TitleClassifier 类
├── api.py               # ④ 接口层：FastAPI HTTP 服务
├── data/                # 数据产物（已提交，可直接使用）
│   ├── train.csv
│   ├── val.csv
│   └── label_map.json
├── model/               # 训练产物【未提交，需自行生成】
└── pretrained/          # 预训练权重【未提交，需自行下载】
```

**依赖方向**：`data_process.py` → `train.py` → `predict.py` → `api.py`，单向不可逆。接口层不重复实现模型加载逻辑，直接复用推理层的 `TitleClassifier`。

### 四个模块各自负责什么

| 模块 | 职责 | 不负责 |
|---|---|---|
| `data_process.py` | 生成仿真标题、清洗文本、切分数据集 | 不涉及任何模型 |
| `train.py` | 加载基础权重、微调、评估、保存到 `model/` | 不做文本清洗 |
| `predict.py` | 加载 `model/`、单条与批量推理 | 不感知 HTTP |
| `api.py` | 请求校验、响应组装、生命周期管理 | 不包含模型加载/训练逻辑 |

---

## 环境依赖

- **Python 3.11**（开发环境为 3.11.9）
- 主要依赖：

```bash
pip install torch transformers pandas numpy scikit-learn fastapi uvicorn pydantic
```

> 如有 NVIDIA 显卡，建议按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 指引安装对应 CUDA 版本的 torch，训练会快很多；无显卡也能跑（自动回退 CPU）。

---

## 快速开始

### 第 0 步：准备预训练权重（**必须做**）

`pretrained/` 未包含在仓库中（约 393MB），需要自行下载 `bert-base-chinese` 到 `pretrained/bert-base-chinese/`：

```bash
pip install huggingface_hub
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download bert-base-chinese \
    --local-dir pretrained/bert-base-chinese
```

代码中已内置 `HF_ENDPOINT=https://hf-mirror.com` 镜像设置（见 `api.py` / `predict.py`），国内网络无需额外配置。

### 第 1 步：数据（可选）

`data/` 已随仓库提交，**可直接跳到第 2 步**。如需重新生成：

```bash
python -X utf8 data_process.py
```

数据生成使用固定种子 `SEED = 42`，每次运行结果完全一致。也可以通过调整 `TOTAL_SAMPLES`、`VAL_RATIO` 等常量改变数据规模与切分比例。

### 第 2 步：训练

```bash
python -X utf8 train.py
```

产物写入 `model/`，包含模型权重与 `label_map.json`。默认超参：`MAX_LEN=48`、`BATCH_SIZE=16`、`EPOCHS=3`、`LR=2e-5`。

### 第 3 步：命令行推理

```bash
python -X utf8 predict.py
```

### 第 4 步：启动 HTTP 服务

```bash
python -X utf8 api.py
```

服务默认监听 `http://127.0.0.1:8000`，交互式文档在 `http://127.0.0.1:8000/docs`。

> `-X utf8` 不能省：标题噪声中包含 emoji，中文 Windows 默认使用 cp936 编码会直接报错。

---

## 接口说明

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查，返回模型是否加载完成、运行设备、类目数 |
| GET | `/labels` | 查看全部类目及其 id |
| POST | `/predict` | 单条标题预测，返回类目、置信度与 Top-K |
| POST | `/predict/batch` | 批量预测 |

示例：

```bash
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"title": "小米 14 Pro 骁龙8Gen3 16+512G 白色", "topk": 3}'
```

模型在服务启动时**加载一次**并挂在全局（见 `api.py` 的 `lifespan`），不会每个请求都重新加载。未加载完成时接口返回 `503` 而非 `500`。

服务默认绑定 `127.0.0.1`，仅本机可访问；如需局域网访问，把 `api.py` 末尾的 host 改为 `"0.0.0.0"`。

---

## 关于未提交的目录

`model/` 与 `pretrained/` 被 `.gitignore` 排除，原因：

1. **体积**：两者合计约 784MB。GitHub 硬性拒收超过 100MB 的单文件（其中的 `model.safetensors` 约 400MB）。
2. **可重建**：`pretrained/` 可从 HuggingFace 下载，`model/` 跑一遍 `train.py` 即可生成，没有必要占用仓库体积。

因此**克隆本仓库后无法直接运行 `predict.py` / `api.py`**，必须先完成上面「第 0 步」与「第 2 步」。

---

## 已知局限

- **数据是仿真的**，指标不具参考价值（见开头声明，这里再次强调）。
- 类目仅 10 个（乳制品、图书、平板电脑、手机、服装、水果、洗发水、热水器、玩具、笔记本电脑），远少于真实 HS 编码体系的规模。
- 文本清洗基于正则规则，针对的是本仿真数据集里的噪声模式（emoji、`&nbsp;`、营销后缀等），**迁移到真实数据需要重写**。
- 未做超参搜索、未做交叉验证，`MAX_LEN`、`EPOCHS` 等取值基于小数据集的快速实验。
- 接口层未做鉴权、限流，仅适合本地演示，**不可直接部署到公网**。
