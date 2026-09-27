# -*- coding: utf-8 -*-
"""
商品标题文本分类 Demo ——【模型训练模块】

输入：data/train.csv、data/val.csv（由 data_process.py 生成）
模型：bert-base-chinese + 10 类分类头（权重已下载到本地，全程不联网）
输出：model/ 目录（模型权重 + 分词器 + label_map.json）

设备：自动检测，有 NVIDIA GPU 走 GPU，没有则降级 CPU。
说明：本模块只做训练与验证，不含推理/接口逻辑。
"""

import os
import sys
import json
import time
from pathlib import Path

# ---- HF 环境变量必须在 import transformers 之前设置 ----
# 权重已经落到本地，这两行只是兜底：万一 transformers 想联网做校验，也会走镜像而不是卡死
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, classification_report

# Windows 控制台默认 GBK，分类报告里有中文类目名会直接抛 UnicodeEncodeError。
# 这里强制 UTF-8 + 容错，终端显示不了也不会让程序崩。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ============================================================
# 1. 全局配置
# ============================================================
SEED = 42                 # 随机种子，保证训练结果可复现
MAX_LEN = 48              # 实测标题最长 45 字，48 可全覆盖（BERT 默认 512，纯属浪费）
BATCH_SIZE = 16           # 4060 的 8G 显存跑这个配置绰绰有余
EPOCHS = 3                # Demo 数据量小，3 轮足够收敛
LR = 2e-5                 # BERT 微调经验学习率，太大会把预训练权重冲垮
WARMUP_RATIO = 0.1        # 前 10% 步数做学习率预热，训练更稳
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0       # 梯度裁剪阈值，防止偶发梯度爆炸

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PRETRAINED_DIR = BASE_DIR / "pretrained" / "bert-base-chinese"
OUTPUT_DIR = BASE_DIR / "model"


# ============================================================
# 2. 设备自适应
# ============================================================
def get_device() -> torch.device:
    """有可用 CUDA 就返回 GPU，否则返回 CPU —— 无 GPU 环境也能跑通，只是慢。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
# 3. 数据集
# ============================================================
class TitleDataset(Dataset):
    """商品标题数据集：title_clean 分词编码，label_id 作为监督标签。"""

    def __init__(self, texts, labels, tokenizer, max_len):
        # 一次性把全部文本编码成定长 id 序列，训练时不用再分词，省时间
        self.encodings = tokenizer(
            list(texts),
            truncation=True,          # 超长截断
            padding="max_length",     # 统一补齐到 max_len，凑 batch 方便
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # input_ids / attention_mask / token_type_ids 原样带上，labels 单独放
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ============================================================
# 4. 训练 / 评估
# ============================================================
def train_one_epoch(model, loader, optimizer, scheduler, device):
    """跑一个 epoch，返回平均训练损失。"""
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch)          # batch 里带了 labels，模型内部自动算交叉熵
        loss = outputs.loss

        loss.backward()
        # 梯度裁剪：偶发的梯度尖峰会被压到 max_norm 以内，避免一步把权重带飞
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

        optimizer.step()
        scheduler.step()                  # 学习率按 warmup + 线性衰减推进

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    """在给定数据集上推理，返回 (真实标签, 预测标签) 两个 numpy 数组。"""
    model.eval()
    all_labels, all_preds = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits            # [batch, num_labels]
        all_preds.append(logits.argmax(dim=-1).cpu())
        all_labels.append(batch["labels"].cpu())
    return torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy()


# ============================================================
# 5. 主流程
# ============================================================
def main():
    print("=" * 62)
    print("商品标题文本分类 Demo —— 模型训练模块")
    print("=" * 62)

    device = get_device()
    if device.type == "cuda":
        print(f"[环境] 设备：cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("[环境] 设备：cpu（未检测到可用 CUDA，自动降级；能跑通，只是慢）")

    torch.manual_seed(SEED)               # 固定随机源，保证每次训练结果一致
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    # ---------- 步骤 1/5：读数据 ----------
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    with open(DATA_DIR / "label_map.json", encoding="utf-8") as f:
        label_map = json.load(f)
    id2label = {int(k): v for k, v in label_map["id2label"].items()}
    num_labels = len(id2label)
    print(f"\n[1/5] 数据加载：train {len(train_df)} 条 / val {len(val_df)} 条 / {num_labels} 个类目")

    # ---------- 步骤 2/5：分词器 + 模型 ----------
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_DIR, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_DIR, num_labels=num_labels, local_files_only=True
    ).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"[2/5] 模型加载：{PRETRAINED_DIR.name}，参数量 {n_param:,}")

    # ---------- 步骤 3/5：DataLoader ----------
    train_ds = TitleDataset(train_df["title_clean"], train_df["label_id"], tokenizer, MAX_LEN)
    val_ds = TitleDataset(val_df["title_clean"], val_df["label_id"], tokenizer, MAX_LEN)
    # num_workers=0：Windows 上多进程 DataLoader 容易出问题，数据量小也用不上
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"[3/5] DataLoader 就绪：batch_size={BATCH_SIZE}，max_len={MAX_LEN}，"
          f"每轮 {len(train_loader)} 步")

    # ---------- 步骤 4/5：优化器 + 调度器 ----------
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    print(f"[4/5] 优化器就绪：AdamW(lr={LR}, wd={WEIGHT_DECAY})，"
          f"总步数 {total_steps}（warmup {warmup_steps} 步）")

    # ---------- 步骤 5/5：训练 ----------
    print(f"\n[5/5] 开始训练（{EPOCHS} epoch）")
    best_acc, best_state = 0.0, None
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        y_true, y_pred = evaluate(model, val_loader, device)
        acc = accuracy_score(y_true, y_pred)
        print(f"      epoch {epoch}/{EPOCHS} | train_loss {train_loss:.4f} | "
              f"val_acc {acc:.4f} | 耗时 {time.time() - t0:.1f}s")

        # 记录验证集上表现最好的那轮权重（存到 CPU，不占显存）
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # 回滚到最佳轮次，再出最终报告——避免报告的是最后一轮而非最好一轮
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\n最佳验证集准确率：{best_acc:.4f}")

    y_true, y_pred = evaluate(model, val_loader, device)
    target_names = [id2label[i] for i in range(num_labels)]
    print("\n分类报告（验证集）：")
    # 10 类任务只看准确率会掩盖小类效果，这里逐类看 precision / recall / f1
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

    # ---------- 保存 ----------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    # label_map.json 一起搬过来，推理模块直接读 model/ 一个目录就够，不用回头找 data/
    with open(OUTPUT_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"模型已保存到：{OUTPUT_DIR}")

    del model                            # 显式释放显存
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\n训练模块执行完毕，可以进入下一模块（推理 / 接口）。")


if __name__ == "__main__":
    main()
