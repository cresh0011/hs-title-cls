# -*- coding: utf-8 -*-
"""
商品标题文本分类 Demo ——【推理模块】

输入：model/ 目录（由 train.py 产出）
功能：加载微调后的模型，对商品标题做单条 / 批量预测，输出中文类目名 + 置信度
设备：自动检测，有 NVIDIA GPU 走 GPU，没有则降级 CPU

说明：本模块只做推理，不含训练和 Web 接口逻辑。
"""

import os
import sys
import json
from pathlib import Path

# 同样必须在 import transformers 之前设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 复用数据处理模块的清洗函数：训练时怎么洗，推理时就得怎么洗，
# 两边不一致会导致输入分布对不上、效果莫名下降。绝不重新复制一份。
from data_process import clean_title

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MAX_LEN = 48                      # 必须与 train.py 保持一致


def get_device() -> torch.device:
    """有可用 CUDA 用 GPU，否则降级 CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TitleClassifier:
    """商品标题分类器：封装模型加载与预测，供脚本或后续接口模块直接调用。"""

    def __init__(self, model_dir=MODEL_DIR, device=None):
        self.device = device or get_device()
        self.model_dir = Path(model_dir)

        # local_files_only=True：只读本地文件，绝不联网
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir, local_files_only=True
        )
        # eval() 关掉 dropout 等训练期行为，保证同一输入每次预测结果一致
        self.model.to(self.device).eval()

        with open(self.model_dir / "label_map.json", encoding="utf-8") as f:
            self.id2label = {int(k): v for k, v in json.load(f)["id2label"].items()}

    # ---------------- 单条预测 ----------------
    @torch.no_grad()
    def predict(self, title: str, topk: int = 3) -> dict:
        """对单条标题预测，返回类目、置信度和 Top-K 概率。"""
        return self.predict_batch([title], topk=topk)[0]

    # ---------------- 批量预测 ----------------
    @torch.no_grad()
    def predict_batch(self, titles, batch_size: int = 32, topk: int = 3) -> list:
        """对一批标题预测，内部分块推理，避免一次喂太多撑爆显存。"""
        titles = list(titles)
        cleaned = [clean_title(t) for t in titles]     # 与训练一致的清洗
        k = min(topk, len(self.id2label))
        results = []

        for start in range(0, len(cleaned), batch_size):
            chunk = cleaned[start:start + batch_size]
            enc = self.tokenizer(
                chunk,
                truncation=True,
                padding="max_length",
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to(self.device)

            probs = F.softmax(self.model(**enc).logits, dim=-1)    # [batch, num_labels]
            top_p, top_i = torch.topk(probs, k=k, dim=-1)          # 取概率最高的 k 个

            for row in range(len(chunk)):
                results.append({
                    "title": titles[start + row],
                    "cleaned": chunk[row],
                    "label": self.id2label[top_i[row, 0].item()],
                    "confidence": top_p[row, 0].item(),
                    "topk": [
                        {"label": self.id2label[top_i[row, j].item()],
                         "prob": top_p[row, j].item()}
                        for j in range(k)
                    ],
                })
        return results


def main():
    print("=" * 62)
    print("商品标题文本分类 Demo —— 推理模块")
    print("=" * 62)

    clf = TitleClassifier()
    print(f"[环境] 设备：{clf.device.type}")
    print(f"[模型] {clf.model_dir}")
    print(f"[类目] {len(clf.id2label)} 个：{'、'.join(clf.id2label.values())}")

    # 演示样本：故意混入脏字符和歧义词，检验清洗与分类是否都正常
    demos = [
        "华为 智能手机 8GB+256GB 骁龙处理器 包邮",
        "★【正品】蒙牛 纯牛奶 250ml*24盒 整箱！！！<br>",
        "苹果 平板电脑 10.4英寸 WiFi版 护眼屏",
        "佳沃 苹果 新鲜 当季 5斤装 顺丰冷链",
        "海尔 电热水器 60升 防电墙 一级能效 包安装",
        "乐高 积木 儿童 3岁以上 环保材质 生日礼物",
    ]

    print("\n" + "-" * 62)
    print("单条预测")
    print("-" * 62)
    for t in demos:
        r = clf.predict(t)
        print(f"\n输入：{r['title']}")
        if r["cleaned"] != r["title"]:
            print(f"清洗：{r['cleaned']}")          # 有脏字符时才有这一行
        print(f"预测：{r['label']}  (置信度 {r['confidence']:.4f})")
        topk_str = " | ".join(f"{x['label']} {x['prob']:.4f}" for x in r["topk"])
        print(f"Top{len(r['topk'])}：{topk_str}")

    print("\n" + "-" * 62)
    print("批量预测")
    print("-" * 62)
    batch = clf.predict_batch(demos, batch_size=4)
    print(f"共 {len(batch)} 条，前 3 条：")
    for r in batch[:3]:
        print(f"  {r['label']:<6} {r['confidence']:.4f}  {r['title'][:30]}")

    print("\n推理模块执行完毕。")


if __name__ == "__main__":
    main()
