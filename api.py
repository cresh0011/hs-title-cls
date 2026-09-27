# -*- coding: utf-8 -*-
"""
商品标题文本分类 Demo ——【接口模块】

把 predict.py 的 TitleClassifier 包成 HTTP 服务，供外部调用。
本模块只负责 HTTP 层：请求校验、响应组装，不包含任何模型加载 / 训练逻辑。

启动方式：
    python -X utf8 api.py
    或者：uvicorn api:app --host 127.0.0.1 --port 8000   （代码改动后自动重载）
交互式文档（启动后浏览器打开）：
    http://127.0.0.1:8000/docs
"""

import os
import sys
from contextlib import asynccontextmanager

# 必须在 import transformers（经由 predict 间接导入）之前设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 复用推理模块的分类器：模型加载逻辑只写一遍，接口层不重复实现
from predict import TitleClassifier

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ============================================================
# 1. 全局状态 + 生命周期
# ============================================================
classifier: TitleClassifier | None = None      # 服务启动时赋值，请求处理时只读


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动时加载一次模型，关闭时释放。

    ⚠ 模型必须在这里加载并挂在全局，不能在请求处理函数里加载——
      加载一次要十几秒，写在请求里等于每个请求都卡十几秒。
    """
    global classifier
    print("[启动] 正在加载模型 ...")
    classifier = TitleClassifier()
    print(f"[启动] 模型就绪 | 设备 {classifier.device.type} | "
          f"类目 {len(classifier.id2label)} 个")
    yield
    classifier = None
    print("[关闭] 模型已释放")


app = FastAPI(
    title="商品标题分类接口",
    description="模拟 HS 编码归类场景：输入中文商品标题，返回商品类目与置信度。",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# 2. 请求 / 响应模型（Pydantic 自动做格式校验和文档生成）
# ============================================================
class PredictRequest(BaseModel):
    title: str = Field(..., min_length=1, description="商品标题文本")
    topk: int = Field(3, ge=1, le=10, description="返回概率最高的前 K 个类目")


class BatchPredictRequest(BaseModel):
    titles: list[str] = Field(..., min_length=1, description="商品标题列表")
    topk: int = Field(3, ge=1, le=10, description="每条返回概率最高的前 K 个类目")


class TopKItem(BaseModel):
    label: str
    prob: float


class PredictResponse(BaseModel):
    title: str          # 原始输入
    cleaned: str        # 清洗后的文本（排查问题时能一眼看出清洗是否正常）
    label: str          # 预测类目
    confidence: float   # 置信度
    topk: list[TopKItem]


class BatchPredictResponse(BaseModel):
    count: int
    results: list[PredictResponse]


def _require_model() -> TitleClassifier:
    """模型未就绪时统一返回 503，而不是抛 500。"""
    if classifier is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成，请稍后重试")
    return classifier


# ============================================================
# 3. 接口
# ============================================================
@app.get("/health", summary="健康检查")
def health():
    """查看服务与模型状态，用于探活。"""
    return {
        "status": "ok",
        "model_loaded": classifier is not None,
        "device": classifier.device.type if classifier else None,
        "num_labels": len(classifier.id2label) if classifier else 0,
    }


@app.get("/labels", summary="查看全部类目")
def labels():
    """返回模型支持的所有类目名及其 id。"""
    clf = _require_model()
    return {"labels": [{"id": i, "label": n} for i, n in clf.id2label.items()]}


@app.post("/predict", response_model=PredictResponse, summary="单条预测")
def predict(req: PredictRequest):
    """输入一条商品标题，返回预测类目、置信度和 Top-K 概率。"""
    clf = _require_model()
    return clf.predict(req.title, topk=req.topk)


@app.post("/predict/batch", response_model=BatchPredictResponse, summary="批量预测")
def predict_batch(req: BatchPredictRequest):
    """输入一组商品标题，一次性返回全部预测结果。"""
    clf = _require_model()
    results = clf.predict_batch(req.titles, topk=req.topk)
    return {"count": len(results), "results": results}


# ============================================================
# 4. 本地启动
# ============================================================
if __name__ == "__main__":
    import uvicorn

    # 127.0.0.1：只允许本机访问。要让局域网内其他机器调用就改成 "0.0.0.0"
    uvicorn.run(app, host="127.0.0.1", port=8000)
