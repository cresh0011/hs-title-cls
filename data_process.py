# -*- coding: utf-8 -*-
"""
商品标题文本分类 Demo ——【数据处理模块】

场景：电商商品标题 -> 商品类目，类比海关 HS 编码归类。

数据来源：本脚本内置随机生成的 2000 条仿真中文商品标题，结构对齐公开数据集
        online_shopping_10_cats（title / label 两列），不需要联网下载。
        ⚠ 仿真数据仅用于打通 Demo 流程，不代表真实业务分布，不能据此得出任何业务结论。

输出文件（默认写到脚本同级的 data/ 目录）：
    data/train.csv       训练集（8）
    data/val.csv         验证集（2）
    data/label_map.json  标签与 id 的双向映射，供后续训练模块复用

本模块只用 CPU（pandas 文本处理），不涉及任何模型训练 / 推理逻辑。
"""

import re
import json
import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ============================================================
# 1. 全局配置
# ============================================================
SEED = 42                                    # 随机种子：保证每次运行结果完全一致
TOTAL_SAMPLES = 2000                         # 仿真样本总数
VAL_RATIO = 0.2                              # 验证集比例，即 8:2 分割

BASE_DIR = Path(__file__).resolve().parent   # 脚本所在目录，避免相对路径踩坑
DATA_DIR = BASE_DIR / "data"                 # 结果输出目录


# ============================================================
# 2. 环境探测
#    本模块纯 CPU 处理，这里只做探测+打印，函数留给后续训练模块直接复用。
# ============================================================
def detect_device() -> str:
    """有 NVIDIA GPU 且 CUDA 可用就返回 GPU 信息，否则自动降级 CPU。"""
    try:
        import torch
    except ImportError:
        return "cpu（未安装 torch，本模块用不到，训练模块再装）"
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})"
    return "cpu（未检测到可用 CUDA，自动降级）"


# ============================================================
# 3. 仿真数据词库
#    10 个商品类目，括号里是对应的 HS 章节，用来模拟“标题 -> 编码归类”的场景。
#    类目命名参考公开数据集 online_shopping_10_cats，把不适用的“酒店”换成了“玩具”。
# ============================================================
CATEGORY_SPECS = {
    # HS 8517 电话机及通信设备
    "手机": {
        "brands": ["华为", "小米", "OPPO", "vivo", "荣耀", "三星", "一加"],
        "heads": ["智能手机", "5G手机", "全网通手机", "双卡双待手机"],
        "attrs": ["8GB+256GB", "12GB+512GB", "骁龙处理器", "6.7英寸", "5000mAh",
                  "拍照旗舰", "超级快充", "曲面屏"],
        "tails": ["正品", "全新", "官方旗舰店", "包邮", "全国联保", "现货速发"],
    },
    # HS 8471 自动数据处理设备
    "笔记本电脑": {
        "brands": ["联想", "戴尔", "惠普", "华硕", "ThinkPad", "机械革命", "宏碁"],
        "heads": ["笔记本电脑", "轻薄本", "游戏本", "商务办公本"],
        "attrs": ["i7处理器", "16G内存", "512G固态", "14英寸", "独立显卡",
                  "高色域屏", "长续航", "背光键盘"],
        "tails": ["正品", "全新", "国行", "包邮", "顺丰发货", "支持分期"],
    },
    # HS 8471 自动数据处理设备
    "平板电脑": {
        "brands": ["苹果", "华为", "小米", "荣耀", "三星", "联想"],
        "heads": ["平板电脑", "平板", "学习平板", "娱乐平板"],
        "attrs": ["10.4英寸", "6GB+128GB", "WiFi版", "护眼屏", "全网通",
                  "高刷屏", "二合一", "带键盘"],
        "tails": ["正品", "全新", "官方标配", "包邮", "全国联保", "现货"],
    },
    # HS 61/62 针织或非针织服装
    "服装": {
        "brands": ["优衣库", "南极人", "恒源祥", "海澜之家", "太平鸟", "森马"],
        "heads": ["T恤", "衬衫", "卫衣", "连衣裙", "牛仔裤", "外套", "针织衫"],
        "attrs": ["纯棉", "短袖", "宽松版型", "加绒", "夏季新款", "加大码",
                  "修身", "百搭"],
        "tails": ["正品", "包邮", "七天无理由", "当季新品", "商场同款"],
    },
    # HS 08 食用水果及坚果
    "水果": {
        "brands": ["百果园", "佳沃", "褚橙", "都乐"],
        "heads": ["苹果", "橙子", "车厘子", "芒果", "猕猴桃", "柚子", "阳光玫瑰葡萄"],
        "attrs": ["新鲜", "当季", "进口", "5斤装", "大果", "现摘现发",
                  "甜度高", "整箱"],
        "tails": ["包邮", "坏果包赔", "顺丰冷链", "产地直发"],
    },
    # HS 04 乳品
    "乳制品": {
        "brands": ["蒙牛", "伊利", "光明", "三元", "君乐宝"],
        "heads": ["纯牛奶", "酸奶", "全脂奶粉", "儿童成长牛奶", "奶酪棒"],
        "attrs": ["250ml*24盒", "整箱装", "高钙", "无蔗糖", "低脂",
                  "早餐奶", "常温保存"],
        "tails": ["正品", "包邮", "旗舰店", "新鲜日期", "整箱特惠"],
    },
    # HS 33 精油及化妆品、盥洗用品
    "洗发水": {
        "brands": ["海飞丝", "潘婷", "清扬", "沙宣", "飘柔", "施华蔻"],
        "heads": ["洗发水", "洗发露", "护发素", "洗护套装", "沐浴露"],
        "attrs": ["去屑止痒", "控油蓬松", "750ml", "无硅油", "持久留香",
                  "柔顺修复", "氨基酸"],
        "tails": ["正品", "包邮", "官方旗舰店", "买一送一", "家庭装"],
    },
    # HS 8516 电热的快速热水器
    "热水器": {
        "brands": ["海尔", "美的", "万和", "林内", "史密斯", "华帝"],
        "heads": ["电热水器", "燃气热水器", "即热式热水器", "储水式热水器"],
        "attrs": ["60升", "家用", "速热", "恒温", "防电墙", "一级能效",
                  "智能预约", "双胆"],
        "tails": ["正品", "包安装", "全国联保", "以旧换新", "厂家直销"],
    },
    # HS 49 印刷品
    "图书": {
        "brands": ["新华书店", "中信出版", "人民邮电", "磨铁"],
        "heads": ["图书", "小说", "教材", "词典", "绘本", "科普读物"],
        "attrs": ["正版", "精装", "全套", "注音版", "双语", "彩图",
                  "大字版", "典藏版"],
        "tails": ["包邮", "现货", "七天无理由", "赠书签"],
    },
    # HS 95 玩具
    "玩具": {
        "brands": ["乐高", "布鲁可", "奥迪双钻", "邦宝"],
        "heads": ["积木", "拼图", "遥控汽车", "毛绒玩具", "益智玩具", "早教玩具"],
        "attrs": ["儿童", "3岁以上", "男孩女孩通用", "环保材质", "大颗粒",
                  "可拆卸", "声光互动"],
        "tails": ["包邮", "生日礼物", "正品", "礼盒装", "安全认证"],
    },
}

# 各类目采样权重：故意造成轻微的类别不均衡，比完全均匀更接近真实电商分布，
# 顺便也能验证后面的分层抽样（stratify）确实起作用。权重和恰好为 10.0。
CATEGORY_WEIGHTS = {
    "手机": 1.3, "笔记本电脑": 1.1, "平板电脑": 1.0, "服装": 1.4,
    "水果": 1.0, "乳制品": 0.9, "洗发水": 1.1, "热水器": 0.7,
    "图书": 0.8, "玩具": 0.7,
}

# 电商标题里常见的“脏数据”，用来验证清洗逻辑真的生效
NOISE_TOKENS = [
    "!!!", "~~~", "★★★", "&nbsp;", "<br>", "😀", "＃", "　", "  ",
]


# ============================================================
# 4. 仿真数据生成
# ============================================================
def _compose_title(spec: dict, rng: random.Random) -> str:
    """把 品牌 / 核心词 / 属性词 / 营销后缀 随机拼成一条电商风格的标题。"""
    parts = []
    if rng.random() < 0.85:                      # 85% 的标题带品牌
        parts.append(rng.choice(spec["brands"]))
    parts.append(rng.choice(spec["heads"]))      # 核心词必带，这是类目的关键信号
    n_attr = rng.randint(1, 3)                   # 随机 1~3 个属性词
    attrs = rng.sample(spec["attrs"], n_attr)    # sample 保证同一标题内属性不重复
    rng.shuffle(attrs)                           # 打乱顺序，让语序更多样
    parts.extend(attrs)
    if rng.random() < 0.6:                       # 60% 的标题带营销后缀
        parts.append(rng.choice(spec["tails"]))
    return " ".join(parts)


def _inject_noise(title: str, rng: random.Random) -> str:
    """随机往标题里插 1~3 个噪声串，模拟真实抓取数据的脏乱情况。"""
    for _ in range(rng.randint(1, 3)):
        pos = rng.randint(0, len(title))         # 随机插入位置（含首尾）
        title = title[:pos] + rng.choice(NOISE_TOKENS) + title[pos:]
    return title


def generate_raw_dataset(total: int = TOTAL_SAMPLES) -> pd.DataFrame:
    """生成仿真原始数据，返回 DataFrame，两列：title / label。"""
    rng = random.Random(SEED)                    # 独立随机源，不污染全局 random 状态
    labels = list(CATEGORY_SPECS.keys())
    weights = [CATEGORY_WEIGHTS[c] for c in labels]

    # 按权重把总数分配到各类目
    counts = [round(total * w / sum(weights)) for w in weights]
    # round 会带来 ±1 的累计误差，差额补到第一个类目，保证总数严格 = total
    counts[0] += total - sum(counts)

    rows, seen = [], set()
    for label, n in zip(labels, counts):
        spec = CATEGORY_SPECS[label]
        made, guard = 0, 0
        while made < n and guard < n * 50:       # guard 防止词库耗尽时死循环
            guard += 1
            title = _compose_title(spec, rng)
            if title in seen:                    # 标题重复就重新生成，保证样本唯一
                continue
            seen.add(title)
            rows.append({"title": _inject_noise(title, rng), "label": label})
            made += 1
        if made < n:
            print(f"[警告] 类目『{label}』词库组合已用尽，实际生成 {made}/{n} 条")

    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)  # 整体打散


# ============================================================
# 5. 文本清洗
# ============================================================
RE_HTML_TAG    = re.compile(r"<[^>]+>")            # <br> </p> 之类标签
RE_HTML_ENTITY = re.compile(r"&[^;]{1,10};")       # HTML 实体，如 &nbsp; &amp;
# 白名单式过滤：只保留 中文 / 英文 / 数字 / 规格符号(+ * . -) / 空白，其余全部删掉。
# 保留 + * . - 是因为标题里的规格会用到，比如 “8GB+256GB”“250ml*24盒”。
# 保留空白(\s) 是为了让下一步能把“多余空格”合并成一个——漏掉它空格会在这一步被直接删光。
RE_KEEP        = re.compile(r"[^一-龥a-zA-Z0-9\+\*\.\-\s]")
RE_MULTI_SPACE = re.compile(r"\s+")                # 连续空白（含全角空格）压成一个


def clean_title(text: str) -> str:
    """文本清洗：去 HTML 标签 -> 去 HTML 实体 -> 去特殊符号 -> 合并多余空格。

    ⚠ 两个必须踩对的点：
      1) 实体必须在“字符过滤”之前处理，否则 &nbsp; 里的 nbsp 是纯字母、会被原样保留；
      2) 实体正则要写成 &[^;]{1,10}; 而不是 &[a-zA-Z]{2,8};——
         因为脏字符可能直接插进实体内部、把 &nbsp; 劈成 &nbs😀p;，
         限定 [a-zA-Z] 就匹配不上，放宽成“& 与 ; 之间的短串”才拦得住。
    """
    text = RE_HTML_TAG.sub(" ", text)        # 1. <br> -> 空格
    text = RE_HTML_ENTITY.sub(" ", text)     # 2. &nbsp; -> 空格
    text = RE_KEEP.sub("", text)             # 3. emoji / ★ / ＃ 等一律删除
    text = RE_MULTI_SPACE.sub(" ", text)     # 4. 多个空格、全角空格合并成一个
    return text.strip()                      # 5. 去掉首尾空白


# ============================================================
# 6. 统计信息打印
# ============================================================
def print_distribution(df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    """打印样本量与各类别在训练/验证集上的分布。"""
    print("\n" + "=" * 62)
    print("数据集统计信息")
    print("=" * 62)
    print(f"总样本数：{len(df)}")
    print(f"训练集  ：{len(train_df)} 条（{len(train_df) / len(df):.1%}）")
    print(f"验证集  ：{len(val_df)} 条（{len(val_df) / len(df):.1%}）")
    print(f"类目数量：{df['label'].nunique()}")

    summary = pd.DataFrame({
        "train": train_df["label"].value_counts(),
        "val": val_df["label"].value_counts(),
    }).fillna(0).astype(int)
    summary["合计"] = summary["train"] + summary["val"]
    summary["占比"] = (summary["合计"] / len(df) * 100).round(1).astype(str) + "%"
    summary = summary.sort_values("合计", ascending=False)

    print("\n各类别分布（按样本量降序）：")
    print(summary.to_string())


# ============================================================
# 7. 主流程
# ============================================================
def main():
    print("=" * 62)
    print("商品标题文本分类 Demo —— 数据处理模块")
    print("=" * 62)
    print(f"[环境] 计算设备：{detect_device()}")

    # ---------- 步骤 1/5：生成仿真原始数据 ----------
    df = generate_raw_dataset()
    print(f"\n[1/5] 生成仿真原始数据：{len(df)} 条")
    print(f"      字段：{list(df.columns)}")

    # ---------- 步骤 2/5：文本清洗 ----------
    df["title_clean"] = df["title"].map(clean_title)
    print(f"\n[2/5] 文本清洗完成，示例（前 3 条）：")
    for _, row in df.head(3).iterrows():
        print(f"      清洗前：{row['title']}")
        print(f"      清洗后：{row['title_clean']}")

    # 清洗后可能产生空文本或重复文本，都需要处理掉
    n_before = len(df)
    df = df[df["title_clean"].str.len() > 0]
    n_empty = n_before - len(df)
    df = df.drop_duplicates(subset=["title_clean"]).reset_index(drop=True)
    n_dup = n_before - n_empty - len(df)
    print(f"      剔除空文本 {n_empty} 条，剔除清洗后重复 {n_dup} 条，剩余 {len(df)} 条")

    # ---------- 步骤 3/5：标签编码 ----------
    # LabelEncoder 会按类别名排序后依次编号，得到 0 ~ 9 的整数标签
    encoder = LabelEncoder()
    df["label_id"] = encoder.fit_transform(df["label"])
    label2id = {str(name): int(idx) for idx, name in enumerate(encoder.classes_)}
    id2label = {str(idx): str(name) for idx, name in enumerate(encoder.classes_)}
    print(f"\n[3/5] 标签编码完成，共 {len(label2id)} 个类目：")
    print(f"      {label2id}")

    # ---------- 步骤 4/5：8:2 分层切分 ----------
    # stratify=df["label_id"] 表示按标签分层抽样，保证验证集里各类目比例与全集一致；
    # 如果去掉它，小类目有可能在验证集里一条都没有。
    train_df, val_df = train_test_split(
        df,
        test_size=VAL_RATIO,
        random_state=SEED,
        stratify=df["label_id"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print(f"\n[4/5] 数据集切分完成：train {len(train_df)} 条 / val {len(val_df)} 条")

    # ---------- 步骤 5/5：保存结果 ----------
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_cols = ["title", "title_clean", "label", "label_id"]   # title 保留原始文本，便于排查
    train_path = DATA_DIR / "train.csv"
    val_path = DATA_DIR / "val.csv"
    map_path = DATA_DIR / "label_map.json"

    # encoding="utf-8-sig"：Windows 下用 Excel 直接打开不会中文乱码
    train_df[out_cols].to_csv(train_path, index=False, encoding="utf-8-sig")
    val_df[out_cols].to_csv(val_path, index=False, encoding="utf-8-sig")
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label},
                  f, ensure_ascii=False, indent=2)

    print(f"\n[5/5] 文件已保存：")
    print(f"      {train_path}")
    print(f"      {val_path}")
    print(f"      {map_path}")

    # ---------- 打印统计信息 ----------
    print_distribution(df, train_df, val_df)
    print("\n数据处理模块执行完毕，可以进入下一模块（模型训练）。")


if __name__ == "__main__":
    main()
