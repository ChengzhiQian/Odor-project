# 建议放在: pythonProject/src/analysis/vis_grover_embedding.py

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

# 尝试导入 UMAP，没有的话就只用 t-SNE
try:
    import umap.umap_ as umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[INFO] umap-learn 未安装，将只使用 t-SNE（可用: pip install umap-learn）")


# ================== 配置区域 ==================

# 项目根目录（本脚本假设从 pythonProject 根目录运行）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

FP_PATH = DATA_DIR / "train_grover_fp.npz"
CSV_PATH = DATA_DIR / "train_dataset.csv"

SMILES_COL = "nonStereoSMILES"

# ☆ 你要观察的气味标签列名（很关键，改成你自己的）
LABEL_COL = "winey"  # 举例：改成你想看的那一列

# 降维时最多采样多少个点（太大会很慢）
MAX_POINTS = 2000

# 随机种子
RANDOM_STATE = 42

# ================== 1. 读取数据 ==================

print(f"[INFO] Loading GROVER fingerprints from: {FP_PATH}")
fp_npz = np.load(FP_PATH)

print("[INFO] npz keys:", fp_npz.files)
# 假设 key 叫 "fps"，如果打印出来是别的，比如 ["arr_0"]，就改成那个
fps = fp_npz["fps"]  # shape [N, D]
print("[INFO] fps shape:", fps.shape)

print(f"[INFO] Loading CSV from: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

if LABEL_COL not in df.columns:
    raise ValueError(f"标签列 {LABEL_COL} 不在 CSV 列里，请检查列名。")

if len(df) != fps.shape[0]:
    raise ValueError(
        f"CSV 行数 ({len(df)}) 与 fingerprint 行数 ({fps.shape[0]}) 不一致，"
        "请检查 train_grover_fp.npz 与 train_dataset.csv 是否对应。"
    )

# 取出标签（0/1），并处理缺失
y_raw = df[LABEL_COL].values
# 简单处理缺失：NaN -> 0（也可以选择丢掉这些样本）
y = np.where(pd.isna(y_raw), 0, y_raw).astype(float)

print(f"[INFO] 正样本数量 = {int((y == 1).sum())} / {len(y)} "
      f"({(y == 1).mean() * 100:.2f}%)")

# ================== 2. 采样子集（为了可视化） ==================

N, D = fps.shape
indices = np.arange(N)

if N > MAX_POINTS:
    print(f"[INFO] N={N} 太大，随机采样 {MAX_POINTS} 个点用于可视化")
    rng = np.random.default_rng(RANDOM_STATE)
    indices = rng.choice(indices, size=MAX_POINTS, replace=False)

fps_sub = fps[indices]
y_sub = y[indices]

print("[INFO] Subsampled fps shape:", fps_sub.shape)

# ================== 3. t-SNE 降维 ==================

print("[INFO] Running t-SNE...")
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    metric="euclidean",
    random_state=RANDOM_STATE,
)
X_tsne = tsne.fit_transform(fps_sub)  # shape [M, 2]

# ================== 4. UMAP 降维（若可用） ==================

X_umap = None
if HAS_UMAP:
    print("[INFO] Running UMAP...")
    umap_model = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=RANDOM_STATE,
    )
    X_umap = umap_model.fit_transform(fps_sub)  # shape [M, 2]

# ================== 5. 可视化：按标签上色 ==================

# 把标签拆成 0/1 两类
pos_mask = (y_sub == 1)
neg_mask = (y_sub == 0)

plt.figure(figsize=(7, 6))
plt.title(f"t-SNE of GROVER embedding (colored by {LABEL_COL})")

plt.scatter(
    X_tsne[neg_mask, 0],
    X_tsne[neg_mask, 1],
    s=8,
    alpha=0.4,
    label=f"{LABEL_COL}=0",
)
plt.scatter(
    X_tsne[pos_mask, 0],
    X_tsne[pos_mask, 1],
    s=16,
    alpha=0.8,
    label=f"{LABEL_COL}=1",
)
plt.legend()
plt.tight_layout()
out_tsne = PROJECT_ROOT / f"tsne_grover_{LABEL_COL}.png"
plt.savefig(out_tsne, dpi=300)
print(f"[INFO] t-SNE 图已保存到: {out_tsne}")

if X_umap is not None:
    plt.figure(figsize=(7, 6))
    plt.title(f"UMAP of GROVER embedding (colored by {LABEL_COL})")

    plt.scatter(
        X_umap[neg_mask, 0],
        X_umap[neg_mask, 1],
        s=8,
        alpha=0.4,
        label=f"{LABEL_COL}=0",
    )
    plt.scatter(
        X_umap[pos_mask, 0],
        X_umap[pos_mask, 1],
        s=16,
        alpha=0.8,
        label=f"{LABEL_COL}=1",
    )
    plt.legend()
    plt.tight_layout()
    out_umap = PROJECT_ROOT / f"umap_grover_{LABEL_COL}.png"
    plt.savefig(out_umap, dpi=300)
    print(f"[INFO] UMAP 图已保存到: {out_umap}")

# ================== 6. 简单统计：相关性 & 分布差异 ==================

print("\n[INFO] ========== 简单统计 ========== ")

# 6.1 各维度与标签的相关性（Pearson）
# y_sub 可能很不平衡，相关系数只是粗略参考
corrs = []
for d in range(D):
    x_d = fps[:, d]
    # 避免全常数
    if x_d.std() < 1e-8:
        corr = 0.0
    else:
        corr = np.corrcoef(x_d, y)[0, 1]
    corrs.append(corr)

corrs = np.array(corrs)
abs_corrs = np.abs(corrs)
topk = 10
top_idx = np.argsort(-abs_corrs)[:topk]

print(f"[INFO] 与标签 {LABEL_COL} 相关性绝对值最高的 {topk} 个维度：")
for i in top_idx:
    print(
        f"  dim {i:4d}: corr = {corrs[i]: .4f}"
    )

# 6.2 正负样本在 embedding 空间的均值差异
fps_pos = fps[y == 1]
fps_neg = fps[y == 0]

if len(fps_pos) > 0 and len(fps_neg) > 0:
    mean_pos = fps_pos.mean(axis=0)
    mean_neg = fps_neg.mean(axis=0)
    diff = mean_pos - mean_neg

    # 看看差异最大的几个维度
    abs_diff = np.abs(diff)
    top_idx_diff = np.argsort(-abs_diff)[:topk]

    print(f"\n[INFO] 正/负样本在 embedding 均值差距最大的 {topk} 个维度：")
    print("      (mean_pos - mean_neg)")
    for i in top_idx_diff:
        print(
            f"  dim {i:4d}: diff = {diff[i]: .4f}"
        )

    # 也可以打印整体 L2 距离
    l2_dist = np.linalg.norm(mean_pos - mean_neg)
    print(f"\n[INFO] 正负样本均值向量的 L2 距离: {l2_dist:.4f}")
else:
    print("[WARN] 正或负样本不足，无法计算均值差异。")
