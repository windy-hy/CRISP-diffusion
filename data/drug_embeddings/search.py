import sys

import pandas as pd
import numpy as np


def fetch_latent_vector(file_path, target_smiles):
    print("正在加载 Parquet 文件进内存，请稍候...")
    # 1. 读取大表
    df = pd.read_parquet(file_path)

    # 2. 检索逻辑
    # 既然截图里 SMILES 列没有名字，大概率它是 DataFrame 的 Index
    if target_smiles in df.index:
        row_data = df.loc[target_smiles]
    else:
        # 万一它不是 Index 而是某一列（比如叫 'SMILES' 或第一列）
        first_col = df.columns[0]
        # 精确匹配
        matched = df[df[first_col] == target_smiles]
        if matched.empty:
            print(f"❌ 检索失败：在表中未找到该 SMILES -> {target_smiles}")
            return None
        # 提取第一条匹配的数据
        row_data = matched.iloc[0, 1:]  # 假设第一列是 SMILES，后面全是 latent

    print(f"✅ 成功找到匹配项！")

    # 3. 提取所有 latent 特征并转为纯数值的 numpy 数组
    # 只提取列名包含 'latent_' 的列，防止把别的元数据混进去
    latent_cols = [col for col in row_data.index if 'latent_' in str(col)]
    latent_vector = row_data[latent_cols].values.astype(np.float32)

    return latent_vector


# ================= 使用演示 =================
if __name__ == "__main__":
    # 替换成你的文件真实路径
    parquet_file = 'fcfp4_1024_embedding_lincs_sciplex3.parquet'

    # 填入你想查的完整 SMILES，比如我们找到的对照组：
    # 'C1CCC(C(CC2CCCCN2)C2CCCCC2)CC1'
    # 'C[S+](C)[O-]'
    smiles_to_find = 'CS(C)=O'


    # 执行检索
    vector = fetch_latent_vector(parquet_file, smiles_to_find)

    if vector is not None:
        print("=" * 50)
        print(f"维度大小: {vector.shape} (应该为 1024)")
        print(f"数值预览: \n{vector}")
        print("=" * 50)
        # 如果你想把它存成单独的 pytorch tensor，可以直接 torch.tensor(vector)

        # 1. 解除 numpy 的打印封印（强行显示全部 1024 个数字）
        np.set_printoptions(threshold=sys.maxsize)
        print("\n👇 完整的 1024 维向量：")
        print(vector)

        # ==========================================
        # 2. 更有价值的分析：透视这个向量的“稀疏度”
        # ==========================================
        # 统计到底有几个 1，几个 0
        num_ones = int(np.sum(vector == 1.0))
        num_zeros = int(np.sum(vector == 0.0))

        print("\n" + "=" * 50)
        print(f"🔬 向量透视报告：")
        print(f"包含 '1' 的数量: {num_ones} 个")
        print(f"包含 '0' 的数量: {num_zeros} 个")

        # 找出具体是哪几个位置（索引）被激活了
        active_indices = np.where(vector == 1.0)[0]
        print(f"激活的特征索引: {active_indices}")
        print("=" * 50)