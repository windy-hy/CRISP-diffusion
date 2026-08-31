import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import os


def generate_fcfp4_parquet(input_parquet, output_parquet, num_Bits=1024):
    print(f"正在读取原始文件: {input_parquet}")
    # 1. 读取原有的 193维 parquet 文件，提取出所有用到的 SMILES (作为 index)
    df_orig = pd.read_parquet(input_parquet)
    smiles_list = df_orig.index.tolist()

    print(f"共找到 {len(smiles_list)} 种独立药物，正在生成 {num_Bits} 维 FCFP4 指纹...")

    # 2. 初始化全 0 数组
    fcfp4_array = np.zeros((len(smiles_list), num_Bits), dtype=np.float32)

    # 3. 计算每个药物的 FCFP4 指纹
    for i, smi in enumerate(smiles_list):
        # SMILES字符串→RDKit分子对象
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            # 你的核心逻辑：半径为2，基于特征的 Morgan 指纹 (即 FCFP4)
            fcfp4 = AllChem.GetMorganFingerprintAsBitVect(
                mol, 2, useFeatures=True, nBits=num_Bits
            ).ToBitString()
            fcfp4_array[i] = np.array(list(fcfp4), dtype=np.float32)
        else:
            print(f"警告: RDKit 无法解析 SMILES {smi}")

    # 4. 保存为新的 parquet 文件，列名设为 latent_0, latent_1 ...
    columns = [f"latent_{i}" for i in range(num_Bits)]
    df_new = pd.DataFrame(fcfp4_array, index=smiles_list, columns=columns)

    df_new.to_parquet(output_parquet)
    print(f"✅ 成功！全新的 {num_Bits} 维特征已保存至: {output_parquet}")


if __name__ == "__main__":
    # 请根据你本地的实际路径进行修改
    INPUT_PATH = "embeddings_lincs_sciplex3.parquet"
    OUTPUT_PATH = "fcfp4_1024_embedding_lincs_sciplex3.parquet"

    # 确保输出目录存在
    # os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    generate_fcfp4_parquet(INPUT_PATH, OUTPUT_PATH)