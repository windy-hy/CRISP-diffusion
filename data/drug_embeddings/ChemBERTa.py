import pandas as pd
import numpy as np
import torch
import os
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def generate_chemberta_parquet(input_parquet, output_parquet, batch_size=32,
                               model_name="seyonec/ChemBERTa-zinc-base-v1"):
    print(f"正在读取原始文件: {input_parquet}")
    # 1. 读取原有的 parquet 文件，提取出所有用到的 SMILES (作为 index)
    df_orig = pd.read_parquet(input_parquet)
    smiles_list = df_orig.index.tolist()

    print(f"共找到 {len(smiles_list)} 种独立药物。")
    print(f"正在加载 ChemBERTa 模型 ({model_name})...")

    # 2. 自动检测计算设备 (优先使用 GPU 以加速推理)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备: {device}")

    # 3. 加载 Tokenizer 和 预训练模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()  # 设置为评估模式，关闭 Dropout 等机制

    all_embeddings = []

    print(f"开始提取特征 (Batch Size = {batch_size})...")
    # 4. 分批次处理 (Batching)，防止显存/内存溢出
    with torch.no_grad():  # 不计算梯度，节省显存并加速
        for i in tqdm(range(0, len(smiles_list), batch_size), desc="Processing Batches"):
            batch_smiles = smiles_list[i: i + batch_size]

            # 对 SMILES 字符串进行分词，自动填充对齐，并截断超长序列
            inputs = tokenizer(
                batch_smiles,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)

            # 将输入送入模型
            outputs = model(**inputs)

            # 提取 [CLS] token 的输出作为整个分子的全局表示
            # outputs.last_hidden_state 的 shape 为 (batch_size, sequence_length, hidden_size)
            # [:, 0, :] 表示取所有样本的第 0 个 token (即 [CLS] 标记)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(batch_embeddings)

    # 5. 将所有批次的特征纵向拼接
    final_embeddings = np.vstack(all_embeddings)
    embedding_dim = final_embeddings.shape[1]  # 384 维

    print(f"特征提取完毕！特征维度: {embedding_dim}。正在保存...")

    # 6. 保存为新的 parquet 文件
    columns = [f"latent_{i}" for i in range(embedding_dim)]
    df_new = pd.DataFrame(final_embeddings, index=smiles_list, columns=columns)

    df_new.to_parquet(output_parquet)
    print(f"✅ 成功！全新的 ChemBERTa {embedding_dim} 维特征已保存至: {output_parquet}")


if __name__ == "__main__":
    # 请根据你本地的实际路径进行修改
    data = 'nips'

    INPUT_PATH = f"embeddings_lincs_{data}.parquet"
    OUTPUT_PATH = f"chemberta_embedding_lincs_{data}.parquet"

    # 确保输出目录存在
    # os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # 你可以根据显存大小调整 batch_size (如 16, 32, 64, 128)
    generate_chemberta_parquet(
        INPUT_PATH,
        OUTPUT_PATH,
        batch_size=32,
        model_name="DeepChem/ChemBERTa-77M-MLM"  # 替换为 v2 版本
    )