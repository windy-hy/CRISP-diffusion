
from pathlib import Path
from typing import List

import pandas as pd
import torch


def get_chemical_representation(
    smiles: List[str],
    embedding_model: str,
    data_df=None,
    device="cuda",
):
    """
    从预计算的化学嵌入数据中提取药物的化学表征，返回PyTorch的Embedding层（避免实时运行嵌入模型，提升效率）

    核心逻辑：
        1. 优先从传入的data_df（或其文件路径）中加载预计算的化学嵌入（支持批量快速获取）
        2. 若未提供data_df，仅支持"zeros"模式（返回固定维度的零向量作为占位嵌入）
        3. 最终将嵌入封装为冻结的Embedding层，可直接嵌入模型作为药物特征输入

    参数说明：
        smiles: List[str] - 药物的标准化SMILES字符串列表（顺序与训练集药物一致，确保索引匹配）
        embedding_model: str - 化学嵌入模型标识（当前仅支持"zeros"，用于无预计算嵌入时的占位）
        data_df: Union[pd.DataFrame, str, None] - 预计算化学嵌入的数据源（默认None）
                - 若为str：视为parquet文件路径，读取该文件作为嵌入DataFrame
                - 若为pd.DataFrame：直接使用该DataFrame（索引为SMILES字符串，列为嵌入维度）
                - 若为None：必须指定embedding_model="zeros"，返回零向量嵌入
        device: str - 嵌入张量的存储设备（默认"cuda"，支持"cpu"）

    返回值：
        torch.nn.Embedding - 药物化学嵌入层（shape: [len(smiles), 嵌入维度]）
                - 嵌入权重由预计算数据或零向量初始化
                - freeze=True：冻结嵌入层权重，训练时不更新（预计算嵌入无需微调）
    """
    """
    Given a list of SMILES strings, returns the embeddings produced by the embedding model.
    The embeddings are loaded from disk without ever running the embedding model.

    :return: torch.nn.Embedding, shape [len(smiles), dim_embedding]. Embeddings are ordered as in `smiles`-list.
    """
    # 处理data_df：若为文件路径，读取parquet文件
    if isinstance(data_df, str):
        df = pd.read_parquet(data_df)
    else:
        df = data_df

    if df is not None:
        # 按smiles列表顺序提取嵌入（确保嵌入与药物列表一一对应）
        # df.loc[smiles]：通过SMILES索引获取对应行，values转为numpy数组
        emb = torch.tensor(df.loc[smiles].values, dtype=torch.float32, device=device)
        assert emb.shape[0] == len(smiles)
    else:
        # 无预计算嵌入时，仅支持"zeros"模式（生成零向量作为占位）
        assert embedding_model == "zeros"
        emb = torch.zeros((len(smiles), 256))
    # 将嵌入张量封装为Embedding层，freeze=True表示训练时不更新权重（预计算嵌入无需优化）
    return torch.nn.Embedding.from_pretrained(emb, freeze=True)
