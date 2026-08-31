from scgpt.tasks import embed_data
import scanpy as sc

def calc_gpt(adata,model_path,gene_name='gene_name',return_key='X_scGPT'):
    """
    利用scGPT预训练模型为单细胞数据生成细胞级嵌入特征，并将嵌入结果存储到AnnData对象的obsm属性中
    核心作用：将原始基因表达谱转换为scGPT的低维嵌入特征，用于下游分析（如细胞聚类、扰动预测、细胞类型分类等）

    Args:
        adata (sc.AnnData): Scanpy的AnnData对象，存储单细胞基因表达数据（行：细胞，列：基因，X：表达矩阵）
        model_path (str): scGPT预训练模型的本地路径/远程地址，用于加载模型进行嵌入计算
        gene_name (str, optional): AnnData对象中存储基因名称的列名（对应var层的列），需与scGPT模型的基因词汇表匹配，默认'gene_name'
        return_key (str, optional): 存储scGPT嵌入结果的obsm键名，obsm用于存放细胞的多维特征矩阵，默认'X_scGPT'

    Returns:
        sc.AnnData: 更新后的原AnnData对象，其obsm[return_key]新增scGPT生成的细胞嵌入特征
    """
    # 调用scGPT的embed_data函数生成嵌入：return_new_adata=True表示返回包含嵌入结果的新AnnData对象
    # 新对象的X属性即为scGPT生成的细胞级嵌入特征
    adata_add = embed_data(adata,model_path,gene_name,return_new_adata=True)
    # 将scGPT嵌入特征复制到原adata的obsm中，保留原始数据结构的同时新增嵌入特征
    adata.obsm[return_key] = adata_add.X.copy()

    return adata