import scanpy as sc
import pandas as pd


def analyze_dataset_splits(data_path, cell_type_col='cell_type', drug_col='condition'):
    # 注意：drug_col 可能是 'condition'，也可能是 'sm_name'，视你的 h5ad 文件而定

    print(f" 正在加载数据集: {data_path} ...")
    adata = sc.read_h5ad(data_path)

    # 打印一下有哪些列，方便核对
    print(f" 数据加载成功！包含细胞数: {adata.shape[0]}, 基因数: {adata.shape[1]}")

    splits = ['split', 'split2', 'split3',"split_drugs","split_drugs2","split_drugs3"]

    for split in splits:
        if split not in adata.obs.columns:
            print(f" 警告: 找不到列 {split}，请检查数据。")
            continue

        print("\n" + "=".ljust(80, "="))
        print(f" 深度解剖切片: 【 {split.upper()} 】 ".center(80, " "))
        print("=".ljust(80, "="))

        # 获取该切片下所有的标签 (如 'train', 'ood_treated', 'test_control' 等)
        unique_labels = adata.obs[split].dropna().unique().astype(str)

        # 逻辑分组：将标签归类为 Train, IID, OOD
        train_labels = [l for l in unique_labels if 'train' in l.lower()]
        iid_labels = [l for l in unique_labels if 'test' in l.lower() or 'val' in l.lower() or 'iid' in l.lower()]
        ood_labels = [l for l in unique_labels if 'ood' in l.lower()]

        groups = {
            ' 训练集 (Train)': train_labels,
            ' 内部测试集 (IID/Test)': iid_labels,
            ' 分布外测试集 (OOD)': ood_labels
        }

        for group_name, labels in groups.items():
            if not labels:
                continue

            print(f"\n {group_name}  (标签: {labels})")
            subset = adata.obs[adata.obs[split].isin(labels)]

            # 获取该分组下的所有细胞类型
            cell_types = subset[cell_type_col].unique()

            for ct in cell_types:
                # 提取特定细胞类型的数据
                ct_subset = subset[subset[cell_type_col] == ct]
                # 获取该细胞类型接触过的所有药物
                drugs = ct_subset[drug_col].unique()

                # 为了打印整齐，提取前 3 个药物名字作为示例
                drug_examples = list(drugs)[:3]
                if len(drugs) > 3:
                    drug_examples_str = f"{drug_examples} ...等"
                else:
                    drug_examples_str = f"{drug_examples}"

                # 重点高亮：如果只有 DMSO，说明这是被扣留了药物反应的“素颜”细胞
                if len(drugs) == 1 and 'dmso' in str(drugs[0]).lower():
                    status = "️ [溶剂]"
                else:
                    status = " [全部药物]"

                print(
                    f"   细胞: {str(ct).ljust(15)} |  药物数: {str(len(drugs)).rjust(3)} | {status.ljust(15)} | 示例: {drug_examples_str}")


if __name__ == "__main__":
    # 替换为你实际的 h5ad 路径
    data_path = 'sciplex3_pp_hvgenes_scFM_resplit.h5ad'

    # ⚠️ 请确保这里的列名与你 adata.obs 中的列名完全一致
    # 比如 NeurIPS 数据集中，细胞类型列通常叫 'cell_type'
    # 药物名称列通常叫 'condition' 或 'sm_name'
    analyze_dataset_splits(data_path, cell_type_col='cell_type', drug_col='condition')

