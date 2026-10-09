# Data Directory — Sources & Download Links

> 本目录包含 **PPI-DynCT** 项目所用的全部数据。由于多数原始文件较大，仓库默认 **不上传原始大数据**（如 `*.h5ad`、`*.pkl`、`*.pt`、`*.zip`、`*.gz`）；请按以下链接自行下载后放入对应子目录。
>
> **Total expected size**: ~8 GB of compressed downloads (h5ad / pkl / pt / txt). 仓库中已包含的派生文件请见末尾「说明」一节。

---

## 目录结构总览

```
Data/
├── *.h5ad / *.pkl              # 单细胞表达数据与 cell-level 张量
├── *_smiles_*.csv              # 药物 SMILES 与 target 映射
├── PPI_data_v2/                # STRING PPI 派生图 (per dataset)
├── BioGRID/                    # BioGRID 原始 PPI 源文件
├── go_cache/                   # GO 语义相似度缓存
├── msigdb/                     # Hallmark 基因集 (MSigDB)
├── stable_graph/               # Bootstrap 图稳定性评估
├── 9606.* / 10090.*            # STRING 蛋白文件 (人/鼠)
├── go-basic.obo / goa_human.gaf # Gene Ontology
├── gene2vec_dim_200_iter_9_w2v.txt # gene2vec 预训练向量
└── ...
```

---

## 1. 单细胞扰动数据集（h5ad）

四个核心 scRNA-seq 扰动数据集是本项目的主要输入。

| 文件 | 数据集 | 物种 | 描述 | 下载链接 | 大小 |
|------|--------|------|------|----------|------|
| `Nault.h5ad` / `Nault_processed.h5ad` | Nault 2021 | Human (Liver) | TCDD 等 6 种化合物处理人肝类器官单细胞测序 | <https://figshare.com/articles/dataset/Measurement_of_single_cell_perturbation_response_to_environmental_toxicants/16638002> | ~185 MB |
| `McFarland_processed.h5ad` | McFarland 2020 | Human (Cell line panel) | 28 个细胞系，387 种化合物的高通量扰动筛选 (PRISM) | <https://figshare.com/articles/dataset/Connecting_Compound_Structure_and_Response_to_Identify_Compounds_with_Efficacy_in_a_Human_Cell_Model_of_Acute_Myeloid_Leukemia/13325115> | ~246 MB |
| `Chang_processed.h5ad` | Chang / Szlamanyi 2024 (NeurIPS dataset) | Human (PC9) | PC9 细胞系 3 种 EGFR/ERBB 抑制剂处理 (GNE-069 / GNE-104 / erlotinib) | 由 NeurIPS 2024 CellComp / PrePR-CT 官方提供: <https://drive.google.com/drive/folders/1HAGNtm7zB-vJOMHwLr2-EiGLcWUC2I7h> (经 `cells_Chang_corrected.pkl` 重新标注) | ~647 MB |

> **引用**:
> - Kang *et al.*, *Nature Biotechnology* 2018 — 10.1038/nbt.4042
> - Nault *et al.*, *bioRxiv* 2021 (Nault2021 figshare 上述链接)
> - McFarland *et al.*, *Cancer Discovery* 2020 (PRISM)
> - Chang *et al.* / PrePR-CT benchmark (NeurIPS 2024 Datasets & Benchmarks)

---

## 2. Cell-level 张量（cells_*.pkl / cells_*_corrected.pkl）

由 `Kang_Data.py`、`generate_nault_graphs.py`、`replogle_*_Data.py` 等预处理脚本生成。每条记录对应一个 `anndata` 对象 + 控制样本 SEACell 配对信息；`_corrected` 后缀版本使用 paper-exact 配种 (seed=42) 重新生成。

- **生成方式**: 运行 `Data_Notebooks/{Dataset}_Data.py` (或对应 `generate_*_graphs.py`)。
- **大小**: 255 MB – 1 GB / 文件（Git 不托管，运行时生成）。

---

## 3. STRING 蛋白-蛋白互作网络（PPI）

人类与小鼠 STRING v11.5 离线数据 (本项目使用 9606 人类版本；10090 仅作备查)：

| 文件 | 用途 | 下载链接 |
|------|------|----------|
| `9606.protein.links.v11.5.txt`  | 蛋白质互作边 + combined_score (0–1000) | <https://stringdb-static.org/download/protein.links.v11.5/9606.protein.links.v11.5.txt.gz> |
| `9606.protein.aliases.v11.5.txt`| ENSP → gene symbol 映射 | <https://stringdb-static.org/download/protein.aliases.v11.5/9606.protein.aliases.v11.5.txt.gz> |
| `9606.protein.info.v11.5.txt`   | protein preferred_name / size | <https://stringdb-static.org/download/protein.info.v11.5/9606.protein.info.v11.5.txt.gz> |
| `10090.protein.*.v11.5.txt`     | 鼠源 PPI (备用) | 替换上面链接中的 `9606` 为 `10090` 即可 |

> **派生文件** `Data/PPI_data_v2/{Dataset}/` (含 `ppi_global.pt`、`gene_node2vec_64d.pt`、`ppi_per_celltype/`) 由 `Data_Notebooks/build_ppi_network_v2.py --threshold 700` 生成，阈值建议 700 (主实验)、400 / 900 (消融)。

---

## 4. BioGRID（备选 PPI 源）

存放于 `Data/BioGRID/`，用于交叉验证或替代 STRING：

- `BIOGRID-ALL-LATEST.tab3.zip` — <https://downloads.thebiogrid.org/Download/BioGRID/Latest-Release/BIOGRID-ALL-LATEST.tab3.zip>
- `BIOGRID-MV-Physical-LATEST.tab3.zip` — <https://downloads.thebiogrid.org/Download/BioGRID/Latest-Release/BIOGRID-MV-Physical-LATEST.tab3.zip>

---


## 5. gene2vec 预训练词向量

`Data/gene2vec_dim_200_iter_9_w2v.txt` —— 24447 个基因 × 200 维的 BioVec 风格嵌入 (Du *et al.* 2019)。

- 下载: <https://github.com/jingchengdu/gene2vec> (release: `gene2vec_dim_200_iter_9_w2v.txt`)
- 引用: Jingcheng Du *et al.*, *NAR* 2019, "gene2vec: distributed representation of genes based on co-expression"

---

## 6. Drug SMILES / Target 注释

- `Data/drug_smiles_Chang.csv` — Chang (NeurIPS) 4 种化合物 SMILES (erlotinib / GNE-069 / GNE-104 / control)  
- `Data/drug_smiles_McFarland.csv` — McFarland 13 种参考化合物 SMILES  
- `Data/drug_target_mapping_Chang.csv` — Drug → target gene → interaction_type (source: DrugBank / ChEMBL)  
- `Data/SMILES_feat_all_datasets.csv` — RDKit 提取的 200+ 维化学描述符（融合数据集用）

> DrugBank 在线数据库: <https://go.drugbank.com/public_users/sign_up>  
> ChEMBL: <https://www.ebi.ac.uk/chembl/>

---

如需批量获取外部源数据，可在 `Data/` 目录下运行：

```bash
# 1) STRING 蛋白文件 (人类)
wget -c https://stringdb-static.org/download/protein.links.v11.5/9606.protein.links.v11.5.txt.gz
wget -c https://stringdb-static.org/download/protein.aliases.v11.5/9606.protein.aliases.v11.5.txt.gz
wget -c https://stringdb-static.org/download/protein.info.v11.5/9606.protein.info.v11.5.txt.gz
gunzip *.gz

- 上表中 *下载链接* 全部由 **本仓库脚本在 `Data_Notebooks/`、`training/config_*.yaml`、`build_*` 等代码中显式或隐含引用** 总结而得；如遇到链接失效请到对应官方网站重新搜索。
- `*.h5ad` / `*.pkl` / `*.pt` 等大文件受 Git LFS / GitHub 100MB 单文件限制，**默认不在仓库中**，请按上表自行获取。
- 项目主要数据均来自公共开放数据 (CC-BY / CC0)，使用时请遵守各原始来源的引用条款与 license。
- 若有任何下载问题或需要实验相关的中间文件 (派生 `.pt`/`.npz`)，可联系 maintainer 或在 GitHub Issue 中反馈。
