#muse_gnn_cv_neural
import torch
from utils import cross_validate
from neural_models.muse_GNN import MuSeGNN
import os
import argparse
parser = argparse.ArgumentParser()
from utils import load_train_test_datasets, load_data, get_num_genes

from sklearn.metrics import confusion_matrix
import numpy as np


# Gene Expression settings
# Options:
# getmm_gene_expression_no_outliers.csv for getmm, no outliers
# getmm_combat_seq_no_outliers_and_singles_gene_expression.csv for getmm, combat-seq, no singles, no outliers
# combat_seq_age_corrected_getmm_gene_expression_no_outliers.csv
# combat_seq_age_corrected_day_1_and_older_getmm_gene_expression_no_outliers.csv
# combat_seq_age_corrected_L4_and_younger_getmm_gene_expression_no_outliers.csv
# combat_seq_getmm_GO_filtered_gene_expression_no_singles_and_outliers.csv
parser.add_argument('--expression_path', type=str,
                    default="/data/bi1/common_datastore/getmm_combat_seq_no_outliers_and_singles_gene_expression.csv",
                    help='path to gene expression data '
                         '(default: /data/bi1/common_datastore/getmm_combat_seq_no_outliers_and_singles_gene_expression.csv)')
parser.add_argument('--label_path', type=str, default="/data/bi1/common_datastore/labels.csv",
                    help='path to labels (default: /data/bi1/common_datastore/labels.csv)')
parser.add_argument('--age_path', type=str, default="/data/bi1/common_datastore/age.csv",
                    help='path to age data (default: /data/bi1/common_datastore/age.csv)')
parser.add_argument('--experiments_path', type=str, default="/data/bi1/common_datastore/sra_to_bioproject.csv",
                    help='path to sra to bioproject mapping (default: /data/bi1/common_datastore/sra_to_bioproject.csv)')
# MLP parameters
parser.add_argument('--mlp_hidden_dim', type=int, default=512,
                    help='embedding dimensions (default: 512)')
parser.add_argument('--num_mlp_layers', type=int, default=3,
                    help='number of MLP layers total, excluding input layer (default: 3)')
# Training / Testing settings
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout (default: 0)')
parser.add_argument('--weight_decay', type=float, default=0.001,
                    help='weight decay (default: 0.001)')
parser.add_argument('--batch_size', type=int, default=1500,
                    help='batch size for training (default: 1500)')
parser.add_argument('--learning_rate', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--epochs', type=int, default=100,
                    help='num training epochs (default: 100)')
parser.add_argument('--seed', type=int, default=42, help="Seed")
parser.add_argument('--eval_model_every', type=int, default=10,
                    help="how often (in # of epochs) to evaluate the model (default: 10)") # 10epoch 진행될 때마다 검증 수행 
parser.add_argument('--train_MLP', action='store_true', help="train the pure MLP (default: False)")
parser.add_argument('--mixsplit', action='store_true', help="perform a mixsplit as described in paper (default: False)") # 기본: non - mixsplit
parser.add_argument('--num_folds', type=int, default=10, help="How many folds for cross validation (default: 10)") # 교차 검증시 10 folds
# Data Filtering
parser.add_argument('--aging_genes_only', action='store_true',
                    help="train the model using aging genes only (default: False)") #노화 관련 유전자만 사용하여 모델 학습 
# GNN parameters
parser.add_argument('--k', type=int, default=22113, help="Number of nodes to keep after Sort Pooling (default: 22113)") # sort pooling 이후 유지할 노드의 수 설정 
parser.add_argument('--num_backbone_layers', type=int, default=1, help="Number of GNN backbone layers (default 1)") # backbone layer 수
parser.add_argument('--backbone_channels', type=int, default=1,
                    help="Number of backbone features / channels (default: 1)")
parser.add_argument('--concat_input_graph', action='store_true',
                    help="Concatenate input graph features (default: False)") # 입력 그래프의 특징을 모델 출력과 결합할지 
parser.add_argument('--train_GNN', action='store_true', help="Train the pure GNN (default: False)")
parser.add_argument('--test_mode', action='store_true', help="Run script in test mode without training")



config = parser.parse_args()

os.makedirs('/data/bi1/results', exist_ok=True)

epochs = config.epochs
batch_size = config.batch_size
learning_rate = config.learning_rate
weight_decay = config.weight_decay
eval_model_every = config.eval_model_every
seed = config.seed
mlp_hidden_dim = config.mlp_hidden_dim
data = config.expression_path.split('/')[-1]
aging_genes_only = config.aging_genes_only
num_folds = config.num_folds
num_mlp_layers = config.num_mlp_layers
mixsplit = config.mixsplit
k = config.k
num_backbone_layers = config.num_backbone_layers
backbone_channels = config.backbone_channels
concat_input_graph = config.concat_input_graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(config.seed)


if config.train_GNN:
    muse_gnn_experiment_name = f"ver4_MuSeGNN-bblayers_{num_backbone_layers}-nchannels_{backbone_channels}-folds_{num_folds}" \
                               f"cat_input_{concat_input_graph}-k_{config.k}-lr_{learning_rate}-wd_{weight_decay}-bs_{batch_size}" \
                               f"-epchs_{epochs}-evalevery_{eval_model_every}-dpout_{config.dropout}" \
                               f"-mlpdim_{mlp_hidden_dim}-mlplayers_{num_mlp_layers}"

    if not (os.path.exists(f"/data/bi1/results/neural/{muse_gnn_experiment_name}/best_model_stats.txt")):
        dataset, gene_expression_graphs, pre_expression_merge_graph = load_data(config)
        train_test_dataset_list = load_train_test_datasets(dataset, gene_expression_graphs)
        train_gene_expression_graphs, labels_train, experiments_train = train_test_dataset_list[0]
        num_genes = get_num_genes(dataset)

        print('Running cross-validation with MuSe-GNN')

        MuSeGNN_model = MuSeGNN(
            input_dim=1, 
            backbone_channels=config.backbone_channels,  
            output_dim=3,  
            num_backbone_layers=config.num_backbone_layers, 
            concat_input_graph=config.concat_input_graph,  
            num_nodes=num_genes,  
            edge_attr=pre_expression_merge_graph.edge_attr,  
            edge_index=pre_expression_merge_graph.edge_index,  
            k=config.k,
            mlp_hidden_dim=config.mlp_hidden_dim,  
            mlp_num_layers=config.num_mlp_layers, 
            dropout=config.dropout, 
            device=device 
        ).to(device)


        # Cross-validation 
        cross_validate(
            MuSeGNN_model,
            train_gene_expression_graphs,
            labels_train,
            device,
            config,
            experiments_train,
            muse_gnn_experiment_name,
            num_folds,
            learning_rate,
            batch_size,
            weight_decay,
            epochs
        )
        

