from .SetTransformer import SAB
from .gnn import GNNGraph

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import torch
import math
import os
import random

from .layers import GraphConvolution
from .SetTransformer import SAB, PMA, MAB, Encoder_SAB

# 降维可视化：
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, ehr_adj, ddi_adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device
        ddi_adj = ddi_adj.cpu()
        ehr_adj = self.normalize(ehr_adj + np.eye(ehr_adj.shape[0]))
        ddi_adj = self.normalize(ddi_adj + np.eye(ddi_adj.shape[0]))

        self.ehr_adj = torch.FloatTensor(ehr_adj).to(device)
        self.ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)
        self.gcn3 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        ehr_node_embedding = self.gcn1(self.x, self.ehr_adj)
        ddi_node_embedding = self.gcn1(self.x, self.ddi_adj)

        ehr_node_embedding = F.relu(ehr_node_embedding)
        ddi_node_embedding = F.relu(ddi_node_embedding)
        ehr_node_embedding = self.dropout(ehr_node_embedding)
        ddi_node_embedding = self.dropout(ddi_node_embedding)

        ehr_node_embedding = self.gcn2(ehr_node_embedding, self.ehr_adj)
        ddi_node_embedding = self.gcn3(ddi_node_embedding, self.ddi_adj)
        return ehr_node_embedding, ddi_node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

class AdjAttenAgger(torch.nn.Module):
    def __init__(self, Qdim, Kdim, mid_dim, *args, **kwargs):
        super(AdjAttenAgger, self).__init__(*args, **kwargs)
        self.model_dim = mid_dim
        self.Qdense = torch.nn.Linear(Qdim, mid_dim)
        self.Kdense = torch.nn.Linear(Kdim, mid_dim)
        # self.use_ln = use_ln

    def forward(self, main_feat, other_feat, mask=None):
        Q = self.Qdense(main_feat)
        K = self.Kdense(other_feat)
        Attn = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.model_dim)
        # Q = main_feat
        # K = other_feat
        # Attn = torch.nn.functional.cosine_similarity(Q, K)
        # Attn[Attn < 0] = -1e9

        if mask is not None:
            Attn = torch.masked_fill(Attn, mask, -(1 << 32))

        # Attn = torch.softmax(Attn, dim=-1)

        return Attn

class SSPNetModel(torch.nn.Module):
    def __init__(
        self,  emb_dim, voc_size,
        ehr_adj, ddi_adj, use_embedding=False,
        device=torch.device('cpu'), dropout=0.5, *args, **kwargs
    ):
        super(SSPNetModel, self).__init__(*args, **kwargs)
        self.device = device
        self.emb_dim = emb_dim
        self.nhead = 2
        self.med_num = voc_size[2]

        score_extractor = [
            torch.nn.Linear(emb_dim, emb_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim // 2, 1)
        ]
        self.score_extractor = torch.nn.Sequential(*score_extractor)
        self.gcn = GCN(voc_size=voc_size[2], emb_dim=emb_dim, ehr_adj=ehr_adj, ddi_adj=ddi_adj, device=device)
        self.med_embedding = nn.Sequential(
            nn.Embedding(voc_size[2], emb_dim),
            nn.Dropout(0.3)
        )
        self.diag_embedding = nn.Sequential(
            nn.Embedding(voc_size[0], emb_dim),
            nn.Dropout(0.3)
        )
        self.proc_embedding = nn.Sequential(
            nn.Embedding(voc_size[1], emb_dim),
            nn.Dropout(0.3)
        )

        self.diag_encoder = Encoder_SAB(emb_dim, emb_dim, self.nhead)
        self.proc_encoder = Encoder_SAB(emb_dim, emb_dim, self.nhead)
        self.decoder = MedTransformerDecoder_all(emb_dim, self.nhead)
        self.pma_d = PMA(emb_dim, self.nhead)
        self.pma_p = PMA(emb_dim, self.nhead)
        self.aggregator = AdjAttenAgger(emb_dim, emb_dim, emb_dim)
        self.W_z = torch.nn.Sequential(*score_extractor)
        self.inter = nn.Parameter(torch.FloatTensor(1))
        self.garm = nn.Parameter(torch.FloatTensor(1))
        self.W_visit = torch.nn.Linear(emb_dim * 2, emb_dim)
        self.seq_encoders = torch.nn.ModuleList([
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True),
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True)])
        self.Out_visit = torch.nn.Linear(emb_dim * 2, emb_dim)

        # todo: FF
        # self.FF = nn.Linear(emb_dim, emb_dim)


    def forward(
        self, med, patient_data,
        ddi_mask_H, tensor_ddi_adj
    ):
        ##################### 药物表征 #########################################
        med_emb = self.med_embedding(med)
        ehr_embedding, ddi_embedding = self.gcn()
        med_ehr_ddi = ehr_embedding - self.inter * ddi_embedding
        med_repr = med_emb + med_ehr_ddi
        # med_repr = med_emb  # Ablation w/o Ge
        # med_repr = med_ehr_ddi

        # 可视化：
        # a = self.visual(med_repr)
        #######################################################################

        #################### 患者表征 #############################################
        diag = torch.LongTensor([patient_data[-1][0]]).to(self.device)
        proc = torch.LongTensor([patient_data[-1][1]]).to(self.device)
        d_emb = self.diag_embedding(diag)
        p_emb = self.proc_embedding(proc)
        d_repr = self.diag_encoder(d_emb)
        p_repr = self.proc_encoder(p_emb)

        '''
        # todo: Ablation SAB->RNN-------------------------------------------------------------
        seq1, seq2 = [], []
        for adm in patient_data:
            Idx1 = torch.LongTensor([adm[0]]).to(self.device)  # 诊断编码ID
            Idx2 = torch.LongTensor([adm[1]]).to(self.device)  # 操作编码ID
            repr1 = self.diag_embedding(Idx1)  # 对诊断编码ID进行embedding
            repr2 = self.proc_embedding(Idx2)  # 对操作编码ID进行embedding
            seq1.append(torch.sum(repr1, keepdim=True, dim=1))
            seq2.append(torch.sum(repr2, keepdim=True, dim=1))
        seq1 = torch.cat(seq1, dim=1)  # 把患者多次就医的诊断编码ID的embedding拼接成一个向量
        seq2 = torch.cat(seq2, dim=1)  # 把患者多次就医的操作编码ID的embedding拼接成一个向量
        output1, hidden1 = self.seq_encoders[0](seq1)
        output2, hidden2 = self.seq_encoders[1](seq2)
        d_repr_rnn, p_repr_rnn = output1[:, -1], output2[:, -1]
        d_repr_rnn = d_repr_rnn.unsqueeze(0)
        p_repr_rnn = p_repr_rnn.unsqueeze(0)
        # todo ---------------------------------------------------------------------------
        '''

        ########################个性化药物表征################################################

        # '''
        # todo --------------------------------------------------------
        visit_len = len(patient_data)
        if visit_len > 1:
            # 可视化：
            a = self.visual(0, med_repr, patient_data[-1][2], patient_data)
            # 诊断、手术级别的筛选
            d_visit = self.pma_d(d_repr)
            p_visit = self.pma_p(p_repr)
            d_visit = d_visit.squeeze(0)
            p_visit = p_visit.squeeze(0)
            d_p_visit = self.W_visit(torch.cat([d_visit, p_visit], dim=-1))

            d_history_all, p_history_all = [], []

            for adm in patient_data:
                Idx1 = torch.LongTensor([adm[0]]).to(self.device)  # 诊断编码ID
                Idx2 = torch.LongTensor([adm[1]]).to(self.device)  # 操作编码ID
                d_emb_h = self.diag_embedding(Idx1)
                p_emb_h = self.proc_embedding(Idx2)
                d_repr_h = self.diag_encoder(d_emb_h)
                p_repr_h = self.proc_encoder(p_emb_h)
                d_history = d_repr_h
                p_history = p_repr_h
                #############################################
                d_history_all.append(d_history)
                p_history_all.append(p_history)

            d_history_pma, p_history_pma = [], []
            d_p_history_pma = []

            for i in range(len(d_history_all)):
            # for i in range(len(d_history_all)-1):
                d_temp = self.pma_d(d_history_all[i]).squeeze(0)
                p_temp = self.pma_p(p_history_all[i]).squeeze(0)
                d_p_temp = self.W_visit(torch.cat([d_temp, p_temp], dim=-1))
                # d_p_temp = torch.cat([d_temp, p_temp], dim=-1)
                d_p_history_pma.append(d_p_temp.squeeze(0))
                d_history_pma.append(d_temp)
                p_history_pma.append(p_temp)

            d_p_history_pma = torch.stack(d_p_history_pma, dim=0)

            # todo:rnn加入时间:先拼接，再进行RNN
            # seq1 = d_p_history_pma.unsqueeze(0)
            # output1, hidden1 = self.seq_encoders(seq1)
            # todo:d和p先进行RNN，再拼接
            d_history_pma = torch.stack(d_history_pma, dim=0)
            p_history_pma = torch.stack(p_history_pma, dim=0)
            seq1 = d_history_pma
            output1, hidden1 = self.seq_encoders[0](seq1)
            seq2 = p_history_pma
            output2, hidden2 = self.seq_encoders[1](seq2)

            output_d_p = self.Out_visit(torch.cat([output1, output2], dim=-1))
            output_d_p = output_d_p.squeeze(1)
            # todo：——————————————————————————————————————————
            score_c = self.aggregator(output_d_p[-1], output_d_p)
            # score_c = self.aggregator(d_p_visit, output_d_p.squeeze(1) + d_p_history_pma)
            score_c = torch.softmax(score_c, dim=-1)
            score_c = score_c.squeeze(0)

            for i in range(visit_len - 1):
                adm = patient_data[i]
                Idx3 = torch.LongTensor([adm[2]]).to(self.device)
                m_emb_h = torch.zeros(self.med_num).to(self.device)
                m_emb_h[Idx3] = 1.0
                m_emb_h = m_emb_h.unsqueeze(0)
                if i == 0:
                    m_h = m_emb_h * score_c[i]
                else:
                    m_h += m_emb_h * score_c[i]
            m_c = torch.zeros(self.med_num).to(self.device) + 1.0
            m_weight = m_c.unsqueeze(0) + m_h * self.garm
            med_repr = med_repr * m_weight.t()

            # med_repr = self.FF(med_repr)
            # 可视化：
            a = self.visual(1, med_repr, patient_data[-1][2], patient_data)
        # todo:--------------------------------------------------------------------------------
        # '''
        ##################################################################################################

        ##################################### 预测药物集合################################################
        hidden = self.decoder(med_repr.unsqueeze(0), d_repr, p_repr)
        # hidden = self.decoder(med_repr.unsqueeze(0), None, p_repr)  # Ablation w/o D
        # hidden = self.decoder(med_repr.unsqueeze(0), d_repr, None)  # Ablation w/o P
        # hidden = self.decoder(med_repr.unsqueeze(0), d_repr_rnn, p_repr_rnn)  # Ablation SAB->RNN
        hidden = hidden.squeeze(0)
        if visit_len > 1:
            a = self.visual(2, hidden, patient_data[-1][2], patient_data)
            b = 1
        score = self.score_extractor(hidden).t()
        neg_pred_prob = torch.sigmoid(score)  # 论文中的前馈神经网络后的sigmod激活函数
        neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)
        batch_neg = 0.0005 * neg_pred_prob.mul(tensor_ddi_adj).sum()
        #############################################################################################

        return score, batch_neg

    def visual(self, num, features, label, history_label):
        # 初始化 t-SNE 模型，将数据降到 2 维以便可视化
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)

        features = features.detach()
        features = features.cpu()
        features = features.numpy()

        med_set = set()
        for i in range(len(history_label) - 1):
            med = history_label[i][2]
            for element in med:
                med_set.add(element)
        med_list = list(med_set)


        # 对特征向量进行降维
        reduced_features = tsne.fit_transform(features)

        # 将降维后的结果归一化到 [0, 1] 范围内
        x_min, x_max = reduced_features.min(0), reduced_features.max(0)
        normalized_features = (reduced_features - x_min) / (x_max - x_min)
        # if num == 1:  # x,y同时进行对称
        #     normalized_features = 1.0 - normalized_features
        if num == 0:  # x 进行对称
            normalized_features[:, 0] = 1.0 - normalized_features[:, 0]
        if num == 2 or num == 1:  # y 进行对称
            normalized_features[:, 1] = 1.0 - normalized_features[:, 1]


        # 准备颜色信息
        # 第 1 到第 10 个特征向量为一种颜色，其余为另一种颜色
        color_map = {0: (227.0 / 255, 0.0 / 255, 57.0 / 255),
                     1: (0.0 / 255, 168.0 / 255, 225.0 / 255)}
        # color_map = {0: (227.0/255, 0.0/255, 57.0/255),
        #              1: (113.0/255, 174.0/255, 70.0/255),
        #              2: (0.0/255, 168.0/255, 225.0/255)}
        # colors = ['red' if i in label else 'blue' for i in range(len(normalized_features))]
        colors = [color_map[0] if i in label else color_map[1] for i in range(len(normalized_features))]
        # colors = [
        #     color_map[0] if i in label else
        #     color_map[1] if i in med_list else
        #     color_map[2]
        #     for i in range(len(normalized_features))
        # ]

        plt.figure(figsize=(5, 5))
        scatter = plt.scatter(normalized_features[:, 0], normalized_features[:, 1],
                              c=colors, alpha=0.7, s=50)
        # plt.title("t-SNE Visualization (Normalized to [0, 1])")
        # plt.xlabel("t-SNE Component 1")
        # plt.ylabel("t-SNE Component 2")
        plt.grid(True)


        # 添加图例
        legend = plt.legend(loc='upper left',
                   handles=[plt.Line2D([0], [0], marker='o', color='w', label='Target drug',
                                       markerfacecolor=color_map[0], markersize=10),
                            plt.Line2D([0], [0], marker='o', color='w', label='Other drug',
                                       markerfacecolor=color_map[1], markersize=10),
                            # plt.Line2D([0], [0], marker='o', color='w', label='Other drug',
                            #            markerfacecolor=color_map[2], markersize=10)
                            ],
                            # title="Feature Vector Groups"
                   )

        # 获取当前坐标轴
        ax = plt.gca()

        # 加粗坐标轴线条
        ax.spines['left'].set_linewidth(3)
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['right'].set_linewidth(3)
        ax.spines['top'].set_linewidth(3)

        # 加粗坐标轴刻度
        ax.tick_params(axis='both', which='major', width=4)

        # 加粗坐标轴标签
        for tick in ax.get_xticklabels():
            tick.set_weight('bold')
            tick.set_size(16)
        for tick in ax.get_yticklabels():
            tick.set_weight('bold')
            tick.set_size(16)

        # 加粗图例
        for text in legend.get_texts():
            text.set_weight('bold')
            text.set_size(14)

        # 保存图形为 PDF 格式
        plt.savefig('/home/wjw/image/output_plot.pdf', format='pdf', bbox_inches='tight')

        plt.show()
        return 0

class MedTransformerDecoder_all(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 layer_norm_eps=1e-5) -> None:
        super(MedTransformerDecoder_all, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.m2d_multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.m2p_multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.m2m_multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.nhead = nhead

    def forward(self, input_med, input_disease_embdding=None, input_proc_embedding=None):
        if input_disease_embdding is None:
            input_disease_embdding = input_proc_embedding
        if input_proc_embedding is None:
            input_proc_embedding = input_disease_embdding

        x = input_med
        x = self.norm1(x + self.self_block(x, attn_mask=None))
        x = self.norm2(x + self._m2d_mha_block(x, input_disease_embdding, attn_mask=None)
                         + self._m2p_mha_block(x, input_proc_embedding, attn_mask=None))
        x = self.norm3(x + self._ff_block(x))

        return x

    def self_block(self, x, attn_mask):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _m2d_mha_block(self, x, mem, attn_mask):
        x = self.m2d_multihead_attn(x, mem, mem,
                                    attn_mask=attn_mask,
                                    need_weights=False)[0]
        return self.dropout2(x)

    def _m2p_mha_block(self, x, mem, attn_mask):
        x = self.m2p_multihead_attn(x, mem, mem,
                                    attn_mask=attn_mask,
                                    need_weights=False)[0]
        return self.dropout2(x)

    def _m2m_mha_block(self, x, mem, attn_mask):
        x = self.m2m_multihead_attn(x, mem, mem,
                                    attn_mask=attn_mask,
                                    need_weights=False)[0]
        return self.dropout2(x)


    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


