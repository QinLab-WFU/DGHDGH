import math
from time import sleep

import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_add

from train._utils import softmax, weights_init_kaiming


class VanillaAttention(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def forward(self, queries, keys, values):
        q, k, v = self.q_linear(queries), self.k_linear(keys), self.v_linear(values)
        d = q.shape[-1]

        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(d)
        attention_weights = F.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(attention_weights), v).squeeze(1)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.constant_(self.q_linear.bias, 0.0)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.constant_(self.k_linear.bias, 0.0)
        nn.init.xavier_uniform_(self.v_linear.weight)
        nn.init.constant_(self.v_linear.bias, 0.0)


class MultiHeadDotProduct(nn.Module):

    def __init__(self, embed_dim, n_heads, mult_attr=0):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.mult_attr = mult_attr

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)

        self.out = nn.Linear(embed_dim, embed_dim)

        self.reset_parameters()

    def forward(self, node_attrs, edge_idxes, edge_attrs, labels):
        bs = labels.shape[0]
        if node_attrs.shape[0] != bs:
            q = node_attrs[bs:, :]
            k = node_attrs[:bs, :]
            v = node_attrs[:bs, :]
        else:
            q = k = v = node_attrs

        k = self.k_linear(k).view(bs, self.n_heads, self.head_dim).transpose(0, 1)
        q = self.q_linear(q).view(bs, self.n_heads, self.head_dim).transpose(0, 1)
        v = self.v_linear(v).view(bs, self.n_heads, self.head_dim).transpose(0, 1)

        node_attrs = self._attention(q, k, v, edge_idxes, edge_attrs, labels)
        node_attrs = node_attrs.transpose(0, 1).contiguous().view(bs, self.n_heads * self.head_dim)
        node_attrs = self.out(node_attrs)

        return node_attrs

    def _attention(self, q, k, v, edge_idxes, edge_attrs, labels):

        r, c, e, bs = edge_idxes[:, 0], edge_idxes[:, 1], edge_idxes.shape[0], labels.shape[0]

        scores = torch.matmul(q.index_select(1, c).unsqueeze(dim=-2), k.index_select(1, r).unsqueeze(dim=-1))
        scores = scores.view(self.n_heads, e, 1) / math.sqrt(self.head_dim)

        mask = labels @ labels.T > 0

        mask_idxes = mask.view(-1).nonzero().squeeze()

        scores[:, mask_idxes, :] = -1e9

        scores = softmax(scores, c, 1, bs)

        if self.mult_attr:
            scores = scores * edge_attrs.unsqueeze(1)

        out = scores * v.index_select(1, r)
        out = scatter_add(out, c, dim=1, dim_size=bs)
        if type(out) == tuple:
            out = out[0]
        return out

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.constant_(self.q_linear.bias, 0.0)
        nn.init.xavier_uniform_(self.v_linear.weight)
        nn.init.constant_(self.v_linear.bias, 0.0)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.constant_(self.k_linear.bias, 0.0)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.constant_(self.out.bias, 0.0)


class NodeMessagePropagation(nn.Module):

    def __init__(self, embed_dim, n_heads, hidden_dim=None):
        super().__init__()

        hidden_dim = 4 * embed_dim if hidden_dim is None else hidden_dim

        self.mmsa_attn = MultiHeadDotProduct(embed_dim, n_heads)

        self.edge_to_node = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU())

        self.norm = nn.Sequential(nn.GELU(), nn.LayerNorm(embed_dim))

        self.FFN = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, node_attrs, edge_idxes, edge_attrs, labels):

        node_feats = self.mmsa_attn(node_attrs, edge_idxes, edge_attrs, labels)

        node_feats = self.norm(node_attrs[: labels.shape[0], :] + node_feats)

        edge_feats = self.edge_to_node(edge_attrs)

        node_feats = self.norm(node_feats + scatter_add(edge_feats, edge_idxes[:, 0], dim=0))

        node_attrs = self.norm(self.FFN(node_feats) + node_feats)

        return node_attrs, edge_idxes, edge_attrs


class EdgeMessagePropagation(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None):
        super().__init__()

        hidden_dim = embed_dim * 4 if hidden_dim is None else hidden_dim

        self.cross_attn = VanillaAttention(embed_dim)

        self.node_to_edge = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU())

        self.norm = nn.Sequential(nn.GELU(), nn.LayerNorm(embed_dim))

        self.FFN = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, node_attrs, edge_idxes, edge_attrs):
        r, c = edge_idxes[:, 0], edge_idxes[:, 1]

        node_feats = self.node_to_edge(node_attrs)
        src_edge_feats = node_feats[r].unsqueeze(1)
        tgt_edge_feats = node_feats[c].unsqueeze(1)

        edge_feats = torch.cat([src_edge_feats, tgt_edge_feats], dim=1)

        edge_feats = self.cross_attn(edge_attrs.unsqueeze(1), edge_feats, edge_feats)

        edge_feats = self.norm(edge_feats + edge_attrs)

        edge_attrs = self.norm(self.FFN(edge_feats) + edge_feats)

        return node_attrs, edge_idxes, edge_attrs


class MetaLayer(nn.Module):
    def __init__(self, edge_model, node_model):
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model]:
            if hasattr(item, "reset_parameters"):
                item.reset_parameters()

    def forward(self, node_attrs, edge_idxes, edge_attrs, labels):
        node_attrs, edge_idxes, edge_attrs = self.node_model(node_attrs, edge_idxes, edge_attrs, labels)
        _, _, edge_attrs = self.edge_model(node_attrs, edge_idxes, edge_attrs)
        return node_attrs, edge_idxes, edge_attrs


class GNNDecoder(nn.Module):
    def __init__(self, embed_dim, out_dim, reduce, n_layers, n_heads, n_classes):
        super().__init__()

        self.reduce_fc = nn.Linear(embed_dim, embed_dim // reduce)
        self.embed_dim = embed_dim // reduce

        self.node_models = [NodeMessagePropagation(embed_dim, n_heads) for _ in range(n_layers)]
        self.edge_models = [EdgeMessagePropagation(embed_dim) for _ in range(n_layers)]
        self.gnn = nn.Sequential(
            *[MetaLayer(node_model=self.node_models[i], edge_model=self.edge_models[i]) for i in range(n_layers)]
        )

        self.bottleneck1 = nn.BatchNorm1d(self.embed_dim)
        self.bottleneck1.bias.requires_grad_(False)
        self.bottleneck1.apply(weights_init_kaiming)

        self.bottleneck2 = nn.BatchNorm1d(self.embed_dim)
        self.bottleneck2.bias.requires_grad_(False)
        self.bottleneck2.apply(weights_init_kaiming)

        self.fc = nn.Linear(self.embed_dim, out_dim)
        self.sigmoid = nn.Sigmoid()
        self.cls = nn.Linear(self.embed_dim, n_classes)

    def forward(self, node_attrs, edge_idxes, edge_attrs, labels):

        node_attrs = self.reduce_fc(node_attrs)

        for layer in self.gnn:
            node_attrs, edge_idxes, edge_attrs = layer(node_attrs, edge_idxes, edge_attrs, labels)

        edge_feats = self.bottleneck1(edge_attrs)
        node_feats = self.bottleneck2(node_attrs)

        node_preds = self.cls(node_feats)
        edge_reprs = self.sigmoid(self.fc(edge_feats))
        node_feats = torch.tanh(node_feats)

        return node_feats, node_preds, edge_reprs


if __name__ == "__main__":
    B, K, C = 2, 8, 10
    from utils.utils import gen_test_data
    from graph_generator import GraphGenerator

    e, t, l = gen_test_data(B, C, K)

    gat = GNNDecoder(K, K, 1, 2, 2, C)
    print(gat)
    gg = GraphGenerator()
    print(gg)

    edge_attrs, edge_idxes, node_attrs = gg.get_graph(e, e)
    node_preds, edge_reprs = gat(node_attrs, edge_idxes, edge_attrs, l)

    print(node_preds.shape, edge_reprs.shape)
