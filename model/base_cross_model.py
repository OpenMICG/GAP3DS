import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import repeat


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, num_q_channels, num_kv_channels, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=num_q_channels,
            num_heads=num_heads,
            kdim=num_kv_channels,
            vdim=num_kv_channels,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv, return_atts=False):
        """
        :param q: (bs, lenq, d_model)
        :param k: (bs, lenv, d_model)
        :param v: (bs, lenv, d_model)
        :param mask:
        :return: (bs, lenq, d_model)
        """
        if not return_atts:
            summed_v = self.attention(q.permute(1, 0, 2), kv.permute(1, 0, 2), kv.permute(1, 0, 2))[0].permute(1, 0, 2)
            return q + self.dropout(summed_v)
        else:

            r = self.attention(q.permute(1, 0, 2), kv.permute(1, 0, 2), kv.permute(1, 0, 2))
            summed_v = r[0].permute(1, 0, 2)
            return q + self.dropout(summed_v), r[1]


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.gelu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        self.d_hid = d_hid
        self.n_position = n_position
        # 初始化时创建位置编码表
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        # 获取当前序列长度和特征维度
        seq_len = x.size(1)
        feat_dim = x.size(-1)
        
        # 如果特征维度不匹配，调整位置编码的维度
        if feat_dim != self.d_hid:
            # 重新生成位置编码表
            pos_table = self._get_sinusoid_encoding_table(seq_len, feat_dim).to(x.device)
            # 确保位置编码的维度与输入匹配
            pos_table = pos_table.expand(x.size(0), -1, -1)  # 扩展到batch维度
            return x + pos_table[:, :seq_len]
        
        # 如果序列长度超过预定义的最大长度，动态生成新的位置编码
        if seq_len > self.n_position:
            pos_table = self._get_sinusoid_encoding_table(seq_len, self.d_hid).to(x.device)
            # 确保位置编码的维度与输入匹配
            pos_table = pos_table.expand(x.size(0), -1, -1)  # 扩展到batch维度
            return x + pos_table[:, :seq_len]
        
        # 否则使用预计算的位置编码
        pos_table = self.pos_table.expand(x.size(0), -1, -1)  # 扩展到batch维度
        return x + pos_table[:, :seq_len].to(x.device)


class SelfAttention(nn.Module):

    def __init__(self, num_heads, num_channels, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.self_att = MultiHeadAttention(num_heads, num_channels, num_channels, dropout)

    def forward(self, enc_input, return_atts=False):
        x = self.norm(enc_input)
        return self.self_att(x, x, return_atts)


class SelfAttentionLayer(nn.Module):

    def __init__(self, num_heads, num_q_channels, dropout=0.1):
        super(SelfAttentionLayer, self).__init__()
        self.self_att = SelfAttention(num_heads, num_q_channels, dropout)
        self.mlp = PositionwiseFeedForward(num_q_channels, num_q_channels, dropout)

    def forward(self, qkv, return_atts=False):
        if return_atts:
            r = self.self_att(qkv, return_atts)
            return self.mlp(r[0]), r[1]
        return self.mlp(self.self_att(qkv))


class CrossAttention(nn.Module):

    def __init__(self, num_heads, num_q_channels, num_kv_channels, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.q_norm = nn.LayerNorm(num_q_channels)
        self.kv_norm = nn.LayerNorm(num_kv_channels)
        self.cross_att = MultiHeadAttention(num_heads, num_q_channels, num_kv_channels, dropout)

    def forward(self, q, kv, return_atts=False):
        q = self.q_norm(q)
        kv = self.kv_norm(kv)
        return self.cross_att(q, kv, return_atts)


class CrossAttentionLayer(nn.Module):

    def __init__(self, num_heads, num_q_channels, num_kv_channels, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        self.cross_att = CrossAttention(num_heads, num_q_channels, num_kv_channels, dropout)
        self.mlp = PositionwiseFeedForward(num_q_channels, num_q_channels, dropout)

    def forward(self, q, kv, return_atts=False):
        if return_atts:
            r = self.cross_att(q, kv, return_atts)
            return self.mlp(r[0]), r[1]
        return self.mlp(self.cross_att(q, kv))


class TransformerEncoder(nn.Module):

    def __init__(self, n_input_channels, n_latent, n_latent_channels=512, n_cross_att_heads=1,
        n_self_att_heads=8, n_self_att_layers=6, dropout=0.1, n_position=400):

        super().__init__()

        self.position_enc = PositionalEncoding(n_input_channels, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.cross_att = CrossAttentionLayer(
                    num_q_channels=n_latent_channels,
                    num_kv_channels=n_input_channels,
                    num_heads=n_cross_att_heads,
                    dropout=dropout)
        self_att = [SelfAttentionLayer(
                    num_q_channels=n_latent_channels,
                    num_heads=n_self_att_heads,
                    dropout=dropout) for _ in range(n_self_att_layers)]
        self.self_att = nn.Sequential(*self_att)
        self.latent = nn.Parameter(torch.empty(n_latent, n_latent_channels))
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, feats_embedding):
        """
        :param feats_embedding: (bs, n_length, dim)
        :param return_atts:
        :return:
        """
        b = feats_embedding.shape[0]
        enc_output = self.dropout(self.position_enc(feats_embedding))
        x_latent = repeat(self.latent, "... -> b ...", b=b)
        # print(x_latent.shape, enc_output.shape)
        x_latent = self.cross_att(x_latent, enc_output)
        x_latent = self.self_att(x_latent)
        return x_latent


class TransformerDecoder(nn.Module):
    # TranformerDecoder

    def __init__(self, n_query, n_query_channels, n_latent_channels, n_cross_att_heads=1, dropout=0.1):

        super().__init__()

        self.cross_att = CrossAttentionLayer(
                    num_q_channels=n_query_channels,
                    num_kv_channels=n_latent_channels,
                    num_heads=n_cross_att_heads,
                    dropout=dropout)

        self.query_latent = nn.Parameter(torch.empty(n_query, n_query_channels))
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.query_latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, query, latent, return_atts=False):
        """
        :param query: (bs, n_q, d_q)
        :param latent: (bs, n_length, d_latent)
        :return:
        """
        b = query.shape[0]
        x_latent = repeat(self.query_latent, "... -> b ...", b=b)
        query_embedding = query + x_latent
        x_latent = self.cross_att(query_embedding, latent, return_atts)
        return x_latent

class Mlp(nn.Module):
    def __init__(self, in_dim, out_dim, expansion=4, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim * expansion)
        self.fc2 = nn.Linear(out_dim * expansion, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        out = self.drop(F.gelu(self.fc1(x)))
        out = self.drop(self.fc2(out))
        return out