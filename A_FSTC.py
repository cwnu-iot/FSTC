import torch
import torch.nn as nn
import torch.nn.functional as F
from Actionsrecognition.Utils import Graph

class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels,
                              out_channels * kernel_size,
                              kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0),
                              stride=(t_stride, 1),
                              dilation=(t_dilation, 1),
                              bias=bias)
    def forward(self, x, A):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous()

class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        embedded = self.embedding(x)
        attention_output, _ = self.attention(embedded, embedded, embedded)
        attention_output = attention_output.permute(1, 0, 2)  # 重排维度
        pooled = torch.mean(attention_output, dim=0)  # 取平均作为池化
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return logits

class rfid_st(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,stride=1,dropout=0,residual=True, selfattention=False):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.rfid_gcn = GraphConvolution(in_channels, out_channels, kernel_size[1])
        self.rfid_tcn = nn.Sequential(nn.BatchNorm2d(out_channels),
                                 nn.GELU(),
                                 nn.Conv2d(out_channels,
                                           out_channels,
                                           (kernel_size[0], 1),
                                           (stride, 1),
                                           padding),
                                 nn.BatchNorm2d(out_channels),
                                 nn.Dropout(dropout, inplace=True)
                                 )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=(stride, 1)),
                                          nn.BatchNorm2d(out_channels))
        if selfattention:
            self.selfattention= SelfAttention(in_channels,int(in_channels/2),out_channels,2,0.6)
        self.gelu = nn.GELU()
    def forward(self, x, A):
        res = self.residual(x)
        if hasattr(self, 'self_attention'):
            x = self.selfattention(x)
        x = self.rfid_gcn(x, A)
        x = self.rfid_tcn(x)
        x = x + res
        return self.gelu(x)

class RFID_ST_Conv(nn.Module):
    def __init__(self, in_channels, graph_args, num_class=None,
                 edge_importance_weighting=True, **kwargs):
        super().__init__()
        graph = Graph(**graph_args)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        self.rfid_st_convs = nn.ModuleList((
            rfid_st(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            rfid_st(64, 128, kernel_size, 2, selfattention=True, **kwargs),
            rfid_st(128, 256, kernel_size, 2, selfattention=True, **kwargs),
        ))
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(A.size()))
                for i in self.rfid_st_convs
            ])
        else:
            self.edge_importance = [1] * len(self.rfid_st_convs)
        if num_class is not None:
            self.cls = nn.Conv2d(256, num_class, kernel_size=1)
        else:
            self.cls = lambda x: x
    def forward(self, x):
        N, T, V, C = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()
        for gcn, importance in zip(self.rfid_st_convs, self.edge_importance):
            x = gcn(x, self.A * importance)
        N, T, V, C = x.size()
        x = x.view(N, V * C, T)
        return x[:, -1, :]

class CosLayer(nn.Module):
    def __init__(self, in_size, out_size, s=15.6):
        super(CosLayer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.W = nn.Parameter(torch.randn(out_size, in_size))
        self.W.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.s = nn.Parameter(torch.randn(1,)) if s is None else s

    def forward(self, input):
        cosine = F.linear(F.normalize(input), F.normalize(self.W))
        output = cosine * self.s
        return output

    def __repr__(self):
        return self.__class__.__name__ +  '(in_size={}, out_size={}, s={})'.format(
                    self.in_size, self.out_size,
                    'learn' if isinstance(self.s, nn.Parameter) else self.s)


class RFIDNetwork(nn.Module):
    def __init__(self, graph_args, num_class, edge_importance_weighting=True, **kwargs):
        super().__init__()
        self.rfid_conv = RFID_ST_Conv(3, graph_args, None, edge_importance_weighting, **kwargs)
        self.fc1 = nn.Linear(256, 128)
        self.cos = CosLayer(128, 21)
    def forward(self, inputs):
        out1 = self.rfid_conv(inputs)
        out2 = F.gelu(self.fc1(out1))
        out = self.cos(out2)
        return out
