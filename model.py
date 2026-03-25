import torch.nn as nn
import torch.nn.functional as F
import torch
from layers import GraphConvolution, MultiHeadAttention, SpGraphAttentionLayer ,HCoN,MultiHeadAttention1,HiLo
from torch_geometric.nn import SAGEConv
import math
from torch.nn.parameter import Parameter
def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

# -----------------------------------------------------------------------------------
class GRU(nn.Module):
    def __init__(self, hidden, dropout):
        super(GRU, self).__init__()
        self.dropout = dropout
        self.Up = Parameter(torch.FloatTensor(hidden, hidden))
        self.Wp = Parameter(torch.FloatTensor(hidden, hidden))
        self.bp = Parameter(torch.FloatTensor(hidden))
        self.Ur = Parameter(torch.FloatTensor(hidden, hidden))
        self.Wr = Parameter(torch.FloatTensor(hidden, hidden))
        self.br = Parameter(torch.FloatTensor(hidden))
        self.Uc = Parameter(torch.FloatTensor(hidden, hidden))
        self.Wc = Parameter(torch.FloatTensor(hidden, hidden))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Up.size(0))
        self.Up.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.Wp.size(0))
        self.Wp.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.bp.size(0))
        self.bp.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.Ur.size(0))
        self.Ur.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.Wr.size(0))
        self.Wr.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.br.size(0))
        self.br.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.Uc.size(0))
        self.Uc.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.Wc.size(0))
        self.Wc.data.uniform_(-stdv, stdv)

    def forward(self, current, short):
        # 更新门       # 先算门PR
        P = torch.sigmoid(torch.matmul(current, self.Up) + torch.matmul(short, self.Wp) + self.bp)
        P = F.dropout(P, self.dropout, training=self.training)
        # 重置门
        R = torch.sigmoid(torch.matmul(current, self.Ur) + torch.matmul(short, self.Wr) + self.br)
        R = F.dropout(R, self.dropout, training=self.training)
        # 候选隐藏状态
        H_tilda = torch.tanh(torch.matmul(current, self.Uc) + R * torch.matmul(short, self.Wc))
        H_tilda = F.dropout(H_tilda, self.dropout, training=self.training)
        # 隐藏状态
        H = (1 - P) * short + P * H_tilda
        return H    # 就是实现12-13-14-15
# 感觉并不好用，因为说白了也是直接拿来用 当然也可以试试

class HyperEncoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout, n_layers,device,args):
        super(HyperEncoder,self).__init__()
        self.n_layers = n_layers    # 编码器层数
        self.hcon1 = HCoN(nfeat, nhid,device=device,args=args)    # 图卷积层
        if n_layers > 1:    # n_layers-1个图卷积层
            self.stack_layers = [HCoN(nhid, nhid,device=device,args=args) for _ in range(n_layers-1)]
            self.stack_layers = nn.ModuleList(self.stack_layers)
        # self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout  # dropout层

    def forward(self, x,y, hg, pad_n, pos_node_idx,pad_e ,pos_edge_idx):   # 原始节点特征  原始+异常边+基序边特征  hg超图  原始节点总数  原始+异常边总数  原始节点id  原始边的id（边id按顺序来）pad_e ,pos_edge_idx
        # x是原始节点嵌入 y是所有普通图边以及虚拟超边的嵌入 ，hg包括了关联矩阵、预先计算的一些东西
        y,x = self.hcon1(x, y, hg)
        y=F.relu(y)
        x=F.relu(x)
        y = F.dropout(y, self.dropout, training=self.training)
        x = F.dropout(x, self.dropout, training=self.training)

        if self.n_layers > 1:   # 剩下的图卷积  实际上都是64-64 都是一样的
            for enc_layer in self.stack_layers:
                y,x = enc_layer(x,y,hg)
                y = F.relu(y)
                x = F.relu(x)

        y1=y[:len(pos_edge_idx)]
        hid_y=y.size(1)
        device_y=x.device
        output_y=torch.zeros(pad_e,hid_y).to(device_y)
        output_y[pos_edge_idx]= y1   # 6079 6768      6079+657+32

        hid = x.size(1)
        device=x.device
        output_x = torch.zeros(pad_n, hid).to(device)  # 每个节点64维全0特征
        output_x[pos_node_idx] = x

        return  output_x ,output_y # 更新原始节点嵌入

# -----------------------------------------------------------------------------------
class Encoder(nn.Module):   # 编码器
    def __init__(self, nfeat, nhid, dropout, n_layers):
        super(Encoder, self).__init__()
        self.n_layers = n_layers    # 编码器层数
        self.gc1 = GraphConvolution(nfeat, nhid)    # 图卷积层
        if n_layers > 1:    # n_layers-1个图卷积层
            self.stack_layers = [GraphConvolution(nhid, nhid) for _ in range(n_layers-1)]
            self.stack_layers = nn.ModuleList(self.stack_layers)
        # self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout  # dropout层

    def forward(self, x, motif_emb, adj, pad_n, pos_idx):   # 输入每个节点的嵌入、每个基序的嵌入，邻接矩阵，初始节点总数，原始节点id
        x = torch.cat((x,motif_emb))    # 拼接原始节点嵌入 和 虚拟节点嵌入
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)  # 一次图卷积
        if self.n_layers > 1:   # 剩下的图卷积  实际上都是64-64 都是一样的
            for enc_layer in self.stack_layers:
                x = F.relu(enc_layer(x, adj))
        # motif_emb = x[-len(motif_emb):]   经过多层GCN
        # x1=x[-len(motif_emb):]
        x = x[:-len(motif_emb)] # 得到处理后的原始节点嵌入
        hid = x.size(1) # 处理后特征维度
        device = x.device
        output = torch.zeros(pad_n, hid).to(device) # 每个节点64维全0特征
        output[pos_idx] = x
        # output[mo_idx]=x1
        # output[motif_idx]=x1
        return output   # 更新原始节点嵌入

class GCNEncoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCNEncoder, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj, pad_n, pos_idx):
        # x = torch.cat((x,motif_emb))
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        # motif_emb = x[-len(motif_emb):]
        hid = x.size(1)
        device = x.device
        output = torch.zeros(pad_n, hid).to(device)
        output[pos_idx] = x
        return output

class GATEncoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GATEncoder, self).__init__()

        self.gat1 = SpGraphAttentionLayer(nfeat,
                                                 nhid*2,
                                                 dropout=dropout,
                                                 alpha=0.1,
                                                 concat=True)
        self.gat2 = SpGraphAttentionLayer(nhid*2,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=0.1,
                                                 concat=True)
        self.dropout = dropout

    def forward(self, x, adj, pad_n, pos_idx):
        # x = torch.cat((x,motif_emb))
        adj= adj.to_dense()
        x = F.relu(self.gat1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gat2(x, adj))
        # motif_emb = x[-len(motif_emb):]
        hid = x.size(1)
        device = x.device
        output = torch.zeros(pad_n, hid).to(device)
        output[pos_idx] = x
        return output


class SAGEEncoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(SAGEEncoder, self).__init__()

        self.gc1 = SAGEConv(nfeat, nhid)
        self.gc2 = SAGEConv(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj, pad_n, pos_idx):
        # x = torch.cat((x,motif_emb))
        device = x.device
        adj = adj.to_dense().type(torch.LongTensor).to(device)
        adj = adj.to_sparse().indices()
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        # motif_emb = x[-len(motif_emb):]
        hid = x.size(1)

        output = torch.zeros(pad_n, hid).to(device)
        output[pos_idx] = x
        return output

class Attribute_Decoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Attribute_Decoder, self).__init__()
        self.gc1 = GraphConvolution(nhid, nhid)
        self.gc2 = GraphConvolution(nhid, nfeat)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))

        return x


class Structure_Decoder(nn.Module):
    def __init__(self, nhid, dropout, n_layers):
        super(Structure_Decoder, self).__init__()
        self.n_layers = n_layers    # 层数
        # self.gc_layers = [GraphConvolution(nhid, nhid) for _ in range(n_layers)]
        # self.gc_layers = nn.ModuleList(self.gc_layers)
        self.dropout = dropout  # dropout

    def forward(self, x, adj):
        # for gc in self.gc_layers:
            # x = F.relu(gc(x, adj))
            #
        # x = F.dropout(x, self.dropout, training=self.training)
        x = x @ x.T # 公式8 差个sigmoid？？？

        return x

# --------------------------------------------------------------------

class Hypergraph_Structure_Decoder(nn.Module):
    def __init__(self, nhid, dropout, n_layers):
        super(Hypergraph_Structure_Decoder, self).__init__()
        self.n_layers = n_layers    # 层数
        # self.gc_layers = [GraphConvolution(nhid, nhid) for _ in range(n_layers)]
        # self.gc_layers = nn.ModuleList(self.gc_layers)
        self.dropout = dropout  # dropout

    def forward(self, x,y,H):
        # for gc in self.gc_layers:
            # x = F.relu(gc(x, adj))
            #
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = x @ x.T # 公式8 差个sigmoid？？？
        x = y @ x.t()

        return x

# --------------------------------------------------------------------

class MotifFeatExtract(nn.Module):
    def __init__(self, hid, num_heads, dropout):
        super().__init__()
        self.motif_token = nn.Embedding(1, hid) # 基序令牌
        self.attn = MultiHeadAttention(hid, num_heads, dropout) # 多头注意力  C. Motif Feature Aggregator

    def forward(self, x):   # 732个基序*每个基序三个节点*每个节点64维特征
        #[motif_batch, 3, dim]
        motif_token_feat = self.motif_token.weight.repeat(x.size(0), 1, 1)  # 要得到 基序数 个token 得到token的初始嵌入

        x = torch.cat((motif_token_feat, x),dim=1)  # 拼接得到论文里的  H = [hi0 , hi1 , hi2 , hi3 ]
        #[motif_batch, 4, dim]
        x = self.attn(x)        # 同样做自注意力
        x = x[:,0,:]    # 得到基序表示
        # [motif_batch, dim]
        return x    # 取hi0 作为基序表示

class MyModelwoM(nn.Module):
    def __init__(self, nodes_num, snap_len, feat_size, hidden_size, num_heads, dropout, encoder='gcn', device ="cuda:0"):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.nodes_num = nodes_num
        self.node = torch.arange(nodes_num, device=self.device)
        self.node_embedding = nn.Embedding(nodes_num, feat_size)
        self.encoder = encoder
        if encoder == 'gcn':
            self.shared_encoder = GCNEncoder(feat_size, hidden_size, dropout)
        elif encoder == 'gat':
            self.shared_encoder = GATEncoder(feat_size, hidden_size, dropout)
        elif encoder == 'graphsage':
            self.shared_encoder = SAGEEncoder(feat_size, hidden_size, dropout)

        self.struct_decoder = Structure_Decoder(hidden_size, dropout,n_layers=1)

        self.time_encoder = nn.Embedding(2*snap_len-1, num_heads)
        self.relative_matrix = self.build_matrix(snap_len)

        self.attn = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.apply(lambda module: init_params(module))

    def build_matrix(self, snap_len):
        pos_matrix_1 = torch.stack([idx*torch.ones(snap_len, device="cuda:0",dtype=torch.int64) for idx in range(snap_len)])
        pos_matrix_2 = torch.stack([torch.arange(snap_len, device=self.device) for _ in range(snap_len)])
        matrix = (pos_matrix_2 - pos_matrix_1) + snap_len - 1
        return matrix

    def forward(self, snapshots):

        x = [self.node_embedding.weight[snap.nodes] for snap in snapshots]

        if self.encoder == 'gcn':
            x = torch.stack([self.shared_encoder(x[i], snap.norm_adj, self.nodes_num, snap.nodes) for i,snap in enumerate(snapshots)])
        else:
            x = torch.stack([self.shared_encoder(x[i], snap.adj, self.nodes_num, snap.nodes) for i,snap in enumerate(snapshots)])

        x = x.transpose(0, 1) # [node_num, snap_len, hid]
        attn_bias = self.time_encoder(self.relative_matrix)        # self.edges = np.unique(edges, axis=0)
        x = self.attn(x, attn_bias) #[node_num, snap_len, hid]
        x = x.transpose(0, 1) #[snap_len, node_num, hid]
        x = [x[i][snapshots[i].nodes] for i in range(len(snapshots))]
        # motif_node_embs = [[x[i][motif] for motif in snap.motifs] for i,snap in enumerate(snapshots)]
        # snap_motif_node_embs = [torch.stack([x[i][motif] for motif in snapshots[i].motifs]) for i in range(len(snapshots))] #[snap_len, motif_num, 3, hid]

        output = [self.struct_decoder(x[i], snapshots[i].norm_adj) for i in range(len(snapshots))]
        return output

class MADG(nn.Module):
    # def __init__(self, motifn , node_n ,nodes_num, snap_len, feat_size, hidden_size, num_heads, dropout, n_layers, device ="cuda:0"):
    def __init__(self, edges_num,nodes_num, snap_len, feat_size, hidden_size, num_heads, dropout, n_layers,device="cuda:0",args=None):
        super().__init__()  # 创建模型

        # self.motifn=motifn
        # self.node_n=node_n
        self.args=args
        self.device = device    # 设备
        self.hidden_size = hidden_size  # 隐藏层维度
        self.nodes_num = nodes_num  # 节点总数
        self.node = torch.arange(nodes_num, device=self.device) # 节点id？ 一维张量 与range相同
        self.node_embedding = nn.Embedding(nodes_num, feat_size)    # 嵌入层 nn.Embedding通常适合离散输入情况  nn.Linear适合连续输入情况

        self.edge_embedding = nn.Embedding(edges_num, feat_size)    # +++++

        # self.shared_encoder = Encoder(feat_size, hidden_size, dropout,n_layers=n_layers)    # 共享的编码器
        # -------------------------------------
        self.hy_shared_encoder=HyperEncoder(feat_size,hidden_size,dropout,n_layers=n_layers,device=self.device,args=args)
        # -------------------------------------

        # self.struct_decoder = Structure_Decoder(hidden_size, dropout,n_layers=1)            # 共享解码器
        self.hypergraph_struct_decoder = Hypergraph_Structure_Decoder(hidden_size, dropout,n_layers=1)            # 共享解码器 +++++++++

        # self.time_encoder = nn.Embedding(2*snap_len-1, num_heads)   # b_Φ（t1,t2）    # 也是嵌入层  2*snap_len-1, num_heads？不懂  头数改1？？？
        # self.relative_matrix = self.build_matrix(snap_len)  # 相关性矩阵  一种相对性矩阵，其中每个元素表示了其对应位置在矩阵中的相对距离？？？ 左下角为0 往右上就+1

        # self.attn = HiLo(hidden_size, 4, window_size=2,alpha=0.5) # 多头注意力层 0.3
        # self.attn = MultiHeadAttention(hidden_size, num_heads, dropout) # 多头注意力层
        self.MotifFE1 = MotifFeatExtract(feat_size, num_heads, dropout) # 基序特征抽取器1
        self.MotifFE2 = MotifFeatExtract(hidden_size, num_heads, dropout)   # 基序特征抽取器2  ？？？

        self.time_encoder1=nn.Linear(3,self.hidden_size)
        # self.time_encoder2=nn.Linear(self.hidden_size,self.hidden_size)
        self.time_decoder1=nn.Linear(self.hidden_size,3)

        # --------------------------------------------------------------------------------------------------------------------------------------
        # 快照编号嵌入到节点嵌入中
        # self.snap_encoder = nn.Linear(1, self.hidden_size)
        # --------------------------------------------------------------------------------------------------------------------------------------


        # self.RNN=GRU(hidden_size,dropout)

        self.l=nn.Linear(2*self.hidden_size,self.hidden_size)
        # self.l1=nn.Linear(self.hidden_size,self.hidden_size)

        # self.attn = MultiHeadAttention1(hidden_size, num_heads, dropout,device) # 多头注意力层

        self.apply(lambda module: init_params(module))  # 对神经网络的每个子模块调用参数初始化

        # self.params1=list(self.l.parameters())+list(self.l1.parameters())+list(self.node_embedding.parameters())+list(self.edge_embedding.parameters())+list(self.hy_shared_encoder.parameters())+list(self.hypergraph_struct_decoder.parameters())+list(self.MotifFE1.parameters())+list(self.MotifFE2.parameters()) # +list(self.attn.parameters())
        self.params1=list(self.l.parameters())+list(self.node_embedding.parameters())+list(self.edge_embedding.parameters())+list(self.hy_shared_encoder.parameters())+list(self.hypergraph_struct_decoder.parameters())+list(self.MotifFE1.parameters())+list(self.MotifFE2.parameters()) # +list(self.attn.parameters())
        self.params2=list(self.time_encoder1.parameters())+list(self.time_decoder1.parameters())#+list(self.snap_encoder.parameters())

    def build_matrix(self, snap_len):
        pos_matrix_1 = torch.stack([idx*torch.ones(snap_len, device=self.device,dtype=torch.int64) for idx in range(snap_len)])
        pos_matrix_2 = torch.stack([torch.arange(snap_len, device=self.device) for _ in range(snap_len)])
        matrix = (pos_matrix_2 - pos_matrix_1) + snap_len - 1
        return matrix

    def forward(self, snapshots):   # 输入是所有快照

        time_rebuild = []
        time_enc=[]
        for i in range(len(snapshots)):
            time_matrix = snapshots[i].final_time
            input = torch.stack(time_matrix)
            input = input.to(snapshots[i].device)

            time_enc1 = self.time_encoder1(input)
            time_enc.append(time_enc1)
            time_dec = self.time_decoder1(time_enc1)

            time_rebuild.append(time_dec)

        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        x = [self.node_embedding.weight[snap.nodes] for snap in snapshots]  # 嵌入层得到每个快照节点的嵌入向量

        y = []
        py=0    # 偏移量 +++++++
        edges_index=[]  #++++++

        for snap in snapshots:  # +++++得到没个快照每个边的初始嵌入
            y.append(self.edge_embedding.weight[[i for i in range(py,len(snap.new_edges)+py)]])
            py+=len(snap.new_edges)
            edges_index.append(py)  # ++++++
        pad_e=py    # 边总数
        edges_index_final=[]    # 记录每个快照的边id
        for i in range(len(edges_index)):
            if i==0:
                edges_index_final.append([j for j in range(0,edges_index[0])])
            else:
                edges_index_final.append([j for j in range(edges_index[i-1],edges_index[i])])

        # x,y是得到了没问题了 motif_embs是每个基序的嵌入，可以先用来当作基序边的表示   那就是要把motif_embs拼在y  这个还有调整的空间+++++把初始边嵌入得到基序嵌入 这个有点麻烦 先不改
        # motif_embs = [self.MotifFE1(torch.stack([x[i][motif] for motif in snapshots[i].motifs])) for i in range(len(x))]    # 先遍历每个快照-->再遍历每个快照的基序-->得到每个基序节点的嵌入-->堆叠快照内所有基序嵌入-->提取基序特征  每个快照都这么做
        motif_embs = [self.MotifFE1(torch.stack([x[i][motif] for motif in snapshots[i].motifs])) for i in range(len(x))]    # 先遍历每个快照-->再遍历每个快照的基序-->得到每个基序节点的嵌入-->堆叠快照内所有基序嵌入-->提取基序特征  每个快照都这么做

        for i in range(len(y)):
            y[i]=torch.cat((y[i],motif_embs[i]),dim=0)


        # if False:
        if self.args.times==1:
            if self.args.mode==0:
                x1 = [self.hy_shared_encoder(x[i], (1.0-self.args.fusion)*y[i]+self.args.fusion*time_enc[i], snap.hg, self.nodes_num, snap.nodes,pad_e,edges_index_final[i]) for i,snap in enumerate(snapshots)] #超图 使用HCoN -----------------
            else:
                for i, snap in enumerate(snapshots):
                    temp=y[i]
                    y[i]=torch.cat((y[i],time_enc[i]),dim=1)
                    y[i] = F.relu(self.l(y[i])+temp)
                    # y[i] = F.relu(self.l1(y[i]) + temp)
                x1 = [self.hy_shared_encoder(x[i], y[i], snap.hg, self.nodes_num, snap.nodes,pad_e,edges_index_final[i]) for i,snap in enumerate(snapshots)]
        else:
            x1 = [self.hy_shared_encoder(x[i], y[i], snap.hg, self.nodes_num, snap.nodes,pad_e,edges_index_final[i]) for i,snap in enumerate(snapshots)] #超图 使用HCoN -----------------

        x = [i[0] for i in x1]
        y = [i[1] for i in x1]
        x=torch.stack(x)
        y=torch.stack(y)

        # ---------------------------------------------------------------------------------------------------------------------------------------
        # x = x.transpose(0, 1) # [node_num, snap_len, hid]   再次重新得到每个节点的嵌入
        #
        # # h = x[:,0,:]
        # # updated_x = x.clone()
        # # for s in range(1,len(snapshots)):
        # #     x_in=x[:,s,:]
        # #     h=self.RNN(x_in,h)
        # #     updated_x[:,s,:]=h
        # # x=updated_x
        #
        # # attn_bias = self.time_encoder(self.relative_matrix) # 得到时间编码    每个头一个10*10的偏置
        # # attn_bias = attn_bias.transpose(1,2).transpose(0,1)
        # x = self.attn(x, None) #[node_num, snap_len, hid]  # 带着时间编码计算attention  B. Temporal Self-Attention H'
        # # x = self.attn(x,14,14) #[node_num, snap_len, hid]  # 带着时间编码计算attention  B. Temporal Self-Attention H'
        # x = x.transpose(0, 1) #[snap_len, node_num, hid]
        # ---------------------------------------------------------------------------------------------------------------------------------------

        x = [x[i][snapshots[i].nodes] for i in range(len(snapshots))]   # 每个快照原始节点的表示  用到的节点

        x_origin=x
        result = []
        y = [y[i][edges_index_final[i]] for i in range(len(snapshots))]
        # if self.args.mode==1:
        #     for i, snap in enumerate(snapshots):
        #         y[i] = torch.cat((y[i], time_enc[i][:len(y[i])]), dim=1)
        #         y[i] = F.relu(self.l(y[i]))
        # y = [y[i]+time_enc[i][:len(y[i])] for i in range(len(snapshots))]
        final_y = []

        result_x=[]
        result_y=[]


        # 循环遍历 x 中的每个张量
        for i in range(len(x)):
            # 从当前张量 x[i] 中提取 motifs 属性对应的张量，并存储在 motif_tensors 列表中
            inputs=[]
            motif_tensors = []
            key_edge_set=snapshots[i].edge2id.keys()
            for motif in snapshots[i].motifs:
                motif_tensors.append(x[i][motif])
                nums_of_motif=len(motif)
                edge_l=[]
                pre_view_edges=[]
                for src in range(nums_of_motif):
                    edge1 = str([min([motif[src], motif[(src+1)%nums_of_motif]]), max([motif[src], motif[(src+1)%nums_of_motif]])])
                    if edge1 in key_edge_set:
                        edge1 = snapshots[i].edge2id[edge1]
                        edge_l.append(edge1)

                inputs.append(y[i][edge_l])


            stack_inputs = torch.stack(inputs)
            processed_inputs = self.MotifFE2(stack_inputs)
            concatenated_inputs = torch.cat((y[i], processed_inputs))
            final_y.append(concatenated_inputs)

            # 将 motif_tensors 列表中的张量堆叠起来，得到一个新的张量
            stacked_motif_tensor = torch.stack(motif_tensors)

            # 使用 MotifFE2 函数处理 stacked_motif_tensor，得到一个新的张量



            processed_tensor = self.MotifFE2(stacked_motif_tensor)

            # 将 x[i] 和 processed_tensor 进行拼接，并将结果添加到结果列表中
            concatenated_tensor = torch.cat((x[i], processed_tensor))
            result.append(concatenated_tensor)

            mt = [torch.cat((tensor1, tensor2), dim=0) for tensor1, tensor2 in zip(motif_tensors, inputs)]
            stack_all=torch.stack(mt)
            processed_all = self.MotifFE2(stack_all)
            concatenated_all_x=torch.cat((x[i], processed_all))
            concatenated_all_y=torch.cat((y[i], processed_all))
            result_x.append(concatenated_all_x)
            result_y.append(concatenated_all_y)






        output_y_all = [self.hypergraph_struct_decoder(x_origin[i], result_y[i], snapshots[i].hg.MT) for i in
                    range(len(snapshots))]  # 关联矩阵重构




        return [] , [] ,[],output_y_all,time_rebuild




