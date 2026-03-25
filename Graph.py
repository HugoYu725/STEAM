import numpy as np
import torch
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from utils import preprocess_adj, adj2tensor
from tqdm import tqdm
import time
import random
from torch_sparse import SparseTensor

class SnapShot: # 快照类
    def __init__(self, edges,time) -> None:
        self.edges_times = time
        self.edges = edges  # 快照中的所有边
        self.edges_num = len(self.edges)    # 边数
        self.nodes = np.unique(edges).tolist()  # 节点列表
        self.nodes.sort()   # 节点id从小到大排序
        self.nodes_num = len(self.nodes)    # 节点总数
        self.node_ids = list(range(self.nodes_num))

        self.edge_ids = list(range(self.edges_num)) # ++++++

        self.id2node = {i:self.nodes[i] for i in range(self.nodes_num)} # 一一对应字典
        self.node2id = {self.nodes[i]:i for i in range(self.nodes_num)} # 同样也是一一对应字典 用于id和nodeindex相互转换

        self.id2edge = {i:self.edges[i] for i in range(self.edges_num)}     # ++++++
        # self.edge2id = {(str)(self.edges[i]): i for i in range(self.edges_num)}    # ++++++  对应于每一个快照的  重复边就考虑最新的

        self.adj1 = self.init_adj()
        self.norm_adj1 = preprocess_adj(self.adj1, is_sparse=True).to_dense()
        self.adj1 = adj2tensor(self.adj1, is_sparse=True)
        self.label_adj1 = self.adj1.to_dense()

        self.adj = self.init_adj0()  # 初始化邻接矩阵
        self.norm_adj = preprocess_adj(self.adj, is_sparse=True).to_dense() # 标准化
        self.adj = adj2tensor(self.adj, is_sparse=True) # 张量
        self.label_adj = self.adj.to_dense()    # 稠密矩阵


    def convert_symmetric(self, X, sparse=True):
        # add symmetric edges
        if sparse:
            X += X.T - sp.diags(X.diagonal())
        else:
            X += X.T - np.diag(X.diagonal())
        return X

    def init_adj0(self):
        values = torch.ones(len(self.edges), dtype=torch.float32)   # 初始边权重都是1
        # values = torch.tensor(self.edges_times)
        indices = self.edges.T  # src，dst
        src_node = indices[0].tolist()  # 源节点node
        dst_node = indices[1].tolist()  # 目标节点node
        src_node = [self.node2id[i] for i in src_node]  # 转id 实际上是一样的
        dst_node = [self.node2id[i] for i in dst_node]
        adj = coo_matrix((values, (src_node, dst_node)),
                         shape=(self.nodes_num, self.nodes_num))    # 由边权和非零元素下标组成邻接稀疏矩阵
        adj = self.convert_symmetric(adj, sparse=True)  # 无向图 对称
        return adj

    def init_adj(self):
        # 创建一个空的字典来存储每条边的权重和出现次数
        edge_weights = {}
        edge_counts = {}

        # 遍历每个边，更新字典中的权重和出现次数
        for i, edge in enumerate(self.edges):
            edge_key = tuple(sorted(edge))
            # edge_key = tuple(sorted(edge_key))
            edge_weights[edge_key] = edge_weights.get(edge_key, 0) + self.edges_times[i]
            edge_counts[edge_key] = edge_counts.get(edge_key, 0) + 1

        # 计算每条边的平均权重
        for edge_key in edge_weights:
            edge_weights[edge_key] /= edge_counts[edge_key]
            edge_weights[edge_key] /= edge_counts[edge_key]



        # 创建邻接矩阵的值，使用平均权重
        values = [edge_weights[tuple(sorted(edge))] for edge in self.edges]

        # 获取边的坐标
        indices = self.edges.T
        src_node = indices[0].tolist()
        dst_node = indices[1].tolist()
        src_node = [self.node2id[i] for i in src_node]
        dst_node = [self.node2id[i] for i in dst_node]

        # 创建 COO 格式的邻接矩阵
        adj = coo_matrix((values, (src_node, dst_node)), shape=(self.nodes_num, self.nodes_num))

        # 将邻接矩阵转换为对称矩阵
        adj = self.convert_symmetric(adj, sparse=True)

        return adj

class CircleMotifSnapShot(SnapShot):
    def __init__(self, edges, motif_list=None, size = 4, time=None) -> None:
        super().__init__(edges, time)
        self.size = size
        self.motif_list = []
        self.time_adj=[]
        if motif_list:
            self.motif_list = motif_list
        else:
            self.search_motif_from_adj()

        self.motif_num = len(self.motif_list)
        if self.motif_num > 500:
            self.motif_list = random.sample(self.motif_list, 500)
            self.motif_num = 500

    def search_motif_from_adj(self):
        start_nodes = random.sample(self.node_ids, len(self.node_ids))  # 随机采样每个节点  就是打乱id
        with tqdm(total=len(start_nodes)) as t:
            for node in start_nodes:    # 随机遍历每个节点
                s = time.time()
                candi_motif = [node]
                self.dfs(candi_motif, self.size-1, t, s)    # 深度优先遍历出深度为4的环

                t.set_postfix(motif_num = len(self.motif_list))
                t.update(1)
                if len(self.motif_list) >= 500:
                    break

        if len(self.motif_list) != 0:
            self.motif_list = np.unique(np.sort(self.motif_list,axis=1), axis=0).tolist()


    def dfs(self, candi_motif, size, t, s):
        node0 = candi_motif[0]
        node1 = candi_motif[-1]
        max_node = max(candi_motif)
        for node_next in self.adj[node1]._indices()[0]:
            if int(node_next) > max_node:
                candi_motif.append(int(node_next))

                if size == 1:
                    if node_next in self.adj[node0]._indices()[0]:
                        self.motif_list.append(candi_motif[:])
                        self.time_adj.append(torch.tensor([self.adj1[candi_motif[i]][candi_motif[(i+1)%self.size]] for i in range(self.size)]))
                        t.set_postfix(motif_num = len(self.motif_list))

                else:
                    self.dfs(candi_motif,size-1, t,s)
                candi_motif.pop(-1)
            if len(self.motif_list) >= 500:
                break
            elif time.time()-s>100:
                break



class MotifSnapShot(SnapShot):
    def __init__(self, edges,time) -> None:
        super().__init__(edges,time)
        self.motif_list,self.time_adj = self.search_motif_from_adj()  # 从邻接矩阵中发现motif
        self.motif_num = len(self.motif_list)   # 正常motif的总数

    def search_motif_from_adj(self):
        motif_list = []
        time_motif=[]
        for node0 in self.node_ids:
            for node1 in self.adj[node0]._indices()[0]:
                if node1 <= node0:
                    continue
                # time_motif.append(self.adj[node0][node1]) # ***************
                for node2 in self.adj[node1]._indices()[0]:
                    if node2 <= node0 or node2 <= node1:
                        continue
                    if node2 in self.adj[node0]._indices()[0]:
                        time_motif.append(torch.tensor([self.adj1[node0][node1],self.adj1[node1][node2],self.adj1[node2][node0]]))
                        motif_list.append([node0, int(node1), int(node2)])
        return motif_list,time_motif

class AnomalyCircleSnapShot(CircleMotifSnapShot):
    def __init__(self, edges, motif_list=None, size = 4, p=0.02,device='cuda:0', time=None) -> None:
        super().__init__(edges, motif_list, size, time)
        self.device=device
        self.anomaly_motifs,self.anomaly_times = self.generate_anomalys(p)
        self.adj,self.norm_adj, self.new_edges = self.rebuild_anomaly_adj()

        self.edge2id = {(str)([min([self.new_edges[i][0],self.new_edges[i][1]]),max([self.new_edges[i][0],self.new_edges[i][1]])]): i  for i in range(len(self.new_edges))}    # ++++++  对应于每一个快照的  重复边就考虑最新的   +++++++


        self.label_adj = self.adj.to_dense()
        self.anomaly_nodes = np.unique(np.array(self.anomaly_motifs)).tolist()
        self.anomaly_motifs_num = len(self.anomaly_motifs)

        self.ano_edges_times = np.array([random.random() for _ in range(len(self.new_edges) - self.edges_num)])
        self.all_edges_times = np.concatenate((self.edges_times, self.ano_edges_times), axis=0)
        self.final_time = self.time_adj + self.anomaly_times

        original_tensor = torch.tensor(self.all_edges_times, dtype=torch.float32)
        tensor = torch.zeros((len(self.all_edges_times), 4), dtype=torch.float32)
        tensor[:, 0] = original_tensor
        tensor[:, 1] = original_tensor
        tensor[:, 2] = original_tensor
        tensor[:, 3] = original_tensor
        self.origin_time_adj = tensor
        self.final_time = list(self.origin_time_adj) + self.time_adj + self.anomaly_times

        self.motifs = self.motif_list + self.anomaly_motifs
        self.augmented_nodes_num = self.nodes_num + len(self.motifs)

        self.augmented_edges_num = len(self.new_edges) + len(self.motifs)  # 正常加异常边加基序边总数+++++
        self.augmented_edges_list = list(range(self.augmented_edges_num))  # 给每个边一个id 也包括了基序边+++++
        self.motif_edges_start_id = len(self.new_edges)  # 基序边起始id+++++
        self.motif_edges_id_list = self.augmented_edges_list[
                                   self.motif_edges_start_id:]  # 基序边的id列表  其实感觉用处不大，最重要的是异常边集，因为基序边是自己增加的+++++

        self.augmented_nodes_list = list(range(self.augmented_nodes_num))
        self.motif_start_id = self.nodes_num
        self.motif_id_list = self.augmented_nodes_list[self.motif_start_id:]

        self.motif_labels = np.array([0]*self.motif_num +[1]*self.anomaly_motifs_num)
        self.nodes_labels = np.zeros(self.nodes_num, dtype=np.int)
        self.nodes_labels[self.anomaly_nodes] = 1
        self.motif_adj, self.motif_norm_adj,self.every_motif_nei = self.build_motif_adj()

        self.final_edges = torch.from_numpy(self.new_edges).transpose(0, 1)
        self.hg = self.dual_hypergraph_trans(self.final_edges, self.nodes_num)

    def dual_hypergraph_trans(self,edge_index, n_node):  #
        # adjacency matrix of graph -> incidence matrix of graph  pyg的稀疏矩阵格式，直接从edge_index得到
        # edge_index=edge_index.to("cuda")    # ++++++++

        num_edge = edge_index.size(1)  # 边的数量，行索引在边上  4
        col = torch.arange(0, num_edge, 1).repeat_interleave(2).view(1, -1).squeeze().to(  # [0,0,1,1,2,2,3,3]
            edge_index.device)  # 列索引：边 0123……   重复两遍-一条边连两端节点
        row = edge_index.T.reshape(1, -1).squeeze().to(edge_index.device)  # [0,1,1,2,0,1,0]

        for i,nei in enumerate(self.every_motif_nei):#+++++
            # i+4219是基序对应的边id nei是邻居集合
            for neigh in nei:
                col=torch.cat((col,torch.tensor([i+num_edge])), dim=0)
                row=torch.cat((row, torch.tensor([neigh])), dim=0)

        val = torch.ones(row.size(0)).to(edge_index.device)  # 每个边连节点的权重
        # print('row:{}, col:{}'.format(row.device, col.device))
        # M = SparseTensor(row=row,
        #                 col=col,
        #                 value=val,
        #                 sparse_sizes=(n_node, num_edge)).coalesce()
        # incidence matrix of graph -> incidence matrix of hypergraph (转置)
        MT = SparseTensor(row=col, col=row, value=val, sparse_sizes=(self.augmented_edges_num, n_node)).coalesce()  # 关联矩阵H
        # node degree, edge degree of hypergraph
        MT_dense=MT.to_dense()
        row_sums = torch.sum(MT_dense, dim=1, keepdim=True)
        normalized_matrix = MT_dense / row_sums
        # norms = torch.norm(MT_dense, p=2, dim=1, keepdim=True)
        # normalized_matrix = MT_dense / norms
        row_indices, col_indices = torch.nonzero(normalized_matrix, as_tuple=True)
        values = normalized_matrix[row_indices, col_indices]
        MT1 = SparseTensor(row=row_indices, col=col_indices, value=values)

        D_e = MT.sum(1)  #
        D_v = MT.sum(0)  # sum(W*MT, dim=1)     # 改了一下，反一下  节点度是每个节点连的超边数
        D_e = torch.pow(D_e, -0.5)
        D_v = torch.pow(D_v, -0.5)  # 标准化
        # B_v B_e 转对角矩阵？
        row_e = col_e = torch.arange(D_e.size(0), dtype=torch.long).to(edge_index.device)
        row_v = col_v = torch.arange(D_v.size(0), dtype=torch.long).to(edge_index.device)
        D_e = SparseTensor(row=row_e, col=col_e, value=D_e, sparse_sizes=(D_e.size(0), D_e.size(0))).coalesce()  # 转对角矩阵
        D_v = SparseTensor(row=row_v, col=col_v, value=D_v, sparse_sizes=(D_v.size(0), D_v.size(0))).coalesce()

        D_e,D_v=D_v,D_e

        B_v = D_v
        B_e = D_e

        hg = HGObject()
        hg.MT = MT
        hg.D_e = D_e
        hg.D_v = D_v
        # hg.BHWDHD=torch.sparse.mm(torch.sparse.mm(torch.sparse.mm(torch.sparse.mm(torch.sparse.mm(D_v, MT), D_e), D_e), MT), D_v)

        hg.DHWDHD = (D_v @ MT @ D_e @ D_e @ MT.t() @ D_v).to(self.device)

        hg.DHWD = (D_v @ MT @ D_e @ D_e).to(self.device)

        hg.BHUBHB = (B_e @ MT.t() @ B_v @ B_v @ MT @ B_e).to(self.device)  #

        hg.BHUB = (B_e @ MT.t() @ B_v @ B_v).to(self.device)

        hg.HD = (MT.t() @ D_v).to(self.device)

        hg.HB = (MT @ B_e).to(self.device)

        hg.norm_MT=normalized_matrix

        return hg


    def generate_anomalys(self, p):
        anomaly_motifs = []
        anomaly_times = []
        anomaly_motifs_num = max(1, int(self.motif_num*p))
        for _ in range(anomaly_motifs_num):
            candi_motif = random.sample(self.node_ids, self.size) #list
            candi_motif.sort()
            while candi_motif in self.motif_list or candi_motif in anomaly_motifs:
                candi_motif = random.sample(self.node_ids, self.size)
                candi_motif.sort()

            anomaly_motifs.append(candi_motif)
        for candi_motif in anomaly_motifs:
            time=[]
            for i in range(self.size):
                if self.adj1[candi_motif[i]][candi_motif[(i+1)%self.size]]==0:
                    time.append(torch.tensor(random.random()))
                else:
                    time.append(self.adj1[candi_motif[i]][candi_motif[(i+1)%self.size]])
            anomaly_times.append(torch.tensor(time))

        return anomaly_motifs ,anomaly_times


    def rebuild_anomaly_adj(self):
        indices = self.edges.T
        src_nodes = indices[0].tolist()
        dst_nodes = indices[1].tolist()
        src_nodes = [self.node2id[i] for i in src_nodes]
        dst_nodes = [self.node2id[i] for i in dst_nodes]
        anomaly_motifs_src = []
        anomaly_motifs_dst = []
        for motif in self.anomaly_motifs:
            anomaly_motifs_src += [motif[i] for i in range(self.size)]
            anomaly_motifs_dst += [motif[(i+1)%self.size] for i in range(self.size)]
            # anomaly_motifs_dst += [motif[self.size-i-1] for i in range(self.size)]
        src_nodes += anomaly_motifs_src
        dst_nodes += anomaly_motifs_dst
        edges = np.vstack((src_nodes, dst_nodes)).T
        values = torch.ones(len(src_nodes), dtype=torch.float32)
        adj = coo_matrix((values, (src_nodes, dst_nodes)),
                         shape=(self.nodes_num, self.nodes_num))
        adj = self.convert_symmetric(adj, sparse=True)
        norm_adj = preprocess_adj(adj, is_sparse=True).to_dense()
        adj = adj2tensor(adj, is_sparse=True)
        return adj, norm_adj, edges


    def build_motif_adj(self):
        indices = self.edges.T
        src_nodes = indices[0].tolist()
        dst_nodes = indices[1].tolist()
        src_nodes = [self.node2id[i] for i in src_nodes]
        dst_nodes = [self.node2id[i] for i in dst_nodes]
        motif_src_nodes = []
        motif_dst_nodes = []
        every_motif_nei = []
        for motif_id, motif in zip(self.motif_id_list, self.motifs):    # 并没有做到基序supernode之间有连接
            neighbor_set = set()
            for node in motif:
                for neighbor in self.adj[node]._indices()[0]:
                    neighbor_set.add(int(neighbor))
                neighbor_set.add(node)
            motif_dst_nodes+=list(neighbor_set)
            motif_src_nodes+=[motif_id]*len(neighbor_set)
            every_motif_nei.append(list(neighbor_set))  # *******

        src_nodes += motif_src_nodes
        dst_nodes += motif_dst_nodes

        values = torch.ones(len(src_nodes), dtype=torch.float32)
        adj = coo_matrix((values, (src_nodes, dst_nodes)),
                         shape=(self.augmented_nodes_num, self.augmented_nodes_num))
        adj = self.convert_symmetric(adj, sparse=True)
        norm_adj = preprocess_adj(adj, is_sparse=True).to_dense()
        adj = adj2tensor(adj, is_sparse=True).to_dense()
        adj = torch.sqrt(adj/adj.sum(-1,keepdim=True))
        return adj, norm_adj,every_motif_nei


class HGObject:
    pass
    # def __init__(self):
    #     self.MT = None
    #     self.D_e = None
    #     self.D_v = None
    #     self.DHWDHD = None
    #     self.DHWD = None
    #     self.BHUBHB = None
    #     self.BHUB = None
    #
    # def compute_properties(self, MT, D_e, D_v, B_e, B_v):
    #     self.MT = MT
    #     self.D_e = D_e
    #     self.D_v = D_v
    #
    #     self.DHWDHD = D_v @ MT.t() @ D_e @ D_e @ MT @ D_v
    #     self.DHWD = D_v @ MT.t() @ D_e @ D_e
    #     self.BHUBHB = B_e @ MT @ B_v @ B_v @ MT.t() @ B_e
    #     self.BHUB = B_e @ MT @ B_v @ B_v
    # def to(self, device):
    #     # 将对象的属性移动到指定设备上
    #     self.some_attribute = self.some_attribute.to(device)
class AnomalyMotifSnapShot(MotifSnapShot):  # 快照类
    def __init__(self, edges, p=0.02,device='cuda:0',time=None) -> None:
        super().__init__(edges,time) # 输入每个基序的边 以及异常概率 默认0.1
        self.edges_times = time
        self.device=device
        self.anomaly_motifs,self.anomaly_times = self.generate_anomalys(p) # 生成异常基序
        self.adj,self.norm_adj, self.new_edges = self.rebuild_anomaly_adj()     # 邻接矩阵 标准化邻接矩阵  注入异常后的边集 +++++new_edge是算上异常边的
        # self.new_edge是加上基序边后的边集
        self.edge2id = {(str)([min([self.new_edges[i][0],self.new_edges[i][1]]),max([self.new_edges[i][0],self.new_edges[i][1]])]): i  for i in range(len(self.new_edges))}    # ++++++  对应于每一个快照的  重复边就考虑最新的   +++++++


        self.label_adj = self.adj.to_dense()    # 稠密矩阵
        self.anomaly_nodes = np.unique(np.array(self.anomaly_motifs)).tolist()  # 属于异常基序的节点总数
        self.anomaly_motifs_num = len(self.anomaly_motifs)  # 异常基序总数 ******==增强超边总数

        self.ano_edges_times = np.array([random.random() for _ in range(len(self.new_edges) - self.edges_num)])
        self.all_edges_times = np.concatenate((self.edges_times, self.ano_edges_times), axis=0)
        self.final_time = self.time_adj + self.anomaly_times    # 关键步骤***  需要在这里加上

        original_tensor=torch.tensor(self.all_edges_times, dtype=torch.float32)
        tensor = torch.zeros((len(self.all_edges_times), 3), dtype=torch.float32)
        tensor[:,0]=original_tensor
        tensor[:,1]=original_tensor
        tensor[:,2]=original_tensor
        self.origin_time_adj=tensor
        self.final_time = list(self.origin_time_adj)+self.time_adj + self.anomaly_times

        self.motifs = self.motif_list + self.anomaly_motifs # 正常基序+异常基序=该快照所有基序
        self.augmented_nodes_num = self.nodes_num + len(self.motifs)    # 增强后节点总数  原始节点数+supernode数  self.augmented_edges_num = self.edges_num + len(self.motifs)+++++差异常边

        self.augmented_edges_num = len(self.new_edges) + len(self.motifs)   # 正常加异常边加基序边总数+++++
        self.augmented_edges_list = list(range(self.augmented_edges_num))   # 给每个边一个id 也包括了基序边+++++
        self.motif_edges_start_id = len(self.new_edges)                     # 基序边起始id+++++
        self.motif_edges_id_list = self.augmented_edges_list[self.motif_edges_start_id:]    # 基序边的id列表  其实感觉用处不大，最重要的是异常边集，因为基序边是自己增加的+++++

        self.augmented_nodes_list = list(range(self.augmented_nodes_num))   # 增强后节点id列表  增强边id列表需要先加上异常边+++++
        self.motif_start_id = self.nodes_num    # 基序supernode的开始id
        self.motif_id_list = self.augmented_nodes_list[self.motif_start_id:]    # supernode的id列表

        self.motif_labels = np.array([0]*self.motif_num +[1]*self.anomaly_motifs_num)   # 基序标签
        self.nodes_labels = np.zeros(self.nodes_num, dtype=np.int)  # 原始节点label全0
        self.nodes_labels[self.anomaly_nodes] = 1   # 属于异常基序的节点的label为1
        self.motif_adj, self.motif_norm_adj ,self.every_motif_nei = self.build_motif_adj()    # 基序邻接矩阵（重构用？  不考虑基序supernode和supernode的连接 我需要在这步得到每个基序边的邻居
        self.a=0    # every_motif_nei是每个基序（a,b,c）对应的基序边要连接的节点
        self.final_edges=torch.from_numpy(self.new_edges).transpose(0, 1)
        self.hg=self.dual_hypergraph_trans(self.final_edges,self.nodes_num)



    # 用矩阵表示 实现对偶超图转换
    def dual_hypergraph_trans(self,edge_index, n_node):  #
        # adjacency matrix of graph -> incidence matrix of graph  pyg的稀疏矩阵格式，直接从edge_index得到
        # edge_index=edge_index.to("cuda")    # ++++++++

        num_edge = edge_index.size(1)  # 边的数量，行索引在边上  4
        col = torch.arange(0, num_edge, 1).repeat_interleave(2).view(1, -1).squeeze().to(  # [0,0,1,1,2,2,3,3]
            edge_index.device)  # 列索引：边 0123……   重复两遍-一条边连两端节点
        row = edge_index.T.reshape(1, -1).squeeze().to(edge_index.device)  # [0,1,1,2,0,1,0]

        for i,nei in enumerate(self.every_motif_nei):#+++++
            # i+4219是基序对应的边id nei是邻居集合
            for neigh in nei:
                col=torch.cat((col,torch.tensor([i+num_edge])), dim=0)
                row=torch.cat((row, torch.tensor([neigh])), dim=0)

        val = torch.ones(row.size(0)).to(edge_index.device)  # 每个边连节点的权重
        # print('row:{}, col:{}'.format(row.device, col.device))
        # M = SparseTensor(row=row,
        #                 col=col,
        #                 value=val,
        #                 sparse_sizes=(n_node, num_edge)).coalesce()
        # incidence matrix of graph -> incidence matrix of hypergraph (转置)
        MT = SparseTensor(row=col, col=row, value=val, sparse_sizes=(self.augmented_edges_num, n_node)).coalesce()  # 关联矩阵H
        # node degree, edge degree of hypergraph
        MT_dense=MT.to_dense()
        row_sums = torch.sum(MT_dense, dim=1, keepdim=True)
        normalized_matrix = MT_dense / row_sums
        # norms = torch.norm(MT_dense, p=2, dim=1, keepdim=True)
        # normalized_matrix = MT_dense / norms
        row_indices, col_indices = torch.nonzero(normalized_matrix, as_tuple=True)
        values = normalized_matrix[row_indices, col_indices]
        MT1 = SparseTensor(row=row_indices, col=col_indices, value=values)

        D_e = MT.sum(1)  #
        D_v = MT.sum(0)  # sum(W*MT, dim=1)     # 改了一下，反一下  节点度是每个节点连的超边数
        D_e = torch.pow(D_e, -0.5)
        D_v = torch.pow(D_v, -0.5)  # 标准化
        # B_v B_e 转对角矩阵？
        row_e = col_e = torch.arange(D_e.size(0), dtype=torch.long).to(edge_index.device)
        row_v = col_v = torch.arange(D_v.size(0), dtype=torch.long).to(edge_index.device)
        D_e = SparseTensor(row=row_e, col=col_e, value=D_e, sparse_sizes=(D_e.size(0), D_e.size(0))).coalesce()  # 转对角矩阵
        D_v = SparseTensor(row=row_v, col=col_v, value=D_v, sparse_sizes=(D_v.size(0), D_v.size(0))).coalesce()

        D_e, D_v = D_v, D_e

        B_v = D_v
        B_e = D_e

        hg = HGObject()
        hg.MT = MT
        hg.D_e = D_e
        hg.D_v = D_v
        # hg.BHWDHD=torch.sparse.mm(torch.sparse.mm(torch.sparse.mm(torch.sparse.mm(torch.sparse.mm(D_v, MT), D_e), D_e), MT), D_v)

        hg.DHWDHD = (D_v @ MT @ D_e @ D_e @ MT.t() @ D_v).to(self.device)

        hg.DHWD = (D_v @ MT @ D_e @ D_e).to(self.device)

        hg.BHUBHB = (B_e @ MT.t() @ B_v @ B_v @ MT @ B_e).to(self.device)  #

        hg.BHUB = (B_e @ MT.t() @ B_v @ B_v).to(self.device)

        hg.HD=(MT.t() @ D_v).to(self.device)

        hg.HB=(MT @ B_e).to(self.device)

        hg.norm_MT=normalized_matrix

        return hg

    def generate_anomalys(self, p):
        anomaly_motifs = []
        anomaly_times = []
        anomaly_motifs_num = max(1, int(self.motif_num*p))  # 要生成的异常基序个数
        for _ in range(anomaly_motifs_num):
            candi_motif = random.sample(self.node_ids, 3) #list 随机采样三个节点
            candi_motif.sort()  # 节点id排序
            while candi_motif in self.motif_list or candi_motif in anomaly_motifs:  # 如果要生成基序是原始基序或者已生成基序 则需要重新采样三个节点
                candi_motif = random.sample(self.node_ids, 3)
                candi_motif.sort()

            anomaly_motifs.append(candi_motif)  # 添加到生成基序列表当中  既然可以记录基序id 那么生成式就有用武之地  这里可以记录基序id 我同样可以记录子图id

        for candi_motif in anomaly_motifs:
            if self.adj1[candi_motif[0]][candi_motif[1]]==0:
                t1=torch.tensor(random.random())
            else:
                t1=self.adj1[candi_motif[0]][candi_motif[1]]
            if self.adj1[candi_motif[1]][candi_motif[2]]==0:
                t2=torch.tensor(random.random())
            else:
                t2=self.adj1[candi_motif[1]][candi_motif[2]]
            if self.adj1[candi_motif[2]][candi_motif[0]]==0:
                t3=torch.tensor(random.random())
            else:
                t3=self.adj1[candi_motif[2]][candi_motif[0]]
            anomaly_times.append(torch.tensor([t1, t2, t3]))

        return anomaly_motifs,anomaly_times

    def generate_anomalys2(self, p):
        anomaly_motifs = []
        anomaly_motifs_num = max(1, int(self.motif_num*p))
        select_zone = list(np.unique(self.motif_list))

        for _ in range(anomaly_motifs_num):
            candi_motif = random.sample(select_zone, 3) #list
            candi_motif.sort()
            while candi_motif in self.motif_list or candi_motif in anomaly_motifs:
                candi_motif = random.sample(select_zone, 3)
                candi_motif.sort()

            anomaly_motifs.append(candi_motif)
        return anomaly_motifs


    def rebuild_anomaly_adj(self):
        indices = self.edges.T  # 边集
        src_nodes = indices[0].tolist() # 源
        dst_nodes = indices[1].tolist() # 目标
        src_nodes = [self.node2id[i] for i in src_nodes]    # 这两行没什么变化×
        dst_nodes = [self.node2id[i] for i in dst_nodes]    # id是快照内的编号 node是所有节点的编号
        anomaly_motifs_src = []
        anomaly_motifs_dst = []
        for motif in self.anomaly_motifs:   # 遍历异常基序列表
            anomaly_motifs_src += [motif[0],motif[1],motif[2]]  # 在源、目标列表中添加 生成异常基序的信息 对应三条边
            anomaly_motifs_dst += [motif[1],motif[2],motif[0]]
        src_nodes += anomaly_motifs_src
        dst_nodes += anomaly_motifs_dst # 拼接上
        edges = np.vstack((src_nodes, dst_nodes)).T # 堆叠得到边集
        values = torch.ones(len(src_nodes), dtype=torch.float32)    # 正+生成 个 1 可能是边权重W 重复边所以权重增加
        adj = coo_matrix((values, (src_nodes, dst_nodes)),
                         shape=(self.nodes_num, self.nodes_num)) # coo格式的邻接矩阵  col row data
        adj = self.convert_symmetric(adj, sparse=True)  # 无向图，所以转为对称矩阵
        norm_adj = preprocess_adj(adj, is_sparse=True).to_dense()   # 标准化邻接矩阵
        adj = adj2tensor(adj, is_sparse=True)   # 转化为张量
        return adj, norm_adj, edges


    def build_motif_adj(self):  # 基序邻接矩阵 用于重构？
        indices = self.edges.T  # [[src id][dst id]]
        src_nodes = indices[0].tolist() # 源节点list
        dst_nodes = indices[1].tolist() # 目标节点list
        src_nodes = [self.node2id[i] for i in src_nodes]
        dst_nodes = [self.node2id[i] for i in dst_nodes]
        motif_src_nodes = []
        motif_dst_nodes = []
        every_motif_nei=[]
        for motif_id, motif in zip(self.motif_id_list, self.motifs):    # motif_id_list是基序虚拟节点id   遍历每个基序 正+负
            neighbor_set = set()
            for node in motif:  # 遍历基序中的每个节点
                for neighbor in self.adj[node]._indices()[0]:   # 遍历该节点的邻居id
                    neighbor_set.add(int(neighbor))
                neighbor_set.add(node)  # 基序中每个节点的邻居+自身
            motif_dst_nodes+=list(neighbor_set) # neighbor_set表示该基序的邻居以及自身节点
            motif_src_nodes+=[motif_id]*len(neighbor_set)   #让虚拟节点和基序的邻居以及基序中的每个节点相连 增强方法1，2
            every_motif_nei.append(list(neighbor_set)) #*******
        src_nodes += motif_src_nodes
        dst_nodes += motif_dst_nodes    # 增加到节点列表  这样就还差一个基序相交 虚拟节点相连
        # 已有边集合  构建超图
        values = torch.ones(len(src_nodes), dtype=torch.float32)    # 边权全0
        adj = coo_matrix((values, (src_nodes, dst_nodes)),
                         shape=(self.augmented_nodes_num, self.augmented_nodes_num))    # 邻接矩阵
        adj = self.convert_symmetric(adj, sparse=True)  # 堆成
        norm_adj = preprocess_adj(adj, is_sparse=True).to_dense()   # 稠密标准
        adj = adj2tensor(adj, is_sparse=True).to_dense()    # 转张量
        adj = torch.sqrt(adj/adj.sum(-1,keepdim=True))  # 标准化
        # adj = adj/adj.sum(-1,keepdim=True)
        return adj, norm_adj ,every_motif_nei    # 所以这个执行完之后 supernode和supernode之间还是没有连接 ******


class AnomalyEdgeSnapShot(SnapShot):
    def __init__(self, edges, p=0.02) -> None:
        super().__init__(edges)
        self.edges_list = self.rebuild_edges()
        self.motif_num = self.edges_num
        self.anomaly_edges = self.generate_anomalys(p)
        self.adj,self.norm_adj = self.rebuild_anomaly_adj()
        self.label_adj = self.adj.to_dense()
        self.anomaly_nodes = np.unique(np.array(self.anomaly_edges)).tolist()
        self.anomaly_motifs_num = len(self.anomaly_edges)

        self.motifs = self.edges_list + self.anomaly_edges
        self.augmented_nodes_num = self.nodes_num + len(self.motifs)
        self.augmented_nodes_list = list(range(self.augmented_nodes_num))
        self.motif_start_id = self.nodes_num
        self.motif_id_list = self.augmented_nodes_list[self.motif_start_id:]

        self.motif_labels = np.array([0]*self.edges_num +[1]*self.anomaly_motifs_num)
        self.nodes_labels = np.zeros(self.nodes_num, dtype=np.int)
        self.nodes_labels[self.anomaly_nodes] = 1
        self.motif_adj, self.motif_norm_adj = self.build_motif_adj()

    def rebuild_edges(self):
        edges = self.edges.tolist()
        edges = [[self.node2id[i] for i in edge] for edge in edges]
        return edges

    def generate_anomalys(self, p):
        anomaly_edges = []
        anomaly_edges_num = max(1, round(self.edges_num*p))
        for _ in range(anomaly_edges_num):
            candi_edges = random.sample(self.node_ids, 2) #list
            candi_edges.sort()
            while candi_edges in self.edges_list or candi_edges in anomaly_edges:
                candi_edges = random.sample(self.node_ids, 2)
                candi_edges.sort()

            anomaly_edges.append(candi_edges)
        return anomaly_edges

    def rebuild_anomaly_adj(self):
        indices = self.edges.T
        src_nodes = indices[0].tolist()
        dst_nodes = indices[1].tolist()
        src_nodes = [self.node2id[i] for i in src_nodes]
        dst_nodes = [self.node2id[i] for i in dst_nodes]

        anomaly_edges_src = []
        anomaly_edges_dst = []
        for edge in self.anomaly_edges:
            anomaly_edges_src += [edge[0]]
            anomaly_edges_dst += [edge[1]]
        src_nodes += anomaly_edges_src
        dst_nodes += anomaly_edges_dst
        values = torch.ones(len(src_nodes), dtype=torch.float32)
        adj = coo_matrix((values, (src_nodes, dst_nodes)),
                         shape=(self.nodes_num, self.nodes_num))
        adj = self.convert_symmetric(adj, sparse=True)
        norm_adj = preprocess_adj(adj, is_sparse=True).to_dense()

        adj = adj2tensor(adj, is_sparse=True)
        return adj, norm_adj

    def build_motif_adj(self):
        indices = self.edges.T
        src_nodes = indices[0].tolist()
        dst_nodes = indices[1].tolist()
        src_nodes = [self.node2id[i] for i in src_nodes]
        dst_nodes = [self.node2id[i] for i in dst_nodes]
        motif_src_nodes = []
        motif_dst_nodes = []
        for motif_id, motif in zip(self.motif_id_list, self.motifs):
            neighbor_set = set()
            for node in motif:
                for neighbor in self.adj[node]._indices()[0]:
                    neighbor_set.add(int(neighbor))
                neighbor_set.add(node)
            motif_dst_nodes+=list(neighbor_set)
            motif_src_nodes+=[motif_id]*len(neighbor_set)

        src_nodes += motif_src_nodes
        dst_nodes += motif_dst_nodes

        anomaly_edges_src = []
        anomaly_edges_dst = []
        for edge in self.anomaly_edges:
            anomaly_edges_src += [edge[0]]
            anomaly_edges_dst += [edge[1]]
        src_nodes += anomaly_edges_src
        dst_nodes += anomaly_edges_dst

        values = torch.ones(len(src_nodes), dtype=torch.float32)
        adj = coo_matrix((values, (src_nodes, dst_nodes)),
                         shape=(self.augmented_nodes_num, self.augmented_nodes_num))
        adj = self.convert_symmetric(adj, sparse=True)
        norm_adj = preprocess_adj(adj, is_sparse=True).to_dense()

        adj = adj2tensor(adj, is_sparse=True).to_dense()
        adj = torch.sqrt(adj/adj.sum(-1,keepdim=True))
        # adj = adj/adj.sum(-1,keepdim=True)
        return adj, norm_adj