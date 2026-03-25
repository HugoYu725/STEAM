# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 21:47:39 2024

@author: chlgao
"""
import torch
from torch_sparse import SparseTensor


class HGObject:
    pass


# 用矩阵表示 实现对偶超图转换
def dual_hypergraph_trans(edge_index, n_node):  #
    # adjacency matrix of graph -> incidence matrix of graph  pyg的稀疏矩阵格式，直接从edge_index得到
    num_edge = edge_index.size(1)  # 边的数量，行索引在边上  4
    col = torch.arange(0, num_edge, 1).repeat_interleave(2).view(1, -1).squeeze().to(   # [0,0,1,1,2,2,3,3]
        edge_index.device)  # 列索引：边 0123……   重复两遍-一条边连两端节点
    row = edge_index.T.reshape(1, -1).squeeze().to(edge_index.device)   # [0,1,1,2,0,1,0]
    val = torch.ones(row.size(0)).to(edge_index.device) # 每个边连节点的权重
    # print('row:{}, col:{}'.format(row.device, col.device))
    # M = SparseTensor(row=row,
    #                 col=col,
    #                 value=val,
    #                 sparse_sizes=(n_node, num_edge)).coalesce()
    # incidence matrix of graph -> incidence matrix of hypergraph (转置)
    MT = SparseTensor(row=col, col=row, value=val, sparse_sizes=(num_edge, n_node)).coalesce()  # 关联矩阵H
    # node degree, edge degree of hypergraph
    D_e = MT.sum(1)  #
    D_v = MT.sum(0)  # sum(W*MT, dim=1)     # 改了一下，反一下  节点度是每个节点连的超边数
    D_e = torch.pow(D_e, -0.5)
    D_v = torch.pow(D_v, -0.5)  # 标准化
    # B_v B_e 转对角矩阵？
    row_e = col_e = torch.arange(D_e.size(0), dtype=torch.long).to(edge_index.device)
    row_v = col_v = torch.arange(D_v.size(0), dtype=torch.long).to(edge_index.device)
    D_e = SparseTensor(row=row_e, col=col_e, value=D_e, sparse_sizes=(D_e.size(0), D_e.size(0))).coalesce() # 转对角矩阵
    D_v = SparseTensor(row=row_v, col=col_v, value=D_v, sparse_sizes=(D_v.size(0), D_v.size(0))).coalesce()

    B_v=D_v
    B_e=D_e

    hg = HGObject()
    hg.MT = MT
    hg.D_e = D_e
    hg.D_v = D_v
    # hg.BHWDHD=torch.sparse.mm(torch.sparse.mm(torch.sparse.mm(torch.sparse.mm(torch.sparse.mm(D_v, MT), D_e), D_e), MT), D_v)

    hg.DHWDHD=D_v @ MT.t()@ D_e@ D_e@ MT@ D_v

    hg.DHWD=D_v @ MT.t()@ D_e@ D_e

    hg.BHUBHB=B_e@MT @B_v@B_v @MT.t()@B_e   #

    hg.BHUB=B_e@MT@B_v@B_v

    return hg

row = torch.tensor([0, 1, 2, 1])  # 边的起始节点索引
col = torch.tensor([1, 2, 0, 0])  # 边的结束节点索引

# 创建边张量
edge_index = torch.stack([row, col], dim=0)
n_node=3
hg=dual_hypergraph_trans(edge_index,n_node)
print(hg)