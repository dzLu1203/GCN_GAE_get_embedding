import pickle as pkl

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

import ndex2
import networkx as nx
import pandas as pd
def read_data(path = '../data/PPI/PCNet.cx'):
    '''
    :param
            path: .cx结构的PPI网络文件，这边我们用的是PCNet.cx
    :return:
            adj: normalized 邻接矩阵 n*n
            features：特征矩阵，大小n*m，m为128维
    '''
    PCNet = ndex2.create_nice_cx_from_file(path)
    Gnx = PCNet.to_networkx()
    Gint = nx.convert_node_labels_to_integers(Gnx, first_label=0, ordering='default', label_attribute=not None)
    G_mtx = nx.to_numpy_matrix(Gint)
    Gmtx = np.array(G_mtx)

    ASD_HC = pd.read_csv('../data/plan3_data/ASD_HC.tsv', sep='\t', index_col='Unnamed: 0')
    ASD_HC = [str(g[1:-1]).strip("'") for g in ASD_HC['seed_genes'].tolist()[0][1:-1].split(', ')]
    CHD_HC = pd.read_csv('../data/plan3_data/CHD_HC.tsv', sep='\t', index_col='Unnamed: 0')
    CHD_HC = [str(g[1:-1]).strip("'") for g in CHD_HC['seed_genes'].tolist()[0][1:-1].split(', ')]

    node_map, gene_name_idx_dict = init_mapping()
    ASD_HC_idx = []
    for x in ASD_HC:
        y = gene_name_idx_dict[x]
        ASD_HC_idx.append(y)
    CHD_HC_idx = []
    for x in CHD_HC:
        y = gene_name_idx_dict[x]
        CHD_HC_idx.append(y)


    feature = Gmtx
    feature = sp.coo_matrix(feature)

    # adj = sp.coo_matrix(Gmtx)
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sp.csr_matrix(Gmtx)
    feature = Gmtx
    feature = torch.FloatTensor(feature)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, feature



def init_mapping():
    node_file = open('../data/plan3_data/node_name.txt','r')
    node = node_file.read()
    node.split(', ')
    node_list = list(nd for nd in node.split(', '))
    n = len(node_list)
    node_map = []
    for i in range(n):
        ff = node_list[i]
        ff = ff[1:-1]
        node_map.append(ff)
    gene_name_idx_dict = dict((node_map[i],i) for i in range(19781))
    return node_map,gene_name_idx_dict


def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        '''
        fix Pickle incompatibility of numpy arrays between Python 2 and 3
        https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
        '''
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as rf:
            u = pkl._Unpickler(rf)
            u.encoding = 'latin1'
            cur_data = u.load()
            objects.append(cur_data)
        # objects.append(
        #     pkl.load(open("data/ind.{}.{}".format(dataset, names[i]), 'rb')))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = torch.FloatTensor(np.array(features.todense()))
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score























# 下面是GCN的utils
'''
import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

import ndex2
import networkx as nx
import pandas as pd
def read_data(path = '../data/PPI/PCNet.cx'):
    \'''
    :param
            path: .cx结构的PPI网络文件，这边我们用的是PCNet.cx
    :return:
            adj: normalized 邻接矩阵 n*n
            features：特征矩阵，大小n*m，m为128维
    \'''
    PCNet = ndex2.create_nice_cx_from_file(path)
    Gnx = PCNet.to_networkx()
    Gint = nx.convert_node_labels_to_integers(Gnx, first_label=0, ordering='default', label_attribute=not None)
    G_mtx = nx.to_numpy_matrix(Gint)
    Gmtx = np.array(G_mtx)

    ASD_HC = pd.read_csv('../data/plan3_data/ASD_HC.tsv', sep='\t', index_col='Unnamed: 0')
    ASD_HC = [str(g[1:-1]).strip("'") for g in ASD_HC['seed_genes'].tolist()[0][1:-1].split(', ')]
    CHD_HC = pd.read_csv('../data/plan3_data/CHD_HC.tsv', sep='\t', index_col='Unnamed: 0')
    CHD_HC = [str(g[1:-1]).strip("'") for g in CHD_HC['seed_genes'].tolist()[0][1:-1].split(', ')]

    node_map, gene_name_idx_dict = init_mapping()
    ASD_HC_idx = []
    for x in ASD_HC:
        y = gene_name_idx_dict[x]
        ASD_HC_idx.append(y)
    CHD_HC_idx = []
    for x in CHD_HC:
        y = gene_name_idx_dict[x]
        CHD_HC_idx.append(y)


    feature = Gmtx
    feature = sp.coo_matrix(feature)

    adj = sp.coo_matrix(Gmtx)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    feature = torch.FloatTensor(np.array(feature.todense()))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, feature



def init_mapping():
    node_file = open('../data/plan3_data/node_name.txt','r')
    node = node_file.read()
    node.split(', ')
    node_list = list(nd for nd in node.split(', '))
    n = len(node_list)
    node_map = []
    for i in range(n):
        ff = node_list[i]
        ff = ff[1:-1]
        node_map.append(ff)
    gene_name_idx_dict = dict((node_map[i],i) for i in range(19781))
    return node_map,gene_name_idx_dict

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
'''