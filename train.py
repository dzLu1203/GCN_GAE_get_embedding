import time
import torch
from utils import *
from GAE_model import *
from torch import optim
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from optimizer import loss_function


'''
    adj：邻接矩阵
    feature：采用的特征，有三个想法，（1）one-hot向量 （2）邻接矩阵 （3）全是1或者0的向量
    首先采用的是邻接矩阵
'''
adj, features = read_data()

#############  init parameters ##############

epochs = 10
hidden1 = 2000
hidden2 = 128
lr = 0.01
dropout = 0.
device = 'cpu'
##############################################

print('get data finish')
n_nodes, feat_dim = features.shape
print('n_nodes: ', n_nodes, 'feat_dim: ', feat_dim)

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()
print('get0')
# adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj_train = adj
adj = adj_train
print('get1')
# Some preprocessing
adj_norm = preprocess_graph(adj)
adj_label = adj_train + sp.eye(adj_train.shape[0])
# adj_label = sparse_to_tuple(adj_label)
adj_label = torch.FloatTensor(adj_label.toarray())
pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())


norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
print('get2')

model = GCNModelVAE(feat_dim, hidden1, hidden2, dropout)
optimizer = optim.Adam(model.parameters(), lr=lr)

embed = None
loss_val = []
for epoch in range(epochs):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    recovered, mu, logvar = model(features,adj_norm)
    # print(recovered.shape)
    loss = loss_function(preds=recovered,labels=adj_label,mu=mu,
                         logvar=logvar, n_nodes=n_nodes,norm = norm, pos_weight=pos_weight)
    loss.backward()
    cur_loss = loss.item()
    optimizer.step()
    loss_val.append(cur_loss)
    embed = mu.data.numpy()
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
          "time=", "{:.5f}".format(time.time() - t)
          )

print('Optimization finish!')
torch.save(embed,'GAE_embed_128_epoch_10.pth')

x_ax = [e for e in range(len(loss_val))]
plt.plot(x_ax,loss_val)
plt.show()
