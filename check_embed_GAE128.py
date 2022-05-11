# import torch
# import numpy as np
# embed = torch.load('GAE_embed_128_epoch_10.pth')
#
# embed = np.array(embed)
# print(embed[0])
# print(embed[1])
# print(embed[1002])


import ndex2
import networkx as nx
import numpy as np
path = '../data/PPI/PCNet.cx'

PCNet = ndex2.create_nice_cx_from_file(path)
Gnx = PCNet.to_networkx()
Gint = nx.convert_node_labels_to_integers(Gnx, first_label=0, ordering='default', label_attribute=not None)
G_mtx = nx.to_numpy_matrix(Gint)
Gmtx = np.array(G_mtx)

ff = []

for i in range(19781):
    for j in range(19781):
        if Gmtx[i][j] == 1:
            tmp = [i,j]
            ff.append(tmp)

path1 = '../data/PCNet_edglist.txt'

output = open(path1,'w+')
for i in range(len(ff)):
    for j in range(len(ff[i])):
        output.write(str(ff[i][j]))
        output.write(' ')
    output.write('\n')
output.close()
