import torch
import torch.nn as nn

import dgl.function as fn


class GNLayer(nn.Module):
    def __init__(self,
                 edge_block,
                 node_block,
                 global_block=None):
    
        super(GNLayer, self).__init__()
        
        # f_e, f_v, f_g
        self.edge_block = edge_block
        self.node_block = node_block
        self.global_block = global_block
                
    def forward(self, graph, global_attr=None, out_edge_key='h_e', out_node_key='h_v'):
        """This is a high-level module.
        Read graph and
        1. update edge-level
        2. update node-level
        3. update global-level
        and return the updated graph
        
        Args:
            graph: DGLGraph
                It has [h_v, h_x] as ndata and edata, respectively.
        """

        #### Note that the following code explicitly allocates messages 
        #### and update edge features.
        graph = self.edge_block(graph, global_attr, out_edge_key)    # edge update
        graph = self.node_block(graph, global_attr, out_node_key)    # node update
                
        if self.global_block is not None:
            out_global_attr = self.global_block(graph, global_attr)    # Global-level update
        else:
            out_global_attr = global_attr

        return (graph, out_global_attr)