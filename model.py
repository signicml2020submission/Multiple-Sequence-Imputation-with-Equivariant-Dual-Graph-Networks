from collections import OrderedDict 

import torch
import torch.nn as nn
import torch.optim as optim

import dgl
import dgl.function as fn

from blocks import EdgeBlock, NodeBlock, GlobalBlock
from modules import GNLayer


class GRU_update(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU_update, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.grucell = nn.GRUCell(input_size, hidden_size)    # h_ij
        
    def forward(self, in_features, hidden_features):
        hx = self.grucell(in_features, hidden_features)
        return hx
    

class RGN(nn.Module):
    def __init__(self,
                 input_edge_dim,
                 input_node_dim,
                 rgn_global_dim,
                 rgn_hidden_dim,
                 num_spatial_hops=1,
                 device='cpu'
                 ):
        super(RGN, self).__init__()
        self.input_edge_dim = input_edge_dim
        self.input_node_dim = input_node_dim
        self.rgn_global_dim = rgn_global_dim
        self.rgn_hidden_dim = rgn_hidden_dim
        self.num_spatial_hops = num_spatial_hops
        self.device = device
        
        
        rgn_layers = OrderedDict()
        for l in range(self.num_spatial_hops):
            if l == 0:
                #### EdgeBlock
                eb_func = GRU_update(self.input_edge_dim+self.input_node_dim+self.rgn_global_dim, 
                                     self.rgn_hidden_dim)
                eb = EdgeBlock("", "", edge_key='e', node_key='x',
                               use_edges=True, use_sender_nodes=True, use_receiver_nodes=False, use_globals=True, 
                               custom_func=eb_func, recurrent=True)
                #### NodeBlock
                nb_func = GRU_update(self.rgn_hidden_dim+self.input_node_dim+self.rgn_global_dim, 
                                     self.rgn_hidden_dim)
                nb = NodeBlock("", "", edge_key='h'+str(l)+'_e', node_key='x',
                               use_nodes=True, use_received_edges=True, use_sent_edges=False, use_globals=True, 
                               custom_func=nb_func, recurrent=True)
            else:
                #### EdgeBlock
                eb_func = GRU_update(self.rgn_hidden_dim+self.rgn_hidden_dim+self.rgn_global_dim, 
                                     self.rgn_hidden_dim)
                eb = EdgeBlock("", "", edge_key='h'+str(l-1)+'_e', node_key='h'+str(l-1)+'_v',
                               use_edges=True, use_sender_nodes=True, use_receiver_nodes=False, use_globals=True, 
                               custom_func=eb_func, recurrent=True)
                #### NodeBlock
                nb_func = GRU_update(self.rgn_hidden_dim+self.rgn_hidden_dim+self.rgn_global_dim, 
                                     self.rgn_hidden_dim)
                nb = NodeBlock("", "", edge_key='h'+str(l)+'_e', node_key='h'+str(l-1)+'_v',
                               use_nodes=True, use_received_edges=True, use_sent_edges=False, use_globals=True, 
                               custom_func=nb_func, recurrent=True)

            #### GlobalBlock
            gb_func = nn.Sequential(nn.Linear(self.rgn_hidden_dim+self.rgn_hidden_dim+self.rgn_global_dim,
                                              self.rgn_global_dim), 
                                    nn.ReLU())
            gb = GlobalBlock("", "", edge_key='h'+str(l)+'_e', node_key='h'+str(l)+'_v', 
                             edges_reducer='mean', nodes_reducer='mean', custom_func=gb_func)
            #### GNLayer
            rgn_layers['gnl'+str(l)] = GNLayer(eb, nb, gb)

        self.rgn_layers = nn.Sequential(rgn_layers)    # stacking
        self.num_layers = len(self.rgn_layers)
        
        
    def forward(self, graph, global_attr):
            
        for l, gnl in enumerate(self.rgn_layers):
            if not 'h'+str(l)+'_e' in graph.edata:
                graph.edata['h'+str(l)+'_e'] = torch.zeros(graph.number_of_edges(), self.rgn_hidden_dim, device=self.device)
            if not 'h'+str(l)+'_v' in graph.ndata:
                graph.ndata['h'+str(l)+'_v'] = torch.zeros(graph.number_of_nodes(), self.rgn_hidden_dim, device=self.device)
                
            graph, global_attr = gnl(graph, global_attr, out_edge_key='h'+str(l)+'_e', out_node_key='h'+str(l)+'_v')

        return graph, global_attr


class SGN(nn.Module):
    def __init__(self,
                 input_edge_dim,
                 input_node_dim,
                 sgn_global_dim,
                 sgn_hidden_dim,
                 rgn_hidden_dim,
                 output_node_dim
                 ):
        super(SGN, self).__init__()
        self.input_edge_dim = input_edge_dim
        self.input_node_dim = input_node_dim
        self.sgn_global_dim = sgn_global_dim
        self.sgn_hidden_dim = sgn_hidden_dim
        self.rgn_hidden_dim = rgn_hidden_dim
        self.output_node_dim = output_node_dim
        
        #### EdgeBlock
        eb_func = nn.Sequential(nn.Linear(self.input_edge_dim+self.rgn_hidden_dim+
                                          self.input_node_dim+self.rgn_hidden_dim+
                                          self.sgn_global_dim,
                                          self.sgn_hidden_dim), 
                                nn.ReLU())
        eb = EdgeBlock("", "", edge_key='cat_e', node_key='cat_x',
                       use_edges=True, use_sender_nodes=True, use_receiver_nodes=False, use_globals=True, 
                       custom_func=eb_func)
        #### NodeBlock
        nb_func = nn.Sequential(nn.Linear(self.sgn_hidden_dim+
                                          self.input_node_dim+self.rgn_hidden_dim+
                                          self.sgn_global_dim,
                                          self.sgn_hidden_dim), 
                                nn.ReLU())
        nb = NodeBlock("", "", edge_key='h_e', node_key='cat_x', 
                       use_nodes=True, use_received_edges=True, use_sent_edges=False, use_globals=True, 
                       custom_func=nb_func)
        #### GlobalBlock
        gb_func = nn.Sequential(nn.Linear(self.sgn_hidden_dim+self.sgn_hidden_dim+self.sgn_global_dim,
                                          self.sgn_global_dim),
                                nn.ReLU())
        gb = GlobalBlock("", "", edge_key='h_e', node_key='h_v', 
                         edges_reducer='mean', nodes_reducer='mean', custom_func=gb_func)
        #### GNLayer
        self.gnl = GNLayer(eb, nb, gb)

        #### Additional NodeBlock
        nb_func2 = nn.Sequential(nn.Linear(self.sgn_hidden_dim+self.sgn_hidden_dim+self.sgn_global_dim,
                                           self.sgn_hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.sgn_hidden_dim, self.output_node_dim))
        nb2 = NodeBlock("", "", edge_key='h_e', node_key='h_v',
                        use_nodes=True, use_received_edges=True, use_sent_edges=False, use_globals=True, 
                        custom_func=nb_func2)
        self.nb2 = nb2
        
        
    def forward(self, graph, global_attr):
        
        graph, global_attr = self.gnl(graph, global_attr, out_edge_key='h_e', out_node_key='h_v')
        graph = self.nb2(graph, global_attr, out_node_key='h_v')
        
        return graph, global_attr



#### Merge RGN + SGN
class SNATA(nn.Module):
    def __init__(self, 
                 rgn_model,
                 sgn_model,
                 edge_embedding=None,
                 device='cpu'):
        super(SNATA, self).__init__()
        
        self.rgn = rgn_model
        self.sgn = sgn_model
        self.edge_embedding = edge_embedding
        self.device = device
        
        # Global parameter
        # self.global_attr = nn.init.xavier_normal_(torch.empty(1, self.rgn.rgn_global_dim))
        # self.global_attr = self.global_attr.squeeze()
        self.global_attr = None
        
    def forward(self, rgn_graph_input_batch_list, graph_input_batch):
        num_steps = len(rgn_graph_input_batch_list)
        # global_attr = self.global_attr
        # if self.global_attr is None:
        #     self.global_attr = nn.init.xavier_normal_(torch.empty(rgn_graph_input_batch_list[0].batch_size, self.rgn.rgn_global_dim, device=self.device))
        # global_attr = self.global_attr
        global_attr = nn.init.xavier_normal_(torch.empty(rgn_graph_input_batch_list[0].batch_size, self.rgn.rgn_global_dim, device=self.device))
        # time-step update
        for ii, rgn_graph_input_batch in enumerate(rgn_graph_input_batch_list):
            if ii == 0:
                for l in range(self.rgn.num_layers):
                    # initiate state vectors
                    rgn_graph_input_batch.edata['h'+str(l)+'_e'] = torch.zeros(rgn_graph_input_batch.number_of_edges(), self.rgn.rgn_hidden_dim,
                                                               device=self.device)
                    rgn_graph_input_batch.ndata['h'+str(l)+'_v'] = torch.zeros(rgn_graph_input_batch.number_of_nodes(), self.rgn.rgn_hidden_dim,
                                                               device=self.device)
            else:
                for l in range(self.rgn.num_layers):
                    # initiate state vectors
                    rgn_graph_input_batch.edata['h'+str(l)+'_e'] = out_graph_batch.edata['h'+str(l)+'_e']
                    rgn_graph_input_batch.ndata['h'+str(l)+'_v'] = out_graph_batch.ndata['h'+str(l)+'_v']
            
            # RGN
            if self.edge_embedding:
                rgn_graph_input_batch.edata['e'] = self.edge_embedding(graph_input_batch.edata['etype'])
            out_graph_batch, global_attr = self.rgn(rgn_graph_input_batch, global_attr)
            
        # Edge embedding
        if self.edge_embedding:
            graph_input_batch.edata['e'] = self.edge_embedding(graph_input_batch.edata['etype'])
            
        # Concatenate
        if num_steps == 0:
            graph_input_batch.edata['cat_e'] = graph_input_batch.edata['e']
            graph_input_batch.ndata['cat_x'] = graph_input_batch.ndata['x']
        else:
            graph_input_batch.edata['cat_e'] = torch.cat([graph_input_batch.edata['e'], 
                                                    out_graph_batch.edata['h'+str(self.rgn.num_layers-1)+'_e']], 
                                                   dim=-1)
            graph_input_batch.ndata['cat_x'] = torch.cat([graph_input_batch.ndata['x'], 
                                                    out_graph_batch.ndata['h'+str(self.rgn.num_layers-1)+'_v']], 
                                                   dim=-1)

        # SGN
        out_graph_batch, global_attr = self.sgn(graph_input_batch, global_attr)
        
        return out_graph_batch, global_attr
        

class RGN_AE(nn.Module):
    def __init__(self, 
                 rgn_model, 
                 ae_model, 
                 device='cpu'):
        super(RGN_AE, self).__init__()
        
        self.rgn = rgn_model
        self.ae = ae_model
        self.device = device
        
        # Global parameter
        self.global_attr = nn.init.xavier_normal_(torch.empty(1, self.rgn.rgn_global_dim))
        self.global_attr = self.global_attr.squeeze()
        
    def forward(self, seq_graphs, graph_input):
        num_steps = len(seq_graphs)
        global_attr = self.global_attr
        # time-step update
        for ii, graph in enumerate(seq_graphs):
            if ii == 0:
                for l in range(self.rgn.num_layers):
                    # initiate state vectors
                    graph.edata['h'+str(l)+'_e'] = torch.zeros(graph.number_of_edges(), self.rgn.rgn_hidden_dim,
                                                               device=self.device)
                    graph.ndata['h'+str(l)+'_v'] = torch.zeros(graph.number_of_nodes(), self.rgn.rgn_hidden_dim,
                                                               device=self.device)
            else:
                for l in range(self.rgn.num_layers):
                    # initiate state vectors
                    graph.edata['h'+str(l)+'_e'] = out_graph.edata['h'+str(l)+'_e']
                    graph.ndata['h'+str(l)+'_v'] = out_graph.ndata['h'+str(l)+'_v']
            
            # RGN
            out_graph, global_attr = self.rgn(graph, global_attr)
            
        # Concatenate
        if num_steps == 0:
            graph_input.ndata['cat_x'] = graph_input.ndata['x']
        else:
            graph_input.ndata['cat_x'] = torch.cat([graph_input.ndata['x'], 
                                                    out_graph.ndata['h'+str(self.rgn.num_layers-1)+'_v']], 
                                                   dim=-1)
        
        # MLP(AE)
        out = self.ae(graph_input.ndata['cat_x'].reshape(-1))
        
        return out, global_attr
