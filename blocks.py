import torch
import torch.nn as nn

import dgl.function as fn
import dgl


class GlobalBlock(nn.Module):
    """Global block, f_g.
    
    A block that updates the global features of each graph based on
    the previous global features, the aggregated features of the
    edges of the graph, and the aggregated features of the nodes of the graph.
    """
    
    def __init__(self,
                 in_feats,
                 out_feats,
                 edge_key='e',
                 node_key='x',
                 edges_reducer="sum",
                 nodes_reducer="sum",
                 custom_func=None,
                 init=True):
        
        super(GlobalBlock, self).__init__()
        
        self.edge_key = edge_key
        self.node_key = node_key
        self.edges_reducer = edges_reducer
        self.nodes_reducer = nodes_reducer
        
        # f_v() is a function: R^in_features -> R^out_features
        if custom_func:
            # Customized function can be used for self.net instead of deafult function.
            # It is highly recommended to use nn.Sequential() type.
            self.net = custom_func
        else:
            self.net = nn.Linear(in_feats, out_feats)
            
        # initialization
        if init:
            for m in self.net.modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
            
    def forward(self, graph, global_attr):
        # agg_edge_attr = getattr(torch, self.edges_reducer)(graph.edata[self.edge_key], dim=0)
        # agg_node_attr = getattr(torch, self.nodes_reducer)(graph.ndata[self.node_key], dim=0)
        agg_edge_attr = getattr(dgl, '{}_edges'.format(self.edges_reducer))(graph, self.edge_key)
        agg_node_attr = getattr(dgl, '{}_nodes'.format(self.nodes_reducer))(graph, self.node_key)

        return self.net(torch.cat([agg_edge_attr, agg_node_attr, global_attr], dim=-1))
    
    
class EdgeBlock(nn.Module):
    """Edge block, f_e.
    Update the features of each edge based on the previous edge features,
    the features of the adjacent nodes, and the global features.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 edge_key='e',
                 node_key='x',
                 use_edges=True,
                 use_sender_nodes=True,
                 use_receiver_nodes=True,
                 use_globals=False,
                 custom_func=None,
                 recurrent=False,
                 init=True):
        
        super(EdgeBlock, self).__init__()
        
        if not (use_edges or use_sender_nodes or use_receiver_nodes or use_globals):
            raise ValueError("At least one of use_edges, use_sender_nodes, "
                             "use_receiver_nodes or use_globals must be True.")

        self.edge_key = edge_key
        self.node_key = node_key
        self._use_edges = use_edges
        self._use_sender_nodes = use_sender_nodes
        self._use_receiver_nodes = use_receiver_nodes
        self._use_globals = use_globals
        self.recurrent = recurrent
    
        # f_e() is a function: R^in_feats -> R^out_feats
        if custom_func:
            # Customized function can be used for self.net instead of deafult function.
            # It is highly recommended to use nn.Sequential() type.
            self.net = custom_func
        else:
            self.net = nn.Linear(in_feats, out_feats)
        
        # initialization
        if init:
            for m in self.net.modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
        
    def forward(self, graph, global_attr=None, out_edge_key='h_e'):
        
        def send_func(edges):
            edges_to_collect = []
            num_edges = edges.data[self.edge_key].shape[0]
            if self._use_edges:
                edges_to_collect.append(edges.data[self.edge_key])
            if self._use_sender_nodes:
                edges_to_collect.append(edges.src[self.node_key])
            if self._use_receiver_nodes:
                edges_to_collect.append(edges.dst[self.node_key])
            if self._use_globals and global_attr is not None:
                # self._global_attr = global_attr.unsqueeze(0)    # make global_attr.shape = (1, DIM)
                # expanded_global_attr = self._global_attr.expand(num_edges, self._global_attr.shape[1])
                expanded_global_attr = edges.data['expanded_global_attr']
                edges_to_collect.append(expanded_global_attr)

            collected_edges = torch.cat(edges_to_collect, dim=-1)
            
            if self.recurrent:
                return {out_edge_key: self.net(collected_edges, edges.data[out_edge_key])}
            else:
                return {out_edge_key: self.net(collected_edges)}
        
        graph.edata['expanded_global_attr'] = dgl.broadcast_edges(graph ,global_attr)
        graph.apply_edges(send_func)
        
        return graph

        
class NodeBlock(nn.Module):
    """Node block, f_v.
    Update the features of each node based on the previous node features,
    the aggregated features of the received edges,
    the aggregated features of the sent edges, and the global features.
    """
    
    def __init__(self,
                 in_feats,
                 out_feats,
                 edge_key='e',
                 node_key='x',
                 use_nodes=True,
                 use_sent_edges=False,
                 use_received_edges=True,
                 use_globals=False,
                 sent_edges_reducer="sum",
                 received_edges_reducer="sum",
                 custom_func=None,
                 recurrent=False,
                 init=True):
        """Initialization of the NodeBlock module.
        
        Args:
            in_features: Input dimension.
                If node, 2*edge(sent, received), and global are used, d_v+(2*d_e)+d_g.
                h'_i = f_v(h_i, AGG(h_ij), AGG(h_ji), u)
            out_features: Output dimension.
                h'_i will have the dimension.
            use_nodes: Whether to condition on node attributes.
            use_sent_edges: Whether to condition on sent edges attributes.
            use_received_edges: Whether to condition on received edges attributes.
            use_globals: Whether to condition on the global attributes.
            reducer: Aggregator. [sum, max, min, prod, mean]
        """
        
        super(NodeBlock, self).__init__()

        if not (use_nodes or use_sent_edges or use_received_edges or use_globals):
            raise ValueError("At least one of use_received_edges, use_sent_edges, "
                             "use_nodes or use_globals must be True.")
        
        self.edge_key = edge_key
        self.node_key = node_key
        self._use_nodes = use_nodes
        self._use_sent_edges = use_sent_edges
        self._use_received_edges = use_received_edges
        self._use_globals = use_globals
        self._sent_edges_reducer = sent_edges_reducer
        self._received_edges_reducer = received_edges_reducer
        self.recurrent = recurrent
        
        # f_v() is a function: R^in_features -> R^out_features
        if custom_func:
            # Customized function can be used for self.net instead of deafult function.
            # It is highly recommended to use nn.Sequential() type.
            self.net = custom_func
        else:
            self.net = nn.Linear(in_feats, out_feats)
            
        # initialization
        if init:
            for m in self.net.modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
    
    def forward(self, graph, global_attr=None, out_node_key='h_v'):
        
        def recv_func(nodes):
            nodes_to_collect = []
            num_nodes = nodes.data[self.node_key].shape[0]
            if self._use_nodes:
                nodes_to_collect.append(nodes.data[self.node_key])
#             if self._use_sent_edges:
#                 agg_edge_attr = getattr(torch, self._sent_edges_reducer)(nodes.mailbox["m"], dim=1)
#                 nodes_to_collect.append(agg_edge_attr.expand(num_nodes, agg_edge_attr.shape[1]))
            if self._use_received_edges:
                agg_edge_attr = getattr(torch, self._received_edges_reducer)(nodes.mailbox["m"], dim=1)
                nodes_to_collect.append(agg_edge_attr.expand(num_nodes, agg_edge_attr.shape[1]))
            if self._use_globals and global_attr is not None:
                # self._global_attr = global_attr.unsqueeze(0)    # make global_attr.shape = (1, DIM)
                # expanded_global_attr = self._global_attr.expand(num_nodes, self._global_attr.shape[1])
                expanded_global_attr = nodes.data['expanded_global_attr']
                nodes_to_collect.append(expanded_global_attr)
                
            collected_nodes = torch.cat(nodes_to_collect, dim=-1)
        
            if self.recurrent:
                return {out_node_key: self.net(collected_nodes, nodes.data[out_node_key])}
            else:
                return {out_node_key: self.net(collected_nodes)}
            
        graph.ndata['expanded_global_attr'] = dgl.broadcast_nodes(graph, global_attr)
        if self._use_received_edges:
            graph.update_all(fn.copy_e(self.edge_key, "m"), recv_func)    # trick
        else:
            graph.apply_nodes(recv_func)
                    
        return graph

    
class NodeBlockInd(nn.Module):
    """Node-level feature transformation.
    Each node is considered independently. (No edge is considered.)
    
    Args:
        in_features: input dimension of node representations.
        out_features: output dimension of node representations.
            (node embedding size)
            
    (N^v, d_v) -> (N^v, out_features)
    NodeBlockInd(graph) -> updated graph
    """
    
    def __init__(self,
                 in_feats,
                 out_feats,
                 custom_func=None,
                 init=True):
        
        super(NodeBlockInd, self).__init__()

        # Customized function
        if custom_func:
            # Customized function can be used for self.net instead of deafult function.
            # It is highly recommended to use nn.Sequential() type.
            self.net = custom_func
        else:
            self.net = nn.Linear(in_feats, out_feats)
            
        # initialization
        if init:
            for m in self.net.modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
    
    def forward(self, graph, node_key='h_v'):
        def recv_func(nodes): return {'h_v': self.net(nodes.data[node_key])}
        
        graph.apply_nodes(recv_func)
        
        return graph
        
        
class EdgeBlockInd(nn.Module):
    """Edge-level feature transformation.
    Each edge is considered independently. (No node is considered.)
    
    Args:
        in_features: input dimension of edge representations.
        out_features: output dimension of edge representations.
            (edge embedding size)
    
    (N^e, d_e) -> (N^e, out_features)
    EdgeBlockInd(graph) -> updated graph
    """
    
    def __init__(self,
                 in_feats,
                 out_feats,
                 custom_func=None,
                 init=True):
        
        super(EdgeBlockInd, self).__init__()
        
        # Customized function
        if custom_func:
            # Customized function can be used for self.net instead of deafult function.
            # It is highly recommended to use nn.Sequential() type.
            self.net = custom_func
        else:
            self.net = nn.Linear(in_feats,out_feats)
            
        # initialization
        if init:
            for m in self.net.modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()

    def forward(self, graph, edge_key='h_e'):
        def send_func(edges): return {'h_e': self.net(edges.data[edge_key])}
        
        graph.apply_edges(send_func)
        
        return graph
        