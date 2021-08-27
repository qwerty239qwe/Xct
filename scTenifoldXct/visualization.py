import igraph as ig
import numpy as np
import pandas as pd
import random
import warnings

def get_Xct_pairs(df):
    return tuple(n.split('_') for n in list(df.index))

visual_style_common = {}
visual_style_common["vertex_size"] = 50
visual_style_common["vertex_label_size"] = 12
visual_style_common["vertex_label_dist"] = 0.0
visual_style_common["edge_curved"] = 0.1
# visual_style_common["edge_arrow_size"] = 1
# visual_style_common["edge_arrow_width"] = 0.3
visual_style_common["margin"] = 50

def plot_pcNet(Xct_obj, view, gene_names, top_edges = 20, show = True, saveas = None, verbose = False, edge_width_scale = None, random_state = 0,
            visual_style = visual_style_common.copy()):
    '''visualize single cell type GRN, only showing direct edges associated with target genes'''
    if view == 'sender':
        net = Xct_obj._net_A
    elif view == 'receiver':
        net = Xct_obj._net_B
    else:
        raise KeyError('view \'sender\' or \'receiver\'')

    if not isinstance(net, pd.DataFrame):
        raise TypeError('convert GRN to dataframe with gene names first')
    else:
        g_to_use = Xct_obj.TFs['TF_symbol'].tolist()
        g_to_use_orig = g_to_use.copy()
        g_to_use = g_to_use + gene_names
      
        net_f = net.loc[net.index.isin(g_to_use), net.columns.isin(g_to_use)].copy() #subset net by TFs and LRs
        net_f.loc[net_f.index.isin(g_to_use_orig), net_f.columns.isin(g_to_use_orig)] = 0 # set !LR row and col = 0, as not interested
        net_f.astype('float64')
        if verbose:
            print(f'identified {len(net_f)} TF(s) along with {len(gene_names)} ligand/receptor target gene(s)')
        
        idx_target = [list(net_f.index).index(g) for g in gene_names]
        is_TF = np.ones(len(net_f), dtype=bool)
        for idx in idx_target:
            is_TF[idx] = False 
                
        mat = net_f.values.copy()
        gene_names_f = net_f.index.tolist()
        del net_f
        g = ig.Graph.Weighted_Adjacency(mat, mode='undirected', attr="weight", loops=True)
        g.vs["name"] = gene_names_f
        g.vs["is_TF"] = is_TF
        
        if len(g.es) == 0:
            gene_names = ' '.join(map(str, gene_names)) #string
            raise ValueError(f'target gene {gene_names} generated 0 edge, stoped...')
        #trim low weight edges
        abs_weight = [abs(w) for w in g.es['weight']]
        edges_delete_ids = sorted(range(len(abs_weight)), key=lambda k: abs_weight[k])[:-top_edges] #idx from bottom
        g.delete_edges(edges_delete_ids)

        #delete isolated nodes
        to_delete_ids = []
        for v in g.vs: 
            if v.degree() == 0:
                to_delete_ids.append(v.index) 
                if v['name'] in gene_names:
                    warnings.warn(f"{v['name']} has been removed due to degree equals to zero among top weighted edges")
        g.delete_vertices(to_delete_ids)

        if verbose:   
            print(f'undirected graph constructed: \n# of nodes: {len(g.vs)}, # of edges: {len(g.es)}\n')
         
        #graph-specific visual_style after graph building
        visual_style["bbox"] = (512, 512)
        visual_style["vertex_label"] = g.vs["name"]
        visual_style["vertex_color"] = ['darkgray' if tf==1 else 'darkorange' for tf in g.vs["is_TF"]]
        visual_style["vertex_shape"] = ['circle' if tf==1 else 'square' for tf in g.vs["is_TF"]]
        if edge_width_scale is None:
            edge_width_scale = 3/max(np.abs(g.es['weight']))
        visual_style["edge_width"] = [edge_width_scale*abs(w) for w in g.es['weight']] 
        visual_style["edge_color"] = ['red' if w>0 else 'blue' for w in g.es['weight']]
        visual_style["layout"] = 'large'
        
        random.seed(random_state) #layout
        if show:
            if saveas is None:
                return ig.plot(g, **visual_style)
            else:
                if verbose:
                    print(f'graph saved as \"{saveas}.png\"')
                return ig.plot(g, f'{saveas}.png', **visual_style)
        else:
            return g


def plot_XNet(g1, g2, Xct_pair = None, saveas = None, verbose = False, edge_width_scale = None, edge_width_max = 5, random_state = 0,
            visual_style = visual_style_common.copy()):
    '''visualize merged GRN from sender and receiver cell types,
        use edge_width_scale to make two graphs width comparable (both using absolute values)'''
    gg = g1.disjoint_union(g2) #merge disjointly
    if verbose:   
            print(f'graphs merged: \n# of nodes: {len(gg.vs)}, # of edges: {len(gg.es)}\n')

    for pair in Xct_pair:
        edges_idx = (gg.vs.find(name = pair[0]).index, gg.vs.find(name = pair[1]).index) #from to
        gg.add_edge(edges_idx[0], edges_idx[1], weight = 1.02) #weight > 1 and be the max
        if verbose:
            print(f'edge from {pair[0]} to {pair[1]} added')
    
    visual_style["bbox"] = (768, 768)
    visual_style["vertex_label"] = gg.vs["name"]
    visual_style["vertex_color"] = ['darkgray' if tf==1 else 'darkorange' for tf in gg.vs["is_TF"]]
    visual_style["vertex_shape"] = ['circle' if tf==1 else 'square' for tf in gg.vs["is_TF"]]

    if edge_width_scale is None:
        edge_width_scale = 3/max(np.abs(gg.es['weight']))
    visual_style["edge_width"] = [edge_width_scale*abs(w) if edge_width_scale*abs(w) < edge_width_max else edge_width_max for w in gg.es['weight']] 
    visual_style["edge_color"] = ['red' if (w>0)&(w<=1) else ('maroon' if w>1 else 'blue') for w in gg.es['weight']]
    visual_style["layout"] = 'kk'
    visual_style["mark_groups"] = [(list(range(0, len(g1.vs))), "whitesmoke")] + [(list(range(len(g1.vs), len(g1.vs)+len(g2.vs))), "whitesmoke")]
 
    random.seed(random_state) #layout
    if saveas is None:
        return ig.plot(gg, **visual_style)
    else:
        if verbose:
            print(f'graph saved as \"{saveas}.png\"')
        return ig.plot(gg, f'{saveas}.png', **visual_style)