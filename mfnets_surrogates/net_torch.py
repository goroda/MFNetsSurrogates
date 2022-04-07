"""MFnets using pytorch."""
import itertools

import torch
import torch.utils
import torch.nn as nn
import networkx as nx
try:
    import queue.SimpleQueue as SimpleQueue
except ImportError:
    from queue import Queue as SimpleQueue

import matplotlib.pyplot as plt

# dtype = torch.float
device = torch.device("cpu")


class ArrayDataset(torch.utils.data.Dataset):
    """Dataset from an array of data."""
    
    def __init__(self, x, y):
        """Initialize. Rows of x are different data points."""
        self.x = x
        self.y = y

    def __getitem__(self, key):
        """Get a row."""
        assert isinstance(key, int)
        return self.x[key, :], self.y[key]

    def __len__(self):
        """Get number of data points."""
        return self.x.size(dim=0)

def make_graph_2():
    """Make a graph with two nodes.

    1 -> 2
    """
    graph = nx.DiGraph()

    # pnodes = torch.randn((2, 2), device=device, dtype=dtype)
    # pedges = torch.randn((1, 2), device=device, dtype=dtype)

    dinput = 1
    graph.add_node(1, func=torch.nn.Linear(dinput, 1, bias=True))
    graph.add_node(2, func=torch.nn.Linear(dinput, 1, bias=True))
    graph.add_edge(1, 2, func=torch.nn.Linear(dinput, 1, bias=True))
    return graph, set([1])


class MFNetTorch(nn.Module):
    """Multifidelity Network."""

    def __init__(self, graph, roots):
        """Initialize a multifidelity surrogate via a graph and its roots.

        Parameters
        ----------
        graph : networkx.graph
            The graphical representation of the MF network

        roots : list
            The ids of every root node in the network

        """
        super(MFNetTorch, self).__init__()
        self.graph = graph
        self.roots = roots
        self.target_node = None
        self.modules_list = nn.ModuleList()

        for node, f in self.graph.nodes.data('func'):
            self.modules_list.add_module(f'node{node}', f)

        for from_n, to_n, f in self.graph.edges.data('func'):
            self.modules_list.add_module(f'edge{from_n}->{to_n}', f)            
            
        # self.modules_list = nn.ModuleList(
        #     [(f'node{ii}', f) for ii, (_, f) in enumerate(self.graph.nodes.data('func'))])

        # self.modules_list.extend([f for _,_,f in self.graph.edges.data('func')])

    def set_target_node(self, target_node):
        """Set the target node for learning."""
        self.target_node = target_node

    def zero_attributes(self):
        """Zero all attributes except 'func' and 'param'."""
        atts = ['eval', 'parents_left']
        for att in atts:
            for node in self.graph.nodes:
                try:
                    self.graph.nodes[node][att] = 0.0
                except: # what exception is it?
                    continue
            for edge in self.graph.edges:
                try:
                    self.graph.edges[edge][att] = 0.0
                except: # what exception is it?
                    continue

    def forward(self, xinput):
        """Evaluate the surrogate output at target_node.

        Parameters
        ----------
        xinput : np.ndarray (nsamples,nparams)
            The independent variables of the model

        target_node : integer
            The id of the node under consideration

        Returns
        -------
        This function adds the following attributes to the underlying graph

        eval :
            stores the evaluation of the function represented by the
            particular node / edge the evaluations at the nodes are
            cumulative (summing up all ancestors) whereas the edges are local

        parents-left : set
            internal attribute needed for accounting

        """
        self.zero_attributes()
        assert self.target_node is not None
        anc = nx.ancestors(self.graph, self.target_node)
        anc_and_target = anc.union(set([self.target_node]))
        relevant_root_nodes = anc.intersection(self.roots)

        # Evaluate the target nodes and all ancestral nodes and put the root
        # nodes in a queue
        queue = SimpleQueue()
        for node in anc_and_target:
            pval = self.graph.nodes[node]['func'](xinput)
            self.graph.nodes[node]['eval'] = pval
            self.graph.nodes[node]['parents_left'] = \
                set(self.graph.predecessors(node))

            if node in relevant_root_nodes:

                queue.put(node)

        while not queue.empty():

            node = queue.get()
            feval = self.graph.nodes[node]['eval']
            for child in self.graph.successors(node):
                if child in anc_and_target:
                    pval = self.graph.edges[node, child]['func'](xinput)

                    self.graph.nodes[child]['eval'] += feval * pval
                    self.graph.edges[node, child]['eval'] = pval

                    self.graph.nodes[child]['parents_left'].remove(node)

                    if self.graph.nodes[child]['parents_left'] == set():
                        queue.put(child)

        return self.graph.nodes[node]['eval']

    def eval_loss(self, data, targets, loss_fns):
        """Evaluate loss function."""
        loss = 0
        for target, dat, loss_fn in zip(targets, data, loss_fns):
            self.set_target_node(target)
            self.zero_attributes()        
            for batch, (X, y) in enumerate(dat):
                with torch.nn.utils.parametrize.cached():
                    pred = self(X).flatten()
                new_loss = loss_fn(pred, y)
                loss += new_loss
            
        return loss

    def train(self, data, targets, loss_fns):
        """Train the model."""
        optimizer = torch.optim.LBFGS(self.parameters(),
                                      max_iter=100,
                                      tolerance_grad=1e-15,
                                      tolerance_change=1e-15,
                                      history_size=500,
                                      line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad()
            self.zero_attributes()
            loss = self.eval_loss(data, targets, loss_fns)
            loss.backward(retain_graph=True)
            return loss

        self.zero_grad()

        optimizer.step(closure)


        
def generate_data(model, ndata):
    """Generate data."""
    ndata_lf, ndata_hf = ndata
    xlf = torch.rand(ndata_lf, 1) * 6 - 3
    xhf = torch.rand(ndata_hf, 1) * 6 - 3
    with torch.no_grad():
        model.set_target_node(1)
        ylf = model(xlf).flatten()
        model.set_target_node(2)
        yhf = model(xhf).flatten()

    d1 = ArrayDataset(xlf, ylf)
    d2 = ArrayDataset(xhf, yhf)
    data = (d1, d2)
    return data

def construct_loss_funcs(model):
    """Create loss functions."""
    num_nodes = len(model.graph.nodes)
    return [torch.nn.MSELoss() for _ in range(num_nodes)]
            
def plot_funcs(model, x, data=None):
    """Plot the univariate functions."""
    with torch.no_grad():
        model.set_target_node(1)
        ylf = model(x)
        model.set_target_node(2)
        yhf = model(x)
        
    plt.figure()
    plt.plot(x, ylf, label='low-fidelity')
    plt.plot(x, yhf, label='high-fidelity')

    if data != None:
        plt.plot(data[0].x, data[0].y, 'ro', label='Low-fidelity Data')
        plt.plot(data[1].x, data[1].y, 'ko', label='High-fidelity Data')
        
    plt.legend()
    

if __name__ == "__main__":

    torch.manual_seed(1)
    graph, root = make_graph_2()

    model = MFNetTorch(graph, root)

    # for param in model.named_parameters():
    #     print(param)
    
    x = torch.linspace(-3, 3, 10).expand(1,10).transpose(0,1)
    plot_funcs(model, x)

    loss_fns = construct_loss_funcs(model)
    data = generate_data(model, [20, 20])
    data_loaders = [torch.utils.data.DataLoader(d, batch_size=len(d), shuffle=False)
               for d in data]


    # loss = eval_loss(model, data_loaders, [1, 2], loss_fns)
    # print("loss = ", loss)


    graph2, root2 = make_graph_2()
    model_trained = MFNetTorch(graph2, root2)
    plot_funcs(model_trained, x)

    model_trained.train(data_loaders, [1, 2], loss_fns)

    plot_funcs(model_trained, x, data)

    with torch.no_grad():
        loss2 = model_trained.eval_loss(data_loaders, [1, 2], loss_fns)
        print("loss2 = ", loss2)
        print(model_trained)
        print("Trained parameters")
        print(torch.nn.utils.parameters_to_vector(model_trained.parameters()))
        print("True parameters")
        print(torch.nn.utils.parameters_to_vector(model.parameters()))
        
    
    
    plt.show()

