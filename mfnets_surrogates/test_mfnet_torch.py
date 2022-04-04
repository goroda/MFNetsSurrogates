import unittest
from net_torch import *
import numpy as np
import torch

def make_graph_8():
    """A graph with 8 nodes.

    3 -> 7 -> 8
              ^
              |
         1 -> 4
            / ^
           /  |
    2 -> 5 -> 6
    """

    graph = nx.DiGraph()

    dinput = 1
    for node in range(1, 9):
        graph.add_node(node, func=torch.nn.Linear(dinput, 1, bias=True))

    graph.add_edge(1, 4, func=torch.nn.Linear(dinput, 1, bias=True))
    graph.add_edge(2, 5, func=torch.nn.Linear(dinput, 1, bias=True))
    graph.add_edge(5, 6, func=torch.nn.Linear(dinput, 1, bias=True))
    graph.add_edge(6, 4, func=torch.nn.Linear(dinput, 1, bias=True))
    graph.add_edge(3, 7, func=torch.nn.Linear(dinput, 1, bias=True))
    graph.add_edge(7, 8, func=torch.nn.Linear(dinput, 1, bias=True))
    graph.add_edge(4, 8, func=torch.nn.Linear(dinput, 1, bias=True))
    graph.add_edge(5, 4, func=torch.nn.Linear(dinput, 1, bias=True))

    roots = set([1, 2, 3])
    return graph, roots


class TestMfnet(unittest.TestCase):

    def test_least_squares_opt(self):
        torch.manual_seed(2)

        graph, roots = make_graph_8()

        node = 8

        ## Truth
        mfsurr_true = MFNetTorch(graph, roots)

        dx = 1
        ndata = [0] * 8
        ndata[7] = 500
        x = torch.rand(ndata[7], 1)
        mfsurr_true.set_target_node(node)
        y =  mfsurr_true.forward(x).flatten()
        std = 1.#1e-4

        graph_learn, roots_learn = make_graph_8()        
        mfsurr_learn = MFNetTorch(graph_learn, roots_learn)

        loss_fns = construct_loss_funcs(mfsurr_learn)    
        dataset = ArrayDataset(x, y)
        data_loaders = [torch.utils.data.DataLoader(dataset,
                                                    batch_size=ndata[7],
                                                    shuffle=False)]
        # print(mfsurr_learn)
        mfsurr_learn.train(data_loaders, [8], loss_fns[7:])

        with torch.no_grad():
            mfsurr_learn.set_target_node(8)
            predict = mfsurr_learn(x).flatten()
            err = torch.linalg.norm(predict-y)**2/2
            print("err = ", err)
            assert err<1e-4

        ntest=1000
        x_test = torch.rand(ntest, dx)
        with torch.no_grad():
            y_test =  mfsurr_true.forward(x_test).flatten()
            predict_test = mfsurr_learn.forward(x_test).flatten()
            err = torch.linalg.norm(predict_test-y_test)/np.sqrt(ntest)
            print("err = ", err)
            assert err<1e-3
        
if __name__== "__main__":    
    mfnet_test_suite = unittest.TestLoader().loadTestsFromTestCase(TestMfnet)
    unittest.TextTestRunner(verbosity=2).run(mfnet_test_suite)
