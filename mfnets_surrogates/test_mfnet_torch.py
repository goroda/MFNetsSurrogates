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
        graph.add_node(node, func=torch.nn.Linear(dinput, 1, bias=True), dim_in=1, dim_out=1)

    graph.add_edge(1, 4, func=torch.nn.Linear(dinput, 1, bias=True), out_rows=1, out_cols=1, dim_in=1)
    graph.add_edge(2, 5, func=torch.nn.Linear(dinput, 1, bias=True), out_rows=1, out_cols=1, dim_in=1)
    graph.add_edge(5, 6, func=torch.nn.Linear(dinput, 1, bias=True), out_rows=1, out_cols=1, dim_in=1)
    graph.add_edge(6, 4, func=torch.nn.Linear(dinput, 1, bias=True), out_rows=1, out_cols=1, dim_in=1)
    graph.add_edge(3, 7, func=torch.nn.Linear(dinput, 1, bias=True), out_rows=1, out_cols=1, dim_in=1)
    graph.add_edge(7, 8, func=torch.nn.Linear(dinput, 1, bias=True), out_rows=1, out_cols=1, dim_in=1)
    graph.add_edge(4, 8, func=torch.nn.Linear(dinput, 1, bias=True), out_rows=1, out_cols=1, dim_in=1)
    graph.add_edge(5, 4, func=torch.nn.Linear(dinput, 1, bias=True), out_rows=1, out_cols=1, dim_in=1)

    roots = set([1, 2, 3])
    return graph, roots


def make_graph_4():
    """A graph with 4 nodes with different output dims

    1- > 4 <- 2 <- 3
    """

    graph = nx.DiGraph()

    dim_in = 1
    dim_out = [2, 3, 6, 4]
    
    graph.add_node(1, func=torch.nn.Linear(dim_in, dim_out[0], bias=True), dim_in=dim_in, dim_out=dim_out[0])
    graph.add_node(2, func=torch.nn.Linear(dim_in, dim_out[1], bias=True), dim_in=dim_in, dim_out=dim_out[1])
    graph.add_node(3, func=torch.nn.Linear(dim_in, dim_out[2], bias=True), dim_in=dim_in, dim_out=dim_out[2])
    graph.add_node(4, func=torch.nn.Linear(dim_in, dim_out[3], bias=True), dim_in=dim_in, dim_out=dim_out[3])
    

    graph.add_edge(1, 4, func=torch.nn.Linear(dim_in, dim_out[0] * dim_out[3], bias=True), out_rows=dim_out[3], out_cols=dim_out[0], dim_in=1)
    graph.add_edge(2, 4, func=torch.nn.Linear(dim_in, dim_out[1] * dim_out[3], bias=True), out_rows=dim_out[3], out_cols=dim_out[1], dim_in=1)
    graph.add_edge(3, 2, func=torch.nn.Linear(dim_in, dim_out[2] * dim_out[1], bias=True), out_rows=dim_out[1], out_cols=dim_out[2], dim_in=1)

    roots = set([1, 4])
    return graph, roots, dim_out


class TestMfnet(unittest.TestCase):

    # @unittest.skip('testing other')
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
        y =  mfsurr_true.forward([x],[8])[0]
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

        print("\n")
        with torch.no_grad():
            predict = mfsurr_learn([x],[8])[0]
            # print(predict.size())
            err = torch.linalg.norm(predict-y)**2/2
            print("err = ", err)
            assert err<1e-4

        ntest=1000
        x_test = torch.rand(ntest, dx)
        with torch.no_grad():
            y_test =  mfsurr_true.forward([x_test],[8])[0]
            predict_test = mfsurr_learn.forward([x_test], [8])[0]
            err = torch.linalg.norm(predict_test-y_test)/np.sqrt(ntest)
            print("err = ", err)
            assert err<1e-3

    def test_least_squares_opt_multi_out(self):
        torch.manual_seed(2)

        # print("\n")
        graph, roots, dim_out = make_graph_4()

        node = 4

        ## Truth
        mfsurr_true = MFNetTorch(graph, roots)

        dx = 1
        ndata = [0] * 4
        ndata[3] = 500
        x = torch.rand(ndata[3], 1)
        # y =  mfsurr_true.forward([x]*4,[1, 2, 3, 4])
        y =  mfsurr_true.forward([x], [4])[0]
        # print("\n")
        # print("yshapes = ", [yy.size() for yy in y])
        # print("y = ", y)
        # exit(1)
        std = 1.#1e-4
        
        graph_learn, roots_learn, _ = make_graph_4()        
        mfsurr_learn = MFNetTorch(graph_learn, roots_learn)

        loss_fns = construct_loss_funcs(mfsurr_learn)    
        dataset = ArrayDataset(x, y)
        data_loaders = [torch.utils.data.DataLoader(dataset,
                                                    batch_size=ndata[3],
                                                    shuffle=False)]
        # print(mfsurr_learn)
        mfsurr_learn.train(data_loaders, [node], loss_fns[(node-1):])

        print("\n")
        with torch.no_grad():
            predict = mfsurr_learn([x],[node])[0]
            # print(dim_out)
            # print(predict.size())
            assert predict.size(dim=1) == dim_out[node-1]

            err = torch.linalg.norm(predict-y)**2/2
            print("err = ", err)
            assert err<1e-4

        ntest=1000
        x_test = torch.rand(ntest, dx)
        with torch.no_grad():
            y_test =  mfsurr_true.forward([x_test],[node])[0]
            predict_test = mfsurr_learn.forward([x_test], [node])[0]
            err = torch.linalg.norm(predict_test-y_test)/np.sqrt(ntest)
            print("err = ", err)
            assert err<1e-3            
        
if __name__== "__main__":    
    mfnet_test_suite = unittest.TestLoader().loadTestsFromTestCase(TestMfnet)
    unittest.TextTestRunner(verbosity=2).run(mfnet_test_suite)
