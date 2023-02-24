import numpy as np
import torch
import networkx as nx
import os
from mfnets_surrogates.net_torch import *
from sklearn.model_selection import train_test_split

def make_graph_4():
    """A graph with 4 nodes with different output dims

    1- > 4 <- 2 <- 3
    
    0- > 3 <- 1 <- 2
    """

    graph = nx.DiGraph()

    dim_in = 1
    # dim_out = [2, 3, 6, 4]
    # dim_out = [2, 3, 6, 4]
    dim_out = [1, 1, 1, 1]
    
    graph.add_node(1, func=torch.nn.Linear(dim_in, dim_out[0], bias=True), dim_in=dim_in, dim_out=dim_out[0])
    graph.add_node(2, func=torch.nn.Linear(dim_in, dim_out[1], bias=True), dim_in=dim_in, dim_out=dim_out[1])
    graph.add_node(3, func=torch.nn.Linear(dim_in, dim_out[2], bias=True), dim_in=dim_in, dim_out=dim_out[2])
    graph.add_node(4, func=torch.nn.Linear(dim_in, dim_out[3], bias=True), dim_in=dim_in, dim_out=dim_out[3])
    

    graph.add_edge(1, 4, func=torch.nn.Linear(dim_in, dim_out[0] * dim_out[3], bias=True), out_rows=dim_out[3], out_cols=dim_out[0], dim_in=1)
    graph.add_edge(2, 4, func=torch.nn.Linear(dim_in, dim_out[1] * dim_out[3], bias=True), out_rows=dim_out[3], out_cols=dim_out[1], dim_in=1)
    graph.add_edge(3, 2, func=torch.nn.Linear(dim_in, dim_out[2] * dim_out[1], bias=True), out_rows=dim_out[1], out_cols=dim_out[2], dim_in=1)

    roots = set([1, 3])
    return graph, roots, dim_out


def gen_data(num_data, seed=5):
    torch.manual_seed(2)

    graph, roots, dim_out = make_graph_4()    
    mfsurr_true = MFNetTorch(graph, roots)        

    ndata = [num_data, num_data, num_data, num_data]
    print(ndata)
    x = torch.rand(ndata[3], 1)
    y =  mfsurr_true.forward([x]*4, [1, 2, 3, 4])

    return x, y


def gen_and_write_data(dirname, frac_test=0.95):

    os.makedirs(dirname, exist_ok=True)  
    
    x, y = gen_data(500, seed=2)
    x = x.detach().numpy()
    y = [yy.detach().numpy() for yy in y]
    
    split_seed = 2
    
    print(x.shape)
    print(y[0].shape)

    # split into training and testing, separate for each

    # Model 1
    X_train, X_test, Y_train, Y_test = train_test_split(x, y[0],
                                                        test_size=frac_test,
                                                        random_state=split_seed)
    headery = " ".join([f"y{ii}" for ii in range(Y_train.shape[1])])
    
    xtrain_f = os.path.join(dirname, "data1_in.txt")
    ytrain_f = os.path.join(dirname, "data1_out.txt")
    xtest_f = os.path.join(dirname, "data1_test.txt")
    ytest_f = os.path.join(dirname, "data1_test_out.txt")


    np.savetxt(xtrain_f, X_train, header="x", comments='')
    np.savetxt(xtest_f, X_test, header="x",  comments='')
    np.savetxt(ytrain_f, Y_train, header=headery, comments='')
    np.savetxt(ytest_f, Y_test, header=headery, comments='')    
    
    # Model 2
    X_train, X_test, Y_train, Y_test = train_test_split(x, y[1],
                                                        test_size=frac_test,
                                                        random_state=split_seed)
    headery = " ".join([f"y{ii}" for ii in range(Y_train.shape[1])])    

    xtrain_f = os.path.join(dirname, "data2_in.txt")
    ytrain_f = os.path.join(dirname, "data2_out.txt")
    xtest_f = os.path.join(dirname, "data2_test.txt")
    ytest_f = os.path.join(dirname, "data2_test_out.txt")
    np.savetxt(xtrain_f, X_train, header="x", comments='')
    np.savetxt(xtest_f, X_test, header="x", comments='')
    np.savetxt(ytrain_f, Y_train, header=headery, comments='')
    np.savetxt(ytest_f, Y_test, header=headery, comments='')      

    # Model 3
    X_train, X_test, Y_train, Y_test = train_test_split(x, y[2],
                                                        test_size=frac_test,
                                                        random_state=split_seed)
    headery = " ".join([f"y{ii}" for ii in range(Y_train.shape[1])])    

    xtrain_f = os.path.join(dirname, "data3_in.txt")
    ytrain_f = os.path.join(dirname, "data3_out.txt")
    xtest_f = os.path.join(dirname, "data3_test.txt")
    ytest_f = os.path.join(dirname, "data3_test_out.txt")
    np.savetxt(xtrain_f, X_train, header="x", comments='')
    np.savetxt(xtest_f, X_test, header="x", comments='')
    np.savetxt(ytrain_f, Y_train, header=headery, comments='')
    np.savetxt(ytest_f, Y_test, header=headery, comments='')      

    # Model 4
    X_train, X_test, Y_train, Y_test = train_test_split(x, y[3],
                                                        test_size=frac_test,
                                                        random_state=split_seed)
    headery = " ".join([f"y{ii}" for ii in range(Y_train.shape[1])])        

    xtrain_f = os.path.join(dirname, "data4_in.txt")
    ytrain_f = os.path.join(dirname, "data4_out.txt")
    xtest_f = os.path.join(dirname, "data4_test.txt")
    ytest_f = os.path.join(dirname, "data4_test_out.txt")
    np.savetxt(xtrain_f, X_train, header="x", comments='')
    np.savetxt(xtest_f, X_test, header="x", comments='')
    np.savetxt(ytrain_f, Y_train, header=headery, comments='')
    np.savetxt(ytest_f, Y_test, header=headery, comments='')     
