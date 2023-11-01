import matplotlib.pyplot as plt
import networkx as nx
import gslearn as gsl
from fileops import get_graph_loss

import datautils

def init_empty_graph():
    graph = nx.DiGraph()
    for ii in range(0, 4):
        graph.add_node(ii)

    return graph

def run_test(data_dir_name):

    loss_fn = lambda g : get_graph_loss(g, data_dir_name)

    graph = init_empty_graph()
    graph.add_edge(0,3)
    # graph.add_edge(1,3)
    graph.add_edge(2,1)

    graph_gsl = gsl.Graph.from_networkx(graph)

    graph_bin = graph_gsl.incidence_matrix.tobytes()

    # very fast when cached!
    loss = loss_fn(graph_bin)
    print("loss = ", loss)
    loss = loss_fn(graph_bin)
    print("loss = ", loss)
    loss = loss_fn(graph_bin)    
    print("loss = ", loss)

def run_sampling(data_dir_name):

    
    log_score = lambda g : -get_graph_loss(g.incidence_matrix.tobytes(),
                                           data_dir_name) *1e10

    num_nodes = 4

    nsamples = 2000
    burnin = 1000

    num_graphs_hist = 20     # include in histogram
    num_edges_hist = 20    # include in histogram
    num_graphs = 10 # visualize    

    graph = init_empty_graph()
    # graph.add_edge(0,3)
    # graph.add_edge(1,3)
    # graph.add_edge(2,1)
    
    dag_learn = gsl.DAGLearning(num_nodes, initial_graph=gsl.Graph.from_networkx(graph))
    prior = gsl.DAGLearning.Prior.UNIFORM
    trace = dag_learn.gen_scored_samples(nsamples, external_score=log_score, prior=prior)
    trace.burnin(burnin)
    graph_counts = trace.unique_graph_counts()
    edge_counts = trace.unique_edge_counts()
    fig1 = gsl.plot_samples(graph_counts, edge_counts, num_graphs_hist,
                            num_edges_hist, num_graphs)

    plt.figure()
    plt.plot(trace.scores)

    plt.show()
    
if __name__ == "__main__":
    
    # print(template_str)
    gen_data = True
    if gen_data == True:
        datautils.gen_and_write_data("data4M", 0.995)

    data_dir_name = "../../data4M"                
    # run_test(data_dir_name)
    
    run_sampling(data_dir_name)
    
        # loss = loss_fn(graph_bin)    
        # print("loss = ", loss)
        # loss = loss_fn(graph_bin)    
        # print("loss = ", loss)
        # loss = loss_fn(graph_bin)    
        # print("loss = ", loss)    
        #dirname_to_graph(dirname)


    
