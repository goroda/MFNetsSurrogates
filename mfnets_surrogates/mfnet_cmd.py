"""Command Line Utility for MFNETS."""

import argparse
import networkx as nx
import torch
import numpy as np

import net_torch as net
import net_pyro

from pyro.infer import MCMC, NUTS
from pyro.infer import Predictive

import pandas as pd


def fill_graph(graph, dim_in):
    """Assign node and edge functions."""
    for node in graph.nodes:
        graph.nodes[node]['func'] = torch.nn.Linear(dim_in, 1, bias=True)

    for e1, e2 in graph.edges:
        graph.edges[e1,e2]['func'] = torch.nn.Linear(dim_in, 1, bias=True)

    return graph

def parse_graph(args, dim_in):
    """Parse the graph."""
    graph_file = args.graph
    assert len(graph_file) == 1, "Only one graph can be used"
    graph_file = graph_file[0]

    try:
        with open(graph_file) as f:
            edge_list = f.read().splitlines()
    except FileNotFoundError:
        print(f"Cannot open file {graph_file}")
        exit(1)

    graph = fill_graph(nx.parse_edgelist(edge_list, create_using=nx.DiGraph), dim_in)

    roots = set([x for x in graph.nodes() if graph.in_degree(x) == 0])

    print(f"Root models: {roots}")
    
    return graph, roots

def parse_data(args):
    """Parse data files."""
    num_models = len(args.data)
    print(f"Number of models: {num_models}")

    datasets = []
    dim_in = None
    for model_data in args.data:
        try:
            data = np.loadtxt(model_data)
        except FileNotFoundError:
            print(f"Cannot open file {model_data}")
            exit(1)
        if dim_in == None:
            dim_in = data.shape[1] - 1
        else:
            assert data.shape[1] - 1 == dim_in, \
                f"All input dimensions must be the same: {dim_in}"
        datasets.append(
            net.ArrayDataset(
                torch.Tensor(data[:,:-1]), # x
                torch.Tensor(data[:,-1]))) # y is last column

    
    return datasets, dim_in


def parse_evaluation_locations(args, dim_in):
    """Parse eval_locatiosn."""
    assert len(args.eval_locs) == 1, "only one evaluation location file allowed"

    try:
        data = np.loadtxt(args.eval_locs[0])
    except FileNotFoundError:
        print(f"Cannot open file {args.eval_locs[0]}")
        exit(1)

    if data.ndim == 1:
        data = data[:, np.newaxis]
        
    assert data.shape[1] == dim_in, "Input dimension not matching for evaluation locations"

    return torch.Tensor(data)

def datasets_to_dataloaders(data):
    """Convert datasets to dataloaders for pytorch training."""
    data_loaders = [torch.utils.data.DataLoader(d, batch_size=len(d), shuffle=False)
                    for d in data]
    return data_loaders

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='mfnet_cmd',
        description="Perform MFNETS",
    )

    parser.add_argument("--data", metavar="FILE", type=str, nargs='+', required=True,
                        help='files containing data ordered from lowest fidelity to highest')

    parser.add_argument("--graph", metavar="Graph", type=str, nargs=1, required=True,
                        help='description of a graph as a list of edges')

    parser.add_argument("--eval_locs", metavar="file", type=str, nargs=1, required=True,
                        help='A list of locations at which to evaluate the trained model')

    parser.add_argument("-o", metavar="output", type=str, nargs=1, required=True,
                        help='output file name')


    parser.add_argument("--type", metavar="type", type=str, nargs=1, required=True,
                        help='running type pytorch or pyro')


    parser.add_argument("--noisevar", metavar="noisevar", type=float, nargs=1,
                        help='noise variance')

    parser.add_argument("--pyro_alg", metavar="pyro_alg", type=str, nargs=1,
                        help='mcmc or xxx')

    parser.add_argument("--num_samples", metavar="num_samples", type=int, nargs=1,
                        default=1000,
                        help='Number of MCMC samples')

    parser.add_argument("--burnin", metavar="type", type=int, nargs=1,
                        default=100,
                        help='Burnin')    

    #########################
    ## Parse Arguments
    args = parser.parse_args()
    
    print("\n\n")
    datasets, dim_in = parse_data(args)
    graph, roots = parse_graph(args, dim_in)

    target_nodes = list(graph.nodes)
    num_nodes = len(target_nodes)
    print("Node names: ", target_nodes)

    eval_locs = parse_evaluation_locations(args, dim_in)

    save_evals_filename = args.o[0]

    run_type = args.type[0]
    

    if run_type == "pytorch":
        #################
        ## Pytorch
        model = net.MFNetTorch(graph, roots)

        ## Training Setup
        loss_fns = net.construct_loss_funcs(model)
        data_loaders = datasets_to_dataloaders(datasets)

        ## Train
        model.train(data_loaders, target_nodes, loss_fns)

        ## Evaluate and save to file
        with torch.no_grad():
            vals = model([eval_locs]*num_nodes, target_nodes)
        vals = np.hstack([v.detach().numpy() for v in vals])
        print(vals.shape)

        print("Saving to: ", save_evals_filename)
        np.savetxt(save_evals_filename, vals)

    elif run_type == "pyro":

        noise_var = args.noisevar[0]
        model = net_pyro.MFNetProbModel(graph, roots, noise_var=noise_var)

        alg = args.pyro_alg[0]
        num_samples = args.num_samples[0]
        
        if alg == 'mcmc':
            warmup = args.burnin[0]
            num_chains = 1
            nuts_kernel = NUTS(model, jit_compile=False)
            mcmc = MCMC(
                nuts_kernel,
                num_samples=num_samples,
                warmup_steps=warmup,
                num_chains=num_chains,
            )

            X = [d.x for d in datasets]
            Y = [d.y for d in datasets]
            print("\n")
            mcmc.run(X, target_nodes, Y)
            print("\n")

            predictive = Predictive(model,  mcmc.get_samples())
            vals = predictive([eval_locs]*num_nodes, target_nodes)

            for ii, node in enumerate(target_nodes):
                v = vals[f"obs{node}"].T # x by num_samples
                fname = f"{save_evals_filename}_model{node}"
                print(f"Saving outputs of Node {node} to: ", fname)
                np.savetxt(fname, v)
                
            df = net_pyro.mcmc_samples_to_pandas(mcmc.get_samples())
            
            sample_filename = f"{save_evals_filename}_param_samples.csv"
            print("Saving samples to ", sample_filename)
            df.to_csv(sample_filename)
            
