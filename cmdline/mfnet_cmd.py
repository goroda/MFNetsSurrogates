"""Command Line Utility for MFNETS."""
import sys
# sys.path.append("../mfnets_surrogates")
sys.path.append("/Users/alex/Software/mfnets_surrogate/mfnets_surrogates")

import argparse
import networkx as nx
import torch
import numpy as np
import matplotlib.pyplot as plt

import net_torch as net
import net_pyro

from pyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.infer.autoguide import (
    # AutoDelta,
    AutoNormal,
    AutoMultivariateNormal,
    # AutoLowRankMultivariateNormal,
    # AutoGuideList,
    AutoIAFNormal,
    # init_to_feasible,
)


import pandas as pd

import logging
logging.basicConfig(level=logging.INFO)


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

    logging.info(f"Root models: {roots}")
    
    return graph, roots

def parse_data(args):
    """Parse data files."""
    num_models = len(args.data)
    logging.info(f"Number of models: {num_models}")

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
    """Parse eval_locations."""

    if args.eval_locs == None:
        return None
    elif len(args.eval_locs) == 0:
        return None
    
    assert len(args.eval_locs) == 1, "only one evaluation location file allowed"
    

    try:
        data = np.loadtxt(args.eval_locs[0])
    except FileNotFoundError:
        logging.error(f"Cannot open file {args.eval_locs[0]}")
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

    parser.add_argument("--eval_locs", metavar="file", type=str, nargs=1, required=False,
                        help='A list of locations at which to evaluate the trained model')

    parser.add_argument("-o", metavar="output", type=str, nargs=1, required=True,
                        help='output file name')

    parser.add_argument("--type", metavar="type", type=str, nargs=1, required=True,
                        help='running type pytorch or pyro')

    parser.add_argument("--noisevar", metavar="noisevar", type=float, nargs=1,
                        help='noise variance')

    parser.add_argument("--pyro_alg", metavar="pyro_alg", type=str, nargs=1,
                        help='mcmc, svi-normal, svi-multinormal, svi-iafflow',
                        default='mcmc')

    parser.add_argument("--iaf_depth", metavar="iaf_depth", type=int,
                        help='number of autoregressive transforms',                        
                        default=4)
    
    parser.add_argument("--iaf_hidden_dim", metavar="iaf_hidden_dim", type=int, nargs='+',
                        help='number of autoregressive transforms',                        
                        default=[40])        
    
    parser.add_argument("--num_samples", metavar="num_samples", type=int,
                        default=1000,
                        help='Number of MCMC samples')

    parser.add_argument("--num_steps", metavar="num_steps", type=int,
                        default=10000,
                        help='Number of SVI steps')    

    parser.add_argument("--burnin", metavar="burnin", type=int,
                        default=100,
                        help='Burnin')    

    #########################
    ## Parse Arguments
    args = parser.parse_args()

    datasets, dim_in = parse_data(args)
    graph, roots = parse_graph(args, dim_in)

    target_nodes = list(graph.nodes)
    num_nodes = len(target_nodes)
    logging.info(f"Node names: {target_nodes}")

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
        if eval_locs != None:
            with torch.no_grad():
                vals = model([eval_locs]*num_nodes, target_nodes)
            vals = np.hstack([v.detach().numpy() for v in vals])
            logging.info(vals.shape)

            logging.info(f"Saving to: {save_evals_filename}")
            np.savetxt(save_evals_filename, vals)

    elif run_type == "pyro":

        noise_var = args.noisevar[0]
        model = net_pyro.MFNetProbModel(graph, roots, noise_var=noise_var)

        alg = args.pyro_alg[0]

        ## Algorithm parameters
        num_samples = args.num_samples
        logging.info(f"Number of samples = {num_samples}")

        # MCMC
        num_chains = 1
        warmup = args.burnin

        # SVI
        adam_params = {"lr": 0.005, "betas": (0.95, 0.999)}
        num_steps = args.num_steps

        
        ## Data setup
        X = [d.x for d in datasets]
        Y = [d.y for d in datasets]
        
        if alg[:3] == 'svi':

            optimizer = Adam(adam_params)
            if alg == 'svi-normal':
                guide = AutoNormal(model)
            elif alg == 'svi-multinormal':
                guide = AutoMultivariateNormal(model)
            elif alg == 'svi-iafflow':
                hidden_dim = args.iaf_hidden_dim # list
                num_transforms = args.iaf_depth
                guide = AutoIAFNormal(model,
                                      hidden_dim=hidden_dim,
                                      num_transforms=num_transforms)
            else:
                logging.info(f"Algorithm \'{alg}\' is not recognized")
                exit(1)
                
            svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

            logging.info(f"Number of steps = {num_steps}")
            #do gradient steps
            for step in range(num_steps):
                elbo = svi.step(X, target_nodes, Y)
                if step % 100 == 0:
                    logging.info(f"Iteration {step}\t Elbo loss: {elbo}")

            predictive = Predictive(model, guide=guide, num_samples=num_samples)
            if eval_locs != None:
                # print(eval_locs.shape)
                pred = predictive([eval_locs]*num_nodes, target_nodes)
                # print(list(pred.keys()))
            else: # just predict on training points
                pred = predictive(X, target_nodes) 

            # param_samples = [print(k, v.squeeze().shape)
            #                  for k,v in pred.items() if k[:3] != "obs"]

            param_samples = {k: v.squeeze() for k,v in pred.items() if k[:3] != "obs"}
            vals = {k: v for k,v in pred.items() if k[:3] == "obs"}                
        

        elif alg == 'mcmc':

            nuts_kernel = NUTS(model, jit_compile=False, full_mass=True)
            mcmc = MCMC(
                nuts_kernel,
                num_samples=num_samples,
                warmup_steps=warmup,
                num_chains=num_chains,
            )
            print("\n")
            mcmc.run(X, target_nodes, Y)
            print("\n")

            param_samples = mcmc.get_samples()

            predictive = Predictive(model,  mcmc.get_samples())            
            if eval_locs != None:
                vals = predictive([eval_locs]*num_nodes, target_nodes)
            else: # predict on training points
                vals = predictive(X, target_nodes)

        else:
            logging.info(f"Algorithm \'{alg}\' is not recognized")
            exit(1)

        for ii, node in enumerate(target_nodes):
            v = vals[f"obs{node}"].T # x by num_samples
            fname = f"{save_evals_filename}_model{node}"
            logging.info(f"Saving outputs of Node {node} to: {fname}")
            np.savetxt(fname, v)



        # fig, axs = plt.subplots(3,1, sharex=True)
        # axs[0].plot(eval_locs, vals["obs1"].transpose(0,1), '-r', alpha=0.2)
        # axs[0].plot(X[0], Y[0], 'ko')

        # axs[1].plot(eval_locs, vals["obs2"].transpose(0,1), '-r', alpha=0.2)
        # axs[1].plot(X[1], Y[1], 'ko')

        # axs[2].plot(eval_locs, vals["obs3"].transpose(0,1), '-r', alpha=0.2)
        # axs[2].plot(X[2], Y[2], 'ko')

        # pred_filename = f"{save_evals_filename}_predict.pdf"
        # plt.savefig(pred_filename)
        # plt.show()
        
        df = net_pyro.samples_to_pandas(param_samples)

        logging.info(df.describe())

        sample_filename = f"{save_evals_filename}_param_samples.csv"
        logging.info(f"Saving samples to {sample_filename}")
        df.to_csv(sample_filename, index=False)

        # df = pd.read_csv(sample_filename)
        # print(df.describe())
            
