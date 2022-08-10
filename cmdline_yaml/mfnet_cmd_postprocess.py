import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly.express as px
import seaborn as sns

import os 
import logging
logging.basicConfig(level=logging.INFO)

def readable_parameter_names(df):

    column_names = df.columns

    new_names = []
    for c in column_names:

        split = c.split('.')

        if "node" in split[0]:
            name = split[0].replace("node", "f_{")
            name += '}'
        elif "edge" in split[0]:
            name = split[0].replace("edge", r"e_{")
            name += '}'
            name = name.replace("->", "\\rightarrow")

        if "weight" in split[1]:
            add_on = split[1].replace("weight", "w")
        elif "bias" in split[1]:
            add_on = split[1].replace("bias", "b")

        name += add_on

        name = '$' + name + '$'
        new_names.append(name)
    return new_names
    
def plot_parameters(filename):
    
    df = pd.read_csv(filename)
    # print(df.columns)
    # df = df[['node1.weight[0]', 'node1.bias[0]', 'node2.weight[0]', 'node2.bias[0]',
    #          'edge1->2.weight[0]', 'edge1->2.bias[0]']]
    # exit(1)
            
    # df = df[['node1.weight[0,0]', 'node1.bias[0]', 'node2.weight[0,0]', 'node2.bias[0]',
    #          'edge1->2.weight[0,0]', 'edge1->2.bias[0]']]
    # df = df[['node1.weight[0]', 'node1.bias[0]', 'node2.weight[0]', 'node2.bias[0]',
    #    'edge1->2.weight[0]', 'edge1->2.bias[0]', 'node3.weight[0]',
    #    'node3.bias[0]', 'edge2->3.weight[0]', 'edge2->3.bias[0]']]
    # df = df[['node1.weight[0,0]', 'node1.bias[0]', 'node2.weight[0,0]', 'node2.bias[0]',
    #    'edge1->2.weight[0,0]', 'edge1->2.bias[0]', 'node3.weight[0,0]',
    #    'node3.bias[0]', 'edge2->3.weight[0,0]', 'edge2->3.bias[0]']]    

    # exit(1)
          
    new_names = readable_parameter_names(df)
    
    df.columns = new_names
    print(df.describe())

    g = sns.PairGrid(df, corner=True)
    g = g.map_diag(sns.histplot, common_norm=False)
    g = g.map_lower(sns.kdeplot)
    # g = g.map_lower(plt.scatterplot)
    g.figure.set_size_inches(8,8)
    g.savefig('posterior.pdf')
    
    # fig = px.scatter_matrix(df)
    # fig.update_traces(showupperhalf=False)
    # fig.show()
    # arr = pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(6,6), diagonal='kde')
    
if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(
        prog='mfnet_cmd',
        description="Perform MFNETS",
    )

    parser.add_argument("dir", 
                        help='directory containing results to process')

    parser.add_argument("--test_evals", metavar="FILE", type=str, nargs='+', required=False,
                        help='file containing predictions in the same order as saved in results file')

    parser.add_argument("--test_inputs", metavar="FILE", type=str, nargs='+', required=False,
                        help='file containing locations of predictions in the same order as saved in results file')


    args = parser.parse_args()
    directory = args.dir

    logging.info(f"Directory: {directory}")
    
    dir_contents = os.listdir(directory)

    logging.info(f"Directory Contents: {dir_contents}")

    fparams = [f for f in dir_contents if "param_samples.csv" in f]

    if len(fparams) != 1:
        logging.info(f"No parameter samples to plot")
        # logging.error(f"Can only have one file containing param_samples, instead have {fparams}")
        # exit(1)
    # else:
    #     plot_parameters(os.path.join(directory, fparams[0]))

    # print(args)
    if 'test_evals' in args:
        fname = args.test_evals[0]
        data_test = np.loadtxt(fname)

        fpred = [f for f in dir_contents if "pyro.out_model" in f]
        if len(fpred) != data_test.shape[1]:
            print("in here ", len(fpred))
            if len(fpred) != 1:
                logging.error(f"Number of model predictions {len(fpred)} does not match number of test predictions {data_test.shape[1]}")

        # expecting one file with each column corresponding to different model fidelity evaluation
        # only for deterministic models
        if len(fpred) == 1: 
            pred = np.loadtxt(os.path.join(directory, fpred[0]))
            print(pred.shape)
            if pred.shape[1] != data_test.shape[1]:
                logging.error(f"Number of model predictions {pred.shape[1]} does not match number of test predictions {data_test.shape[1]}")

            for ii in range(pred.shape[1]):
                print(ii)
                plt.figure()
                plt.plot(data_test[:, ii], pred[:, ii], 'o')
        else:
            # handles multiple samples
            for pred in fpred:

                model_number = int(pred.split("pyro.out_model")[1]) - 1
                print(model_number)
                pred_model = np.loadtxt(os.path.join(directory, pred))
                print(pred_model.shape)
                
                indsort = np.argsort(data_test[:, model_number])
                dsort = data_test[indsort, model_number]
                
                pred_sorted = pred_model[indsort, :]
                quantiles = np.quantile(pred_sorted, (0.1, 0.5, 0.9), axis=1).T
                print(quantiles.shape)
                
                plt.figure()
                # plt.plot(dsort, quantiles[:, 1], '-r')
                plt.fill_between(dsort, quantiles[:, 0], quantiles[:, 2], alpha=1.0)
                plt.plot(dsort, dsort, '-k')
                plt.xlabel("True Value")
                plt.ylabel("Predicted Value")
                # plt.plot(data_test[indsort, model_number], quantiles[:, 0], '--')
                # plt.plot(data_test[indsort, model_number], quantiles[:, 2], '--')
        plt.show()

            
        


    plt.show()
