import os
import sys
import copy
import networkx as nx
import gslearn as gsl
import subprocess 
sys.path.append("../../cmdline_yaml")

import functools

RESULTS_DIR_BASE="./results"

# template_file="templates/template.yaml"
# with open(template_file, 'r') as file:
#     template_str = file.read()

template_file="templates/template_nn.yaml"
with open(template_file, 'r') as file:
    template_str = file.read()    

def gen_input_file(graph_file_name, data_dir="../../example_data"):

    new_str = copy.deepcopy(template_str)
    new_str = new_str.replace("GRAPH_FILE_TEMP", graph_file_name)
    new_str = new_str.replace("DATA_TEMP", data_dir)

    return new_str

def graph_to_dirname(graph_gsl_bin):

    dirname = graph_gsl_bin.hex()
    return dirname
    
def gen_new_directory(graph_gsl_bin):
    
    results_dir = os.path.join(RESULTS_DIR_BASE, graph_to_dirname(graph_gsl_bin))
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def write_input_file(graph_gsl_bin, dirname, data_dir_name):

    filename = os.path.join(dirname, "edges.edge_list")
    net = gsl.Graph.from_incidence_matrix_buffer(graph_gsl_bin).to_networkx()
    nx.write_edgelist(net, filename, data=False)
    new_str = gen_input_file("edges.edge_list", data_dir_name)
    input_filename = os.path.join(dirname, "input.yaml")
    with open(input_filename, 'w') as file:
        file.write(new_str)
    
def dirname_to_graph(dirname):

    _, graph_encode = dirname.split("results/")
    graph = gsl.Graph.from_incidence_matrix_buffer(bytes.fromhex(graph_encode))
    return graph

def run_dir(dirname):

    cwd = os.getcwd()
    # print(cwd)
    # print(dirname)
    fullpath = os.path.join(cwd, dirname)
    # print(fullpath)
    os.chdir(fullpath)
    out = subprocess.run(["ls", "-l"], capture_output=True, text=True)
    # out = subprocess.run(["ls", "-l"], capture_output=True, text=True)    
    # print(out)

    # print("\n\n\n")
    out = subprocess.run(["python", "../../../../cmdline_yaml/mfnet_cmd.py", "input.yaml"], capture_output=True, text=True)


    loss = None
    with open("log.log") as f:
        logfile = f.readlines()
        for k, line in enumerate(logfile):
            if line.find("INFO:root:Model Loss:") != -1:
                _, loss = line.split("Model Loss:")
                loss = float(loss)
                # print(line)
            
    os.chdir(cwd)
    if loss == None:
        print(out)
        raise Exception("something wrong with the run")
    # out = subprocess.run(["ls", "-l"], capture_output=True, text=True)
    # print(out)
    cwd = os.getcwd()
    # print(cwd)
    # print("LOSS = ", loss)
    return loss

@functools.cache
def get_graph_loss(graph_gsl_bin, data_dir_name):
    graph_dirname = gen_new_directory(graph_gsl_bin)
    write_input_file(graph_gsl_bin, graph_dirname, data_dir_name)

    print(gsl.Graph.from_incidence_matrix_buffer(graph_gsl_bin))
    loss = run_dir(graph_dirname)
    print(loss)
    return loss
