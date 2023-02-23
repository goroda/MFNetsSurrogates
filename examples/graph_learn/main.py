import os
import sys
import copy
import networkx as nx
import gslearn as gsl
import subprocess 
sys.path.append("../../cmdline_yaml")

RESULTS_DIR_BASE="./results"

template_file="templates/template.yaml"
with open(template_file, 'r') as file:
    template_str = file.read()

def gen_input_file(graph_file_name):

    new_str = copy.deepcopy(template_str)
    new_str = new_str.replace("GRAPH_FILE_TEMP", graph_file_name)
    return new_str

def graph_to_dirname(graph_gsl):

    dirname = graph_gsl.incidence_matrix.tobytes().hex()
    return dirname
    
def gen_new_directory(graph_gsl):
    
    results_dir = os.path.join(RESULTS_DIR_BASE, graph_to_dirname(graph_gsl))
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def write_input_file(graph, dirname):

    filename = os.path.join(dirname, "edges.edge_list")
    net = graph.to_networkx()
    nx.write_edgelist(net, filename, data=False)
    new_str = gen_input_file("edges.edge_list")
    input_filename = os.path.join(dirname, "input.yaml")
    with open(input_filename, 'w') as file:
        file.write(new_str)
    
def dirname_to_graph(dirname):

    _, graph_encode = dirname.split("results/")
    graph = gsl.Graph.from_incidence_matrix_buffer(bytes.fromhex(graph_encode))
    return graph

def run(dirname):

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
    # print(out)

    with open("log.log") as f:
        logfile = f.readlines()
        for k, line in enumerate(logfile):
            if line.find("INFO:root:Model Loss:") != -1:
                _, loss = line.split("Model Loss:")
                loss = float(loss)
                # print(line)
    
    os.chdir(cwd)
    # out = subprocess.run(["ls", "-l"], capture_output=True, text=True)
    # print(out)
    cwd = os.getcwd()
    # print(cwd)
    # print("LOSS = ", loss)
    return loss

if __name__ == "__main__":
    
    # print(template_str)

    graph = nx.DiGraph()
    for ii in range(0, 2):
        graph.add_node(ii)
    graph.add_edge(0,1)

    graph_gsl = gsl.Graph.from_networkx(graph)

    dirname = gen_new_directory(graph_gsl)
    write_input_file(graph_gsl, dirname)
    loss = run(dirname)
    print("loss = ", loss)
    #dirname_to_graph(dirname)


    
