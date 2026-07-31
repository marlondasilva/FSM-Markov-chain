import numpy as np
import random as rm
import yaml
import progressbar

import argparse
PARSER = argparse.ArgumentParser(description="Prioritization of test cases - Markov Chain")
PARSER.add_argument("train_configs",
                    help="Path to a YAML file that configures the prioritization code")

args = PARSER.parse_args()

def find_element(l, elem):
    for row, i in enumerate(l):
        try:
            column = i.index(elem)
        except ValueError:
            continue
        return row, column
    return -1

# Function that calculates the limit probability of the transition matrix.
def prob_limit(matrix):
    i = 0
    epsilon  = 1e-8
    matrixAux = matrix
    while(True):
        C = np.dot(matrix,matrixAux)
        D = matrixAux[0][0] - C[0][0]
        if (abs(D) < epsilon):
            break
        matrixAux = C
        i = i+1
    return C

def probLimit(P):
    n = P.shape[0] # number of states
    A = (P - np.eye(n)).T # transposed matrix from pi.(P - I) to A.b

    # swap the last line to sum(pi) = 1
    A[-1, :] = 1
    b = np.zeros(n)
    b[-1] = 1

    try:
        pi = np.linalg.solve(A, b)
        # ~ print("Steady-state Probability (pi):")
        # ~ print(pi)
        return pi
    except np.linalg.LinAlgError:
        print("Error: The chain has not unique solution. Check if the chain is irreductible.")
        return 1

def randon_walk(initial_state, matrix, arcs_name, states_list):
    arc_now = initial_state
    transition_way = []
    arc_list = [arc_now]
    prob = 1

    while (True):
        # Find the prob
        present_state_idx = states_list.index(arc_now)
        arc = np.random.choice(arcs_name[present_state_idx],
                               replace=True,
                               p= matrix[present_state_idx])
        transition_way.append(arc)

        # Find the index of arc
        i, j = find_element(arcs_name, arc)
        prob = prob * matrix[i][j]
        arc_now = arc.split("_")[1]
        arc_list.append(arc_now)

        if arc_now in final_state:
            break
    return [transition_way,prob],[arc_list,prob]


if __name__ == "__main__":
    yaml_name = ''
    widgets = [' [',
         progressbar.Timer(format= 'elapsed time: %(elapsed)s'),
         '] ', progressbar.FormatLabel('%(value)f'), ' ',
           progressbar.Bar('*'),' (',
           progressbar.AdaptiveETA(), ') ',
          ]
    # Parses the experiment settings
    with open(args.train_configs) as yaml_file:
        code_configs = yaml.load(yaml_file, Loader=yaml.FullLoader)
        yaml_name = yaml_file.name
    # Parses the states list
    with open(code_configs["states_path"]) as f:
        for line in f:
            states = [elt.strip() for elt in line.split(',')]
    # Parses the transition matrix
    transition_matrix = np.loadtxt(code_configs["transition_matrix_path"], delimiter=",")
    # Parsing the initial and final state
    initial_state, final_state = code_configs["initial_state"], code_configs["final_state"]
    # Parsing the stop criteria percentage
    stop_criteria = code_configs["test_case_percentage"]
    bar = progressbar.ProgressBar(max_value=stop_criteria*1.05, widgets=widgets).start()

    # Creating arcs name matrix
    N = len(states)
    rows, cols = (N, N)
    arcs_name = []
    for origin_state in states:
        aux_list = []
        for destiny_state in states:
            aux_list.append(origin_state+'_'+destiny_state)
            if len(aux_list) == N:
                arcs_name.append(aux_list)
                break

    # Checking if the transition matrix is correct, that is, the sum of the lines is equal to 1.
    for i in range(len(transition_matrix)):
        sum = 0
        for j in range(len(transition_matrix)):
            sum += transition_matrix[i][j]
        if(sum != 1):
            print("Incorrect distribution!!")
            break

    list_transitions = []
    list_states = []
    stop = 0.0
    count = 0.0
    last_probability = 0.0

    i = 0
    # Main loop - here random_walk is used
    while stop < stop_criteria:
        sequence_arc,sequence_state = randon_walk(initial_state, transition_matrix, arcs_name, states)
        if sequence_arc in list_transitions:
            pass
        else:
            list_transitions.append(sequence_arc)
            list_states.append(sequence_state)
            stop += sequence_arc[1]
        # print(f"{(stop*100):.3f}% of test cases found.")
        if sequence_arc[1] < 1.e-25:
            last_probability = sequence_arc[1]
            break
        bar.update(stop)

    percent = stop*100
    bar.finish()
    print(f"\n{percent:.3f}% of test cases found.\nLast probability: {last_probability:.35f}")
    list_transitions.sort(key=lambda x:x[1],reverse=True)
    list_states.sort(key=lambda x:x[1],reverse=True)

    out_arcs = open(f"{yaml_name.split('.')[0]}arcs.txt", "w")
    out_arcs.write("Test cases showing the arcs traveled with their respective probabilities:\n")
    for line in list_transitions:
      out_arcs.write(str(line)+"\n")
    out_arcs.close()

    out_states = open(f"{yaml_name.split('.')[0]}states.txt", "w")
    out_states.write("Test cases showing the states traveled with their respective probabilities:\n")
    for line in list_states:
        out_states.write(str(line)+"\n")
    out_states.close()
    
    # ~ print(probLimit(transition_matrix))
    # ~ print(prob_limit(transition_matrix))
