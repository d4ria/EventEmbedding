import pandas as pd
import numpy as np

from typing import Tuple, Dict
import tqdm


def calculate_event_graph_weights(df: pd.DataFrame,
                                  delta_t: int = 1) -> Tuple[Dict[Tuple[int, int], float],
                                                             Dict[Tuple[int, int, int, int], int],
                                                             Dict[int, Tuple[int, int]]]:
    """
    Calculates path weights and co-occurrance weights for the event graph.

    :param df: a Pandas DataFrame with 3 columns in order: source, target, timestamp
    :param delta_t: delta t from the article, needed for the definition of the delta_t-adjacent events
    :return: three dictionaries:
            1. w_path_dict, with path weights that has 2-element tuples as keys,
               eg. w_path_dict[(event_i, event_j)] = 1 / (1 + |timestamp_1 - timestamp_2| ),
            2. w_co_dict, with co-occurance weight that has 4-element tuples as keys,
               eg. w_co_dict[(event_i_source, event_i_target, event_j_source, event_j_target)] = x,
               where x is the total number of the delta_t-adjacent events on the underlying edges
               (event_i_source, event_i_target) and (event_j_source, event_j_target),
            3 event_to_nodes, with event indices as keys, and 2-element tuples as values, denoting
              source node and target node of an event.
    """
    n = len(df)
    w_path_dict = {}
    w_co_dict = {}
    event_to_nodes = {}

    for index, row in tqdm.tqdm(df.iterrows(), desc='Calculating weights', total=len(df)):
        src, trt, timestamp = row["i"], row["j"], row["t"]
        event_to_nodes[index] = (src, trt)

        node_sharing = df[((df['i'].isin([src, trt])) | (df['j'].isin([src, trt]))) & df['t'] > timestamp]
        w_path = 1 / (1 + abs(node_sharing["t"] - timestamp))

        for other_index, other_row in node_sharing.iterrows():
            src_j, trt_j, timestamp_j = other_row["i"], other_row["j"], other_row["t"]
            w_path_dict[(index, other_index)] = w_path[other_index]
            if timestamp_j - timestamp <= delta_t:
                w_co_dict[(src, trt, src_j, trt_j)] = w_co_dict.get((src, trt, src_j, trt_j), 0) + 1
                w_co_dict[(src_j, trt_j, src, trt)] = w_co_dict[(src, trt, src_j, trt_j)]

    return w_path_dict, w_co_dict, event_to_nodes


def _create_edges_list(w_path_dict: Dict[Tuple[int, int], float]) -> np.array:
    """
    Based on a dictionary with edge weights computes the matrix of edges.

    :param w_path_dict: a dictionary with path weights that has 2-element tuples as keys and weights as values
    :return: an array 2x(number of edges) with indices of nodes connected by an edge
    """
    edges = np.empty((2, len(w_path_dict.keys())))
    for i, (event_a, event_b) in enumerate(w_path_dict.keys()):
        edges[0, i] = event_a
        edges[1, i] = event_b
    return edges


def sample_neighbourhoods(w_path_dict: Dict[Tuple[int, int], float],
                          w_co_dict: Dict[Tuple[int, int, int, int], int],
                          event_to_nodes: Dict[int, Tuple[int, int]],
                          alpha: float = 0.5,
                          nb: int = 10,
                          s: int = 5):

    def _sample_node_neighbourhoods(node, neighbors, probs):
        if len(neighbors) < s:
            return [[node] + list(np.random.choice(neighbors, size=s, replace=True, p=probs))
                    for _ in range(nb)]
        else:
            return [[node] + list(np.random.choice(neighbors, size=s, replace=True, p=probs))
                    for _ in range(nb)]

    neighbourhoods = []
    edges = _create_edges_list(w_path_dict)
    nodes = np.unique(edges)

    for node in tqdm.tqdm(nodes, desc='Sampling neighbourhoods'):
        src, trt = event_to_nodes[node]

        predecessors = edges[0, edges[1, :] == node]
        successors = edges[1, edges[0, :] == node]
        neighbors = np.concatenate((predecessors, successors), axis=None)

        if neighbors.size == 0:
            continue
        F_path_denominator = sum([w_path_dict[(pred, node)] for pred in predecessors]) + \
                             sum([w_path_dict[(node, succ)] for succ in successors])
        F_co_denominator = sum([w_co_dict.get((*event_to_nodes[pred], src, trt), 0) for pred in predecessors]) + \
                           sum([w_co_dict.get((src, trt, *event_to_nodes[succ]), 0) for succ in successors])

        if F_path_denominator == 0:
            probabilities = [(1 - alpha) * w_co_dict.get((*event_to_nodes[pred], src, trt), 0) / F_co_denominator
                             for pred in predecessors] + \
                            [(1 - alpha) * w_co_dict.get((src, trt, *event_to_nodes[succ]), 0) / F_co_denominator
                             for succ in successors]
        elif F_co_denominator == 0:
            probabilities = [alpha * w_path_dict[(pred, node)] / F_path_denominator for pred in predecessors] + \
                            [alpha * w_path_dict[(node, succ)] / F_path_denominator for succ in successors]
        else:
            probabilities = [alpha * w_path_dict[(pred, node)] / F_path_denominator +
                             (1 - alpha) * w_co_dict.get((*event_to_nodes[pred], src, trt), 0) / F_co_denominator
                             for pred in predecessors] + \
                            [alpha * w_path_dict[(node, succ)] / F_path_denominator +
                             (1 - alpha) * w_co_dict.get((src, trt, *event_to_nodes[succ]), 0) / F_co_denominator
                             for succ in successors]
        probabilities = probabilities / sum(probabilities)

        node_nbrhds = _sample_node_neighbourhoods(node, neighbors, probabilities)
        neighbourhoods.extend(node_nbrhds)

    return neighbourhoods

