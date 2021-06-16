from src.eventgraph import sample_neighbourhoods

import yaml
import pickle

with open("params.yaml", "r") as fd:
    params = yaml.safe_load(fd)

alpha = params["sampling"]["alpha"]
nb = params["sampling"]["nb"]
s = params["sampling"]["s"]


if __name__ == "__main__":

    with open('data/w_path_dict.pkl', 'rb') as handle:
        w_path_dict = pickle.load(handle)

    with open('data/w_co_dict.pkl', 'rb') as handle:
        w_co_dict = pickle.load(handle)

    with open('data/event_to_nodes.pkl', 'rb') as handle:
        event_to_nodes = pickle.load(handle)

    nbs = sample_neighbourhoods(w_path_dict, w_co_dict, event_to_nodes,
                                alpha, nb, s)

    with open('data/sampled_neighbourhoods.pkl', 'wb') as handle:
        pickle.dump(nbs, handle)

