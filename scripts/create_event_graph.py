from src.eventgraph import calculate_event_graph_weights
import pandas as pd

import yaml
import pickle

with open("params.yaml", "r") as fd:
    params = yaml.safe_load(fd)

DATA_RATIO = params["data_ratio"]
DATA_PATH = "data/preprocessed_dataframe.pkl"

if __name__ == "__main__":
    df = pd.read_pickle(DATA_PATH)
    df = df.iloc[:int(DATA_RATIO * len(df)), :]

    w_path_dict, w_co_dict, event_to_nodes = calculate_event_graph_weights(df)

    with open('data/w_path_dict.pkl', 'wb') as handle:
        pickle.dump(w_path_dict, handle)

    with open('data/w_co_dict.pkl', 'wb') as handle:
        pickle.dump(w_co_dict, handle)

    with open('data/event_to_nodes.pkl', 'wb') as handle:
        pickle.dump(event_to_nodes, handle)