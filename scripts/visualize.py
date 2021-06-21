from itertools import product
import umap
import numpy as np
import pandas as pd
import plotly.express as px

import yaml
import pickle

with open("params.yaml", "r") as fd:
    params = yaml.safe_load(fd)

DATA_PATH = "data/preprocessed_dataframe.pkl"

if __name__ == "__main__":
    with open(f'data/embeddings.pkl', 'rb') as handle:
        embeddings = pickle.load(handle)

    preprocessed_df = pd.read_pickle(DATA_PATH)

    X, keys = [], []
    for key in embeddings:
        X.append(embeddings[key])
        keys.append(key)

    X = np.array(X)
    reducer = umap.UMAP(n_neighbors=200)
    X_embedded = reducer.fit_transform(X)
    embeddings_df = pd.DataFrame(X_embedded, columns=['x1', 'x2'])
    embeddings_df['t'] = preprocessed_df.iloc[[
        int(k) for k in keys]]['t'].values

    fig = px.scatter(embeddings_df, x="x1", y="x2",
                     color="t", hover_data=['t'])
    fig.write_html("data/embeddings.html")
