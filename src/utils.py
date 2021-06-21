import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import plotly.express as px
import umap

from typing import Literal, Optional, Dict, List


def preprocess_data(data_path: str = 'data/comments.csv',
                    timedelta: Literal['s', 'm', 'h', 'D', 'W'] = 's') -> Optional[pd.DataFrame]:
    """
    Converts the given data to the form required by the Teneto library.

    :param data_path: path to the csv file, NECESSARY containing the columns 'author_login', 'receiver' and 'date'
    :param timedelta: a single letter indicating the type of time delta with 's' - seconds, 'm' - minutes, 'h' - hours,
                      'D' - days, 'W' - weeks
    :return: Pandas data frame with columns i, j and t representing source node, target node and timestamp, respectively
    """
    df = pd.read_csv('data/comments.csv').dropna()
    df = df[['author_login', 'receiver', 'date']]
    df = df[df.author_login != df.receiver]

    usernames = np.unique(
        np.array(df['author_login'].tolist() + df['receiver'].tolist())
    ).reshape(-1, 1)

    oe = OrdinalEncoder()
    oe = oe.fit(usernames)

    df.author_login = oe.transform(df[['author_login']]).astype(int)
    df.receiver = oe.transform(df[['receiver']]).astype(int)

    df['date'] = pd.to_datetime(df['date'])
    df['date'] = df.date.values.astype(np.int64) // 10 ** 9
    df['date'] = df['date'] - df['date'].min()

    timedelta_dict = {'s': 1, 'm': 60, 'h': 60 * 60, 'D': 24 * 60 * 60, 'W': 7 * 24 * 60 * 60}
    if timedelta not in timedelta_dict.keys():
        return None
    df['date'] = (df['date'] / timedelta_dict[timedelta]).apply(np.floor).astype(int)
    df.columns = ['i', 'j', 't']
    df = df.sort_values(by='t').reset_index()

    return df

def visualize_embeddings(embeddings: Dict[int, List[int]], 
              events_df: pd.DataFrame,
              path: str = "data/embeddings.html") -> None:
    """Generates plotly visualization of embeddings reduced to 2-D space with UMAP algorithm.

    :param embeddings: embeddings generated with skipgram.
    :type embeddings: Dict[int, List[int]]
    :param events_df: dataframe with times of events. i-th row contains
    data for event with key i. Column 't' contains time of the event.
    :type events_df: pd.DataFrame
    :param path: path of embeddings visualzation file, defaults to "data/embeddings.html"
    :type path: str, optional
    """
    X, keys = [], []
    for key in embeddings:
        X.append(embeddings[key])
        keys.append(key)

    X = np.array(X)
    reducer = umap.UMAP(n_neighbors=200)
    X_embedded = reducer.fit_transform(X)
    embeddings_df = pd.DataFrame(X_embedded, columns=['x1', 'x2'])
    embeddings_df['t'] = events_df.iloc[[
        int(k) for k in keys]]['t'].values

    fig = px.scatter(embeddings_df, x="x1", y="x2",
                     color="t", hover_data=['t'])
    fig.write_html(path)