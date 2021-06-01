import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from typing import Literal, Optional


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
