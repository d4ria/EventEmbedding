from gensim.models import Word2Vec
import numpy as np

from typing import List, Dict

def get_embeddings(nbs: List[List[int]],
                   vector_size: int = 30,
                   window_size: int = 10,
                   min_event_count: int = 25,
                   epochs: int = 20) -> Dict[int, List[float]]:
    """
    Creates skipgram embeddings for sampled event neighbourhoods

    :param nbs: sampled sequences of events.
    :type nbs: List[List[int]]
    :param vector_size: size of skipgram embeddings, defaults to 30
    :type vector_size: int, optional
    :param window_size: window size for skipgram model, defaults to 10
    :type window_size: int, optional
    :param min_event_count: minimum number occurences of event. Events 
        with fewer occurrences will be removed from the sequences, defaults to 25
    :type min_event_count: int, optional
    :param epochs: number for epochs for skipgram training, defaults to 20
    :type epochs: int, optional
    :return: dictionary of embeddings.
    :rtype: Dict[int, List[float]]
    """
    model = Word2Vec(sentences=nbs, 
                    vector_size=vector_size, 
                    window=window_size, 
                    min_count=min_event_count, 
                    epochs=epochs,
                    sg=1)
    
    X = model.wv.get_normed_vectors()
    keys = model.wv.index_to_key

    embeddings = {keys[idx]: X[idx] for idx in range(len(keys))}

    return embeddings