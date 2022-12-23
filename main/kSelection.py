import argparse
import numpy as np
import pandas as pd
from dataLoad import DataLoad
from wordVec import WordToVec
from clustering import Clustering

def args():
    description = 'chunksize(int): the chunksize of pandas json reader; train_path(str): the path of training data; k_values(list): candidates of k(int)'
    parse = argparse.ArgumentParser(description=description)
    parse.add_argument('--chunksize', type=int, default=1e4)
    parse.add_argument('--train_path', type=str, default='~/otto/data/train.jsonl')
    parse.add_argument('--k_value', type=int, nargs='+', default=[1e4, 1e5, 1e6]) #--k_value 1e4 1e5 1e6
    parse.add_argument('--ts_duration', type=int, default=2)
    arg = parse.parse_args()
    return arg

if __name__ == '__main__':
    arg = args()
    dl = DataLoad(arg.train_path)
    train_df = dl.get_data_with_chunk(chunksize=arg.chunksize)
    print('data has been loaded')
    train, test, test_res = dl.data_split(train_df)
    print('data has been splited')

    wv = WordToVec(train)
    sentences = wv.time_session(arg.ts_duration)
    print('articles sentences has been created')
    vectors = wv.train(sentences)
    print('vectors obtained')
    c = Clustering('km', vectors)

    k_values = arg.k_value
    res = c.fine_tune(k_values, train, test, test_res)
    print(res)