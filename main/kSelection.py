import argparse
import numpy as np
import pandas as pd
from dataLoad import DataLoad
from wordVec import WordToVec
from clustering import Clustering
import json

def args():
    description = 'chunksize(int): the chunksize of pandas json reader; train_path(str): the path of training data; k_values(list): candidates of k(int)'
    parse = argparse.ArgumentParser(description=description)
    parse.add_argument('--chunksize', type=int, default=1e4)
    parse.add_argument('--train_path', type=str, default='~/otto/data/train.jsonl')
    parse.add_argument('--flattendf_path', type=str, default='~/otto/data/flattenTrain.csv')
    parse.add_argument('--k_value', type=int, nargs='+', default=[1e4, 1e5, 1e6]) #--k_value 1e4 1e5 1e6
    parse.add_argument('--ts_duration', type=int, default=2)
    parse.add_argument('--sentences_path', type=str, default='~/otto/data/sentences.jsonl')
    parse.add_argument('--traindf', type=str, default='~/otto/data/train.csv')
    parse.add_argument('--testdf', type=str, default='~/otto/data/test.csv')
    parse.add_argument('--testresdf', type=str, default='~/otto/data/testres.csv')
    arg = parse.parse_args()
    return arg

if __name__ == '__main__':
    arg = args()
    dl = DataLoad(arg.train_path)
    # train_df = dl.get_data_with_chunk(chunksize=arg.chunksize)
    # train_df.to_csv(arg.flattendf_path, index=False)
    # print('data has been loaded and saved at ' + arg.flattendf_path)

    train_df = pd.read_csv(arg.flattendf_path)
    print('flattened data has been loaded')
    train, test, test_res = dl.data_split(train_df, train_size=0.9)
    print('data has been splited')
    del train_df
    del test
    del test_res
    del dl
    #train.to_csv(arg.traindf, index=False)
    #train.to_csv(arg.testdf, index=False)
    #train.to_csv(arg.testresdf, index=False)
    #print('splited data has been stored')
    #train = pd.read_csv(arg.traindf)
    #print('train set has been loaded')
    wv = WordToVec(train)
    sentences = wv.time_session(arg.ts_duration)
    print('sentences generation is done')
    s_str = json.dumps(sentences)
    with open(arg.sentences_path, 'w') as json_file:
        json_file.write(s_str)
    # sentences.to_csv(arg.sentences_path, index=False)
    print('articles sentences has been created and stored in' + arg.sentences_path)
    #sentences = pd.read_csv(arg.sentences_path)
    #vectors = wv.train(sentences)
    #print('vectors obtained')
    #c = Clustering('km', vectors)

    #k_values = arg.k_value
    #res = c.fine_tune(k_values, train, test, test_res)
    print(res)
