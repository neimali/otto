import argparse
import numpy as np
import pandas as pd
from dataLoad import DataLoad
from wordVec import WordToVec
from clustering import Clustering
import json
from tqdm import tqdm

def args():
    description = 'chunksize(int): the chunksize of pandas json reader; train_path(str): the path of training data; k_values(list): candidates of k(int)'
    parse = argparse.ArgumentParser(description=description)
    parse.add_argument('--chunksize', type=int, default=1e4)
    parse.add_argument('--train_path', type=str, default='~/otto/data/train.jsonl')
    parse.add_argument('--test_path', type=str, default='~/otto/data/test.jsonl')
    parse.add_argument('--flattentrain_path', type=str, default='~/otto/data/flattenTrain.csv')
    parse.add_argument('--flattentest_path', type=str, default='~/otto/data/flattenTest.csv')
    parse.add_argument('--ts_duration', type=int, default=2)
    parse.add_argument('--sentences_path', type=str, default='/home/qiaodawang19/otto/data/sentences.jsonl')
    parse.add_argument('--cluster_candidates_path', type=str, default='/home/qiaodawang19/otto/data/cluster_candidates.jsonl')
    parse.add_argument('--co_vi_candidates_path', type=str, default='/home/qiaodawang19/otto/data/co_vi_candidates.jsonl')
    parse.add_argument('--final_candidates_path', type=str, default='/home/qiaodawang19/otto/data/final_candidates.jsonl')
    parse.add_argument('--traindf', type=str, default='~/otto/data/train.csv')
    parse.add_argument('--testdf', type=str, default='~/otto/data/test.csv')
    parse.add_argument('--testresdf', type=str, default='~/otto/data/testres.csv')
    arg = parse.parse_args()
    return arg

if __name__ == '__main__':
    arg = args()
    # dl = DataLoad(arg.train_path)
    # train_df = dl.get_data_with_chunk(chunksize=arg.chunksize)
    # train_df.to_csv(arg.flattendf_path, index=False)
    # print('data has been loaded and saved at ' + arg.flattendf_path)

    train_df = pd.read_csv(arg.flattentrain_path)
    print('flattened training data has been loaded')
    #dl = DataLoad(arg.test_path)
    #test_df = dl.get_data_with_chunk(chunksize=arg.chunksize)
    #test_df.to_csv(arg.flattentest_path, index=False)
    #print('data has been loaded and saved at ' + arg.flattentest_path)

    # wv = WordToVec(train_df)
    # sentences = wv.time_session(arg.ts_duration)
    # print('sentences generation is done')
    # s_str = json.dumps(sentences)
    # with open(arg.sentences_path, 'w') as json_file:
    #     json_file.write(s_str)
    # print('articles sentences has been created and stored in' + arg.sentences_path)
    # print('Loading sentneces...')
    # with open(arg.sentences_path, 'r') as json_file:
    #     sentences = json.load(json_file)
    # print('training started')
    # sentences = list(sentences.values())
    # vectors = wv.train(sentences)
    # print('word2vec training complete')
    #
    # del sentences
    #
    # c = Clustering('km', vectors)
    # k_values = int(np.floor(1855603/50))
    # clusters = c.kmCluster(k_values)
    # rank = c.articles_rank_by_label(train_df, clusters)
    #
    # del train_df
    #
    # test_df = pd.read_csv(arg.flattentest_path)
    # print('flattened testing data has been loaded')
    #
    # candidates = c.get_candidates(test_df,clusters,rank)
    # s_str = json.dumps(candidates)
    # with open(arg.cluster_candidates_path, 'w') as json_file:
    #     json_file.write(s_str)
    # print('candidates from clustering saved')


    with open(arg.co_vi_candidates_path, 'r') as json_file:
        co_vi_candidates = json.load(json_file)

    with open(arg.cluster_candidates_path, 'r') as json_file:
        cluster_candidates = json.load(json_file)

    most_popular = train_df.aid.value_counts()

    candidates = {}
    for k,v in tqdm(cluster_candidates):
        cans = []
        cans.extend(v)
        cans.extend(co_vi_candidates[k])
        if len(cans) >= 50:
            cans = cans[:50]
        else :
            cans.extend(list(most_popular.index[:50-len(cans)]))

    s_str = json.dumps(candidates)
    with open(arg.final_candidates_path, 'w') as json_file:
        json_file.write(s_str)

