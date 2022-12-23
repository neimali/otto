from gensim.models import Word2Vec
import pandas as pd
from dataLoad import DataLoad
import polars as pl
from annoy import AnnoyIndex

class WordToVec:
    def __init__(self, data):
            self.df = data

    def sentence_session(self):
        # sentences_df = self.df.groupby('session').agg(
        #     pl.col('aid').alias('sentence')
        # )
        # sentences = sentences_df['sentence'].to_list()
        # return sentences
        sentences = self.df.groupby('session')['aid'].apply(list)
        return sentences

    def time_session(self, interval):
        interval_ts = interval/2.7778e-7
        session = -1
        sentence = 0
        ts = -1
        # new session and ts_interval>interval
        for i in self.df.index:
            if self.df.loc[i, 'session'] == session and self.df.loc[i, 'ts']-ts < interval_ts:
                ts = self.df.loc[i, 'ts']
                self.df.loc[i, 'sentence'] = sentence
            elif self.df.loc[i, 'session'] != session:
                sentence += 1
                session = self.df.loc[i, 'session']
                ts = self.df.loc[i, 'ts']
                self.df.loc[i, 'sentence'] = sentence
            else:
                sentence += 1
                ts = self.df.loc[i, 'ts']
                self.df.loc[i, 'sentence'] = sentence

        sentences = self.df.groupby('sentence')['aid'].apply(list)
        return sentences

    def train(self, sentences):
        w2vec = Word2Vec(sentences=sentences, vector_size=32, min_count=1, workers=4)
        # aid2idx = {aid: i for i, aid in enumerate(w2vec.wv.index_to_key)}
        # index = AnnoyIndex(32, 'euclidean')
        # for aid, idx in aid2idx.items():
        #     index.add_item(idx, w2vec.wv.vectors[idx])
        # index.build(10)

        vectors = {}
        for aid, vector in zip(w2vec.wv.index_to_key, w2vec.wv.vectors.tolist()):
            vectors[aid] = vector

        return vectors

