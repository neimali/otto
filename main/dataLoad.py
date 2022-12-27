import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm

class DataLoad:

    def __init__(self, path='D:\OTTO\Data\\train.jsonl'):
        self.path = path
        self.sample = None
        self.sample_size = 0

    def sample_data(self, sample_size=2):
        if sample_size == -1:
            self.sample = pd.read_json(self.path, lines=True)
            self.sample_size = len(self.sample)
        else:
            df = pd.read_json(self.path, lines=True, chunksize=sample_size)
            for chunk in df:
                chunk.set_index('session', drop=True, inplace=True)
                break
            self.sample = chunk
            self.sample_size = sample_size

    def get_data_with_chunk(self, chunksize=10000):
        df = pd.read_json(self.path, lines=True, chunksize=chunksize)
        to_return_df = pd.DataFrame()
        for chunk in df:
            time_start = time.time()
            chunk.set_index('session', drop=True, inplace=True)
            flattened_df = self.to_dataframe(chunk)
            to_return_df = pd.concat([to_return_df, flattened_df], ignore_index=True)
            time_end = time.time()
            print('time cost:' + str(time_end-time_start))
        return to_return_df
    
    def to_dataframe(self, data):
        session = []
        aid = []
        ts = []
        atype = []
        for i in range(len(data)):
            s = data.index[i]
            for action in data.iloc[i].item():
                session.append(s)
                aid.append(action['aid'])
                ts.append(action['ts'])
                atype.append(action['type'])

        data_dict = {'session': session, 'aid': aid, 'ts': ts, 'type': atype}
        df = pd.DataFrame.from_dict(data_dict)

        return df

    def random_sample(self, sample_size, random_seed=24):
        np.random.seed(random_seed)
        sample_indexs = np.random.randint(0, 12899778, size=sample_size).tolist()
        df = pd.read_json(self.path, lines=True, chunksize=3000000)
        for i, chunk in enumerate(df):
            chunk.set_index('session', drop=True, inplace=True)
            chunk = chunk.iloc[sample_indexs]
            if self.sample is None:
                self.sample = chunk
            else:
                self.sample = pd.concat([self.sample, chunk])

    def data_split(self, data, train_size=None, random_seed=24):
        session = data.session.unique().tolist()
        train, test = train_test_split(session, train_size=train_size, random_state=random_seed)
        train_df = data.loc[data.session.isin(train)].copy()
        test_res_df = data.loc[data.session.isin(test)].copy()
        np.random.seed(random_seed)
        truncate_index = []
        for i, s in enumerate(tqdm(test_res_df.groupby('session'))):
            truncate_point = np.random.randint(1, len(s[1]))
            index = s[1].index.tolist()
            truncate_index.extend(index[truncate_point:])

        test_df = test_res_df.drop(truncate_index)

        return train_df, test_df, test_res_df








if __name__ == '__main__':
    dl = DataLoad('D:\OTTO\Data\\train.jsonl')

    dl.sample_data(10)

    df = dl.to_dataframe(dl.sample)
    print(df)
    train, test, test_res = dl.data_split(df)
    print(train)
    print(test)
    print(test_res)
    # print(df)
    # dl.random_sample(10)
    # tmp = pd.to_datetime(df['ts'],unit='ms')
    # print(tmp.head())
    # print(len(df.index==0))

    #count aid
    # print(df.value_counts('aid'))

    #test on test set
    # dlt = DataLoad('D:\OTTO\Data\\test.jsonl')
    # dlt.sample_data(10)
    # df = dlt.to_dataframe()
    # print(df)

    # dl = DataLoad('D:\OTTO\Data\\train.jsonl')
    # dl.get_data_with_chunk(10000)


#sample data
#extract aid