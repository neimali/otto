import xgboost as xgb
from sklearn.model_selection import GroupKFold
import json
import argparse
import pandas as pd
from feature import Feature

class Reranking:
    def __init__(self, model, canidates_path, k_fold):
        self.model = model
        self.k_fold = k_fold
        self.candidates_path = canidates_path

    def flatten_candidates(self):
        with open(self.candidates_path, 'r') as json_file:
            candidates = json.load(json_file)
        flattened_can = pd.DataFrame(columns=['session', 'candidates'])
        for k, v in candidates.items():
            session = [k]*len(v)
            tmp = pd.DataFrame([session, v], columns=['session', 'candidates'])
            flattened_can = pd.concat([flattened_can, tmp])

        return flattened_can

    def train_test_split(self):
        pass

    def get_feature(self, data):
        f = Feature(data)
        # f.target()
        f.drop_type()
        f.user_action_count()
        f.user_action_ratio()
        print(f.data.columns)
        return f.data

    def train(self, candidates, features, model_path):
        skf = GroupKFold(n_splits=self.k_fold)
        for fold,(train_idx, valid_idx) in enumerate(skf.split(candidates, candidates['click'], groups=candidates['session'] )):
            X_train = candidates.loc[train_idx, features]
            y_train = candidates.loc[train_idx, 'click']
            X_valid = candidates.loc[valid_idx, features]
            y_valid = candidates.loc[valid_idx, 'click']

            # IF YOU HAVE 50 CANDIDATE WE USE 50 BELOW
            dtrain = xgb.DMatrix(X_train, y_train, group=[100] * (len(train_idx)//100) )
            dvalid = xgb.DMatrix(X_valid, y_valid, group=[100] * (len(valid_idx)//100) )

            xgb_parms = {'objective':'rank:pairwise', 'tree_method':'gpu_hist'}
            model = xgb.train(xgb_parms,
                dtrain=dtrain,
                evals=[(dtrain,'train'),(dvalid,'valid')],
                num_boost_round=1000,
                verbose_eval=100)
            model.save_model(model_path)

    def inference(self):
        pass


if __name__ == '__main__':

    def args():
        description = 'can_path is the path of 100 candidates, k_fold is the k for k_fold validation'
        parse = argparse.ArgumentParser(description=description)
        parse.add_argument('--can_path', type=str, default='/home/qiaodawang19/otto/data/final_candidates.jsonl')
        parse.add_argument('--model_path', type=str, default='/home/qiaodawang19/otto/model/t_model.xgb')
        parse.add_argument('--train_path', type=str, default='/home/qiaodawang19/otto/data/flattenTrain.csv')
        arg = parse.parse_args()
        return arg
    arg = args()
    r = Reranking('XGBoost', canidates_path=arg.can_path, k_fold=5)
    print('start loading training data')
    # train_df = pd.read_csv(arg.train_path)
    train_df = pd.read_parquet('/home/qiaodawang19/otto/data/memoryopt/train.parquet')
    print('start adding features')
    train_df = r.get_feature(train_df)
    features = ['cl_cnt', 'ca_cnt', 'or_cnt', 'cl_ca_ratio', 'cl_or_ratio', 'ca_or_ratio']
    print('start training')
    r.train(train_df, features=features, model_path=arg.model_path)
    print('training compelet model saved at ' + arg.model_path)
