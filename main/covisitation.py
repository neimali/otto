import pandas as pd
import numpy as np
import itertools
from collections import Counter
import argparse
import json


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
    parse.add_argument('--traindf', type=str, default='~/otto/data/train.csv')
    parse.add_argument('--testdf', type=str, default='~/otto/data/test.csv')
    parse.add_argument('--testresdf', type=str, default='~/otto/data/testres.csv')
    parse.add_argument('--co_vi_candidates_path', type=str, default='/home/qiaodawang19/otto/data/co_vi_candidates.jsonl')
    arg = parse.parse_args()
    return arg

def pqt_to_dict(df):
    return df.groupby('aid_x').aid_y.apply(list).to_dict()


def obtain_candidates(df):
    # USER HISTORY AIDS AND TYPES
    aids=df.aid.tolist()
    types = df.type.tolist()
    unique_aids = list(dict.fromkeys(aids[::-1] ))
    # RERANK CANDIDATES USING WEIGHTS
    if len(unique_aids)>=50:
        weights=np.logspace(0.1,1,len(aids),base=2, endpoint=True)-1
        aids_temp = Counter()
        # RERANK BASED ON REPEAT ITEMS AND TYPE OF ITEMS
        for aid,w,t in zip(aids,weights,types):
            aids_temp[aid] += w * type_weight_multipliers[t]
        sorted_aids = [k for k,v in aids_temp.most_common(50)]
        return sorted_aids
    # USE "CLICKS" CO-VISITATION MATRIX
    aids2 = list(itertools.chain(*[top_20_clicks[aid] for aid in unique_aids if aid in top_20_clicks]))
    # USE "CART ORDER" CO-VISITATION MATRIX
    aids3 = list(itertools.chain(*[top_20_buys[aid] for aid in unique_aids if aid in top_20_buys]))
    # USE "BUY2BUY" CO-VISITATION MATRIX
    aids4 = list(itertools.chain(*[top_20_buy2buy[aid] for aid in unique_aids if aid in top_20_buy2buy]))
    # RERANK CANDIDATES
    top_aids2 = [aid2 for aid2, cnt in Counter(aids2).most_common(10) if aid2 not in unique_aids]
    top_aids3 = [aid3 for aid3, cnt in Counter(aids3).most_common(20) if aid3 not in unique_aids]
    top_aids4 = [aid4 for aid4, cnt in Counter(aids4).most_common(20) if aid4 not in unique_aids]
    top_aids2.extend(top_aids3)
    top_aids2.extend(top_aids4)
    result = unique_aids + top_aids2[:50 - len(top_aids2)]
    # USE TOP20 TEST CLICKS
    return result



if __name__ == '__main__':
    VER = 5
    DISK_PIECES = 4
    type_weight_multipliers = {'clicks': 1, 'carts': 6, 'orders': 3}

    arg = args()
    # LOAD THREE CO-VISITATION MATRICES
    matrices_dir = '/home/qiaodawang19/otto/data/co_visitation/'
    top_20_clicks = pqt_to_dict(pd.read_parquet(matrices_dir + f'top_20_clicks_v{VER}_0.pqt'))
    for k in range(1, DISK_PIECES):
        top_20_clicks.update(
            pqt_to_dict(pd.read_parquet(matrices_dir + f'top_20_clicks_v{VER}_{k}.pqt')))
    top_20_buys = pqt_to_dict(pd.read_parquet(matrices_dir + f'top_15_carts_orders_v{VER}_0.pqt'))
    for k in range(1, DISK_PIECES):
        top_20_buys.update(
            pqt_to_dict(pd.read_parquet(matrices_dir + f'top_15_carts_orders_v{VER}_{k}.pqt')))
    top_20_buy2buy = pqt_to_dict(pd.read_parquet(matrices_dir + f'top_15_buy2buy_v{VER}_0.pqt'))
    print('CO-VISITATION MATRICES is loaded')

    test_df = pd.read_csv(arg.flattentest_path)
    print('flattened testing data has been loaded')

    print('start getting candidates')
    candidates = test_df.sort_values(["session", "ts"]).groupby(["session"]).apply(
        lambda x: obtain_candidates(x)
    )
    candidates = candidates.to_dict()
    s_str = json.dumps(candidates)
    with open(arg.co_vi_candidates_path, 'w') as json_file:
         json_file.write(s_str)
    print('co-visitation matrix canidates has been created and stored in' + arg.co_vi_candidates_path)
