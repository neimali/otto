from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import recall_score
from wordVec import WordToVec
from dataLoad import DataLoad
import pandas as pd
import numpy as np
from tqdm import tqdm

class Clustering:
    def __init__(self, model, art_vectors):
        self.model_name = model
        self.art_vec_dict = art_vectors
        self.vectors = list(self.art_vec_dict.values())
        print(self.vectors)
        self.article = list(self.art_vec_dict.keys())
        print(self.article)

    def kmCluster(self, k):
        # km = KMeans(n_clusters=k, random_state=24)
        km = MiniBatchKMeans(n_clusters=k, batch_size=4096,random_state=24)
        print('clustering on k=' + str(k) + ' is started')
        km.fit(self.vectors)
        print('clustering finished')
        label = km.labels_
        res = pd.DataFrame({'aid': self.article, 'label': label})
        return res

    def articles_rank_by_label(self, data, cluster_res):
        res = {}
        print('start calculating rank by label')
        article_rank = data.aid.value_counts()
        article_rank = pd.DataFrame({'aid': article_rank.index, 'counts': article_rank.values})
        print(article_rank)
        article_rank = pd.merge(article_rank, cluster_res)
        print(article_rank)
        # for l in cluster_res['label'].unique():
        #     articles = cluster_res.loc[cluster_res.label == l, 'aid'].unique()
        #     res[l] = article_rank.loc[article_rank.index.isin(articles)].index.tolist()
        for l in tqdm(article_rank.groupby('label')):
            res[l[0]] = l[1].aid.tolist()
        return res

    def get_candidates(self, test, clusters, rank):
        # print('start generating candidates from clustering')
        # can = {}
        # test_cluster = pd.merge(test, clusters, how='left', on='aid')
        # for s in tqdm(test_cluster.groupby('session')):
        #     session_label = s[1].label.mode().iloc[0]
        #     candidates = rank[session_label]
        #     can[s[0]] = candidates
        # return can

        # get distribution(percentage) of label
        can = {}
        test_cluster = pd.merge(test, clusters, how='left', on='aid')
        for s in tqdm(test_cluster.groupby('session')):
            candidates_s = []
            count = s[1].label.value_counts()
            ratio = count.div(count.sum())
            for i in ratio.index:
                num = int(np.ceil(ratio[i]*50))
                candidates_s.extend(rank[i][:num])
            can[s[0]] = candidates_s

        return can

    def validation(self, test, test_res, clusters, rank):
        # get label for each session(majority vote)
        # calculate the numbers of articles to be filled
        # get top k article in article rank
        # return score of prediction
        def metric(pred, test):
            intersect = list(set(pred).intersection(set(test)))
            return len(intersect)/min(20,len(test_res))

        pred = {}
        scores = []
        # clusters_df = pd.DataFrame(clusters).set_index('aid')
        # test_cluster = test.join(clusters_df, on='aid', how='left')
        test_cluster = pd.merge(test, clusters, how='left', on='aid')
        # for s in tqdm(test_cluster.session.unique()):
        #     session_label = test_cluster.loc[test_cluster.session == s, 'label'].value_counts().index[0]
        #     candidates = rank[session_label]
        #     pred[s] = candidates[:20-len(test.loc[test.session == s])]
        #     score = metric(pred[s], test_res.loc[test_res.session == 's', 'aid'].tolist())
        #     scores.append(score)
        print('validation started')
        for s, res in tqdm(zip(test_cluster.groupby('session'), test_res.groupby('session'))):
            session_label = s[1].label.mode().iloc[0]
            candidates = rank[session_label]
            pred[s[0]] = candidates[:20-len(s[1])]
            score = metric(pred[s[0]], res[1].aid)
            scores.append(score)

        return sum(scores)/len(scores)

    def fine_tune(self, k_values, train, test, test_res):
        # select proper k for k_means
        best_k = 0
        best_score = -1
        for k in k_values:
            clusters = self.kmCluster(k)
            rank = self.articles_rank_by_label(train, clusters)
            score = self.validation(test, test_res, clusters, rank)
            if score > best_score:
                best_k = k
                best_score = score
        print('best k value is ' + str(best_k) + ' and the score is ' + str(best_score) + ' best_k has been return')
        return best_k

if __name__ == '__main__':
    dl = DataLoad('D:\OTTO\Data\\train.jsonl')
    dl.sample_data(10)
    df = dl.to_dataframe(dl.sample)
    train, test, test_res = dl.data_split(df)
    wv = WordToVec(df)
    s = wv.time_session(2)
    s = list(s.values())
    vector = wv.train(s)
    c = Clustering('km', vector)
    cres = c.kmCluster(2)
    rank = c.articles_rank_by_label(df, cres)
    candidates = c.get_candidates(test, cres, rank)
    print(candidates)
    # score = c.validation(test, test_res, cres, rank)
    # k = c.fine_tune([2,3], df, test, test_res)




