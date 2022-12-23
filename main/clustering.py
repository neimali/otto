from sklearn.cluster import KMeans
from sklearn.metrics import recall_score
from wordVec import WordToVec
from dataLoad import DataLoad
import pandas as pd

class Clustering:
    def __init__(self, model, art_vectors):
        self.model_name = model
        self.art_vec_dict = art_vectors
        self.vectors = list(self.art_vec_dict.values())
        self.article = list(self.art_vec_dict.keys())

    def kmCluster(self, k):
        km = KMeans(n_clusters=k, random_state=24)
        km.fit(self.vectors)
        label = km.labels_
        res = pd.DataFrame({'aid': self.article, 'label': label})
        return res

    def articles_rank_by_label(self, data, cluster_res):
        res = {}
        article_rank = data.aid.value_counts()
        for l in cluster_res['label'].unique():
            articles = cluster_res.loc[cluster_res.label == l, 'aid'].unique()
            res[l] = article_rank.loc[article_rank.index.isin(articles)].index.tolist()
        return res

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
        clusters_df = pd.DataFrame(clusters).set_index('aid')
        test_cluster = test.join(clusters_df, on='aid', how='left')
        for s in test_cluster.session.unique():
            session_label = test_cluster.loc[test_cluster.session == s, 'label'].value_counts().index[0]
            candidates = rank[session_label]
            pred[s] = candidates[:20-len(test.loc[test.session == s])]
            score = metric(pred[s], test_res.loc[test_res.session == 's', 'aid'].tolist())
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





