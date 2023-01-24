import pandas as pd

class Feature:
    def __init__(self, data):
        self.data = data

    # def down_cast(self):
    #     self.data['session'] = pd.to_numeric(self.data['session'], downcast='unsigned')
    #     self.data['ts'] = pd.to_numeric(self.data['ts']/1000, downcast='unsigned')

    # def target(self):
    #     self.data['click'] = self.data['type'].apply(lambda x: 1 if x == 'clicks' else 0)
    #     self.data['click'] = pd.to_numeric(self.data['click'], downcast='unsigned')
    #     self.data['cart'] = self.data['type'].apply(lambda x: 1 if x == 'carts' else 0)
    #     self.data['cart'] = pd.to_numeric(self.data['cart'], downcast='unsigned')
    #     self.data['order'] = self.data['type'].apply(lambda x: 1 if x == 'orders' else 0)
    #     self.data['order'] = pd.to_numeric(self.data['order'], downcast='unsigned')
    #     print('target generated')

    def drop_type(self):
        self.data.drop(['type','ts'], axis=1)
        print('type and ts droped')

    def user_action_count(self):
        print('user_action_count feature start generating')
        self.data['cl_cnt'] = self.data[self.data['type'] == 0].groupby('session')['type'].transform('count')
        self.data['cl_cnt'] = self.data.groupby('session')['cl_cnt'].transform(lambda x: x.fillna(x.min()))
        self.data['cl_cnt'] = self.data.groupby('session')['cl_cnt'].transform(lambda x: x.fillna(0)).astype('Int8')

        self.data['ca_cnt'] = self.data[self.data['type'] == 1].groupby('session')['type'].transform('count')
        self.data['ca_cnt'] = self.data.groupby('session')['cl_cnt'].transform(lambda x: x.fillna(x.min()))
        self.data['ca_cnt'] = self.data.groupby('session')['cl_cnt'].transform(lambda x: x.fillna(0)).astype('Int8')

        self.data['or_cnt'] = self.data[self.data['type'] == 2].groupby('session')['type'].transform('count')
        self.data['or_cnt'] = self.data.groupby('session')['or_cnt'].transform(lambda x: x.fillna(x.min()))
        self.data['or_cnt'] = self.data.groupby('session')['or_cnt'].transform(lambda x: x.fillna(0)).astype('Int8')

        print('user_action_count feature complete')

    def user_action_ratio(self):
        self.data['cl_ca_ratio'] = (self.data['ca_cnt'] / self.data['cl_cnt']).astype('float32')
        self.data['cl_or_ratio'] = (self.data['or_cnt'] / self.data['cl_cnt']).astype('float32')
        self.data['ca_or_ratio'] = (self.data['or_cnt'] / self.data['ca_cnt']).astype('float32')

        self.data['cl_ca_ratio'] = self.data['cl_ca_ratio'].fillna(0)
        self.data['cl_or_ratio'] = self.data['cl_or_ratio'].fillna(0)
        self.data['ca_or_ratio'] = self.data['ca_or_ratio'].fillna(0)

        print('user_action_ratio feature complete')

    def user_time_features(self):
        self.data['ss_ts_max'] = self.data.groupby('session')['ts'].transform('max')
        self.data['ss_ts_min'] = self.data.groupby('session')['ts'].transform('min')
        self.data['ss_ts_mean'] = self.data.groupby('session')['ts'].transform('mean')
        self.data['ss_ts_mean'] = self.data['ss_ts_mean'].astype('int32')
        print('user_time_features feature complete')

    def session_count(self):
        self.data['sess_cnt'] = self.data.groupby('session')['session'].transform('count')
        self.data['sess_cnt'] = self.data['sess_cnt'].astype('int16')
        print('session_count feature complete')

    def aid_count(self):
        self.data['aid_cnt'] = self.data.groupby('aid')['aid'].transform('count')
        self.data['aid_cnt'] = self.data['aid_cnt'].astype('int16')
        print('aid_cnt feature complete')

    def aid_action_count(self):
        self.data['aid_cl_cnt'] = self.data[self.data['type'] == 'clicks'].groupby('aid')['type'].transform('count')
        self.data['aid_ca_cnt'] = self.data[self.data['type'] == 'carts'].groupby('aid')['type'].transform('count')
        self.data['aid_or_cnt'] = self.data[self.data['type'] == 'orders'].groupby('aid')['type'].transform('count')

        self.data['aid_cl_cnt'] = self.data.groupby('aid')['aid_cl_cnt'].transform(lambda x: x.fillna(x.min()))
        self.data['aid_cl_cnt'] = self.data.groupby('aid')['aid_cl_cnt'].transform(lambda x: x.fillna(0))
        self.data['aid_ca_cnt'] = self.data.groupby('aid')['aid_ca_cnt'].transform(lambda x: x.fillna(x.min()))
        self.data['aid_ca_cnt'] = self.data.groupby('aid')['aid_ca_cnt'].transform(lambda x: x.fillna(0))
        self.data['aid_or_cnt'] = self.data.groupby('aid')['aid_or_cnt'].transform(lambda x: x.fillna(x.min()))
        self.data['aid_or_cnt'] = self.data.groupby('aid')['aid_or_cnt'].transform(lambda x: x.fillna(0))
        print('aid_action_count feature complete')

    def aid_action_ration(self):
        self.data['aid_cl_ca_ratio'] = self.data['aid_ca_cnt'] / self.data['aid_cl_cnt']
        self.data['aid_cl_or_ratio'] = self.data['aid_or_cnt'] / self.data['aid_cl_cnt']
        self.data['aid_ca_or_ratio'] = self.data['aid_or_cnt'] / self.data['aid_ca_cnt']

        self.data['aid_cl_ca_ratio'] = self.data['aid_cl_ca_ratio'].fillna(0)
        self.data['aid_cl_or_ratio'] = self.data['aid_cl_or_ratio'].fillna(0)
        self.data['aid_ca_or_ratio'] = self.data['aid_ca_or_ratio'].fillna(0)
        print('aid_action_ration feature complete')

    def aid_time_features(self):
        self.data['aid_ts_max'] = self.data.groupby('aid')['ts'].transform('max')
        self.data['aid_ts_min'] = self.data.groupby('aid')['ts'].transform('min')
        self.data['aid_ts_mean'] = self.data.groupby('aid')['ts'].transform('mean')
        self.data['aid_ts_mean'] = self.data['aid_ts_mean'].astype('int32')
        print('aid_time_features feature complete')
