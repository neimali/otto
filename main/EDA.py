import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from dataLoad import DataLoad

#timestamp count each session
#session duration
#iterm co-occurrence

class EDA:

    def __init__(self):
        self.dl = DataLoad()
        self.dl.sample_data(20000)
        self.df = self.dl.to_dataframe()

    # count over all session
    def count(self, variable, isprint=False):
        if variable == 'session':
            c = len(self.df.index.unique())
        else:
            c = len(self.df[variable].unique())
        if isprint:
            print('the total number of ' + variable + ' in ' + str(self.dl.sample_size) + ' samples is ' + str(c))
        else:
            return c

    def check_session(self, session):
        df = self.df.loc[self.df.index == session]
        print(df)

    # count the number of variables in each session
    def count_session(self, session, variable, isprint=False):
        df = self.df.loc[self.df.index == session]
        c = len(df[variable].unique())
        if isprint:
            print('the total number of ' + variable + ' in session ' + str(session) + ' is ' + str(c))
        else:
            return c

    # calculate the duration of each session
    def session_duration(self, session, unit='minute', isprint=False):
        df = self.df.loc[self.df.index == session]
        duration = df.ts.max() - df.ts.min()
        if unit == 'minute':
            duration *= 1.66667e-5
        elif unit == 'hour':
            duration *= 2.7778e-7
        elif unit == 'second':
            duration *= 0.001
        duration = round(duration, 2)
        if isprint:
            print('the duration of session ' + str(session) + ' is ' + str(duration) + ' ' + unit)
        else:
            return duration

    # given event duration create co_occurance matrix for articles
    def co_occurance(self, session, duration):
        df = self.df.loc[self.df.index == session]
        df['hour'] = df['ts'].apply(lambda x: round(x*2.7778e-7, 2))
        df['session'] = df.index
        df.index = range(len(df))

        aid_count = self.count_session(session, 'aid')
        aids = df.aid.unique().tolist()
        matrix = [[0]*aid_count for _ in range(aid_count)]
        for i in range(len(df)):
            time = df.hour[i]
            aid = df.aid[i]
            ind = aids.index(aid)
            co_oc = df.aid.loc[(df.hour < time+duration) & (df.hour > time-duration)]
            for co_ocArticle in co_oc:
                matrix[ind][aids.index(co_ocArticle)] += 1

        return matrix

