import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict

from utils import equiprob_bin_edges

class NB_text_to_numeric():
    def __init__(self, analyzer='char_wb', ngram_range = (4,4), min_df = 5):
        vectorizer = CountVectorizer(analyzer=analyzer, 
                                     ngram_range = ngram_range, 
                                     min_df = min_df)
        model = BernoulliNB(alpha = 1)
        self.pipe = make_pipeline(vectorizer, model)  

    def fit_transform(self, X, Y, cv = 100):
        train_feat = cross_val_predict(self.pipe, X, Y, cv = 100, 
                                       method = 'predict_proba')
        self.pipe.fit(X, Y)

        return train_feat

    def transform(self, X):
        return self.pipe.predict_proba(X)


def generate_place_of_study_nb(train, test):
    clean_pattern = re.compile(r'[\'"\.]')

    sen_train = train['Уч_Заведение'] \
                        .fillna('') \
                        .apply(lambda x : clean_pattern.sub('', x.lower())) \
                        .tolist()

    sen_test = test['Уч_Заведение'] \
                        .fillna('') \
                        .apply(lambda x : clean_pattern.sub('', x.lower())) \
                        .tolist()
    Y = train['Статус']
    
    transfomer = NB_text_to_numeric()
    train_feat = transfomer.fit_transform(sen_train, Y)
    test_feat = transfomer.transform(sen_test)

    train[[f'NaiveBayes_{c}' for c in [-1, 3, 4]]] = pd.DataFrame(train_feat)
    test[[f'NaiveBayes_{c}' for c in [-1, 3, 4]]] = pd.DataFrame(test_feat)
    train.drop(columns =  'Уч_Заведение', inplace = True)
    test.drop(columns =  'Уч_Заведение', inplace = True)

    return train, test

def generate_dev_from_group_average_birth(train, test):
    ft = ['Дата_Рождения', 'Код_группы']
    group_means = pd.concat([train[ft], test[ft]]).groupby('Код_группы').mean().reset_index()
    train['Разн_со_средн_по_группе'] = train['Дата_Рождения'] - pd.merge(train['Код_группы'], group_means, how = 'left', on = 'Код_группы')['Дата_Рождения']
    test['Разн_со_средн_по_группе'] = test['Дата_Рождения'] - pd.merge(test['Код_группы'], group_means, how = 'left', on = 'Код_группы')['Дата_Рождения']
    return train, test

def generate_age(train, test):
    train['Возраст_поступления'] = train['Год_Поступления'] - train['Дата_Рождения']
    test['Возраст_поступления'] = test['Год_Поступления'] - test['Дата_Рождения']
    return train, test

def generate_gap_year_dur(train, test):
    train['Продолж_между_учебо'] = train['Год_Поступления'] - train['Год_Окончания_УЗ']
    train['Продолж_между_учебо'].fillna(-100, inplace = True)
    train.drop(columns = 'Год_Окончания_УЗ', inplace = True)

    test['Продолж_между_учебо'] = test['Год_Поступления'] - test['Год_Окончания_УЗ']
    test['Продолж_между_учебо'].fillna(-100, inplace = True)
    test.drop(columns = 'Год_Окончания_УЗ', inplace = True)
    return train, test 

def generate_is_school_certificate(train, test):
    train['Аттестат'] = ((train['СрБаллАттестата'] <= 5) & (train['СрБаллАттестата'] >= 0)).astype('int8')
    test['Аттестат'] = ((test['СрБаллАттестата'] <= 5) & (test['СрБаллАттестата'] >= 0)).astype('int8')
    return train, test


def generate_relative_rating(train, test, rel_col, fet_name, binarize, eps = 1e-4):
    abnorm_score_train_idx = train[train['СрБаллАттестата'] > 100].index
    abnorm_score_test_idx = test[test['СрБаллАттестата'] > 100].index

    ft = [rel_col, 'СрБаллАттестата', 'Аттестат']
    stats= pd.concat([
                    train[ft].drop(abnorm_score_train_idx), 
                    test[ft].drop(abnorm_score_test_idx)
                    ]).groupby([rel_col, 'Аттестат'])['СрБаллАттестата'].agg([np.mean, np.std]).fillna(0)

    train = pd.merge(train, stats.reset_index(), how = 'left', on = [rel_col, 'Аттестат'])
    test = pd.merge(test, stats.reset_index(), how = 'left', on = [rel_col, 'Аттестат'])

    train[fet_name] = (train['СрБаллАттестата'] - train['mean']) / (train['std'] + eps)
    test[fet_name] = (test['СрБаллАттестата'] - test['mean']) / (test['std'] + eps)

    train.drop(columns = ['mean', 'std'], inplace = True)
    test.drop(columns = ['mean', 'std'], inplace = True)
    
    if binarize:
        f_values = pd.concat([train.drop(abnorm_score_train_idx), test.drop(abnorm_score_test_idx)])[fet_name]
        bins = equiprob_bin_edges(f_values, 10)
        train[fet_name] = np.digitize(train[fet_name], bins = bins)
        test[fet_name] = np.digitize(test[fet_name], bins = bins)
    return train, test

def generate_group_freq(train, test):
    group_filling = pd.concat([train, test]).groupby('Код_группы')['ID'].count().rename('Сколько_еще_в_группе')
    train = pd.merge(train, group_filling, how = 'left', on = 'Код_группы')
    test = pd.merge(test, group_filling, how = 'left', on = 'Код_группы')
    return train, test

def generate_in_year_diff(train, test):
    def f(x):
        values, counts = np.unique(x, return_counts = True)
        return values[counts == counts.max()].min()
    group_years = pd.concat([train, test]).groupby('Код_группы')['Год_Поступления'] \
        .agg(f) \
        .rename('year_mode') \
        .reset_index()
    train['Разница_года_поступления'] = train['Год_Поступления'] - pd.merge(train, group_years, how = 'left', on = 'Код_группы')['year_mode']
    test['Разница_года_поступления'] = test['Год_Поступления'] - pd.merge(test, group_years, how = 'left', on = 'Код_группы')['year_mode']
    return train, test







