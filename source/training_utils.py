import numpy as np
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score

def catboost_cv_predict(model, X, Y, cat_cols, transformers = None, n_splits = 10, early_stopping = False):
    kf = StratifiedKFold(n_splits = n_splits, shuffle = True)
    pred = np.zeros((Y.size, 3))
    for tr_id, ts_id in kf.split(X, Y):
        X_tr, Y_tr = X.iloc[tr_id], Y.iloc[tr_id]
        X_ts, Y_ts = X.iloc[ts_id], Y.iloc[ts_id]

        if transformers:
            for transf in transformers:
                X_tr = transf.fit_transform(X_tr, Y_tr.values)
                X_ts = transf.transform(X_ts)

        tr_pool = Pool(X_tr, Y_tr, cat_features=cat_cols)
        ts_pool = Pool(X_ts, Y_ts, cat_features=cat_cols)

        if early_stopping:
            model.fit(tr_pool, eval_set = ts_pool, 
                      verbose = False, early_stopping_rounds = 100)
        else:
            model.fit(tr_pool, verbose = False)

        pred[ts_id] = model.predict_proba(ts_pool)


    return pred

def cross_val_catboost(catboost_config, X, Y, cat_cols, transformers = None, n_splits = 10, average = 'macro', verbose = False):
    kf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state=159)
    train_scores = []
    test_scores = []    
    models = []
    i = 0
    for tr_id, ts_id in kf.split(X, Y):
        model = CatBoostClassifier(**catboost_config)
        X_tr, Y_tr = X.iloc[tr_id], Y.iloc[tr_id]
        X_ts, Y_ts = X.iloc[ts_id], Y.iloc[ts_id]

        if transformers:
            for transf in transformers:
                X_tr = transf.fit_transform(X_tr, Y_tr.values)
                X_ts = transf.transform(X_ts)

        tr_pool = Pool(X_tr, Y_tr, cat_features=cat_cols)
        ts_pool = Pool(X_ts, Y_ts, cat_features=cat_cols)

        model.fit(tr_pool, eval_set = ts_pool, verbose = verbose)

        pr_tr = np.squeeze(model.predict(tr_pool))
        pr_ts = np.squeeze(model.predict(ts_pool))

        train_scores.append(f1_score(Y.iloc[tr_id], pr_tr, average=average))
        test_scores.append(f1_score(Y.iloc[ts_id], pr_ts, average=average))
        models.append(model.copy())
        i += 1
        if i == 100:
            break

    return {'train' : train_scores, 'test' : test_scores, 'models' : models}