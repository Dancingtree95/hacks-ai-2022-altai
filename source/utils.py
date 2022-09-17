import numpy as np

def cat_inspect(train, test, columns):
    result = {}
    for col in columns:
        train_set = set(train[col].unique())
        test_set = set(test[col].unique())
        res = {}
        res['train_card'] = len(train_set)
        res['test_card'] = len(test_set)
        res['common'] = len(train_set.intersection(test_set))
        result[col] = res
    return result




def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]



def equiprob_bin_edges(array, n_bins):
    h = 1 / n_bins
    return np.quantile(array, [h * i for i in range(n_bins + 1)])

    

