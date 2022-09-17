import pandas as pd

def dict_apply(x, map_dict):
        if x in map_dict:
            return map_dict[x]
        else:
            return x

    

def gender_prep(*dfs):
    gender_normalize_map = {'муж' : 'Муж', 
                        'жен' : 'Жен'
                            }
    for df in dfs:
        df['Пол'] = df['Пол'].apply(dict_apply, map_dict = gender_normalize_map)
        df['Пол'].fillna('nanan', inplace = True)


def lang_prep(*dfs):
    lang_normalize_map = {'Англиийский': 'Английский язык', 
                            'Иностранный язык (Английский)' : 'Английский язык', 
                            'Иностранный язык (Немецкий)' : 'Немецкий язык', 
                            'Английский, немецкий языки' : 'Немецкий язык', 
                            'Англиийский' : 'Английский язык'
                            }
    for df in dfs:
        df['Изучаемый_Язык'] = df['Изучаемый_Язык'].apply(dict_apply, map_dict = lang_normalize_map)
        df['Изучаемый_Язык'].fillna('nanan', inplace = True)

def pay_prep(*dfs):
    for df in dfs:
        df.drop(columns = 'Пособие', inplace = True)


def country_prep(*dfs):
    country_normalize_map = {'Кыргызстан' : 'Киргизия', 
                         'Кыргызская Республика' : 'Киргизия',
                         'Кыргызия' : 'Киргизия',
                         'Республика Казахстан' : 'Казахстан', 
                         'Казахстан Респ' : 'Казахстан',
                         'Казахстан респ' : 'Казахстан', 
                         'Казахстан ВКО' : 'Казахстан', 
                         'КАЗАХСТАН' : 'Казахстан', 
                         'Таджикистан Респ' : 'Таджикистан', 
                         'Республика Таджикистан' : 'Таджикистан',
                         'КИТАЙ' : 'Китай', 
                         'Росссия' : 'Россия', 
                         'РОССИЯ' : 'Россия'
                         }
    for df in dfs:
        df['Страна_ПП'] = df['Страна_ПП'].apply(dict_apply, map_dict = country_normalize_map)
        df['Страна_ПП'].fillna('nanan', inplace = True)

def residen_prep(*dfs):
    for df in dfs:
        df['Общежитие'].fillna(-1, inplace = True)

def cust_prep(*dfs):
    for df in dfs:
        df.drop(columns = 'Опекунство', inplace = True)


def vill_prep(*dfs):
    for df in dfs:
        df['Село'].fillna(-1, inplace = True)


def fore_prep(*dfs):
    for df in dfs:
        df['Иностранец'].fillna(-1, inplace = True)

def bdate_prep(*dfs):
    for df in dfs:
        df['Дата_Рождения'] = pd.to_datetime(df['Дата_Рождения']).apply(lambda x : x.year)

def school_prep(*dfs):
    for df in dfs:
        df['Уч_Заведение'].fillna('nanan', inplace = True)


def preprocess(train, test):
    gender_prep(train, test)
    lang_prep(train, test)
    pay_prep(train, test)
    country_prep(train, test)
    residen_prep(train, test)
    cust_prep(train, test)
    vill_prep(train, test)
    fore_prep(train, test)
    bdate_prep(train, test)
    school_prep(train, test)

    train.drop(train[train['Год_Поступления'] > 2022].index, inplace = True)
    train.reset_index(drop = True, inplace = True)

    drop_columns = ['Где_Находится_УЗ', 'Регион_ПП', 'Город_ПП', 'Страна_Родители']
    train.drop(columns = drop_columns, inplace = True)
    test.drop(columns = drop_columns, inplace = True)
