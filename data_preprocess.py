import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import linear_model
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
import timeit

data_source = "Video_Games_Sales_as_at_22_Dec_2016.csv"
data = pd.read_csv(data_source)


# Сначала считаем издателей
data = data.dropna(axis='index', how='any', subset=['Publisher'])
print(len(data))
d = pd.Series(data['Publisher']).value_counts()
print(len(d))
data['Pub_prod'] = data.apply(lambda row: d[row['Publisher']], axis=1)

# Затем считаем разработчиков
data = data.dropna(axis='index', how='any', subset=['Developer'])
print(len(data))
d = pd.Series(data['Developer']).value_counts()
print(len(d))
data['Dev_prod'] = data.apply(lambda row: d[row['Developer']], axis=1)


# Высчитываем среднюю оценку издателей
def take_score(score_listm, id):
    return score_list[id]

data = data.dropna(axis='index', how='any', subset=['Critic_Score'])
# переведем текстовые признаки в индексы
data['Developer_id'] = pd.factorize(data.Developer)[0]
data['Publisher_id'] = pd.factorize(data.Publisher)[0]
print(len(data))
Pub_ser = pd.Series(data['Publisher_id']).value_counts()
Dev_ser = pd.Series(data['Developer_id']).value_counts()

score_list = []
for index in range(len(Pub_ser)):
    print(index)
    data_score = data[data.Publisher_id == index]
    sum_score = sum(data_score['Critic_Score'])
    score_list.append(sum_score/Pub_ser[index])
data['Pub_score'] = data.apply(lambda row: take_score(score_list, row['Publisher_id']), axis=1)

score_list = []
for index in range(len(Dev_ser)):
    print(index)
    data_score = data[data.Developer_id == index]
    sum_score = sum(data_score['Critic_Score'])
    score_list.append(sum_score/Dev_ser[index])
data['Dev_score'] = data.apply(lambda row: take_score(score_list, row['Developer_id']), axis=1)

data.to_csv("my_data.csv")
print(d)
