import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier

data_source = "my_data.csv"
data = pd.read_csv(data_source)
#data = data.dropna(axis='index', how='any', subset=['Critic_Score'])
#data = data.dropna(axis='index', how='any', subset=['User_Count'])
#data = data.drop(data[data["Global_Sales"] > 5].index)
#data = data.drop(data[data["Global_Sales"] < 0.1].index)

def task_1(): # Сколько записей в базе

    print(data.info(), end="\n-------------\n")
    print(len(data))


def task_2(data): # Построение гистограмм
    data.drop(['Name', 'Platform', 'Genre', 'Publisher', 'User_Score', 'Developer', 'Rating'], axis='columns', inplace=True)  # Удаление проблемных столбцов
    groups = list(data.columns)
    for group in groups:
        plt.figure(num=group)
        plt.hist(x=data[group], bins=None)
        plt.xlabel('value')
        plt.ylabel('quantity')
    plt.show()



def corr(data): # task_5 Матрица корреляции
    data.drop(['Name'], axis='columns', inplace=True)  # Удаление проблемных столбцов
    data['Platform_id'] = pd.factorize(data.Platform)[0]
    data['Genre_id'] = pd.factorize(data.Genre)[0]
    data['Publisher_id'] = pd.factorize(data.Publisher)[0]
    data['User_Score_id'] = pd.factorize(data.User_Score)[0]
    data['Developer_id'] = pd.factorize(data.Developer)[0]
    data['Rating_id'] = pd.factorize(data.Rating)[0]
    corr = data.corr()
    sns.heatmap(corr,annot=True,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    plt.show()

def ExtreTrees(): # расчет инормативности признаков
    y = data["weather"] # вписываем тот признак, который хотим предсказать
    X = data[["par1", "par2", "par4", "par5", "par7", "par8"]]
    model = ExtraTreesClassifier()
    model.fit(X, y)
    # display the relative importance of each attribute
    print(model.feature_importances_)

corr(data)
#plt.figure(num='longitude')
#plt.hist(x=data['longitude'], bins=None)
#plt.show()
