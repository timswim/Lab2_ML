import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import linear_model
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
import timeit

data_source = "my_data.csv"
data = pd.read_csv(data_source)
data = data.dropna(axis='index', how='any', subset=['Critic_Score'])
data = data.dropna(axis='index', how='any', subset=['User_Count'])

# Преобразование данных
data = data.drop(data[data["Global_Sales"] > 5].index)
data = data.drop(data[data["Global_Sales"] < 0.1].index)
print(len(data))
data["Critics"] = data["Critic_Score"]*data["Critic_Count"]

X = data[["User_Count", "Critics", "Pub_prod", "Dev_prod", "Pub_score", "Dev_score"]]

y = data["JP_Sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

N_train, _ = X_train.shape
N_test,  _ = X_test.shape
print(N_train, N_test)

# Создаем экземпляр регрессора

reg = linear_model.LinearRegression()
a = timeit.default_timer()
# Обучение регрессора
reg.fit(X_train, y_train)
print("Время обучения: {}".format(timeit.default_timer()-a))

a = timeit.default_timer()
time = 0
for num in range(100):
    KNN_prediction = reg.predict(X_test)
    time += timeit.default_timer()-a
    print(num, "--", timeit.default_timer() - a)
    a = timeit.default_timer()

time = time / 100
print("Время работы: {}".format(time))


predict = reg.predict(X_test)
print("Показатель отклонения: {}".format(explained_variance_score(y_test, predict)))
print("Абсолютное значение ошибки: {}".format(mean_absolute_error(y_test, predict)))
print("Кросс-валидация ----------------")
scores = cross_val_score(reg, X, y, cv=5, scoring='explained_variance')
print("Точнсть кросс-валидациz показателя отклонения: {}".format(sum(scores)/len(scores)))
print(scores)
scores = cross_val_score(reg, X, y, cv=5, scoring='neg_mean_absolute_error')
print("Точнсть кросс-валидация абсолютного значения ошибки: {}".format(-sum(scores)/len(scores)))
print(scores)