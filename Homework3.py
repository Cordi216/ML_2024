import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

df = pd.read_csv('bikes_rent.csv')


def secondTask():
    data = np.array(df["weathersit"]).reshape((-1, 1))
    model = LinearRegression().fit(data, np.array(df["cnt"]))
    predictions = model.predict(data)
    plt.scatter(data, np.array(df["cnt"]), color='black', alpha=0.5)
    plt.plot(data, predictions, color='blue', linewidth=3)
    plt.xticks(())
    plt.yticks(())
    plt.title("secondTask")
    plt.xlabel("weather")
    plt.ylabel("cnt")
    plt.show()

def thirdTask(data, season, yr, mnth, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed_mph, windspeed_ms):
    x = data.drop(columns=['cnt'])
    y = data['cnt']
    model = LinearRegression().fit(x, y)
    input_data = pd.DataFrame({
        'season': [season],
        'yr': [yr],
        'mnth': [mnth],
        'holiday': [holiday],
        'weekday': [weekday],
        'workingday': [workingday],
        'weathersit': [weathersit],
        'temp': [temp],
        'atemp': [atemp],
        'hum': [hum],
        'windspeed(mph)': [windspeed_mph],
        'windspeed(ms)': [windspeed_ms]
    })
    print(model.predict(input_data)[0]," - прогноз")

def fourthTask(data):
    pca = PCA(n_components=1)
    reduced_data = pca.fit_transform(data.drop(columns=['cnt']))
    plt.scatter(df["cnt"],reduced_data)
    plt.title("thirdTask")
    plt.xlabel("cnt")
    plt.ylabel("decomposed_data")
    plt.show()
def fifthTask(data):
    model = data.drop(columns=['cnt'])
    x_train, x_test, y_train, y_test = train_test_split(model.values, df["cnt"], test_size=0.2)
    lasso = Lasso(alpha=0.1)
    lasso.fit(x_train, y_train)
    coefficients = pd.DataFrame({'feature': model.columns, 'coefficient': lasso.coef_})
    most_influential_feature = coefficients.loc[coefficients['coefficient'].idxmax()]
    return most_influential_feature['feature']

secondTask()
thirdTask(df,1,0,1,0,2,1,1,8.2,10.6061,59.0435,10.739832,4.80099776486)
fourthTask(df)
print(fifthTask(df)," - оказывает наибольшее влияние на cnt")
