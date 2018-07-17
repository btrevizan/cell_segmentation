from pandas import read_csv
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


def train():
    x, y = __load()

    models = {
        "knn": KNeighborsClassifier()  # had the best accuracy for this data set
    }

    for model_name in models:
        model = models[model_name]
        scores = cross_val_score(model, X, y, cv=10)

        print("%s Accuracy: %0.2f (+/- %0.2f)" % (model_name,
                                                  scores.mean(),
                                                  scores.std() * 2))

        model.fit(x, y)
        joblib.dump(model, 'data/model/{}.pkl'.format(model_name))


def __load():
    data = read_csv('data/dataset.csv', header=None)
    data = data.values

    x = data[:, :-1]
    y = data[:, -1]

    return x, y
