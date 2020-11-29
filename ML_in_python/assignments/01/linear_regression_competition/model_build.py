#!/usr/bin/env python
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

# So we can use the HGBR model... experimental feature huh :D
from sklearn.experimental import enable_hist_gradient_boosting

from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="linear_regression_competition.model", type=str, help="Model path")
parser.add_argument("--seed", default=42, type=int, help="Random seed")

def pick_algorithm():
    # Choose the right model
    pipelines = []
    pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',LinearRegression())])))
    pipelines.append(('ScaledSGDR', Pipeline([('Scaler', StandardScaler()),('SGDR',SGDRegressor())])))
    pipelines.append(('ScaledHR', Pipeline([('Scaler', StandardScaler()),('HR',HuberRegressor())])))
    pipelines.append(('ScaledLARS', Pipeline([('Scaler', StandardScaler()),('LARS',Lars())])))
    pipelines.append(('ScaledLL', Pipeline([('Scaler', StandardScaler()),('LL',LassoLars())])))
    pipelines.append(('ScaledORP', Pipeline([('Scaler', StandardScaler()),('ORP',OrthogonalMatchingPursuit())])))
    pipelines.append(('ScaledPAR', Pipeline([('Scaler', StandardScaler()),('PAR',PassiveAggressiveRegressor())])))
    pipelines.append(('ScaledBR', Pipeline([('Scaler', StandardScaler()),('BR',BayesianRidge())])))
    pipelines.append(('ScaledRIDGE', Pipeline([('Scaler', StandardScaler()),('RIDGE', Ridge())])))
    pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
    pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
    pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
    pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
    pipelines.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))
    pipelines.append(('ScaledABR', Pipeline([('Scaler', StandardScaler()),('ABR', AdaBoostRegressor())])))
    pipelines.append(('ScaledBAR', Pipeline([('Scaler', StandardScaler()),('BAR', BaggingRegressor())])))
    pipelines.append(('ScaledHGBR', Pipeline([('Scaler', StandardScaler()),('HGBR', HistGradientBoostingRegressor())])))

    results = []
    names = []
    for name, model in pipelines:
        kfold = KFold(n_splits=10, random_state=args.seed)
        cv_results = cross_val_score(model, xtrain, ytrain, cv=kfold, scoring='neg_mean_squared_error')
        results.append(cv_results)
        names.append(name)
        msg = f"{name}: {cv_results.mean()}"
        print(msg)

def twist_params():
    # Choose the right estimator
    scaler = StandardScaler().fit(xtrain)
    rescaledX = scaler.transform(xtrain)
    param_grid = dict(learning_rate=np.arange(.001,.8,.1))

    model = HistGradientBoostingRegressor(random_state=args.seed)
    kfold = KFold(n_splits=10, random_state=args.seed)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kfold)
    grid_result = grid.fit(rescaledX, ytrain)

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print(f"{mean} with estimator: {param}")

    print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

def train_model():
    # Train the model
    scaler = StandardScaler().fit(xtrain)
    rescaled_xtrain = scaler.transform(xtrain)
    model = GradientBoostingRegressor(random_state=args.seed,alpha=0.69, n_estimators=550, max_depth=6, max_features=1.0, min_samples_leaf=3, learning_rate=0.1)
    #model = HistGradientBoostingRegressor(random_state=args.seed, max_iter=200, learning_rate=0.2)
    model.fit(rescaled_xtrain, ytrain)
    return model, scaler

if __name__ == "__main__":
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Load the data to train["data"] and train["target"]
    train = np.load("linear_regression_competition.train.npz")
    train = {entry: train[entry] for entry in train}
    xtrain, x, ytrain, y = train_test_split(train["data"], \
            train["target"], test_size=.05, random_state=args.seed)

    #pick_algorithm()

    #twist_params()

    model, scaler = train_model()

    # Validate the model
    rescaled_x = scaler.transform(x)
    pred = model.predict(rescaled_x)
    print (np.sqrt(mean_squared_error(y, pred)))

    with open(args.model_path, "wb") as model_file:
        pickle.dump(model, model_file)
