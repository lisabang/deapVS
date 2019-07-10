import numpy as np
import sklearn.model_selection as skms
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def mlr_r2(X, y):
    model = LinearRegression()
    model.fit(X, y)
    # compute with formulas from the theory
    yhat = model.predict(X)
    SS_Residual = sum((y - yhat) ** 2)
    SS_Total = sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (float(SS_Residual)) / SS_Total
    adjusted_r_squared = 1 - (1 - r_squared) * (len(y) - 1) / (len(y) - X.shape[1] - 1)
    return r_squared, adjusted_r_squared


def mlr_RMSE(x, y):
    columnnames = list(x.columns.values)
    npones = np.ones(len(y), float)
    A_sl = x.values
    A = np.column_stack([A_sl, npones])
    lstsq, residuals, rank, something = np.linalg.lstsq(A, y, rcond=-1)
    degfreedom = y.size - 1

    r2 = 1 - residuals / (y.size * y.var())
    r2adj = 1 - (((1 - r2) * degfreedom) / (y.size - rank - 2))
    RMSE = np.sqrt(1 - r2) * np.std(y)

    return RMSE


def kfoldmlr(xi, yi, **kwargs):
    """gives the y-hats for a q2LOO calculation"""
    x = xi.values
    y = yi.values
    nfolds = kwargs["nfolds"]
    mean = kwargs["mean"]
    kf = skms.KFold(n_splits=nfolds)  # indices=None, shuffle=False, random_state=None)
    y_hats = []
    print(kf)
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        coefficients = mlrr(x_train, y_train)[0]
        resids = mlrr(x_train, y_train)[1]
        y_hats.append(resids)
    # for e in y_hats:
    #    cleanyhats.append(float(e))
    stack = np.asarray(y_hats)
    if mean == True:
        return np.mean(stack)
    else:
        return stack


def kfoldmlrplot(xi, yi, **kwargs):
    """gives the y-hats for a q2LOO calculation"""
    x = xi.values
    y = yi.values
    nfolds = kwargs["nfolds"]
    kf = skms.KFold(n_splits=nfolds)  # indices=None, shuffle=False, random_state=None)
    y_hats = []
    print(kf)
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        coefficients = mlrr(x_train, y_train)[0]
        resids = mlrr(x_train, y_train)[1]
        plt.plot(x_train, y_train, "o", label="Original data", markersize=5)
        plt.plot(
            x_train,
            coefficients[0] * x_train + coefficients[1],
            "r",
            label="Fitted line",
        )
        plt.legend()
        plt.show()
