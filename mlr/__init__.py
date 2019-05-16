import numpy as np
import sklearn.model_selection as skms
import matplotlib.pyplot as plt

def mlr(x_of_trainingset, y_actual):
    columnnames = list(x_of_trainingset.columns.values)
    npones = np.ones(len(y_actual), float)
    A_sl = x_of_trainingset.as_matrix()
    A = np.column_stack([A_sl, npones])
    lstsq, residuals, rank, something = np.linalg.lstsq(A, y_actual)
    degfreedom = y_actual.size - 1

    r2 = 1 - residuals / (y_actual.size * y_actual.var())
    r2adj = 1 - (((1 - r2) * degfreedom) / (y_actual.size - rank - 2))
    RMSE = np.sqrt(1 - r2) * np.std(y_actual)

    # fitness=collections.namedtuple([x_of_trainingset],[r2,r2adj,RMSE])
    return lstsq, rank, r2, r2adj, RMSE

    # y_predicted=(lstsq[0]*liu_train(0))+(lstsq[1]*liu_train(1))+(lstsq[2])+(lstsq[3])+(lstsq[4])+(lstsq[5])+lstsq[6]
    # print "y-predicted:"
    # print y_predicted


def mlrr(x_of_trainingset, y_actual):
    #usethis
    npones = np.ones(len(x_of_trainingset), float)
    A_sl = x_of_trainingset
    A = np.column_stack([A_sl, npones])
    lstsq, residuals, rank, something = np.linalg.lstsq(A, y_actual)
    return lstsq, residuals
def pmlr(x_of_trainingset, y_actual):
    npones = np.ones(len(y_actual), float)
    A = np.column_stack([x_of_trainingset, npones])
    lstsq = np.dot(np.linalg.pinv(A), y_actual)
    return lstsq


def kfoldmlr(xi, yi, **kwargs):
    '''gives the y-hats for a q2LOO calculation'''
    x = xi.values
    y = yi.values
    nfolds=kwargs["nfolds"]
    mean=kwargs["mean"]
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
    if mean==True:
        return np.mean(stack)
    else:
        return stack


def kfoldmlrplot(xi, yi, **kwargs):
    '''gives the y-hats for a q2LOO calculation'''
    x = xi.values
    y = yi.values
    nfolds=kwargs["nfolds"]
    kf = skms.KFold(n_splits=nfolds)  # indices=None, shuffle=False, random_state=None)
    y_hats = []
    print(kf)
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        coefficients = mlrr(x_train, y_train)[0]
        resids = mlrr(x_train, y_train)[1]
        plt.plot(x_train, y_train, 'o', label='Original data', markersize=5)
        plt.plot(x_train, coefficients[0]*x_train + coefficients[1], 'r', label='Fitted line')
        plt.legend()
        plt.show()
