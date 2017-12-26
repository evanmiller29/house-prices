
def rmse(y_actual, y_pred):

    """

    :param y_actual: Ground truth values
    :param y_pred:  Predicted values
    :return: Double
    """

    from math import sqrt
    from sklearn.metrics import mean_squared_error

    rmse = sqrt(mean_squared_error(y_valid, y_pred))

    return rmse

    'This wont work for some reason. The error is around numpy arrays and normal floats not being able to be processed like this'