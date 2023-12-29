import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

class SimplifiedBoostingRegressor:
    def __init__(self):
        pass

    @staticmethod
    def loss(targets, predictions):
        loss = np.mean((targets - predictions)**2)
        return loss


    @staticmethod
    def loss_gradients(targets, predictions):
        gradients = -2 * (targets - predictions)
        assert gradients.shape == targets.shape
        return gradients

