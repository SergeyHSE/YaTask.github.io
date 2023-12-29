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

    def fit(self, model_constructor, data, targets, num_steps=10, lr=0.1, max_depth=5, verbose=False):
        self.models_list = []
        self.lr = lr
        self.loss_log = []

        residuals = targets.copy()

        for step in range(num_steps):
            try:
                model = model_constructor(max_depth=max_depth)
            except TypeError:
                print('max_depth keyword is not found. Ignoring')
                model = model_constructor()

            # Ensure that the fit method returns the model
            self.models_list.append(model.fit(data, residuals))

            predictions = self.predict(data)
            self.loss_log.append(self.loss(targets, predictions))

            if verbose:
                print(f'Step {step+1}/{num_steps}, Loss: {self.loss_log[-1]}')

            residuals = targets - predictions  # Update residuals

        if verbose:
            print('Finished! Final Loss=', self.loss_log[-1])

        return self

