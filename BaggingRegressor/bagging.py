import numpy as np

class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob
        self.indices_list = []
        self.models_list = []
        self.data = None
        self.target = None
        self.list_of_predictions_lists = None
        self.oob_predictions = None
