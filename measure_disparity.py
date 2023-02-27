import sys

sys.path.insert(0, "../")


import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display

# Fairness metrics
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric

# Explainers
from aif360.explainers import MetricTextExplainer
from pandas import DataFrame


class data():
    def __init__(self, pre, outcome, model_label, sample_weight, protect_feature):
        self.pre = pre
        self.outcome = outcome
        self.model_label = model_label
        self.sample_weight = sample_weight
        self.protect_feature = protect_feature


class DataSet():
    def __init__(self, data_array):
        self.data_array = data_array
        self.unprotected_weight = self.num_a_is_0()
        self.protected_weigth = self.num_a_is_1()
        self.pre_favourite_unprotect_weight = self.num_y_hat_is_1_a_is_0()
        self.pre_favourite_protect_weight = self.num_y_hat_is_1_a_is_1()
        self.tru_unfavourite_unprotect_weight = self.num_y_is_0_a_is_0()
        self.tru_unfavourite_protect_weight = self.num_y_is_0_a_is_1()
        self.tru_favourite_unprotect_weight = self.num_y_is_1_a_is_0()
        self.tru_favourite_protect_weight = self.num_y_is_1_a_is_1()
        self.pre_favourite_tru_unfavourite_unprotect_weight = (
            self.num_y_hat_is_1_y_is_0_a_is_0()
        )
        self.pre_favourite_tru_unfavourite_protect_weight = (
            self.num_y_hat_is_1_y_is_0_a_is_1()
        )
        self.pre_favourite_tru_favourite_unprotect_weight = (
            self.num_y_hat_is_1_y_is_1_a_is_0()
        )
        self.pre_favourite_tru_favourite_protect_weight = (
            self.num_y_hat_is_1_y_is_1_a_is_1()
        )

    def num_a_is_0(self):
        sum_weight = 0
        for data in self.data_array:
            if data.protect_feature == 0:
                sum_weight = sum_weight + data.sample_weight
        return sum_weight

    def num_a_is_1(self):
        sum_weight = 0
        for data in self.data_array:
            if data.protect_feature == 1:
                sum_weight = sum_weight + data.sample_weight
        return sum_weight

    def num_y_hat_is_1_a_is_0(self):
        sum_weight = 0
        for data in self.data_array:
            if data.protect_feature == 0 and data.model_label == 1:
                sum_weight = sum_weight + data.sample_weight
        return sum_weight

    def num_y_hat_is_1_a_is_1(self):
        sum_weight = 0
        for data in self.data_array:
            if data.protect_feature == 1 and data.model_label == 1:
                sum_weight = sum_weight + data.sample_weight
        return sum_weight

    def num_y_is_0_a_is_0(self):
        sum_weight = 0
        for data in self.data_array:
            if data.protect_feature == 0 and data.outcome == 0:
                sum_weight = sum_weight + data.sample_weight
        return sum_weight

    def num_y_is_0_a_is_1(self):
        sum_weight = 0
        for data in self.data_array:
            if data.protect_feature == 1 and data.outcome == 0:
                sum_weight = sum_weight + data.sample_weight
        return sum_weight

    def num_y_is_1_a_is_0(self):
        sum_weight = 0
        for data in self.data_array:
            if data.protect_feature == 0 and data.outcome == 1:
                sum_weight = sum_weight + data.sample_weight
        return sum_weight

    def num_y_is_1_a_is_1(self):
        sum_weight = 0
        for data in self.data_array:
            if data.protect_feature == 1 and data.outcome == 1:
                sum_weight = sum_weight + data.sample_weight
        return sum_weight

    def num_y_hat_is_1_y_is_0_a_is_0(self):
        sum_weight = 0
        for data in self.data_array:
            if (
                data.protect_feature == 0
                and data.outcome == 0
                and data.model_label == 1
            ):
                sum_weight = sum_weight + data.sample_weight
        return sum_weight

    def num_y_hat_is_1_y_is_0_a_is_1(self):
        sum_weight = 0
        for data in self.data_array:
            if (
                data.protect_feature == 1
                and data.outcome == 0
                and data.model_label == 1
            ):
                sum_weight = sum_weight + data.sample_weight
        return sum_weight

    def num_y_hat_is_1_y_is_1_a_is_0(self):
        sum_weight = 0
        for data in self.data_array:
            if (
                data.protect_feature == 0
                and data.outcome == 1
                and data.model_label == 1
            ):
                sum_weight = sum_weight + data.sample_weight
        return sum_weight

    def num_y_hat_is_1_y_is_1_a_is_1(self):
        sum_weight = 0
        for data in self.data_array:
            if (
                data.protect_feature == 1
                and data.outcome == 1
                and data.model_label == 1
            ):
                sum_weight = sum_weight + data.sample_weight
        return sum_weight

    def SPD(self):
        return abs(
            self.pre_favourite_unprotect_weight / self.unprotected_weight
            - self.pre_favourite_protect_weight / self.protected_weigth
        )

    def AOD(self):
        return (
            abs(
                self.pre_favourite_tru_unfavourite_unprotect_weight
                / self.tru_unfavourite_unprotect_weight
                - self.pre_favourite_tru_unfavourite_protect_weight
                / self.tru_unfavourite_protect_weight
            )
            + abs(
                self.pre_favourite_tru_favourite_unprotect_weight
                / self.tru_favourite_unprotect_weight
                - self.pre_favourite_tru_favourite_protect_weight
                / self.tru_favourite_protect_weight
            )
        ) / 2

    def EDO(self):
        return abs(
            self.pre_favourite_tru_favourite_unprotect_weight
            / self.tru_favourite_unprotect_weight
            - self.pre_favourite_tru_favourite_protect_weight
            / self.tru_favourite_protect_weight
        )


def measure_disparity(data_frame):
    data_array = []
    for index, row in data_frame.iterrows():
        this_data = data(
            eval(row["Model prediction"]),
            row["Binary outcome"],
            row["Model label"],
            row["Sample weights"],
            row["Demographic data on protected and reference classes"],
        )
        data_array.append(this_data)
    dataset = DataSet(data_array)
    print("This dataset's SPD is " + str(dataset.SPD()))
    print("This dataset's AOD is " + str(dataset.AOD()))
    print("This dataset's EOD is " + str(dataset.EDO()))
