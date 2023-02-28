import matplotlib.pyplot as plt
import numpy as np

# Explainers
from aif360.explainers import MetricTextExplainer

# Fairness metrics
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from pandas import DataFrame


class data:
    """
    This class can become the object of every sample.

    Args:
        pre: the probability of prediction ,like [0.2,0.8]
        outcome: the ground truth of the sample
        model_label: the label of model prediction
        sample_weight: the weight of the sample
        protect_feature: the value of protected feature
    """

    def __init__(self, pre, outcome, model_label, sample_weight, protect_feature):
        self.pre = pre  # [0.2,0.8]
        self.outcome = outcome  # 1
        self.model_label = model_label  # 1
        self.sample_weight = sample_weight  # 0.8
        self.protect_feature = protect_feature  # 1 or 0


class DataSet:
    """
    This class is the contioner of dataset.We can get some characteristics of the dataset
    Args:
        data_array: An array composed of data
    """

    def __init__(self, data_array):
        self.data_array = data_array
        self.unprotected_weight = self.num_a_is_0()  # sum of unprotected sample'sweight
        self.protected_weight = self.num_a_is_1()
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

    """
        Return : the sum of all samples' weights
    """

    def sum_weight(self):
        sum_weight = 0
        for data in self.data_array:
            sum_weight = sum_weight + data.sample_weight
        return sum_weight

    """
        Return : the sum of unprotected samples' weights
    """

    def num_a_is_0(self):
        sum_weight = 0
        for data in self.data_array:
            if data.protect_feature == 0:
                sum_weight = sum_weight + data.sample_weight
        return sum_weight

    """
        Return : the sum of protected samples' weights
    """

    def num_a_is_1(self):
        sum_weight = 0
        for data in self.data_array:
            if data.protect_feature == 1:
                sum_weight = sum_weight + data.sample_weight
        return sum_weight

    """
        Return : the sum of unprotected and predicted favourite samples' weights
    """

    def num_y_hat_is_1_a_is_0(self):
        sum_weight = 0
        for data in self.data_array:
            if data.protect_feature == 0 and data.model_label == 1:
                sum_weight = sum_weight + data.sample_weight
        return sum_weight

    """
        Return : the sum of protected and predicted favourite samples' weights
    """

    def num_y_hat_is_1_a_is_1(self):
        sum_weight = 0
        for data in self.data_array:
            if data.protect_feature == 1 and data.model_label == 1:
                sum_weight = sum_weight + data.sample_weight
        return sum_weight

    """
        Return : the sum of unprotected and unfavourite samples' weights
    """

    def num_y_is_0_a_is_0(self):
        sum_weight = 0
        for data in self.data_array:
            if data.protect_feature == 0 and data.outcome == 0:
                sum_weight = sum_weight + data.sample_weight
        return sum_weight

    """
        Return : the sum of protected and unfavourite samples' weights
    """

    def num_y_is_0_a_is_1(self):
        sum_weight = 0
        for data in self.data_array:
            if data.protect_feature == 1 and data.outcome == 0:
                sum_weight = sum_weight + data.sample_weight
        return sum_weight

    """
        Return : the sum of unprotected and favourite samples' weights
    """

    def num_y_is_1_a_is_0(self):
        sum_weight = 0
        for data in self.data_array:
            if data.protect_feature == 0 and data.outcome == 1:
                sum_weight = sum_weight + data.sample_weight
        return sum_weight

    """
        Return : the sum of protected and favourite samples' weights
    """

    def num_y_is_1_a_is_1(self):
        sum_weight = 0
        for data in self.data_array:
            if data.protect_feature == 1 and data.outcome == 1:
                sum_weight = sum_weight + data.sample_weight
        return sum_weight

    """
        Return : the sum of unprotected and predicted favourite but ture unfavourite samples' weights
    """

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

    """
        Return : the sum of protected and predicted favourite but ture unfavourite samples' weights
    """

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

    """
        Return : the sum of unprotected and predicted favourite and ture favourite samples' weights
    """

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

    """
        Return : the sum of protected and predicted favourite and ture favourite samples' weights
    """

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

    """
        Return : the sum of true positive samples' weights
    """

    def num_y_hat_and_y_is_1(self):
        sum_weight = 0
        for data in self.data_array:
            if data.outcome == 1 and data.model_label == 1:
                sum_weight = sum_weight + data.sample_weight
        return sum_weight

    """
        Return : the sum of true negivate samples' weights
    """

    def num_y_hat_and_y_is_0(self):
        sum_weight = 0
        for data in self.data_array:
            if data.outcome == 0 and data.model_label == 0:
                sum_weight = sum_weight + data.sample_weight
        return sum_weight

    """
        Return : the sum of positive samples' weights
    """

    def num_y_is_1(self):
        sum_weight = 0
        for data in self.data_array:
            if data.outcome == 1:
                sum_weight = sum_weight + data.sample_weight
        return sum_weight

    """
        Return : the sum of negivate samples' weights
    """

    def num_y_is_0(self):
        sum_weight = 0
        for data in self.data_array:
            if data.outcome == 0:
                sum_weight = sum_weight + data.sample_weight
        return sum_weight

    """
        Return : Statistical Parity Difference
    """

    def SPD(self):
        return abs(
            self.pre_favourite_unprotect_weight / self.unprotected_weight
            - self.pre_favourite_protect_weight / self.protected_weight
        )

    """
        Return : Average Odds Difference
    """

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

    """
        Return : Equal Opportunity Difference
    """

    def EDO(self):
        return abs(
            self.pre_favourite_tru_favourite_unprotect_weight
            / self.tru_favourite_unprotect_weight
            - self.pre_favourite_tru_favourite_protect_weight
            / self.tru_favourite_protect_weight
        )

    """
        Return : ths true positive rate
    """

    def true_positive_rate(self):
        return self.num_y_hat_and_y_is_1() / self.num_y_is_1()

    """
        Return : ths true negitive rate
    """

    def true_negetive_rate(self):
        return self.num_y_hat_and_y_is_0() / self.num_y_is_0()


def measure_disparity(data_frame):
    data_array = []
    for index, row in data_frame.iterrows():  #
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
    spd = []
    aod = []
    eod = []
    balence_rate = []
    thresh_arr = np.linspace(0.01, 1, 100)

    for thresh in thresh_arr:
        data_array = []
        for index, row in data_frame.iterrows():
            prediction = eval(row["Model prediction"])
            outcome = row["Binary outcome"]
            label = row["Model label"]
            weights = row["Sample weights"]
            protected = row["Demographic data on protected and reference classes"]
            if prediction[1] >= thresh:
                label = 1
            else:
                label = 0
            this_data = data(prediction, outcome, label, weights, protected)
            data_array.append(this_data)
        dataset = DataSet(data_array)
        spd.append(dataset.SPD())
        aod.append(dataset.AOD())
        eod.append(dataset.EDO())
        balence_rate.append(
            (dataset.true_negetive_rate() + dataset.true_positive_rate()) / 2
        )
    plot(
        thresh_arr,
        "Classification Thresholds",
        balence_rate,
        "Balanced Accuracy",
        spd,
        "SPD",
    )
    plot(
        thresh_arr,
        "Classification Thresholds",
        balence_rate,
        "Balanced Accuracy",
        aod,
        "AOD",
    )
    plot(
        thresh_arr,
        "Classification Thresholds",
        balence_rate,
        "Balanced Accuracy",
        eod,
        "EOD",
    )


def plot(x, x_name, y_left, y_left_name, y_right, y_right_name):
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(x, y_left)
    ax1.set_xlabel(x_name, fontsize=16, fontweight="bold")
    ax1.set_ylabel(y_left_name, color="b", fontsize=16, fontweight="bold")
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)
    ax1.set_ylim(0.5, 0.8)

    ax2 = ax1.twinx()
    ax2.plot(x, y_right, color="r")
    ax2.set_ylabel(y_right_name, color="r", fontsize=16, fontweight="bold")
    if "DI" in y_right_name:
        ax2.set_ylim(0.0, 0.7)
    else:
        ax2.set_ylim(0, 0.7)

    best_ind = np.argmax(y_left)
    ax2.axvline(np.array(x)[best_ind], color="k", linestyle=":")
    ax2.yaxis.set_tick_params(labelsize=14)
    ax2.grid(True)
    plt.savefig(y_right_name + ".png")
