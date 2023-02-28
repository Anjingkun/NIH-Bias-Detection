# Scalers
import numpy as np
from aif360.algorithms import Transformer
from aif360.metrics import utils

# Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class BiasRemoverModel:
    """
    BiasRemoverModel is a model that can be trained by a dataset.Before using this model ,
    the dataset will be reweighted by MultiLevelReweighing. After reweighing the dataset,
    we can train the model.
    """

    def __init__(self):
        self.lr_model = make_pipeline(
            StandardScaler(), LogisticRegression(solver="liblinear", random_state=1)
        )

    """
        This function can train the model.
        Args:
            dataset (BinaryLabelDataset): Dataset containing true labels.
        Return:
            BiasRemoverModel (self)
    """

    def fit(self, dataset):
        privileged_groups = []
        unprivileged_groups = []
        for i in range(len(dataset.privileged_protected_attributes)):
            privileged_groups.append(
                {
                    "feature_name": dataset.protected_attribute_names[i],
                    "privileged_value": dataset.privileged_protected_attributes[i][0],
                    "level": 1,  # we can change this level，but this level must be positive integar
                }
            )
            unprivileged_groups.append(
                {
                    "feature_name": dataset.protected_attribute_names[i],
                    "unprivileged_value": dataset.unprivileged_protected_attributes[i][
                        0
                    ],
                    "level": 1,  # we can change this level，but this level must be positive integar
                }
            )
        rw = MultiLevelReweighing(unprivileged_groups, privileged_groups)
        trans_dataset = rw.fit_transform(dataset)
        fit_params = {
            "logisticregression__sample_weight": trans_dataset.instance_weights
        }
        lr_trans_dataset_model = self.lr_model.fit(
            dataset.features, dataset.labels.ravel(), **fit_params
        )
        self.lr_model = lr_trans_dataset_model
        return self

    """
        This function can output the Probability of prediction
        Args:
            features (numpy):ths features of samples.
            such as :[
            [1,1,0,1,0],
            [1,1,0,0,0]
            ]
        Return :
            the Probability of prediction
    """

    def predic_prob(self, features):
        return self.lr_model.predict_proba(features)


class MultiLevelReweighing(Transformer):
    """MultiLevelReweighing is a preprocessing technique that Weights the examples in each
    (group, label) combination differently to ensure fairness before
    classification .This technique can compute the protected level of every sample , and then it can
    give every sample a new weight.
    """

    def __init__(self, unprivileged_groups, privileged_groups):
        """
        Args:
            unprivileged_groups (list(dict)): Representation for unprivileged group.
            privileged_groups (list(dict)): Representation for privileged group.
            such as:
            [{'feature_name':'sex','privileged_value':1,'level':2},
            {'feature_name':'race','privileged_value':1,'level':1}]
        """
        super(MultiLevelReweighing, self).__init__(
            unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups
        )

        self.unprivileged_groups = unprivileged_groups
        self.privileged_groups = privileged_groups
        self.protect_feature_count = len(self.privileged_groups)
        self.w_p_fav = 1.0
        self.w_p_unfav = 1.0
        self.w_up_fav = 1.0
        self.w_up_unfav = 1.0

    def fit(self, dataset):
        """Compute the weights for reweighing the dataset.

        Args:
            dataset (BinaryLabelDataset): Dataset containing true labels.

        Returns:
            MultiLevelReweighing: Returns self.
        """

        (
            protect_level_dic,
            fav_cond,
            unfav_cond,
            combination_p_fav_dic,
        ) = self._obtain_conditionings(dataset)

        level_max = 0
        for group in self.privileged_groups:
            level_max = level_max + group["level"]
        n = np.sum(dataset.instance_weights, dtype=np.float64)
        n_p_level_dic = {}
        # weight:[1,2,1,1,2]
        #'n_p1':2
        for level in range(0, level_max + 1):
            n_p_level_dic["n_p" + str(level)] = np.sum(
                dataset.instance_weights[
                    protect_level_dic["protect_level" + str(level) + "_cond"]
                ],
                dtype=np.float64,
            )

        n_fav = np.sum(dataset.instance_weights[fav_cond], dtype=np.float64)
        n_unfav = np.sum(dataset.instance_weights[unfav_cond], dtype=np.float64)

        n_p_level_fav_or_unfav_dic = {}
        # "n_p1_unfav":3
        for level in range(0, level_max + 1):
            n_p_level_fav_or_unfav_dic["n_p" + str(level) + "_fav"] = np.sum(
                dataset.instance_weights[
                    combination_p_fav_dic["cond_p" + str(level) + "_fav"]
                ],
                dtype=np.float64,
            )
            n_p_level_fav_or_unfav_dic["n_p" + str(level) + "_unfav"] = np.sum(
                dataset.instance_weights[
                    combination_p_fav_dic["cond_p" + str(level) + "_unfav"]
                ],
                dtype=np.float64,
            )

        self.w_p_level_fav_or_unfav_dic = {}

        # reweighing weights
        # dataset.instance_weights
        # "w_p3_fav":3
        for level in range(0, level_max + 1):
            self.w_p_level_fav_or_unfav_dic["w_p" + str(level) + "_fav"] = (
                n_fav
                * n_p_level_dic["n_p" + str(level)]
                / (n * n_p_level_fav_or_unfav_dic["n_p" + str(level) + "_fav"])
            )
            self.w_p_level_fav_or_unfav_dic["w_p" + str(level) + "_unfav"] = (
                n_unfav
                * n_p_level_dic["n_p" + str(level)]
                / (n * n_p_level_fav_or_unfav_dic["n_p" + str(level) + "_unfav"])
            )

        return self

    def transform(self, dataset):
        """Transform the dataset to a new dataset based on the estimated
        transformation.

        Args:
            dataset (BinaryLabelDataset): Dataset that needs to be transformed.
        Returns:
            dataset (BinaryLabelDataset): Dataset with transformed
                instance_weights attribute.
        """
        protect_level_dic = {}
        level_max = 0
        for group in self.privileged_groups:
            level_max = level_max + group["level"]
        dataset_transformed = dataset.copy(deepcopy=True)

        (
            protect_level_dic,
            fav_cond,
            unfav_cond,
            combination_p_fav_dic,
        ) = self._obtain_conditionings(dataset)

        # apply reweighing
        for level in range(1 + level_max):
            dataset_transformed.instance_weights[
                combination_p_fav_dic["cond_p" + str(level) + "_fav"]
            ] *= self.w_p_level_fav_or_unfav_dic["w_p" + str(level) + "_fav"]
            dataset_transformed.instance_weights[
                combination_p_fav_dic["cond_p" + str(level) + "_unfav"]
            ] *= self.w_p_level_fav_or_unfav_dic["w_p" + str(level) + "_unfav"]

        return dataset_transformed

    ##############################
    #### Supporting functions ####
    ##############################
    def _obtain_conditionings(self, dataset):
        """Obtain the necessary conditioning boolean vectors to compute
        instance level weights.
        [{'feature_name':'sex','privileged_value':1,'level':2},
            {'feature_name':'race','privileged_value':1,'level':1}]
        """
        # conditioning

        protect_level_dic = {}
        level_max = 0
        for group in self.privileged_groups:
            level_max = level_max + group["level"]
        for i in range(0, level_max + 1):
            protect_level_dic["protect_level" + str(i) + "_cond"] = np.zeros(
                dataset.protected_attributes.shape[0], dtype=bool
            )
        #'protect_level3_cond':[0,1,0]
        #'protect_level2_cond':[0,0,1]
        # [
        #     [1,1],
        #     [0,1],
        #     [1,0],
        #     [0,0]
        # ]
        # ['sex','race']
        for sample_index in range(len(dataset.protected_attributes)):  # 3,5
            protect_feature = dataset.protected_attributes[sample_index]
            this_sample_protect_level = 0
            for group in self.privileged_groups:
                name = group["feature_name"]
                val = group["privileged_value"]
                level = group["level"]
                index = dataset.protected_attribute_names.index(name)
                if protect_feature[index] == val:
                    this_sample_protect_level = this_sample_protect_level + level  # 3
            protect_level_dic[
                "protect_level" + str(this_sample_protect_level) + "_cond"
            ][sample_index] = True
            # "protect_level" + str(3) + "_cond":[0,0,0,1,0]

        # [[1],[0],[1]],[1,0,1]
        fav_cond = dataset.labels.ravel() == dataset.favorable_label  # [0,0,1,1,0]
        unfav_cond = dataset.labels.ravel() == dataset.unfavorable_label  # [1,1,0,0,1]

        # combination of label and privileged/unpriv. groups
        combination_p_fav_dic = {}
        # "cond_p3_fav":[0,0,1,0,0]

        for i in range(0, level_max + 1):  # 3

            combination_p_fav_dic["cond_p" + str(i) + "_fav"] = np.logical_and(
                fav_cond, protect_level_dic["protect_level" + str(i) + "_cond"]
            )
            combination_p_fav_dic["cond_p" + str(i) + "_unfav"] = np.logical_and(
                unfav_cond, protect_level_dic["protect_level" + str(i) + "_cond"]
            )

        return (protect_level_dic, fav_cond, unfav_cond, combination_p_fav_dic)
