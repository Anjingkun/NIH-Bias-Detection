from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import (
    load_preproc_data_adult,
)
from sklearn.pipeline import make_pipeline
from measure_disparity import measure_disparity
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

privileged_groups = [{"sex": 1}]
unprivileged_groups = [{"sex": 0}]
dataset_orig = load_preproc_data_adult(["sex"])

for i in range(len(dataset_orig.features)):
    if dataset_orig.features[i][1] != dataset_orig.protected_attributes[i][0]:
        print("no")

dataset = dataset_orig
model = make_pipeline(
    StandardScaler(), LogisticRegression(solver="liblinear", random_state=1)
)
fit_params = {"logisticregression__sample_weight": dataset.instance_weights}

lr_orig_panel19 = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)
y_val_pred_prob = model.predict_proba(dataset.features)
pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
y_val_pred = (y_val_pred_prob[:, pos_ind] > 0.5).astype(np.float64)

str_y_val_pred_prob = []
for i in y_val_pred_prob:
    str_y_val_pred_prob.append(str(i.tolist()))
import pandas as pd

data_dic = {
    "Model prediction": str_y_val_pred_prob,
    "Binary outcome": dataset.labels.ravel(),
    "Model label": y_val_pred,
    "Sample weights": dataset.instance_weights,
    "Demographic data on protected and reference classes": dataset.protected_attributes.ravel(),
}
dataframe = pd.DataFrame(data_dic)
measure_disparity(dataframe)


from aif360.datasets import AdultDataset, MEPSDataset19
from mitigate_disparity import MultiLevelReweighing as Reweighing, BiasRemoverModel
from aif360.metrics import BinaryLabelDatasetMetric

chose_dataset = "MEPS"
if chose_dataset == "adult":
    dataset = AdultDataset()
    multi_privileged_groups = [
        {"feature_name": "race", "privileged_value": 1, "level": 1},
        {"feature_name": "sex", "privileged_value": 1, "level": 1},
    ]
    multi_unprivileged_groups = [
        {"feature_name": "race", "unprivileged_value": 0, "level": 1},
        {"feature_name": "sex", "unprivileged_value": 0, "level": 1},
    ]
    privileged_groups1 = [{"sex": 1}]
    unprivileged_groups1 = [{"sex": 0}]
    privileged_groups2 = [{"race": 1}]
    unprivileged_groups2 = [{"race": 0}]
else:
    dataset = MEPSDataset19()

    multi_privileged_groups = [
        {"feature_name": "RACE", "privileged_value": 1, "level": 1},
    ]
    multi_unprivileged_groups = [
        {"feature_name": "RACE", "unprivileged_value": 0, "level": 1},
    ] 
    privileged_groups2 = [{"RACE": 1}]
    unprivileged_groups2 = [{"RACE": 0}]
rw = Reweighing(multi_unprivileged_groups, multi_privileged_groups)
trans_adult_dataset = rw.fit_transform(dataset)
if chose_dataset=='adult':
    metric_orig_adult = BinaryLabelDatasetMetric(
        dataset,
        unprivileged_groups=unprivileged_groups1,
        privileged_groups=privileged_groups1,
    )
    print('before reweighing ,sex disparate impact is' +str(metric_orig_adult.disparate_impact()))
    metric_trans_adult = BinaryLabelDatasetMetric(
        trans_adult_dataset,
        unprivileged_groups=unprivileged_groups1,
        privileged_groups=privileged_groups1,
    )
    print('after reweighing ,sex disparate impact is '+str(metric_trans_adult.disparate_impact()))
metric_orig_adult = BinaryLabelDatasetMetric(
    dataset,
    unprivileged_groups=unprivileged_groups2,
    privileged_groups=privileged_groups2,
)
print('before reweighing ,race disparate impact is '+str(metric_orig_adult.disparate_impact()))
metric_trans_adult = BinaryLabelDatasetMetric(
    trans_adult_dataset,
    unprivileged_groups=unprivileged_groups2,
    privileged_groups=privileged_groups2,
)
print('after reweighing ,race disparate impact is '+str(metric_trans_adult.disparate_impact()))
brm_model = BiasRemoverModel()
brm_model.fit(dataset)
predic_prob = brm_model.predic_prob(dataset.features)
print('the probability of prediction is')
print(predic_prob)