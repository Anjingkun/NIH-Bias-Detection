
def DisparateImpact(dataset, protect_feature_name, privileged_value, fav_value):
    '''
        The function is used to calculate the disparate impact of the dataset.
        Args:
            dataset : A dataset that needs to calculate disparate impact.
            protect_feature_name : Name of the protected attribute.
            privileged_value : Privileged value of protected attribute.
            fav_value : The value of favourite label.
    '''
    index = 0
    for i in range(len(dataset.feature_names)):
        if protect_feature_name == dataset.feature_names[i]:
            index = i
            break
    weight_protect = 0
    weight_unprotect = 0
    for i in range(len(dataset.features)):
        if dataset.features[i][index] == privileged_value:
            weight_protect += dataset.instance_weights[i]
        else:
            weight_unprotect += dataset.instance_weights[i]

    weight_fav_pro = 0
    weight_fav_unpro = 0
    for i in range(len(dataset.features)):
        if (
            dataset.features[i][index] == privileged_value
            and dataset.labels[i] == fav_value
        ):
            weight_fav_pro += dataset.instance_weights[i]
        elif (
            dataset.features[i][index] != privileged_value
            and dataset.labels[i] == fav_value
        ):
            weight_fav_unpro += dataset.instance_weights[i]
    disparate_impact = (weight_fav_unpro / weight_unprotect) / (
        weight_fav_pro / weight_protect
    )
    if disparate_impact > 1:
        disparate_impact = 1 / disparate_impact
    return disparate_impact