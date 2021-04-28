#     """
#     This function is used to select features based on pearson correlation.
#     :param df: training data
#     :num_features: number of features to select
#     :return: selected features
#     """


def pearson_corr_feature_selection(df, num_features):
    corr = df.corr()
    corr_target = abs(corr["Output"])
    most_relevant_features = corr_target.sort_values(ascending=False)
    most_relevant_features = most_relevant_features.index.tolist()
    most_relevant_features.pop(0)
    most_relevant_features = most_relevant_features[:num_features]
    most_relevant_features = df[most_relevant_features]
    if 'Output' in most_relevant_features.columns:
        most_relevant_features = most_relevant_features.drop('Output', axis=1)
    if 'Target' in most_relevant_features.columns:
        most_relevant_features = most_relevant_features.drop('Target', axis=1)
    return most_relevant_features