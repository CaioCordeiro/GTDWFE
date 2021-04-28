import pandas as pd
# .to_csv('src/data/filtered_datasets/GT_feature_selection.csv')

def write_full_data_frame():
    aac_all_1 = pd.read_csv("src/data/Ac_Sa_Ca_Kl_Ec/aac_all.csv")
    bla_all_1 = pd.read_csv("src/data/Ac_Sa_Ca_Kl_Ec/bla_all.csv")
    dfr_all_1 = pd.read_csv("src/data/Ac_Sa_Ca_Kl_Ec/dfr_all.csv")
    aac_all_2 = pd.read_csv("src/data/Ps_Vb_En/aac_all.csv")
    bla_all_2 = pd.read_csv("src/data/Ps_Vb_En/bla_all.csv")
    dfr_all_2 = pd.read_csv("src/data/Ps_Vb_En/dfr_all.csv")

    pd.concat([aac_all_1, bla_all_1, dfr_all_1, aac_all_2, bla_all_2, dfr_all_2]).to_csv('src/data/full_dataset.csv')

def get_full_data_frame():
    df = pd.read_csv("src/data/full_dataset.csv")
    # print(df["Output"])
    return df

def get_sample_dataframe():
    return pd.read_csv("src/data/Ac_Sa_Ca_Kl_Ec/bla_all.csv")

def get_single_dataframe(bac, ds_name):
    if bac == "ac":
        if ds_name == "aac":
            return pd.read_csv("src/data/Ac_Sa_Ca_Kl_Ec/aac_all.csv")
        if ds_name == "bla":
            return pd.read_csv("src/data/Ac_Sa_Ca_Kl_Ec/bla_all.csv")
        if ds_name == "dfr":
            return pd.read_csv("src/data/Ac_Sa_Ca_Kl_Ec/dfr_all.csv")
    if bac == "ps":
        if ds_name == "aac":
            return pd.read_csv("src/data/Ps_Vb_En/aac_all.csv")
        if ds_name == "bla":
            return pd.read_csv("src/data/Ps_Vb_En/bla_all.csv")
        if ds_name == "dfr":
            return pd.read_csv("src/data/Ps_Vb_En/dfr_all.csv")

def get_selected_feature_data_frame(algo_name):
    if algo_name == "GT":
        return pd.read_csv("src/data/filtered_datasets/GT_feature_selection.csv").drop(["Unnamed: 0"], axis=1)
    elif algo_name == "pearson":
        return pd.read_csv("src/data/filtered_datasets/pearson_corr_feature_selection.csv")
    elif algo_name == "relieff":
        return pd.read_csv("src/data/filtered_datasets/relieff_feature_selection.csv")
        