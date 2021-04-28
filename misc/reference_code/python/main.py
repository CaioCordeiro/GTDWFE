import pandas as pd
import numpy as np
import threading
from multiprocessing import Process
from scipy.stats import ttest_ind
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.datasets import load_iris
from data.data_provider import get_full_data_frame, get_sample_dataframe, get_selected_feature_data_frame, get_single_dataframe
from feature_selection.GT_feature_selection import GT_feature_selection
from feature_selection.relieff_algorithm import Relieff
from feature_selection.pearson_corr_selection import pearson_corr_feature_selection

FEATURE_SIZE = 124

def create_thread(func, arg1=None, arg2=None, arg3=None):
    if arg1 is None and arg2 is None and arg3 is None:
        t = Process(target=func)
        t.start()
    elif arg2 is None:
        t = Process(target=func, args=(arg1,))
        t.start()
    elif arg3 is None:
        t = Process(target=func, args=(arg1, arg2))
        t.start()
    else:
        t = Process(target=func, args=(arg1, arg2, arg3))
        t.start()


def get_mean_scores(scores: dict, name: str) -> dict:
    for i in scores.keys():
        scores[i] = sum(scores[i])/len(scores[i])
    scores["name"] = name
    return scores

def store_parameters(scores, n=None, p=None):
    scores["n"] = n
    if p is not None:
        scores["p"] = n
    return scores


def compare_classifiers(a, b):
    p = ttest_ind(a, b, equal_var=False)
    return p

def return_x_y(df):
    y = df["Output"]
    X = df.drop([ "Output"], axis=1)
    return X, y

def discretize(df):
    for i in df.columns:
        if i != "Feature" and i != "Output":
            df[i] = pd.qcut(df[i], q=5,  labels=False, precision=0, duplicates='drop')
    return df

def get_iris():
    df = load_iris()
    df = pd.DataFrame(data= np.c_[df['data'], df['target']],
                        columns= df['feature_names'] + ['target'])
    df = df.dropna()
    df = discretize(df)

    y = df['target']
    X = df.drop(['target'], axis=1)
    return X, y

def get_AMR_dataset_sample():
    df = get_sample_dataframe()
    df = discretize(df)
    y = df["Output"]
    X = df.drop(["Output","Feature"], axis=1) 
    return X, y

def run_SVM_with_features(X, y, name):
    clf = svm.SVC()
    return cross_validate(clf, X, y, cv=10, return_train_score=True)

def get_AMR_dataset():
    df = get_full_data_frame()
    df = discretize(df)
    df.drop(["Feature"], axis=1, inplace=True)
    return df

def get_single_df(bac, ds_name):
    df = get_single_dataframe(bac, ds_name)
    df = discretize(df)
    y = df["Output"]
    X = df.drop(["Output","Feature"], axis=1)
    return pd.concat([X, y], axis=1)

def run_GT_feature_selection(n, X, y):
    return GT_feature_selection(pd.concat([X, y], axis=1), X, y, n)

def run_relieff_feature_selection(n, X, y):
    X, y = Relieff(X.to_numpy(), y, n)
    return X

def run_pearson_corr_feature_selection(n, df):
    X = pearson_corr_feature_selection(df, n)
    return X

def run_GT_model(n, p):
    run_GT_feature_selection(n, p)
    X_GT = get_selected_feature_data_frame("GT")
    X, y = get_AMR_dataset_sample()
    print(run_SVM_with_features(X_GT, y, "GT"))

def run_pearson_model(n):
    run_pearson_corr_feature_selection(n)
    df = get_selected_feature_data_frame("pearson")
    y = df["Output"]
    X = df.drop([ "Output"], axis=1)
    print(run_SVM_with_features(X, y, "pearson"))

def run_relieff_model(n):
    run_relieff_feature_selection(n)
    df = get_selected_feature_data_frame("relieff")
    y = df["Output"]
    X = df.drop([ "Output"], axis=1)
    print(run_SVM_with_features(X, y, "relieff"))

def run_default_model():
    X, y = get_AMR_dataset_sample()
    print(run_SVM_with_features(X, y, "default"))

def get_GT_best_parameters(df, bac, ds_name):
    param_store = {}
    y = df["Output"]
    X = df.drop(["Output"], axis=1) 
    b_n = 1
    print("==== STARTING GT ==== {} -> {} ====".format(bac, ds_name))
    X_b = pd.read_csv('src/data/filtered_datasets/'+bac+'/GT/GT_feature_selection_'+ds_name+'.csv')
    b_n = len(X_b.columns)
    best_clf_res = run_SVM_with_features(X_b, y, "GT")
    best_test_score = best_clf_res["test_score"]
    for n in range(460, len(X.columns)):
        print("Testing {} Features GT".format(n))
        X_c = run_GT_feature_selection(n, X, y) 
        
        candidate_clf_res = run_SVM_with_features(X_c, y, "GT")
        param_store[n] = (candidate_clf_res, len(X_c.columns))
        candidate_test_score = candidate_clf_res["test_score"]
        t_test = compare_classifiers(best_test_score, candidate_test_score)
        if (t_test[0] < 0 and t_test[1] < 0.05):
            X_b = X_c
            best_clf_res = candidate_clf_res
            best_test_score = best_clf_res["test_score"]
            b_n = n
            print("New best score for {} - {} is |{}| with n -> |{}|".format(bac, ds_name, best_test_score, b_n))
            X_b.to_csv('src/data/filtered_datasets/'+bac+'/GT/GT_feature_selection_'+ds_name+'.csv')
            pd.DataFrame([b_n]).to_csv('src/data/filtered_datasets/'+bac+'/GT/GT_feature_selection_parameters_'+ds_name+'.csv')
        pd.DataFrame(param_store).to_csv('src/data/filtered_datasets/'+bac+'/GT/GT_hyper_data_'+ds_name+'.csv')
    print("GT FINISHED")
    print("=============================")
    print(best_clf_res)
    print("=============================")

def get_RRelieff_best_parameters(df, bac, ds_name):
    param_store = {}
    y = df["Output"]
    X = df.drop(["Output"], axis=1) 
    b_n = 3
    print("==== STARTING RRelief ==== {} -> {} ====".format(bac, ds_name))
    X_b = run_relieff_feature_selection(b_n, X, y)
    b_clf_res = run_SVM_with_features(X_b, y, "relieff")
    b_test_score = b_clf_res["test_score"]
    for n in range(1, len(X.columns)):
        print("Testing {} Features RRelief".format(n))
        X_c = run_relieff_feature_selection(n, X, y)
        c_clf_res = run_SVM_with_features(X_c, y, "relieff")
        param_store[n] = (c_clf_res, n)
        c_test_score = c_clf_res["test_score"]
        t_test = compare_classifiers(b_test_score, c_test_score)
        if (t_test[0] < 0 and t_test[1] < 0.05):
            X_b = X_c
            b_clf_res = c_clf_res
            b_test_score = b_clf_res["test_score"]
            b_n = n
    pd.DataFrame(X_b).to_csv('src/data/filtered_datasets/'+bac+'/relieff/relieff_feature_selection_'+ds_name+'.csv')
    pd.DataFrame([b_n]).to_csv('src/data/filtered_datasets/'+bac+'/relieff/relieff_feature_selection_parameters_'+ds_name+'.csv')
    pd.DataFrame(param_store).to_csv('src/data/filtered_datasets/'+bac+'/relieff/relieff_hyper_data_'+ds_name+'.csv')
    print("RRelief FINISHED")
    print("=============================")
    print(b_clf_res)
    print("=============================")

def get_pearson_best_parameters(df, bac, ds_name):
    y = df["Output"]
    param_store = {}
    t_values = [ 0.52, 0.5, 0.48, 0.46, 0.45, 0.43, 0.42, 0.4, 0.38, 0.36, 0.35, 0.33 , 0.32, 0.32, 0.28, 0.26, 0.25, 0.23, 0.2, 0.15, 0.11, 0.05, 0.011, 0]
    b_t = 0.52
    print("==== Pearson STARTED ==== {} -> {} ====".format(bac, ds_name))
    X_b = run_pearson_corr_feature_selection(b_t, df)
    b_clf_res = run_SVM_with_features(X_b, y, "pearson")
    b_test_score = b_clf_res["test_score"]
    for t in t_values:
        print("Testing {} Threshold Pearson".format(t))
        X_c = run_pearson_corr_feature_selection(t, df)
        c_clf_res = run_SVM_with_features(X_c, y, "pearson")
        param_store[t] = (c_clf_res, len(X_c.columns))
        c_test_score = c_clf_res["test_score"]
        t_test = compare_classifiers(b_test_score, c_test_score)
        if (t_test[0] <= 0 and t_test[1] <= 0.05):
            X_b = X_c
            b_clf_res = c_clf_res
            b_test_score = b_clf_res["test_score"]
            b_t = t
    X_b.to_csv('src/data/filtered_datasets/'+bac+'/pearson/pearson_feature_selection_'+ds_name+'.csv')
    pd.DataFrame([b_t]).to_csv('src/data/filtered_datasets/'+bac+'/pearson/pearson_feature_selection_parameters_'+ds_name+'.csv')
    pd.DataFrame(param_store).to_csv('src/data/filtered_datasets/'+bac+'/pearson/pearson_hyper_data_'+ds_name+'.csv')
    print("Pearson FINISHED")
    print("=============================")
    print(b_clf_res)
    print("=============================")

def run_all_models():
    datasets = ["aac", "bla", "dfr"]
    for i in datasets:
        df_ac = get_single_df("ac", i)
        create_thread(get_pearson_best_parameters, df_ac, "ac", i)
        # create_thread(get_RRelieff_best_parameters, df_ac, "ac", i)
        # create_thread(get_GT_best_parameters, df_ac, "ac", i)
        # if (i != "dfr" and i != "aac"):
            # create_thread(get_pearson_best_parameters, df_ps, "ps", i)
            # create_thread(get_RRelieff_best_parameters, df_ps, "ps", i)
    df_ps = get_single_df("ps", "bla")
    create_thread(get_GT_best_parameters, df_ps, "ps", "bla")


def main():
    run_all_models()

# if __name__ == "__main__":
#     main()

'''
{
    acc: [a,a,a,a,a,a,a]
}

{
    acc: [a,a,a,a,a,a,a]
}

func = stats.t_test(acc1,acc2) -> p (qual a probabilidade dos conjuntos serem diferentes)

 p> 0 -> acc1 melhor
 p< 0 -> acc2 melhor
 p = 0 -> igual
'''

# https://www.kaggle.com/ogrellier/parameter-tuning-5-x-2-fold-cv-statistical-test
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html

X, y = get_iris()
run_GT_feature_selection(3, X, y)
