import pandas as pd
import numpy as np
import threading
from multiprocessing import Process
from scipy.stats import ttest_ind
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.datasets import load_iris
from data.data_provider import get_full_data_frame, get_sample_dataframe, get_selected_feature_data_frame, get_single_dataframe
from feature_selection.GT_feature_selection import GT_feature_selection
from feature_selection.relieff_algorithm import Relieff
from feature_selection.pearson_corr_selection import pearson_corr_feature_selection

FEATURE_SIZE = 125

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

def discretize(df):
    for i in df.columns:
        if i != "Feature" and i != "Output":
            df[i] = pd.qcut(df[i], q=5,  labels=False, precision=0, duplicates='drop')
    return df

def get_single_df(bac, ds_name):
    df = get_single_dataframe(bac, ds_name)
    df = discretize(df)
    y = df["Output"]
    X = df.drop(["Output","Feature"], axis=1)
    return pd.concat([X, y], axis=1)

def run_SVM_with_features(X, y):
    clf = svm.SVC()
    return cross_validate(clf, X, y, cv=10, return_train_score=True)

def run_LR_with_features(X, y):
    clf = LogisticRegression(max_iter=200)
    return cross_validate(clf, X, y, cv=10, return_train_score=True)

def run_GT_feature_selection(n, X, y):
    return GT_feature_selection(pd.concat([X, y], axis=1), X, y, n)

def run_relieff_feature_selection(n, X, y):
    X = Relieff(X.to_numpy(), y, n)
    return X

def run_pearson_corr_feature_selection(n, df):
    X = pearson_corr_feature_selection(df, n)
    return X

def compare_classifiers(a, b):
    p = ttest_ind(a, b, equal_var=False)
    return p

def get_GT_best_parameters(df, bac, ds_name):
    param_store_SVM = {}
    param_store_LR = {}

    y = df["Output"]
    X = df.drop(["Output"], axis=1) 

    best_n_SVM = 1
    best_n_LR = 1

    X_b_SVM = X.sample(n=best_n_SVM,axis='columns')
    X_b_LR = X.sample(n=best_n_LR,axis='columns')

    print("==== STARTING GT ==== {} -> {} ====".format(bac, ds_name))

    best_clf_res_SVM = run_SVM_with_features(X_b_SVM, y)
    best_clf_res_LR = run_LR_with_features(X_b_LR, y)

    best_test_score_SVM = best_clf_res_SVM["test_score"]
    best_test_score_LR = best_clf_res_LR["test_score"]

    for n in range(1, FEATURE_SIZE + 1):
        print("Testing {} Features GT".format(n))
        X_c = run_GT_feature_selection(n, X, y)
        
        candidate_clf_res_SVM = run_SVM_with_features(X_c, y)
        candidate_clf_res_LR = run_LR_with_features(X_c, y)

        param_store_SVM[n] = (candidate_clf_res_SVM, len(X_c.columns))
        param_store_LR[n] = (candidate_clf_res_LR, len(X_c.columns))

        candidate_test_score_SVM = candidate_clf_res_SVM["test_score"]
        candidate_test_score_LR = candidate_clf_res_LR["test_score"]

        # Test SVM clf with candidate features
        t_test = compare_classifiers(best_test_score_SVM, candidate_test_score_SVM)
        if (t_test[0] < 0 and t_test[1] < 0.05):
            X_b_SVM = X_c
            best_clf_res_SVM = candidate_clf_res_SVM
            best_test_score_SVM = best_clf_res_SVM["test_score"]
            best_n_SVM = n
            print("New best GT SVM score for {} - {} is |{}| with n -> |{}|".format(bac, ds_name, best_test_score_SVM, best_n_SVM))
            X_b_SVM.to_csv('src/data/filtered_datasets/'+bac+'/SVM/GT/GT_feature_selection_'+ds_name+'.csv')
            pd.DataFrame([best_n_SVM]).to_csv('src/data/filtered_datasets/'+bac+'/SVM/GT/GT_feature_selection_parameters_'+ds_name+'.csv')
        
        # Test LR clf with candidate features
        t_test = compare_classifiers(best_test_score_LR, candidate_test_score_LR)
        if (t_test[0] < 0 and t_test[1] < 0.05):
            X_b_LR = X_c
            best_clf_res_LR = candidate_clf_res_LR
            best_test_score_LR = best_clf_res_LR["test_score"]
            best_n_LR = n
            print("New best GT LR score for {} - {} is |{}| with n -> |{}|".format(bac, ds_name, best_test_score_LR, best_n_LR))
            X_b_LR.to_csv('src/data/filtered_datasets/'+bac+'/LR/GT/GT_feature_selection_'+ds_name+'.csv')
            pd.DataFrame([best_n_LR]).to_csv('src/data/filtered_datasets/'+bac+'/LR/GT/GT_feature_selection_parameters_'+ds_name+'.csv')
        
        pd.DataFrame(param_store_SVM).to_csv('src/data/filtered_datasets/'+bac+'/SVM/GT/GT_hyper_data_'+ds_name+'.csv')
        pd.DataFrame(param_store_LR).to_csv('src/data/filtered_datasets/'+bac+'/LR/GT/GT_hyper_data_'+ds_name+'.csv')
    print("GT FINISHED")
    print("=============================")
    print(best_clf_res_SVM)
    print(best_clf_res_LR)
    print("=============================")

def get_RRelieff_best_parameters(df, bac, ds_name):
    param_store_SVM = {}
    param_store_LR = {}

    y = df["Output"]
    X = df.drop(["Output"], axis=1) 

    best_n_SVM = 1
    best_n_LR = 1

    X_b_SVM = X.sample(n=best_n_SVM,axis='columns')
    X_b_LR = X.sample(n=best_n_LR,axis='columns')

    print("==== STARTING RRelief ==== {} -> {} ====".format(bac, ds_name))

    best_clf_res_SVM = run_SVM_with_features(X_b_SVM, y)
    best_clf_res_LR = run_LR_with_features(X_b_LR, y)

    best_test_score_SVM = best_clf_res_SVM["test_score"]
    best_test_score_LR = best_clf_res_LR["test_score"]

    for n in range(1, FEATURE_SIZE + 1):
        print("Testing {} Features RRelief --- {} {}".format(n, bac, ds_name))
        X_c = run_relieff_feature_selection(n, X, y)
        candidate_clf_res_SVM = run_SVM_with_features(X_c, y)
        candidate_clf_res_LR = run_LR_with_features(X_c, y)

        param_store_SVM[n] = (candidate_clf_res_SVM, n)
        param_store_LR[n] = (candidate_clf_res_LR, n)

        candidate_test_score_SVM = candidate_clf_res_SVM["test_score"]
        candidate_test_score_LR = candidate_clf_res_LR["test_score"]

        t_test = compare_classifiers(best_test_score_SVM, candidate_test_score_SVM)
        if (t_test[0] < 0 and t_test[1] < 0.05):
            X_b_SVM = X_c
            best_clf_res_SVM = candidate_clf_res_SVM
            best_test_score_SVM = best_clf_res_SVM["test_score"]
            best_n_SVM = n
            print("New best Relieff SVM score for {} - {} is |{}| with n -> |{}|".format(bac, ds_name, best_test_score_SVM, best_n_SVM))
            pd.DataFrame(X_b_SVM).to_csv('src/data/filtered_datasets/'+bac+'/SVM/relieff/relieff_feature_selection_'+ds_name+'.csv')
            pd.DataFrame([best_n_SVM]).to_csv('src/data/filtered_datasets/'+bac+'/SVM/relieff/relieff_feature_selection_parameters_'+ds_name+'.csv')
        
        # Test LR clf with candidate features
        t_test = compare_classifiers(best_test_score_LR, candidate_test_score_LR)
        if (t_test[0] < 0 and t_test[1] < 0.05):
            X_b_LR = X_c
            best_clf_res_LR = candidate_clf_res_LR
            best_test_score_LR = best_clf_res_LR["test_score"]
            best_n_LR = n
            print("New best Relieff LR score for {} - {} is |{}| with n -> |{}|".format(bac, ds_name, best_test_score_LR, best_n_LR))
            pd.DataFrame(X_b_LR).to_csv('src/data/filtered_datasets/'+bac+'/LR/relieff/relieff_feature_selection_'+ds_name+'.csv')
            pd.DataFrame([best_n_LR]).to_csv('src/data/filtered_datasets/'+bac+'/LR/relieff/relieff_feature_selection_parameters_'+ds_name+'.csv')
        
        pd.DataFrame(param_store_SVM).to_csv('src/data/filtered_datasets/'+bac+'/SVM/relieff/relieff_hyper_data_'+ds_name+'.csv')
        pd.DataFrame(param_store_LR).to_csv('src/data/filtered_datasets/'+bac+'/LR/relieff/relieff_hyper_data_'+ds_name+'.csv')
    print("RRelief FINISHED")
    print("=============================")
    print(best_clf_res_SVM)
    print(best_clf_res_LR)
    print("=============================")

def get_pearson_best_parameters(df, bac, ds_name):
    param_store_SVM = {}
    param_store_LR = {}
    # df_c = df.copy(deep=True)
    y = df["Output"]

    best_n_SVM = 1
    best_n_LR = 1

    X_b_SVM = pearson_corr_feature_selection(df, best_n_SVM)
    X_b_LR = pearson_corr_feature_selection(df, best_n_LR)
    
    print("==== STARTING Pearson ==== {} -> {} ====".format(bac, ds_name))

    best_clf_res_SVM = run_SVM_with_features(X_b_SVM, y)
    best_clf_res_LR = run_LR_with_features(X_b_LR, y)

    best_test_score_SVM = best_clf_res_SVM["test_score"]
    best_test_score_LR = best_clf_res_LR["test_score"]

    for n in range(1, FEATURE_SIZE + 1):
        print("Testing {} Features Pearson --- {} {}".format(n, bac, ds_name))
        X_c = pearson_corr_feature_selection(df, n)
        candidate_clf_res_SVM = run_SVM_with_features(X_c, y)
        candidate_clf_res_LR = run_LR_with_features(X_c, y)

        param_store_SVM[n] = (candidate_clf_res_SVM, n)
        param_store_LR[n] = (candidate_clf_res_LR, n)

        candidate_test_score_SVM = candidate_clf_res_SVM["test_score"]
        candidate_test_score_LR = candidate_clf_res_LR["test_score"]

        t_test = compare_classifiers(best_test_score_SVM, candidate_test_score_SVM)
        if (t_test[0] < 0 and t_test[1] < 1):
            X_b_SVM = X_c
            best_clf_res_SVM = candidate_clf_res_SVM
            best_test_score_SVM = best_clf_res_SVM["test_score"]
            best_n_SVM = n
            print("New best Pearson SVM score for {} - {} is |{}| with n -> |{}|".format(bac, ds_name, best_test_score_SVM, best_n_SVM))
            pd.DataFrame(X_b_SVM).to_csv('src/data/filtered_datasets/'+bac+'/SVM/pearson/pearson_feature_selection_'+ds_name+'.csv')
            pd.DataFrame([best_n_SVM]).to_csv('src/data/filtered_datasets/'+bac+'/SVM/pearson/pearson_feature_selection_parameters_'+ds_name+'.csv')
        
        # Test LR clf with candidate features
        t_test = compare_classifiers(best_test_score_LR, candidate_test_score_LR)
        if (t_test[0] < 0 and t_test[1] < 1):
            X_b_LR = X_c
            best_clf_res_LR = candidate_clf_res_LR
            best_test_score_LR = best_clf_res_LR["test_score"]
            best_n_LR = n
            print("New best Pearson LR score for {} - {} is |{}| with n -> |{}|".format(bac, ds_name, best_test_score_LR, best_n_LR))
            pd.DataFrame(X_b_LR).to_csv('src/data/filtered_datasets/'+bac+'/LR/pearson/pearson_feature_selection_'+ds_name+'.csv')
            pd.DataFrame([best_n_LR]).to_csv('src/data/filtered_datasets/'+bac+'/LR/pearson/pearson_feature_selection_parameters_'+ds_name+'.csv')
        
        pd.DataFrame(param_store_SVM).to_csv('src/data/filtered_datasets/'+bac+'/SVM/pearson/pearson_hyper_data_'+ds_name+'.csv')
        pd.DataFrame(param_store_LR).to_csv('src/data/filtered_datasets/'+bac+'/LR/pearson/pearson_hyper_data_'+ds_name+'.csv')
    print("Pearson FINISHED")
    print("=============================")
    print(best_clf_res_SVM)
    print(best_clf_res_LR)
    print("=============================")

def run_all_models():
    datasets = ["aac"]
    for i in datasets:
        df_ac = get_single_df("ac", i)
        create_thread(get_pearson_best_parameters, df_ac, "ac", i)
        # create_thread(get_RRelieff_best_parameters, df_ac, "ac", i)
        # create_thread(get_GT_best_parameters, df_ac, "ac", i)
        # if (i != "dfr" and i != "aac"):
        #     create_thread(get_pearson_best_parameters, df_ps, "ps", i)
        #     create_thread(get_RRelieff_best_parameters, df_ps, "ps", i)
    # df_ps = get_single_df("ps", "bla")
    # create_thread(get_GT_best_parameters, df_ps, "ps", "bla")


def main():
    run_all_models()

if __name__ == "__main__":
    main()

# df_ac = get_single_df("ac", "aac")
# get_pearson_best_parameters(df_ac, "ac", "aac")