from mlxtend.evaluate import paired_ttest_5x2cv

def paired_5x2_cv_test(clf1,clf2,X,y):
    t, p = paired_ttest_5x2cv(estimator1=clf1,
                            estimator2=clf2,
                            X=X, y=y,
                            random_seed=42)
    return [t,p]
    