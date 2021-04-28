from mlxtend.evaluate import paired_ttest_resampled

def resampled_paired_t_test(clf1, clf2, X, y):
    t, p = paired_ttest_resampled(estimator1=clf1,
                                estimator2=clf2,
                                X=X, y=y,
                                random_seed=42)
    return [t,p]