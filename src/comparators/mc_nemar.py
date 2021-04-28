from sklearn.metrics import confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar

def mc_nemar_test(clf1_pred,clf2_pred,y_true1, y_true2):
    matrix_clf1 = confusion_matrix(y_true=y_true1, y_pred=clf1_pred).ravel()
    matrix_clf2 = confusion_matrix(y_true=y_true2, y_pred=clf2_pred).ravel()
    clf1_error = matrix_clf1[1] + matrix_clf1[2]
    clf1_correct = matrix_clf1[0] + matrix_clf1[3]
    clf2_error = matrix_clf2[1] + matrix_clf2[2]
    clf2_correct = matrix_clf2[0] + matrix_clf2[3]
    result = mcnemar([[clf1_correct,clf1_error],[clf2_correct,clf2_error]])
    return result

