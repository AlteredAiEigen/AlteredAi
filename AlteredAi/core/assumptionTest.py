import statsmodels.api as sm
import numpy as np
import scipy.stats as stats
from statsmodels.stats.stattools import durbin_watson
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def check_linear_regression_assumptions(X, y,plot=False):
    '''
    :param X: Training Features - numpy array
    :param y: Target - numpy array
    :param plot: True - if you want plots for every assumption and false otherwise
    :return: assumption test result

    >> np.random.seed(0)
    >> n = 100
    >> X = np.random.normal(0, 1, (n, 3))
    >> y = 10 + 2*X[:, 0] + 3*X[:, 1] + 4*X[:, 2] + np.random.normal(0, 1, n)
    >> check_linear_regression_assumptions(X, y,plot=False)

    '''
    # Add a constant term to the input data
    X = sm.add_constant(X)

    # Fit the linear regression model
    model = sm.OLS(y, X).fit()

    # Check the linearity assumption
    y_pred = model.predict(X)
    residual = y - y_pred
    linearity = np.allclose(y_pred, X.dot(model.params), rtol=1e-3)
    if linearity:
        print("Linearity:", '\033[32m' + '✔' + '\033[0m')
    else:
        print("Linearity:", '\033[31m' + '✘' + '\033[0m')
    if plot:
        plt.scatter(y, y_pred)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Linearity Check")
        plt.show()

    # Check the normality assumption
    _, p_value = stats.normaltest(residual)
    normality = p_value > 0.05
    if normality:
        print("Normality:", '\033[32m' + '✔' + '\033[0m')
    else:
        print("Normality:", '\033[31m' + '✘' + '\033[0m')
    if plot:
        sns.displot(residual, kind='kde')
        fig = sm.qqplot(residual, line='r')
        plt.title("Normality Check")
        plt.show()


    res = model.resid
    # Check the homoscedasticity assumption
    homoscedasticity = np.allclose(np.std(residual), np.sqrt(np.mean(residual**2)), rtol=1e-3)
    if homoscedasticity:
        print("Homoscedasticity:", '\033[32m' + '✔' + '\033[0m')
    else:
        print("Homoscedasticity:", '\033[31m' + '✘' + '\033[0m')
    if plot:
        plt.scatter(y_pred, res)
        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        plt.title("Homoscedasticity Check")
        plt.show()

    # Check the independence assumption
    d = durbin_watson(residual)
    independence = d > 1.5 and d < 2.5
    if independence:
        print("Independence:", '\033[32m' + '✔' + '\033[0m')
    else:
        print("Independence:", '\033[31m' + '✘' + '\033[0m')
    if plot:
        fig = sm.graphics.tsa.plot_acf(residual, lags=40)
        plt.title("Independence Check")
        plt.show()

    # Check for multicollinearity using the variance inflation factor (VIF)
    vif = np.array([sm.OLS(X[:, i], X[:, :i]).fit().rsquared for i in range(1, X.shape[1])])
    multicollinearity = np.all(vif < 5)
    if multicollinearity:
        print("Multicollinearity:", '\033[32m' + '✔' + '\033[0m')
    else:
        print("Multicollinearity:", '\033[31m' + '✘' + '\033[0m')
    if plot:
        vif = np.array([sm.OLS(X[:, i], X[:, :i]).fit().rsquared for i in range(1, X.shape[1])])
        print("Variance inflation factors (VIF):",vif)



def evaluate_metrics(y_true, y_pred,plot=False):
    '''

    :param y_true: Truth value
    :param y_pred: model prediction
    :param plot: True if you want plot otherwise false
    :return:acc,prec,rec,f1,auc

    >> y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0])
    >> y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 0, 1, 1])
    >> acc, prec, rec, f1, auc = evaluate_metrics(y_true, y_pred,plot=True)

    '''
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    # Print the metrics
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 score:", f1)
    print("ROC AUC score:", auc)

    # Check if the model is overfitting or generalizing well
    if acc > 0.95 and auc > 0.95:
        print("The model is likely overfitting the training data.")
    elif acc < 0.80 or auc < 0.80:
        print("The model may not be generalizing well to new data.")
    else:
        print("The model is likely generalizing well to new data.")

    # Plot the ROC curve
    if plot:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.show()

    # Plot the confusion matrix
    if plot:
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()

    return acc, prec, rec, f1, auc
