import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from bayes_opt import BayesianOptimization
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


df = pd.read_csv('Training_dataset.csv')
X_trn = df.iloc[:, 10:]
y_trnn = df['MGMT_status']
y_trn = y_trnn.copy()
y_trn = y_trn.replace({'UNMETHYLATED': 0}, regex=True).replace({'METHYLATED': 1}, regex=True)
X = X_trn.dropna(axis='columns')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y_trn, test_size = 0.2, stratify = y_trn)

def opt_svc(C, gamma):
    model = SVC(
        C=float(C),
        gamma=float(gamma),
        kernel="rbf",
        class_weight="balanced"
    )
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring="recall")
    return scores.mean()

pbounds = {'C': (0.001, 10), 'gamma': (0.001, 0.1)}

bayopt = BayesianOptimization(f=opt_svc, pbounds=pbounds, random_state=42)
bayopt.maximize(init_points=10, n_iter=30)

best_params = bayopt.max['params']
print("Best hyperparameters:", best_params)

svc = SVC(
    C=float(best_params['C']),
    gamma=float(best_params['gamma']),
    kernel="rbf",
    class_weight="balanced"
)
svc.fit(X_train, y_train)


y_pred = svc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

y_scores = svc.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
