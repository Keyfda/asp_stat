import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import DataGenerator as dg


mu = [[0, 2, 3], [3, 5, 1]]
sigma = [[2, 1, 2], [1, 2, 1]]
N = 1000
col = len(mu[0])

X, Y, class0, class1 = dg.norm_database(mu, sigma, N)

trainCount = round(0.7*N*2)
Xtrain = X[0:trainCount]
Xtest = X[trainCount:N*2 + 1]
Ytrain = Y[0:trainCount]
Ytest = Y[trainCount:N*2 + 1] 

Nvar = 13
clf = LogisticRegression(random_state=Nvar,solver='saga').fit( Xtrain, Ytrain)

pred_train = clf.predict(Xtrain)
pred_train_proba = clf.predict_proba(Xtrain)
pred_test = clf.predict(Xtest)
pred_test_proba = clf.predict_proba(Xtest)

# acc_train = clf.score(Xtrain, Ytrain)
# acc_test = clf.score(Xtest, Ytest)

train_acc = clf.score(Xtrain, Ytrain)
test_acc = clf.score(Xtest, Ytest)

acc_test_m = sum(pred_test == Ytest)/len(Ytest)

test_tp = sum(pred_test[Ytest == 1])
test_tn = sum(~pred_test[Ytest == 0])
test_fp = sum(pred_test[Ytest == 0])
test_fn = sum(~pred_test[Ytest == 1])
test_recall = test_tp / (test_tp + test_fn)
test_precision = test_tp / (test_tp + test_fp)

train_tp = sum(pred_train[Ytrain == 1])
train_tn = sum(~pred_train[Ytrain == 0])
train_fp = sum(pred_train[Ytrain == 0])
train_fn = sum(~pred_train[Ytrain == 1])
train_recall = train_tp / (train_tp + train_fn)
train_precision = train_tp / (train_tp + train_fp)

table = [[" ", "Число объектов", "Точность, %", "Чувствительность, %", "Специфичность, %"],
        ["Train", len(Ytrain), round(train_acc*100, 2), round(train_recall*100,2), round(train_precision*100, 2)],
        ["Test", len(Ytest), round(test_acc*100, 2), round(test_recall*100, 2), round(test_precision*100, 2)]]

t = []
for row in table:
    s = ""
    for x in row:
        s += str(x) + ' ' * (25 - len(str(x)))
    t.append(s)
print('\n'.join(t))

plt.hist(pred_train_proba[Ytrain, 1], bins='auto',alpha=0.7)
plt.hist(pred_train_proba[~Ytrain, 1], bins='auto',alpha=0.7)
plt.title("Результаты классификации, Train")
plt.savefig('hist_1_train.png')
plt.show(block=False)
plt.figure()
plt.hist(pred_test_proba[Ytest, 1], bins='auto',alpha=0.7)
plt.hist(pred_test_proba[~Ytest, 1], bins='auto',alpha=0.7)
plt.title("Результаты классификации, Test")
plt.savefig('hist_1_test.png')
plt.show(block=True)