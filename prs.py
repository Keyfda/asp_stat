'''pr1'''

# import csv

# with open('diabetes_data_upload.csv', newline='') as csvfile:
#     data = list(csv.reader(csvfile))
# # print(data)

# sum_num = 0
# for i in data[1:]:
#     sum_num = sum_num + float(i[0])
# avg = sum_num/(len(data)-1)
# print(avg)

# sum_fem = 0 
# sum_male = 0
# male_count=0
# female_count=0

# for i in data[1:]:
#     if i[1] == "Male":
#         sum_male = sum_male + float(i[0])
#         male_count +=1
#     else:
#         sum_fem = sum_fem + float(i[0])
#         female_count+=1
# avg_male = sum_male/male_count
# avg_female = sum_fem/female_count

# print(avg_female)
# print(avg_male)


# has_none = []
# has_diabetes = []
# has_obesity = []
# has_both = []

# obesity_index = data[0].index("Obesity")
# diabetes_index = data[0].index("class")

# for idx, i in enumerate(data[1:], start=1):
#     obesity = i[obesity_index] == "Yes"
#     diabetes = i[diabetes_index] == "Positive"
    
#     if diabetes and obesity:
#         has_both.append(idx)
#     elif diabetes and not obesity:
#         has_diabetes.append(idx)
#     elif not diabetes and obesity:
#         has_obesity.append(idx)
#     else:
#         has_none.append(idx)

# print("Both diabetes and obesity:", has_both)
# print("Diabetes only:", has_diabetes)
# print("Obesity only:", has_obesity)
# print("Neither:", has_none)



'''pr2'''

# import pandas as pd
# import numpy as np

# table = pd.read_csv('diabetes_data_upload.csv')
# # print(table)


# age = table["Age"]
# print (np.mean(age))

# age_male = table.loc[table["Gender"] == "Male", "Age"]
# age_female = table.loc[table["Gender"] == "Female", "Age"]

# print (np.mean(age_female))
# print (np.mean(age_male))

# print(age.mean())
# print(age_female.mean())
# print(age_male.mean())



'''pr3'''

'''pr4'''



'''pr5'''

# import pandas as pd
# from sklearn import preprocessing, model_selection
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
# import matplotlib.pyplot as plt
# import seaborn as sns

# df = pd.read_csv('diabetes_data_upload.csv')

# X = df.drop('class', axis=1)
# y = df['class'].map({'Negative': 0, 'Positive': 1})

# X['Age'] = preprocessing.scale(X[['Age']])

# categorical_cols = X.select_dtypes(include='object').columns
# enc = preprocessing.OneHotEncoder(sparse_output=False, drop='first')
# X_encoded = enc.fit_transform(X[categorical_cols])

# X_encoded_df = pd.DataFrame(X_encoded,
#                             columns=enc.get_feature_names_out(categorical_cols),
#                             index=X.index)

# X = X.drop(categorical_cols, axis=1)
# X = pd.concat([X, X_encoded_df], axis=1)

# X.to_csv("X.csv", index=False)
# y.to_csv("Y.csv", index=False)

# X_array = X.to_numpy()
# y_array = y.to_numpy()

# random_state = 13
# test_size = 0.2
# X_train, X_test, y_train, y_test = model_selection.train_test_split(
#     X_array, y_array, test_size=test_size, random_state=random_state)

# print("X_train ", X_train.shape)
# print("X_test ", X_test.shape)
# print("y_train ", y_train.shape)
# print("y_test ", y_test.shape)

# model = LogisticRegression(random_state=16, max_iter=1000)
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)
# y_pred_prob = model.predict_proba(X_test)[:, 1]

# accuracy = accuracy_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# roc_auc = roc_auc_score(y_test, y_pred_prob)

# print("Точность ", accuracy)
# print("F1 ", f1)
# print("ROC AUC ", roc_auc)

# conf_matrix = confusion_matrix(y_test, y_pred)
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
#             xticklabels=[0, 1], yticklabels=[0, 1])

# plt.xlabel('Предсказанные значения')
# plt.ylabel('Истинные значения')
# plt.title('Матрица ошибок')
# plt.show()

# fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# plt.plot(fpr, tpr, label=f'ROC-кривая (AUC = {roc_auc})')
# plt.plot([0, 1], [0, 1])
# plt.xlabel('Доля ложноположительных')
# plt.ylabel('Доля истинноположительных')
# plt.title('ROC-кривая')
# plt.legend()
# plt.show()

'''optimized'''

import pandas as pd
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('diabetes_data_upload.csv')

X = df.drop('class', axis=1)
y = df['class'].map({'Negative': 0, 'Positive': 1})

X['Age'] = preprocessing.scale(X[['Age']])

categorical_cols = X.select_dtypes(include='object').columns
enc = preprocessing.OneHotEncoder(sparse_output=False, drop='first')
X_encoded = enc.fit_transform(X[categorical_cols])

X_encoded_df = pd.DataFrame(X_encoded,
                            columns=enc.get_feature_names_out(categorical_cols),
                            index=X.index)

X = X.drop(categorical_cols, axis=1)
X = pd.concat([X, X_encoded_df], axis=1)

X_array = X.to_numpy()
y_array = y.to_numpy()

random_state = 13
test_size = 0.2
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X_array, y_array, test_size=test_size, random_state=random_state)

param_range = [100, 10, 1, 0.1, 0.01, 0.001]
best_accuracy = 0
best_C = None

for C in param_range:
    model = LogisticRegression(random_state=16, max_iter=1000, C=C)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    
    print(f"C = {C}")
    print(f"Точность: {accuracy:.3f}, F1: {f1:.3f}, ROC AUC: {roc_auc:.3f}")
    print("-------------------------------")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_C = C