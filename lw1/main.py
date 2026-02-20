import numpy as np
import matplotlib.pyplot as plt
import DataGenerator as dg

mu = [[0, 2, 3, 4], [3, 5, 1, 6]]
sigma = [[2, 1, 2, 3], [1, 2, 1, 3]]
N = 1000
col = len(mu[0])

#---linear

# X, Y, class0, class1 = dg.norm_database(mu, sigma, N)

# trainCount = round(0.7*N*2)
# Xtrain = X[0:trainCount]
# Xtest = X[trainCount:N*2 + 1]
# Ytrain = Y[0:trainCount]
# Ytest = Y[trainCount:N*2 + 1] 


# for i in range(0, col):
    
#     _ = plt.hist(class0[:, i], bins='auto', alpha=0.7, label='Класс 0')
#     _ = plt.hist(class1[:, i], bins='auto', alpha=0.7, label='Класс 1')
#     plt.title('Гистограмма распределения признака ' + str(i+1))
#     plt.xlabel('Значение признака')
#     plt.ylabel('Частота')
#     plt.legend()
#     plt.savefig('hist2_'+str(i+1)+'.png')
#     plt.show()



# plt.scatter(class0[:, 1], class0[:, 2], marker=".", alpha=0.7,
#             label='Класс 0')
# plt.scatter(class1[:, 1], class1[:, 2], marker=".", alpha=0.7,
#             label='Класс 1')
# plt.title('Скатерограмма распределения признаков 2 и 3 ')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# plt.savefig('scatter2_'+str(col)+'.png')
# plt.show()

#---nonlinear
N = 1000

X, Y, class0, class1 = dg.nonlinear_dataset_13(N)

col = X.shape[1]

trainCount = round(0.7 * X.shape[0])

Xtrain = X[:trainCount]
Xtest = X[trainCount:]

Ytrain = Y[:trainCount]
Ytest = Y[trainCount:]

for i in range(col):
    plt.figure()

    plt.hist(class0[:, i], bins='auto', alpha=0.7, label='Класс 0')
    plt.hist(class1[:, i], bins='auto', alpha=0.7, label='Класс 1')

    plt.title(f'Гистограмма распределения признака {i}')
    plt.xlabel('Значение')
    plt.ylabel('Частота')
    plt.legend()
    plt.savefig(f'Hist_{i}.png')
    plt.show()

plt.figure()
plt.scatter(class0[:, 0], class0[:, 1], marker=".", alpha=0.7,
            label='Класс 0')
plt.scatter(class1[:, 0], class1[:, 1],
            marker=".", alpha=0.7, label='Класс 1')
plt.title('Нелинейный датасет (вариант 13)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.axis('equal') 
plt.savefig('Scatter.png')
plt.show()
