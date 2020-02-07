from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
import pickle
import numpy as np


print("\n 1st Q Table: \n")

with open("/Users/mgavrilov/Study/ENSEMBLEALGS/learn/main_QTable.pkl", 'rb') as f:
    Q_table_1st = pickle.load(f)
    # print(Q_table_1st)

print("\n 2nd Q Table: \n")

with open("/Users/mgavrilov/Study/ENSEMBLEALGS/learn/rotated_QTable.pkl", 'rb') as f2:
    Q_table_2nd = pickle.load(f2)
    # print(Q_table_2nd)

print(Q_table_1st.shape)

Q_table_adaboost = np.zeros([76, 7])

qtable_X = Q_table_1st
qtable_Y = Q_table_2nd

qtable_decisioned = np.zeros(76)

qtable_decisioned_X = np.zeros(76)
qtable_decisioned_Y = np.zeros(76)
max_index_X = -1
max_value_X = -1
for i in range (76):
    max_index_X = -1
    max_value_X = -1
    for j in range (7):
        if (qtable_X[i][j] > max_value_X):
            max_value_X = qtable_X[i][j]
            max_index_X = j
            print('Index: ', max_index_X, '\n')
            qtable_decisioned_X[i] = max_index_X


print(qtable_decisioned_X)

print('\n')

max_index_Y = -1
max_value_Y = -1
for i in range (76):
    max_index_Y = -1
    max_value_Y = -1
    for j in range (7):
        if (qtable_Y[i][j] > max_value_Y):
            max_value_Y = qtable_Y[i][j]
            max_index_Y = j
    qtable_decisioned_Y[i] = max_index_Y

print(qtable_decisioned_Y)

# print('qtable_decisioned_result: \n', qtable_decisioned ,'\n' )
# print('qtable_decisioned_shape \n', qtable_decisioned.shape, '\n')
#
#
# print('Result shape: \n', qtable_decisioned.shape, '\n')
# print('Result Qtable \n', qtable_decisioned, '\n')
#
#
# print('Test x: \n', qtable_X.shape, '\n')
# print('Test y: \n', qtable_decisioned.shape, '\n')
#
# print('Fact x: \n', qtable_X, '\n')
# print('Fact y: \n', qtable_decisioned, '\n')


abc = AdaBoostClassifier(n_estimators=50, learning_rate=1)

model = abc.fit(qtable_X, qtable_decisioned_X)

y_pred = model.predict(qtable_Y)

print("Accuracy:", metrics.accuracy_score(qtable_decisioned_X, y_pred))

print ('y_pred ', y_pred , '\n')
# print('y_pred.shape ', y_pred.shape,'\n')

for i in range (76):
        for j in range (7):
            if (y_pred[i] == 0):
                Q_table_adaboost[i][0] = 0

            if (y_pred[i] == 1):
                Q_table_adaboost[i][1] = 111

            if (y_pred[i] == 2):
                Q_table_adaboost[i][2] = 222

            if (y_pred[i] == 3):
                Q_table_adaboost[i][3] = 333

            if (y_pred[i] == 4):
                Q_table_adaboost[i][4] = 444

            if (y_pred[i] == 5):
                Q_table_adaboost[i][5] = 555

            if (y_pred[i] == 6):
                Q_table_adaboost[i][6] = 666

            if (y_pred[i] == 7):
                Q_table_adaboost[i][7] = 777


for i in range (76):
        for j in range (7):
            if (Q_table_adaboost[i][j] == 0):
                if(qtable_decisioned_X[i] >= qtable_decisioned_Y[i]):

                    if (qtable_decisioned_X[i] == 0):
                        Q_table_adaboost[i][0] = 0

                    if (qtable_decisioned_X[i] == 1):
                        Q_table_adaboost[i][1] = 111

                    if (qtable_decisioned_X[i] == 2):
                        Q_table_adaboost[i][2] = 222

                    if (qtable_decisioned_X[i] == 3):
                        Q_table_adaboost[i][3] = 333

                    if (qtable_decisioned_X[i] == 4):
                        Q_table_adaboost[i][4] = 444

                    if (qtable_decisioned_X[i] == 5):
                        Q_table_adaboost[i][5] = 555

                    if (qtable_decisioned_X[i] == 6):
                        Q_table_adaboost[i][6] = 666

                    if (qtable_decisioned_X[i] == 7):
                        Q_table_adaboost[i][7] = 777

                elif (qtable_decisioned_X[i] < qtable_decisioned_Y[i]):

                    if (qtable_decisioned_Y[i] == 0):
                        Q_table_adaboost[i][0] = 0

                    if (qtable_decisioned_Y[i] == 1):
                        Q_table_adaboost[i][1] = 111

                    if (qtable_decisioned_Y[i] == 2):
                        Q_table_adaboost[i][2] = 222

                    if (qtable_decisioned_Y[i] == 3):
                        Q_table_adaboost[i][3] = 333

                    if (qtable_decisioned_Y[i] == 4):
                        Q_table_adaboost[i][4] = 444

                    if (qtable_decisioned_Y[i] == 5):
                        Q_table_adaboost[i][5] = 555

                    if (qtable_decisioned_Y[i] == 6):
                        Q_table_adaboost[i][6] = 666

                    if (qtable_decisioned_Y[i] == 7):
                        Q_table_adaboost[i][7] = 777



print('AdaBoost QTable: \n', Q_table_adaboost, '\n')
print(Q_table_adaboost.shape)


with open("Ada_Boost_QTable.pkl", 'wb') as f3:
    pickle.dump(Q_table_adaboost, f3)