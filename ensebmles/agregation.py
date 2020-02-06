import pickle
import numpy as np

print("\n 1st Q Table: \n")

with open("/Users/mgavrilov/Study/ENSEMBLEALGS/learn/se11.pkl", 'rb') as f:
    Q_table_1st = pickle.load(f)
    print(Q_table_1st)

print("\n 2nd Q Table: \n")

with open("/Users/mgavrilov/Study/ENSEMBLEALGS/learn/se11_rotated.pkl", 'rb') as f2:
    Q_table_2nd = pickle.load(f2)
    print(Q_table_2nd)

print(Q_table_1st.shape)

Q_table_agregated = np.zeros([76, 7])

print(Q_table_agregated)



print(Q_table_agregated)
for i in range (76):
    for j in range (7):
        Q_table_agregated[i][j] = (Q_table_1st[i][j] + Q_table_2nd[i][j]) / 2
print(Q_table_agregated)
print(Q_table_agregated.shape)

with open("Agregation.pkl", 'wb') as f3:
    pickle.dump(Q_table_agregated, f3)