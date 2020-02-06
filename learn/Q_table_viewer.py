import pickle


with open("main_QTable.pkl", 'rb') as f:
    Q_table = pickle.load(f)
    print('Q_table first: \n',Q_table)

with open("rotated_QTable.pkl", 'rb') as f2:
    Q_table_2 = pickle.load(f2)
    print('Q_table second: \n', Q_table_2)
    