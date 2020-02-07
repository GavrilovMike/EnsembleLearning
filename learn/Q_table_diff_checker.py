import pickle


with open("/Users/mgavrilov/Study/ENSEMBLEALGS/ensebmles/Bagging/Bagging_QTable.pkl", 'rb') as f:
    Q_table = pickle.load(f)
    print('Q_table first: \n',Q_table)

with open("/Users/mgavrilov/Study/ENSEMBLEALGS/ensebmles/Boosting/Ada_Boost_QTable.pkl", 'rb') as f2:
    Q_table_2 = pickle.load(f2)
    print('Q_table second: \n', Q_table_2)



print('Diff Checker: \n', Q_table == Q_table_2, '\n')