import pandas as pd

train_data = pd.read_csv("/Users/cpeixin/PycharmProjects/Kaggle_Brent/titanic/data/train.csv")
test_data = pd.read_csv("/Users/cpeixin/PycharmProjects/Kaggle_Brent/titanic/data/test.csv")



# pclass_fare = ['Pclass','Fare']
# print(test_data[pclass_fare])

# print(test_data.info())
# 筛选出Fare为空的列

print(test_data.loc[test_data['Pclass'] == 3])
print(test_data['Fare'].fillna((test_data['Pclass'] == 3).mean(), inplace=True))
# print(test_data.loc[test_data['Fare'].isnull()])


print(test_data.info())