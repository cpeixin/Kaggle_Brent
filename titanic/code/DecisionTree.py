import pandas as pd
from sklearn.feature_extraction import DictVectorizer

def main():
    train_data = pd.read_csv("/Users/cpeixin/PycharmProjects/Kaggle_Brent/titanic/data/train.csv")
    test_data = pd.read_csv("/Users/cpeixin/PycharmProjects/Kaggle_Brent/titanic/data/test.csv")
    # 用平均值补齐缺失值
    train_data["Age"].fillna(train_data["Age"].median(), inplace=True)
    # 用大多数值补齐缺失值
    train_data['Embarked'].fillna('S', inplace=True)

    # 用平均值补齐缺失值
    test_data["Age"].fillna(test_data["Age"].median(), inplace=True)
    # 平均值补齐测试数据中的缺失值
    test_data['Fare'].fillna((test_data['Pclass'] == 3).median(), inplace=True)
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

    train_features = train_data[features]
    train_labels = train_data['Survived']
    test_features = test_data[features]

    # 对分类值特征，采用DictVectorizer来处理
    dvec = DictVectorizer(sparse=False)
    # 处理符号化对象，并且特征向量转化为特征值矩阵
    train_features = dvec.fit_transform(train_features.to_dict(orient='record'))

    # 创建决策树模型
    from sklearn.tree import DecisionTreeClassifier
    # 创建决策树
    clf = DecisionTreeClassifier(criterion='entropy')
    # 训练
    clf.fit(train_features, train_labels)

    test_features = dvec.fit_transform(test_features.to_dict(orient='record'))
    pred_labels = clf.predict(test_features)

    PassengerIds = test_data['PassengerId']
    # print(pred_labels)
    # print(PassengerIds)

    result = pd.DataFrame(
        {'PassengerId': test_data['PassengerId'], 'Survived': pred_labels})
    result.to_csv("/Users/cpeixin/PycharmProjects/Kaggle_Brent/titanic/data/result.csv", index=False)


if __name__ == '__main__':
    main()