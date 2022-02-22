"""
Titanic Survival Prediction
by Justin Hu
2022/02/19
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import seaborn as sb

class TitanicSurvival:
    """
    a class for training three models to predict the survival status for Titanic shipwreck
    Including:
        1. load data
        2. explore data
        3. data preprocessing
        4. model training and selection(performance comparison)
        5. make predications and output it to kaggle submission format

    Algorithms compared here:
        1. logistic regression
        2. Random forest
        3. XGBoost

    Data set columns:
        survival	Survival	0 = No, 1 = Yes
        pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
        sex	Sex
        Age	Age in years
        sibsp	# of siblings / spouses aboard the Titanic
        parch	# of parents / children aboard the Titanic
        ticket	Ticket number
        fare	Passenger fare
        cabin	Cabin number
        embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton
    """

    def __init__(self):
        self.df_train = None
        self.df_test = None

    def load_data(self):
        """
        load data from csv
        :return:
        """
        self.df_train = pd.read_csv('train.csv')
        self.df_train.name = 'Training Set'
        print(f'train data loaded. data shape: {self.df_train.shape}')
        self.df_test = pd.read_csv('test.csv')
        self.df_test.name = 'Test Set'
        print(f'test data loaded. data shape: {self.df_test.shape}')

    def explore_data(self):
        """
        further explore data set
        :return:
        """
        # General info about the dataset
        # the train data is a 891 x 12 matrix.
        # of 891 rows,  342 survived.
        print(f'data set info:')
        print(self.df_train.info())

        # Show Summary Stats for Age, SibSp, Parch, Fare
        #            Age       SibSp       Parch        Fare
        # count  714.000000  891.000000  891.000000  891.000000
        # mean    29.699118    0.523008    0.381594   32.204208
        # std     14.526497    1.102743    0.806057   49.693429
        # min      0.420000    0.000000    0.000000    0.000000
        # 25%     20.125000    0.000000    0.000000    7.910400
        # 50%     28.000000    0.000000    0.000000   14.454200
        # 75%     38.000000    1.000000    0.000000   31.000000
        # max     80.000000    8.000000    6.000000  512.329200

        # Wow,  there are also some missing for age column
        print(self.df_train.describe())

        number_survived = self.df_train.loc[self.df_train['Survived'] == 1].shape[0]
        number_unsurvived = self.df_train.loc[self.df_train['Survived'] == 0].shape[0]

        print(f"total rows: {self.df_train.shape[0] }, number of survived: "
              f"{ number_survived }, "
              f"number of unsurvived: { number_unsurvived }, "
              f"survival ratio { number_survived / (number_survived + number_unsurvived)}")

        # pclass types
        #Pclass values: [1 2 3]
        print(f" Pclass values: {np.unique(self.df_train['Pclass'])}")

        # Sex values
        # ['female' 'male']
        print(f" Sex values: {np.unique(self.df_train['Sex'])}")

        #Missing values
        #              Total     %
        # Cabin          687  77.1
        # Age            177  19.9
        # Embarked         2   0.2
        missing = self.df_train.isnull().sum().sort_values(ascending=False)
        percent = self.df_train.isnull().sum() / self.df_train.isnull().count() * 100
        percent = (round(percent, 1)).sort_values(ascending=False)
        missing = pd.concat([missing, percent], axis=1, keys=['Total', '%'])
        print(missing)


        # exlore data further with visualization
        # survived vs unsurvied
        f, ax = plt.subplots()
        ax.set_title('Survived')

        sb.countplot('Survived', data=self.df_train, ax=ax)
        plt.show()

        # Pclass, Age vs Survived
        f, ax = plt.subplots()
        sb.violinplot('Pclass', 'Age', data=self.df_train, split=True, hue='Survived', ax=ax)
        ax.set_title('Plass and Age vs Survived')
        ax.set_yticks(range(0, 110, 10))
        plt.show()

        # Pclass
        sb.barplot(x='Pclass', y='Survived', hue='Sex', data=self.df_train)
        plt.show()


        # Sex and Age vs Survivied
        f, ax = plt.subplots()
        sb.violinplot('Sex', 'Age', data=self.df_train, split=True, hue='Survived', ax=ax)
        ax.set_title('Sex and Age vs Survived')
        ax.set_yticks(range(0, 110, 10))
        plt.show()

        # Embarked vs Survived
        sb.factorplot('Embarked', 'Survived', data=self.df_train)
        plt.show()

        # SibSp and Parch, combined into family size
        data1 = self.df_train.copy()
        data1['Family_size'] = data1['SibSp'] + data1['Parch'] + 1
        print(data1['Family_size'].value_counts().sort_values(ascending=True))
        sb.factorplot('Family_size', 'Survived', data=data1, aspect=2.5)
        plt.show()


    def data_preprocessing(self, df):
        """
        data preprocessing
        :param df:
        :return:
        """
        pass


def main():
    c = TitanicSurvival()
    c.load_data()
    c.explore_data()


if __name__ == '__main__':
    main()
