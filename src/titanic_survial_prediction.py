"""
Titanic Survival Prediction
by Justin Hu
2022/02/19
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
        self.df = None

    def load_data(self):
        """
        load data from csv
        :return:
        """
        self.df = pd.read_csv('train.csv')
        print(f'data loaed. data shape: {self.df.shape}')

    def explore_data(self):
        """
        further explore data set
        :return:
        """
        # General info about the dataset
        # the train data is a 891 x 12 matrix.
        # of 891 rows,  342 survived.
        print(f'data set info: {self.df.info}')
        print(f'columns: {self.df.columns}')
        print(f"total rows: {self.df.shape[0] }, number of survived: "
              f"{ self.df.loc[self.df['Survived'] == 1].shape[0]}, "
              f"number of unsurvived: {self.df.loc[self.df['Survived'] == 0].shape[0]}")

        # check null value in data, seem like we have some missing value for Cabin number
        # 687 rows have missing cabin number.
        # Since the majority rows have missed  cabin number,  I will not use it as a feature
        null_check = self.df.isnull()
        print(null_check)
        print(f"{ null_check.loc[null_check['Cabin'] == True].shape[0]} rows have missing cabin number")


        # pclass types
        #Pclass values: [1 2 3]
        print(f" Pclass values: {np.unique(self.df['Pclass'])}")

        # Sex values
        # ['female' 'male']
        print(f" Sex values: {np.unique(self.df['Sex'])}")


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
        print(self.df[['Age', 'SibSp', 'Parch', 'Fare']].describe())

        #explore correlations with different columns


    def data_preprocessing(self):
        pass


def main():
    c = TitanicSurvival()
    c.load_data()
    c.explore_data()



if __name__ == '__main__':
    main()
