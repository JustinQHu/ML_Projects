"""
Titanic Survival Prediction
by Justin Hu
2022/02/19
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

# For debugging purpose, allowing more columns be printed out in the console
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 20)


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
        self.df_all = None

    def load_data(self):
        """
        load data from csv
        To load data correctly, make sure csv files are in the same folder with py file.
        Otherwise, please change the path in pd.read_csv() function.
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
        # Conclusion:
        # 1. the train data is a 891 x 12 matrix. of 891 rows,  342 survived.
        # 2. of 12 columns,  Survived is targets.  name and passengerID are IDs.
        # 3. Pclass,  Age,  Sex do impact the probability of survival
        # 4. Family_size(SibSP + Parch) does impact the chance of survival
        # 5. Embarked may impact the probability of survival
        # 6. Cabin has a missing rate of 77.1%,  can't use it directly
        # 7. Age has a missing rate of 19.9, something needs to be done.
        # 8. Fare may impact the survival probability

        print(f'training data set info:')
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

        # Survival distribution across Age and Fare
        cont_feature = ['Age', 'Fare']
        surv = self.df_train['Survived'] == 1
        fig, axs = plt.subplots(ncols=2)
        for i, feature in enumerate(cont_feature):
            sb.histplot(self.df_train[~surv][feature], label='Not Survived', color='#e74c3c', ax=axs[i])
            sb.histplot(self.df_train[surv][feature], label='Survived', color='#2ecc71', ax=axs[i])
            axs[i].set_xlabel('')
            axs[i].legend(loc='upper right')
            axs[i].set_title(f'Distribution of Survival in {feature}')
        plt.show()

        # Survival Distribution across 'Embarked', 'Parch', 'Pclass', 'Sex', 'SibSp'
        cat_feature =['Embarked', 'Parch', 'Pclass', 'Sex', 'SibSp']
        fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(20, 20))
        for i, feature in enumerate(cat_feature, 1):
            plt.subplot(2, 3, i)
            sb.countplot(x=feature, hue='Survived', data=self.df_train)
            plt.ylabel('Passenger Count', size=20, labelpad=15)
            plt.tick_params(axis='x', labelsize=20)
            plt.tick_params(axis='y', labelsize=20)
            plt.legend(['Not Survived', 'Survived'], loc='upper center', prop={'size': 18})
            plt.title(f'Count of Survival in {feature} Feature',size=20, y=1.05)
        plt.show()

        # correlation between features# data.corr()-->correlation matrix
        sb.heatmap(self.df_train.corr(), annot=True, cmap='RdYlGn', linewidths=0.2)
        fig = plt.gcf()
        fig.set_size_inches(10, 8)
        plt.show()

    def data_preprocessing(self):
        """
        data preprocessing/feature engineering based on the result of exploratory data analysis
        preprocess both the training and test set together
        :param:
        :return:
        """
        self.df_all = pd.concat([self.df_train, self.df_test], sort=True).reset_index(drop=True)
        self.df_all.name = 'All Set'
        print()
        print('_' * 25)
        print(f'total data set info:')
        print(self.df_all.info())
        print(self.df_all.head(10))

        # filling Missing values in Age, Embarked and Fare with descriptive statistical measures

        # filling missing values in Age with median age of the same pclass and sex
        print()
        median_age_by_pclass_sex = self.df_all.groupby(['Sex', 'Pclass']).median()['Age']
        for pclass in range(1, 4):
            for sex in ['female', 'male']:
                print(f'Median age of Pclass {pclass} {sex}s: {median_age_by_pclass_sex[sex][pclass].astype(int)} ')

        # Filling the missing values in Age with the medians of Sex and Pclass groups
        self.df_all['Age'] = self.df_all.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))

        # Filling 2 missing values in Embarked with 'S'
        self.df_all['Embarked'] = self.df_all['Embarked'].fillna('S')

        # Filling the missing value in Fare with the median Fare of 3rd class alone passenger
        med_fare = self.df_all.groupby(['Pclass', 'Parch', 'SibSp'])['Fare'].median()[3][0][0]
        self.df_all['Fare'] = self.df_all['Fare'].fillna(med_fare)

        # Recheck to make sure no missing values in the data set
        #  0   Age          1309 non-null   float64
        #  1   Cabin        295 non-null    object
        #  2   Embarked     1309 non-null   object
        #  3   Fare         1309 non-null   float64
        #  4   Name         1309 non-null   object
        #  5   Parch        1309 non-null   int64
        #  6   PassengerId  1309 non-null   int64
        #  7   Pclass       1309 non-null   int64
        #  8   Sex          1309 non-null   object
        #  9   SibSp        1309 non-null   int64
        #  10  Survived     891 non-null    float64
        #  11  Ticket       1309 non-null   object
        #  As shown above, all feature except for Cabin have 1209 non-null values.
        #  Right now i just leave out Cabin, dealing with it later maybe
        print()
        print('_' * 25)
        print('All set info after filling missing values:')
        print(self.df_all.info())
        print(self.df_all.head(10))

        # Binning continuous feature to remove noise
        self.df_all['Fare'] = pd.qcut(self.df_all['Fare'], 13)
        self.df_all['Age'] = pd.qcut(self.df_all['Age'], 10)

        # Combine Sibsp and Parch to Family_size which makes sense
        self.df_all['Family_Size'] = self.df_all['SibSp'] + self.df_all['Parch'] + 1

        # Mapping Family Size
        family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large',
                      11: 'Large'}
        self.df_all['Family_Size_Grouped'] = self.df_all['Family_Size'].map(family_map)

        # Grouping ticket by their frequency
        self.df_all['Ticket_Frequency'] = self.df_all.groupby('Ticket')['Ticket'].transform('count')

        # Interpreting Name to Title and IsMarried
        self.df_all['Title'] = self.df_all['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

        self.df_all['Is_Married'] = 0
        self.df_all['Is_Married'].loc[self.df_all['Title'] == 'Mrs'] = 1

        # Grouping titles
        self.df_all['Title'] = self.df_all['Title'].replace(['Miss', 'Mrs', 'Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'],
                                                  'Miss/Mrs/Ms')
        self.df_all['Title'] = self.df_all['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'],
                                                  'Dr/Military/Noble/Clergy')

        # Label Encoding non-numerical features with LabelEncoder
        non_numeric_features = ['Embarked', 'Sex', 'Title', 'Family_Size_Grouped', 'Age', 'Fare']
        for feature in non_numeric_features:
            self.df_all[feature] = LabelEncoder().fit_transform(self.df_all[feature])

        # Correct misunderstanding brought by labelling categorical features with OneHotEncoder
        # For example,
        # sex is denoted with 0 and 1,  the model can be fooled by the fact
        # that 0 < 1 can thinks one sex is smalled than other sex.  There is no such relationship between two sexes.
        onehot_features = ['Pclass', 'Sex',  'Embarked', 'Title', 'Family_Size_Grouped']
        encoded_features = []
        for feature in onehot_features:
            encoded_feat = OneHotEncoder().fit_transform(self.df_all[feature].values.reshape(-1, 1)).toarray()
            n = self.df_all[feature].nunique()
            cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]
            encoded_df = pd.DataFrame(encoded_feat, columns=cols)
            encoded_df.index = self.df_all.index
            encoded_features.append(encoded_df)

        self.df_all = pd.concat([self.df_all, *encoded_features[:6]], axis=1)

        # Dropping unneeded features
        drop_cols = ['Embarked', 'Family_Size', 'Family_Size_Grouped', 'Cabin',
                     'Name', 'Parch', 'PassengerId', 'Pclass', 'Sex', 'SibSp', 'Ticket', 'Title']
        self.df_all.drop(columns=drop_cols, inplace=True)

        print()
        print('_' * 25)
        print('All set info after data preprocessing and feature engineering:')
        print(self.df_all.info())
        print(self.df_all.head(10))

        # Split All Set to Training and Test after all data preprocessing and feature engineering
        # we know training set has 891 rows.  0 to 890 are training set,  the rest 418 rows are test set
        self.df_train, self.df_test = self.df_all.loc[:890], self.df_all.loc[891:].drop(['Survived'], axis=1)

        print()
        print('_' * 25)
        print('Training set info after data preprocessing and feature engineering:')
        print(self.df_train.info())
        print(self.df_train.head(10))

        print()
        print('_' * 25)
        print('Test set info after data preprocessing and feature engineering:')
        print(self.df_test.info())
        print(self.df_test.head(10))


def main():
    c = TitanicSurvival()
    c.load_data()
    c.explore_data()
    c.data_preprocessing()


if __name__ == '__main__':
    main()
