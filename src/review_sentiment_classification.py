"""
Reviews Sentiment Classification
By Justin Hu
2022/04/12
Tested on Python 3.9
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

# For debugging purpose, allowing more columns be printed out to the console
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 20)


class SentimentCls:
    """
    Sentiment Classification on restaurant reviews

    Tasks include:
        1. load data and split test and valid dataset
        2. train the model and predict classification
        3. valid the model performance

    Original dataset:  reviews.csv
        format:  Name	RatingValue	DatePublished	Review
            separated by tap

    """

    # binning rules:
    # rating 1 & 2 --> negative --> code 0
    #                 rating 3  --> neutral --> code 1
    #                 rating 4 & 5 --> positive --> code 2
    binning_map = {
        1: 0,
        2: 0,
        3: 1,
        4: 2,
        5: 2,
    }

    def __init__(self):
        """
        initialization of instance
        """
        self.df_review = None
        self.df_train = None
        self.df_valid = None
        self.df_test = None
        self.model = None

    def load_data(self):
        """
        load the initial data set
        :return:
        """
        self.df_review = pd.read_csv('reviews.csv', delimiter='\t')
        self.df_review.name = 'Initial Set'
        print('Initial Dataset Info:')
        print(self.df_review.info())

        # info of the original set
        #  #   Column         Non-Null Count  Dtype
        # ---  ------         --------------  -----
        #  0   Name           1920 non-null   object
        #  1   RatingValue    1920 non-null   int64
        #  2   DatePublished  1920 non-null   object
        #  3   Review         1920 non-null   object
        # print out the first 5 rows to examine the data structure
        print(self.df_review.head(5))

    def transform_data(self):
        """
        data transformation, including:
            1. binning rating value to target values according to the following rules:
                rating 1 & 2 --> negative --> code 0
                rating 3  --> neutral --> code 1
                rating 4 & 5 --> positive --> code 2
            2. splitting  the set into train (70%) and validation (30%) set
                training set:  1920 * 70% = 1344
                validation set: 1920 * 30% = 576
            3. save the train set and validation set to csv files : review_train.csv, and review_valid.csv respectively
        :return:
        """

        # bin RatingValue to Sentiment following the rule
        self.df_review['Sentiment'] = self.df_review['RatingValue'].apply(lambda x: SentimentCls.binning_map[x])

        # print out the first 10 row to check the binning results
        print(self.df_review.head(10))

        # partition the dataset
        self.df_train = self.df_review[:1344]
        self.df_valid = self.df_review[1344:]
        self.df_train.name = 'Training Set'
        self.df_valid.name = 'Validation Set'

        # drop unnecessary columns in both training and validation set
        drop_cols = ['Name', 'RatingValue', 'DatePublished']
        self.df_train.drop(columns=drop_cols, inplace=True)
        self.df_valid.drop(columns=drop_cols, inplace=True)

        # print out the first 5 rows to check the two datasets
        print('\nThe training set:')
        print(self.df_train.head(5))
        print('\nThe validation set:')
        print(self.df_valid.head(5))

        # Save the two new datasets to csv, which is unnecessary frankly
        self.df_train.to_csv('review_train.csv', index=False)
        self.df_valid.to_csv('review_valid.csv', index=False)

    def train_model(self):
        """
        train and evaluate the model of sentiment classification:
            1. TF-IDF is used as text representation
            2. classification models used: SVM
        :return:
        """
        sentiment_clf = Pipeline([
            ('vect', TfidfVectorizer(analyzer='word', ngram_range=(1, 1), use_idf=True)),
            ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                  alpha=1e-3, random_state=42,
                                  max_iter=5, tol=None))
        ])

        # Parameter tuning using grid search
        # Comment out parameter save performance
        # parameters = {
        #     'vect__analyzer': ('word', 'char'),
        #     'vect__ngram_range': [(1, 1), (1, 2),(2,2)],
        #     'vect__use_idf': (True, False),
        #     'clf__alpha': (1e-2, 1e-3),
        #     'clf__max_iter': (5, 200),
        # }
        # gs_clf = GridSearchCV(sentiment_clf, parameters, cv=5, n_jobs=-1)
        # gs_clf = gs_clf.fit(self.df_train['Review'], self.df_train['Sentiment'])
        #
        # print('best score:')
        # print(gs_clf.best_score_)
        # for param_name in sorted(parameters.keys()):
        #     print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

        # best score:
        # 0.803581534705654
        # clf__alpha: 0.001
        # clf__max_iter: 5
        # vect__analyzer: 'word'
        # vect__ngram_range: (1, 1)
        # vect__use_idf: True

        sentiment_clf.fit(self.df_train['Review'], self.df_train['Sentiment'])
        self.model = sentiment_clf

    def evaluate(self, target_set):
        """
        evaluate the model on the target dataset
        target set could be:
            1. validation set
            2. test set
        :param target_set:
        :return:
        """
        predicted = self.model.predict(target_set['Review'])
        print('accuracy on the validation set')
        print(np.mean(predicted == target_set['Sentiment']))

        print(metrics.classification_report(target_set['Sentiment'], predicted,
                                            target_names=['negative', 'neutral', 'positive']))

        print('confusion matrix:')
        print(metrics.confusion_matrix(target_set['Sentiment'], predicted))

    def evaluate_model_on_validation(self):
        """
        evaluate the model of sentiment classification
        :return:
        """
        print('evaluate model on the validation set:')
        self.evaluate(self.df_valid)

    def evaluate_model_on_test(self):
        """
        The function is for instructor to evaluate the code on a test dataset test.csv
        Assuming test.csv is separated by comma, otherwise please change parameters of read_csv accordingly.
        :return:
        """
        self.df_test = pd.read_csv('test.csv')
        if self.df_test.empty:
            print('test set is empty!')
        else:
            print('evaluate model on the test set:')
            self.evaluate(self.df_test)


def main():
    """
    please use evaluate_model_on_test to evaluate the model on test dataset
    :return:
    """
    c = SentimentCls()
    c.load_data()
    c.transform_data()
    c.train_model()
    c.evaluate_model_on_validation()
    # c.evaluate_model_on_test()


if __name__ == '__main__':
    main()
