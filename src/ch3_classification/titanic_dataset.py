import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import f1_score
from sklearn.model_selection import (
    cross_val_score,
    cross_val_predict,
    # train_test_split,
    GridSearchCV
)


def load_dataset():
    data_dir = os.path.join(os.path.dirname(
        os.path.dirname(
            os.path.dirname(__file__)
        )
    ), "data")
    dataset = pd.read_csv(os.path.join(data_dir, "titanic", "train.csv"))
    dataset.drop(columns=['Ticket', 'Cabin', 'Name'], inplace=True)
    return dataset


def load_test_dataset():
    data_dir = os.path.join(os.path.dirname(
        os.path.dirname(
            os.path.dirname(__file__)
        )
    ), "data")
    dataset = pd.read_csv(os.path.join(data_dir, "titanic", "test.csv"))
    dataset.drop(columns=['Cabin', 'Name'], inplace=True)
    return dataset


def get_stats(dataset):
    """Show common dataset values to get some insight on the data."""

    print("get_stats")
    print("dataset", dataset)
    print("dataset.head", dataset.head())
    print("dataset.info", dataset.info())
    print("dataset.describe", dataset.describe())
    print("non-number fields:")
    print("Pclass", dataset['Pclass'].value_counts())
    print("Sex", dataset['Sex'].value_counts())
    print("Embarked", dataset['Embarked'].value_counts())
    # print("Day_of_Week", dataset['Day_of_Week'].value_counts())
    # print("Time_of_Day", dataset['Time_of_Day'].value_counts())
    corr_matrix = dataset.corr(numeric_only=True)
    print(
        "correlation matrix:",
        corr_matrix["Survived"].sort_values(
            ascending=False
        ),
    )


class AgeBinner(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        bins = [0, 15, 60, np.inf]
        return np.digitize(X, bins) - 1


def encode_sex(X):
    return (X == 'female').astype(int)


def train(dataset):

    numeric_age_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        # ('binner', AgeBinner()),
        # ('onehot', OneHotEncoder(sparse_output=True, handle_unknown='ignore'))
    ])

    categorical_embarked_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(sparse_output=True, handle_unknown='ignore'))
    ])

    sex_pipe = Pipeline([
        (
            'encoder',
            FunctionTransformer(
                encode_sex,
                feature_names_out='one-to-one'
            )
        )
    ])

    preprocessor = ColumnTransformer([
        ('age_bin', numeric_age_pipe, ['Age']),
        ('embarked', categorical_embarked_pipe, ['Embarked']),
        ('sex', sex_pipe, ['Sex']),
    ], remainder='passthrough')

    X = dataset.drop(columns=['Survived'])
    y = dataset['Survived']
    y = y == 1

    # sgd = SGDClassifier()
    sgd = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    X_transformed = preprocessor.fit_transform(X)
    # import pdb; pdb.set_trace()
    sgd.fit(X_transformed, y)
    # y_predicted = lreg.predict(X_transformed)
    y_train_pred = cross_val_predict(sgd, X_transformed, y, cv=3)

    # import pdb; pdb.set_trace()
    f1 = f1_score(y, y_train_pred, average="macro")
    print("F1 score (macro):", f1)
    # 0.7981693810809821


dataset = load_dataset()
# get_stats(dataset)
train(dataset)
