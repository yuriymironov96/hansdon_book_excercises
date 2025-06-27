import numpy as np
from scipy.ndimage import shift
from sklearn.datasets import fetch_openml
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import (
    cross_val_score,
    cross_val_predict,
    # train_test_split,
    GridSearchCV
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', as_frame=False)
X, y = mnist.data, mnist.target

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

some_digit = X[0]


def plot_digit(image_data):
    # display dataset item as an image
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")
    plt.show()


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)


def score_measuring_example():
    knc = KNeighborsClassifier(
        algorithm="auto",
        n_neighbors=5,
        weights="distance",
        p=1  # 1 = Manhattan, 2 = Euclidean
    )
    knc.fit(X_train, y_train)
    score = cross_val_score(
        knc,
        X_train_scaled,
        y_train,
        cv=3,
        scoring="accuracy"
    )
    print("cross_val_score", score)

    y_train_pred = cross_val_predict(knc, X_train_scaled, y_train, cv=3)

    f1 = f1_score(y_train, y_train_pred, average="macro")
    print("F1 score (macro):", f1)

    X_test_scaled = scaler.transform(X_test)
    y_test_pred = cross_val_predict(knc, X_test_scaled, y_test, cv=3)

    f1_test = f1_score(y_test, y_test_pred, average="macro")
    print("F1 test score (macro):", f1_test)


def naive_knn_classifier():
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_train)
    baseline_accuracy = knn_clf.score(X_test, y_test)
    print(baseline_accuracy)


def tune_hyperparameters_best():
    param_grid = [
        {'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5, 6]}
    ]

    knn_clf = KNeighborsClassifier()
    grid_search = GridSearchCV(knn_clf, param_grid, cv=5)
    grid_search.fit(X_train[:10_000], y_train[:10_000])
    print("grid_search.best_params_", grid_search.best_params_)
    print("grid_search.best_score_", grid_search.best_score_)


def tune_hyperparameters():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knc", KNeighborsClassifier())
    ])

    param_grid = {
        "knc__n_neighbors": [3, 5, 7, 9],
        "knc__weights": ["uniform", "distance"],
        "knc__algorithm": ["auto", "ball_tree", "kd_tree"],
        "knc__p": [1, 2]  # 1 = Manhattan, 2 = Euclidean
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    # Best parameters and score
    print("Best params:", grid_search.best_params_)
    print("Best F1 score (macro):", grid_search.best_score_)


def confusion_matrix_example(knc):
    y_train_pred = cross_val_predict(knc, X_train_scaled, y_train, cv=3)
    ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)
    ConfusionMatrixDisplay.from_predictions(
        y_train,
        y_train_pred,
        normalize="true",
        values_format=".0%"
    )
    sample_weight = (y_train_pred != y_train)
    ConfusionMatrixDisplay.from_predictions(
        y_train,
        y_train_pred,
        sample_weight=sample_weight,
        normalize="true",
        values_format=".0%"
    )
    ConfusionMatrixDisplay.from_predictions(
        y_train,
        y_train_pred,
        sample_weight=sample_weight,
        normalize="pred",
        values_format=".0%"
    )
    plt.show()


def final_classifier():
    # final classifier with the best parameters
    knn_clf = KNeighborsClassifier(n_neighbors=4, weights="distance")
    knn_clf.fit(X_train, y_train)
    tuned_accuracy = knn_clf.score(X_test, y_test)
    print("tuned_accuracy", tuned_accuracy)  # 0.9714


def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])


def final_classifier_with_augmented_dataset():
    X_train_shifted_right = np.array(
        [shift_image(img, 0, 1) for img in X_train]
    )
    X_train_shifted_left = np.array(
        [shift_image(img, 0, -1) for img in X_train]
    )
    X_train_shifted_up = np.array(
        [shift_image(img, -1, 0) for img in X_train]
    )
    X_train_shifted_down = np.array(
        [shift_image(img, 1, 0) for img in X_train]
    )
    X_train_augmented = np.concatenate(
        [X_train,
         X_train_shifted_right,
         X_train_shifted_left,
         X_train_shifted_up,
         X_train_shifted_down,]
    )
    y_train_augmented = np.concatenate(
        [
            y_train.copy(),
            y_train.copy(),
            y_train.copy(),
            y_train.copy(),
            y_train.copy(),
        ]
    )
    shuffle_idx = np.random.permutation(len(X_train_augmented))
    X_train_augmented = X_train_augmented[shuffle_idx]
    y_train_augmented = y_train_augmented[shuffle_idx]
    knn_clf = KNeighborsClassifier(n_neighbors=4, weights="distance")
    knn_clf.fit(X_train_augmented, y_train_augmented)
    tuned_accuracy = knn_clf.score(X_test, y_test)
    print("tuned_accuracy", tuned_accuracy)


final_classifier_with_augmented_dataset()
