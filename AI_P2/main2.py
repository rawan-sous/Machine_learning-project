import numpy as np
import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

TEST_SIZE = 0.3


def load_data(filename):
    """
    Load spam data from a CSV file `filename` and convert it into a list of
    feature vectors and a list of target labels. Return a tuple (features, labels).

    Feature vectors should be a list of lists, where each inner list contains the
    57 features.

    Labels should be the corresponding list of labels, where each label
    is 1 if spam, and 0 otherwise.
    """
    features = []
    labels = []

    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            # Extract features (first 57 numbers) and label (last number) from each row
            feature_vector = [float(value) for value in row[:-1]]
            label = int(row[-1])

            features.append(feature_vector)
            labels.append(label)

    return features, labels


def preprocess(features):
    """
    Normalize each feature by subtracting the mean value in each
    feature and dividing by the standard deviation.
    """
    features = np.array(features)
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)

    # Avoid division by zero by setting small stds to a non-zero value
    stds[stds < 1e-8] = 1.0

    normalized_features = (features - means) / stds
    return normalized_features.tolist()


def train_knn_model(filename):
    # Load data from spreadsheet
    features, labels = load_data(filename)

    # Preprocess features
    features = preprocess(features)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SIZE)

    # Create and train k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    knn.fit(X_train, y_train)

    return knn


def train_mlp_model(filename):
    # Load data from spreadsheet
    features, labels = load_data(filename)

    # Preprocess features
    features = preprocess(features)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SIZE)

    # Create and train MLP classifier
    mlp = MLPClassifier(hidden_layer_sizes=(10, 5), activation='logistic', max_iter=1000)
    mlp.fit(X_train, y_train)

    return mlp


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python template.py ./spambase.csv")

    # Train k-NN classifier
    knn = train_knn_model(sys.argv[1])

    # Print the trained k-NN classifier
    print("k-NN Classifier:")
    print(knn)

    # Train MLP classifier
    mlp = train_mlp_model(sys.argv[1])

    # Print the trained MLP classifier
    print("MLP Classifier:")
    print(mlp)


if __name__ == "__main__":
    main()
