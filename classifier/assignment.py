import csv, random
import numpy as np

def main():
    X_train, y_train, X_test = load_data()

def load_data():

    content = extract_data('../assignment1_2016S1/training_data.csv')
    X_train = [x[1:] for x in content]
    X_train = np.asarray(X_train, dtype='f')

    content = extract_data('../assignment1_2016S1/training_labels.csv')
    y_train = [y[1] for y in content]

    content = extract_data('../assignment1_2016S1/test_data.csv')
    X_test = [x[1:] for x in content]
    X_test = np.asarray(X_test, dtype='f')

    return X_train, y_train, X_test

def extract_data(filename):

    content = []
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            content.append(row)
    return content

def cross_validation(k, m):
    items = list(range(m))
    random.shuffle(items)
    slices = [items[i::k] for i in range(k)]
    for i in range(k):
        validation = slices[i]
        training = [item
                    for s in slices if s is not validation
                    for item in s]
        yield training, validation


if __name__ == "__main__":
    main()