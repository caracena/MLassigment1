import csv, random
from classifier.perceptron import *
from sklearn.metrics import precision_recall_fscore_support
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import logging, time
import os.path

def main():
    logging.basicConfig(filename='results.log', filemode='w', level=logging.INFO)
    logging.info('Started')
    ## load data

    start_time = time.time()
    logging.info('Loading data')
    X_train, y_train, X_test = load_data()
    list_classes = list(set(y_train))
    list_classes.sort()
    m, n = X_train.shape
    k = 10
    logging.info("--- %s seconds ---" % (time.time() - start_time))


    models = [RandomForestClassifier(n_estimators=30), MultinomialNB(), GaussianNB(), SVC(), LogisticRegression(C=1e5),
              KNeighborsClassifier(n_neighbors=3)]

    for model in models:
        start_time = time.time()
        logging.info(model.__name__)
        results = get_results_algorithms(X_train, y_train, m, k, model)
        logging.info(results)
        logging.info("--- %s seconds ---" % (time.time() - start_time))

    ## perceptron algorithm result: 60% with PCA at 90%: 60%
    start_time = time.time()
    logging.info('perceptron algorithm')
    iterations = 4
    results = []
    c_cross = 0
    for training, validation in cross_validation(k, m):
        print('cross validation iteration {}'.format(c_cross))
        y_train_cross = [y_train[y] for y in training]
        y_val_cross = [y_train[y] for y in validation]
        w_avg = perceptron_train(X_train[training], y_train_cross, iterations, list_classes)
        y_pred = perceptron_test(X_train[validation], w_avg, list_classes)
        res = precision_recall_fscore_support(y_val_cross, y_pred, average='micro')
        results.append(res)
        c_cross += 1

    logging.info(get_precision_recall_fscore_overall(results, k))
    logging.info("--- %s seconds ---" % (time.time() - start_time))

    ## PCA
    logging.info('PCA')
    start_time = time.time()
    pca = PCA(n_components=0.95)
    X_train = pca.fit_transform(X_train)
    logging.info("--- %s seconds ---" % (time.time() - start_time))


    for model in models:
        start_time = time.time()
        logging.info(model.__name__ + ' with PCA at 95%')
        results = get_results_algorithms(X_train, y_train, m, k, model)
        logging.info(results)
        logging.info("--- %s seconds ---" % (time.time() - start_time))

    ## perceptron algorithm result: 60% with PCA at 90%: 60%
    start_time = time.time()
    logging.info('perceptron algorithm with PCA at 95%')
    iterations = 4
    results = []
    c_cross = 0
    for training, validation in cross_validation(k, m):
        print('cross validation iteration {}'.format(c_cross))
        y_train_cross = [y_train[y] for y in training]
        y_val_cross = [y_train[y] for y in validation]
        w_avg = perceptron_train(X_train[training], y_train_cross, iterations, list_classes)
        y_pred = perceptron_test(X_train[validation], w_avg, list_classes)
        res = precision_recall_fscore_support(y_val_cross, y_pred, average='micro')
        results.append(res)
        c_cross += 1

    logging.info(get_precision_recall_fscore_overall(results, k))
    logging.info("--- %s seconds ---" % (time.time() - start_time))

    logging.info('Finished')


def load_data():

    if os.path.isfile('../assignment1_2016S1/training_data_order.csv'):
        content = extract_data('../assignment1_2016S1/training_data_order.csv')
    else:
        content = extract_data('../assignment1_2016S1/training_data.csv')
        content.sort(key=lambda x: x[0])
        save_data(content, '../assignment1_2016S1/training_data_order.csv')
    X_train = [x[1:] for x in content]
    X_train = np.asarray(X_train, dtype='f')

    if os.path.isfile('../assignment1_2016S1/training_labels_order.csv'):
        content = extract_data('../assignment1_2016S1/training_labels_order.csv')
    else:
        content = extract_data('../assignment1_2016S1/training_labels.csv')
        content.sort(key=lambda x: x[0])
        save_data(content, '../assignment1_2016S1/training_labels_order.csv')
    y_train = [y[1] for y in content]

    if os.path.isfile('../assignment1_2016S1/training_labels_order.csv'):
        content = extract_data('../assignment1_2016S1/test_data_order.csv')
    else:
        content = extract_data('../assignment1_2016S1/test_data.csv')
        content.sort(key=lambda x: x[0])
        save_data(content, '../assignment1_2016S1/test_data_order.csv')
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

def save_data(content, filename):
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for row in content:
            writer.writerow(row)

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

def get_precision_recall_fscore_overall(results, k):
    precision, recall, fscore = 0, 0, 0
    for res in results:
        precision += res[0]
        recall += res[1]
        fscore += res[2]
    return precision/k, recall/k, fscore/k

def get_results_algorithms(X_train, y_train, m, k, model):
    results = []
    c_cross = 0
    for training, validation in cross_validation(k, m):
        print('cross validation iteration {}'.format(c_cross))
        y_train_cross = [y_train[y] for y in training]
        y_val_cross = [y_train[y] for y in validation]
        model.fit(X_train[training], y_train_cross)
        y_pred = model.predict(X_train[validation])
        res = precision_recall_fscore_support(y_val_cross, y_pred, average='micro')
        results.append(res)
        c_cross += 1

    return get_precision_recall_fscore_overall(results, k)


if __name__ == "__main__":
    main()