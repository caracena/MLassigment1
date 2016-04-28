import csv, random, statistics as stat
import numpy as np
from classifier.perceptron import perceptron_test, perceptron_train
from sklearn.metrics import precision_recall_fscore_support
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import logging, time
import os.path

def main():
    logging.basicConfig(filename='resultsperceptron.log', filemode='w', level=logging.INFO)
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
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    for thres in thresholds:
        X_train = make_binary(X_train, thres)
        start_time = time.time()
        logging.info('perceptron algorithm threshold {}'.format(thres))
        print('perceptron algorithm threshold {}'.format(thres))
        iterations = 3
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

def main():
    logging.basicConfig(filename='resultsBST.log', filemode='w', level=logging.INFO)
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

    models = []
    learning_rate = [0.1, 0.5, 1.0]
    n_estimators = [100, 150, 200]
    max_depth = [1, 3, 5]
    param = [(i,j,k) for i in learning_rate for j in n_estimators for k in max_depth]
    for p in param:
        models.append(GradientBoostingClassifier(n_estimators=p[1], learning_rate=p[0], max_depth=p[2]))

    i = 0
    for model in models:
        start_time = time.time()
        logging.info('GradientBoostingClassifier with {} estimators, {} of learning rate, '
                     'and max depth {}'.format(param[i][1],param[i][0], param[i][2]))
        print('GradientBoostingClassifier with {} estimators, {} of learning rate, '
                     'and max depth {}'.format(param[i][1], param[i][0], param[i][2]))
        results = get_results_algorithms(X_train, y_train, m, k, model)
        logging.info(results)
        logging.info("--- %s seconds ---" % (time.time() - start_time))
        i += 1

    logging.info(get_precision_recall_fscore_overall(results, k))
    logging.info("--- %s seconds ---" % (time.time() - start_time))


    logging.info('Finished')

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
    precision, recall, fscore = [], [], []
    for res in results:
        precision.append(res[0])
        recall.append(res[1])
        fscore.append(res[2])
    return stat.mean(precision), stat.stdev(precision),stat.mean(recall), \
           stat.stdev(recall), stat.mean(fscore), stat.stdev(fscore)

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

def make_binary(X_train, t):
    return X_train >= t

if __name__ == "__main__":
    main()