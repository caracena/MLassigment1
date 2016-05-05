import logging, time
from logisticregression.Base import Base
from logisticregression.LogisticRegression import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support


def main():

    b = Base("../assignment1_2016S1/training_data.csv",
             "../assignment1_2016S1/training_labels.csv",
             "../assignment1_2016S1/test_data.csv")
    clf = LogisticRegression()

    k = 10
    ls = [0.3]

    logging.basicConfig(filename='resultslogistic.log', filemode='w', level=logging.INFO)

    logging.info('Started')

    start_time = time.time()
    logging.info('Loading data')
    X_train, y_train, X_test = b.load_data()
    logging.info("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    logging.info('Reduce matrix')
    columns = (X_train != 0).sum(0)
    X_train = X_train[:, columns < 500]
    m, n = X_train.shape
    logging.info('X_train shape {},{}'.format(m,n))
    logging.info("--- %s seconds ---" % (time.time() - start_time))

    for l in ls:
        results = []
        c_cross = 0

        logging.info('Logistic Regression with lambda {}'.format(l))
        print('Logistic Regression with lambda {}'.format(l))
        start_time = time.time()

        for training, validation in b.cross_validation(k, m):
            print('cross validation iteration {}'.format(c_cross))

            y_train_cross = [y_train[y] for y in training]
            y_val_cross = [y_train[y] for y in validation]

            clf.fit(X_train[training], y_train_cross, l)
            y_pred = clf.predict(X_train[validation])

            res = precision_recall_fscore_support(y_val_cross, y_pred, average='micro')

            results.append(res)
            c_cross += 1

        logging.info(b.get_precision_recall_fscore_overall(results, k))
        logging.info("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()