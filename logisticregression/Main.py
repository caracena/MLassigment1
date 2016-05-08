import logging, time, argparse, random
from logisticregression.Base import Base
from logisticregression.LogisticRegression import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

def main():

    ## load positional parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("l", type=float, help="lambda parameter for regularisation [0 - Inf]", default=0.3,
                        nargs='?',const=0.3)
    parser.add_argument("p", help="process to be done (cross, test or predict)",
                        choices=['cross', 'test', 'predict'] ,default='cross', const=0.3, nargs='?')
    parser.add_argument("r", help="type of reduction to be applied",
                        choices=['pca', 'none', 'common'], default='none', const='none', nargs='?')
    parser.add_argument("--value_reduction", type = float, help="value of dimensionality reduction",
                        default=0, const=0, nargs='?')
    args = parser.parse_args()

    ## append results to resultslogistic.log
    logging.basicConfig(filename='resultslogistic.log', level=logging.INFO)
    logging.info('Starting Logistic Regression proccess {} with lambda {}'.format(args.l, args.p))

    ## create base with files
    b = Base("../assignment1_2016S1/training_data.csv",
             "../assignment1_2016S1/training_labels.csv",
             "../assignment1_2016S1/test_data.csv")
    clf = LogisticRegression()

    ## load data
    start_time = time.time()
    logging.info('Loading data')
    X_train, y_train, X_test, test_names = b.load_data()
    logging.info("--- %s seconds ---" % (time.time() - start_time))


    if args.r != 'none':
        ## dimensionality reduction
        start_time = time.time()
        logging.info('Reduce matrix using {}'.format(args.r))
        X_train = b.dimension_reduction(X_train, args.reduction, args.value_reduction)
        m, n = X_train.shape
        if args.p == 'predict':
            X_test = b.dimension_reduction(X_test, args.reduction, args.value_reduction)
        logging.info('X_train shape {},{}'.format(m, n))
        logging.info("--- %s seconds ---" % (time.time() - start_time))

    args.p == 'predict'

    ## get lambda
    lmda = args.l
    m, n = X_train.shape

    ## cross validation process
    if args.p == 'cross':
        logging.info('Starting 10-fold cross validation')
        start_time = time.time()

        k = 10
        results = []
        c_cross = 0

        ## call training and validation indexes
        for training, validation in b.cross_validation(k, m):
            print('cross validation iteration {}'.format(c_cross))

            ## list of labels in training and validation
            y_train_cross = [y_train[y] for y in training]
            y_val_cross = [y_train[y] for y in validation]

            ## train and predict in validation data
            clf.fit(X_train[training], y_train_cross, lmda)
            y_pred = clf.predict(X_train[validation])

            ## calculate macro average precision, recall and fscore
            res = b.get_precision_recall_fscore(y_val_cross, y_pred)

            ## append results
            results.append(res)
            c_cross += 1

        ## calculate and store averaged results and duration
        logging.info(b.get_precision_recall_fscore_overall(results, k))
        logging.info("--- %s seconds ---" % (time.time() - start_time))

    ## Random sample for testing and analyze results
    elif args.p == 'test':
        logging.info('Training in 2/3 of data and testing in 1/3 of data')
        start_time = time.time()

        ## split the dataset
        random.seed(1)
        items = list(range(m))
        random.shuffle(items)
        training = items[m // 3:]
        testing = items[:m // 3]

        ## list of labels in training and testing
        y_val = [y_train[y] for y in training]
        y_test = [y_train[y] for y in testing]

        ## train and predict in testing data
        clf.fit(X_train[training], y_val, lmda)
        y_pred = clf.predict(X_train[testing])

        ## calculate macro average precision, recall and fscore
        res = b.get_precision_recall_fscore(y_test, y_pred)
        logging.info(res)
        logging.info("--- %s seconds ---" % (time.time() - start_time))
        ## confusion matrix
        b.plot_confusion_matrix(y_test, y_pred, clf.list_classes)

    ## training in whole dataset predict labels in testing data
    elif args.p == 'predict':
        logging.info('Training in 2/3 of data and testing in 1/3 of data')
        start_time = time.time()

        ## train and predict in testing data
        clf.fit(X_train, y_train, lmda)
        y_pred = clf.predict(X_test)
        results = zip(test_names,y_pred)
        b.save_data(results,'test_data.csv')
        logging.info("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()