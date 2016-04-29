import numpy as np
from scipy.optimize import fmin_bfgs
import  time, logging, random
from classifier.assignment import load_data
from sklearn.metrics import precision_recall_fscore_support

def main():
    X_train = np.array([[1,0],[0,200],[2,0],[3,0], [4,0]])
    y_train = ['casa', 'auto', 'casa', 'casa', 'tree']
    list_classes = ['auto', 'casa', 'tree']

    logging.basicConfig(filename='resultslogistic.log', filemode='w', level=logging.INFO)
    logging.info('Started')

    start_time = time.time()
    logging.info('Loading data')
    X_train, y_train, X_test = load_data()
    m, n = X_train.shape
    list_classes = list(set(y_train))
    list_classes.sort()

    random.seed(123)
    items = list(range(m))
    random.shuffle(items)
    X_train = X_train[items[:m//100]]
    y_train = [y_train[i] for i in items[:m//100]]
    logging.info("--- %s seconds ---" % (time.time() - start_time))


    logging.info('Logistic Regression')
    print('Logistic Regression')
    start_time = time.time()
    all_theta = logistic_train(X_train, y_train, list_classes)
    y_pred = logistic_test(X_train, all_theta, list_classes)
    print(y_pred)
    res = precision_recall_fscore_support(y_train, y_pred, average='micro')
    logging.info(res)
    print(res)
    logging.info("--- %s seconds ---" % (time.time() - start_time))

def logistic_train(X, y, list_classes):
    m, n = X.shape
    classes = len(list_classes)
    X = add_theta0(X)
    all_theta = np.zeros((len(list_classes), n + 1))
    l = 1
    start_time = time.time()
    for i in range(classes):
        print('training for class {} and took {}'.format(list_classes[i], (time.time() - start_time)))
        initial_theta = np.zeros(n+1)
        y_class = get_y_class(y, list_classes, i)
        def decorated_cost(theta):
            return cost_function_reg(theta, X, y_class, l)

        def decorated_grad(theta):
            return grad_function_reg(theta, X, y_class, l)

        theta = fmin_bfgs(decorated_cost, initial_theta, maxiter=10, fprime=decorated_grad)
        all_theta[i,:] = theta
        start_time = time.time()

    return all_theta

def logistic_test(X, all_theta, list_classes):
    m, n = X.shape
    X = add_theta0(X)
    y_pred = []
    for i in range(m):
        max_index = np.argmax(sigmoid(all_theta.dot(np.transpose(X[i,:]))))
        y_pred.append(list_classes[max_index])

    return y_pred

def cost_function_reg(theta, X, y, l):
    m, n = X.shape
    J = (1/m) * (-y.T.dot(np.log(sigmoid(X.dot(theta)))) - (1-y.T).dot(np.log(1 - sigmoid(X.dot(theta))))) + \
        (l/m)* 0.5 * np.sum(theta[1:]**2)
    return J

def grad_function_reg(theta, X, y, l):
    m, n = X.shape
    grad = (1/m) * X.T.dot(sigmoid(X.dot(theta)) - y)
    grad[1:] = grad[1:] + (l/m)*theta[1:]
    return grad

def predict(X, theta):
    y = sigmoid(X.dot(theta));
    return y>0.5

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def add_theta0(X):
    m, n = X.shape
    X_aux = np.zeros((m, n + 1))
    X_aux[:, 1:] = X
    return X_aux

def get_y_class(y, list_classes, i):
    return np.asarray([b == list_classes[i] for b in y])

if __name__ == "__main__":
    main()
