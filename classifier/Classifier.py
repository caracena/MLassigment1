import numpy as np
import codecs
import pandas as pd
import sklearn.cross_validation as cross_validation



class MultinomialNaiveBayes():

    def __init__(self):
        self.trained = False
        self.likelihood =  0
        self.prior = 0
        self.smooth = False
        self.smooth_param = 1
        
    def train(self, x, y):
        # n_docs = no. of documents
        # n_words = no. of unique words    
        n_docs, n_words = x.shape
        
        # classes = a list of possible classes
        classes = np.unique(y)

        # words = list of possible words
        words = np.unique(x.columns.values)
        
        # n_classes = no. of classes
        n_classes = np.unique(y).shape[0]


        # TODO: This is where you have to write your code!
        # You need to compute the values of the prior and likelihood parameters
        # and place them in the variables called "prior" and "likelihood".
        # Examples:
            # prior[0] is the prior probability of a document being of class 0
            # likelihood[4, 0] is the likelihood of the fifth(*) feature being 
            # active, given that the document is of class 0
            # (*) recall that Python starts indices at 0, so an index of 4 
            # corresponds to the fifth feature!
        
        ###########################
        # Calculating the Prior Probabilities for the classes
        # pos_count and neg_count gives the count for each word for a particular class
        class_word_count = {}
        word_count = {}
        class_count = {}
        prior = {}
        pos_count,neg_count = np.zeros(n_words),np.zeros(n_words)
        # examining each word and fining the above mentioned values
        cwLength = x.shape[1]
        cdLength = x.shape[0]
        for c_w in range(cwLength):
            word_count[c_w] = 0
            for c_d in range(cdLength):
                clazz = y[c_d][0]
                value = x.iloc[c_d][c_w]

                if clazz in class_word_count :
                    class_word_count[clazz][c_w] += value
                else:
                    class_word_count[clazz] = np.zeros(n_words)

                if clazz in class_count:
                    class_count[clazz] += 1
                else:
                    class_count[clazz] = 0

                word_count[c_w] += value



        # Finding likelihood for each word with respective to a class
        for c_w in range (cwLength):
            for clazz in class_count:
                numerator = class_word_count[clazz][c_w]
                denominator = class_word_count[clazz].sum()
                class_word_count[clazz][c_w] = np.nan_to_num(np.log(numerator +1 / denominator + n_words ))

        ###########################
        self.likelihood = class_count
        self.prior = prior
        self.trained = True
        return params

if __name__ == '__main__':
    X = pd.read_csv('../zoo.csv', header=None)
    y = pd.read_csv('../target.csv', header=None)
    X_scaled = X.applymap(lambda x: 1 if x > 0 else 0)
    nb = MultinomialNaiveBayes()

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_scaled, y, test_size=0.33, random_state=42)

    params = nb.train(X_train, y_train.as_matrix())

    predict_train = nb.test(X_train, params)
    eval_train = nb.evaluate(predict_train, y_train)

    predict_test = nb.test(X_test, params)
    eval_test = nb.evaluate(predict_test, y_test)

    print("Accuracy on training set: {0}, on test set: {1}".format(eval_train, eval_test))
