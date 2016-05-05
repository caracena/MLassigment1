import os.path
import csv, random, statistics as stat
import numpy as np


class Base:

    def __init__(self, filename, targetname, test_name):
        self.dataset_name = filename
        self.test_name = test_name
        self.target_name = targetname

    def load_data(self):

        if os.path.isfile('../data/training_data_order.csv'):
            content = self.extract_data('../data/training_data_order.csv')
        else:
            content = self.extract_data(self.dataset_name)
            content.sort(key=lambda x: x[0])
            self.save_data(content, '../data/training_data_order.csv')

        X_train = [x[1:] for x in content]
        X_train = np.asarray(X_train, dtype='f')

        if os.path.isfile('../data/training_labels_order.csv'):
            content = self.extract_data('../data/training_labels_order.csv')
        else:
            content = self.extract_data(self.target_name)
            content.sort(key=lambda x: x[0])
            self.save_data(content, '../data/training_labels_order.csv')
        y_train = [y[1] for y in content]

        if os.path.isfile('../data/test_data_order.csv'):
            content = self.extract_data('../data/test_data_order.csv')
        else:
            content = self.extract_data(self.test_name)
            content.sort(key=lambda x: x[0])
            self.save_data(content, '../data/test_data_order.csv')

        X_test = [x[1:] for x in content]
        X_test = np.asarray(X_test, dtype='f')

        return X_train, y_train, X_test

    def extract_data(self,filename):

        content = []
        with open(filename) as f:
            reader = csv.reader(f)
            for row in reader:
                content.append(row)
        return content

    def save_data(self,content, filename):
        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile)
            for row in content:
                writer.writerow(row)

    def cross_validation(self,k, m):
        items = list(range(m))
        random.shuffle(items)
        slices = [items[i::k] for i in range(k)]
        for i in range(k):
            validation = slices[i]
            training = [item
                        for s in slices if s is not validation
                        for item in s]
            yield training, validation

    def get_precision_recall_fscore_overall(self,results, k):
        precision, recall, fscore = [], [], []
        for res in results:
            precision.append(res[0])
            recall.append(res[1])
            fscore.append(res[2])
        return stat.mean(precision), stat.stdev(precision), stat.mean(recall), \
               stat.stdev(recall), stat.mean(fscore), stat.stdev(fscore)
