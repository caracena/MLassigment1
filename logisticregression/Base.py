import os.path
import csv, random, statistics as stat
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class Base:

    def __init__(self, filename, targetname, test_name):
        self.dataset_name = filename
        self.test_name = test_name
        self.target_name = targetname

    def load_data(self):

        X_train = self.extract_data(self.dataset_name)
        y_train = self.extract_data(self.target_name, array = False)
        X_test, test_names = self.extract_data(self.test_name, sort = False, names = True)

        return X_train, y_train, X_test, test_names

    def dimension_reduction(self, X_train, option, value):
        if option == 'common':
            if not value:
                value = 500
            columns = (X_train != 0).sum(0)
            X_train = X_train[:, columns < value]
        elif option == 'pca':
            if not value:
                value = 0.95
            pca = PCA(n_components=value)
            X_train = pca.fit_transform(X_train)
        return X_train

    def extract_data(self,filename, sort= True, array = True, names = False):
        content = []
        with open(filename) as f:
            reader = csv.reader(f)
            for row in reader:
                content.append(row)

            if sort: content.sort(key=lambda x: x[0])

            if array:
                content_array = [x[1:] for x in content]
                content_array = np.asarray(content_array, dtype='f')
            else:
                content_array = [y[1] for y in content]

            if names:
                names_list = [n[0] for n in content]
                return content_array, names_list

        return content_array


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

    def plot_confusion_matrix(self, y_test, y_pred, list_classes):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(len(list_classes))
        plt.xticks(tick_marks, list_classes, rotation=90)
        plt.yticks(tick_marks, list_classes)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.grid(True)
        width, height = len(list_classes), len(list_classes)
        for x in range(width):
            for y in range(height):
                if cm[x][y] > 100:
                    color = 'white'
                else:
                    color = 'black'
                if cm[x][y] > 0:
                    plt.annotate(str(cm[x][y]), xy=(y, x),
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 color=color)
        plt.show()