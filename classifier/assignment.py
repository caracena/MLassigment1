import csv

def main():
    content = []
    with open('assignment1_2016S1/training_data.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            content.append(row)

    X_train = [x[1:] for x in content]

    content = []
    with open('assignment1_2016S1/training_labels.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            content.append(row)

    y_train = [y[1] for y in content]

    




if __name__ == "__main__":
    main()