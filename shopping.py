import csv
import sys
import calendar

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        evidence = []
        labels = []
        # create a month name to int dict, `if num` checks if num is not less than 0, `num - 1` because built-in month starts from 1
        monthToInt = {name: num - 1 for num, name in enumerate(calendar.month_abbr) if num}
        # remove `Jun` and replace with `June` to follow the csv file
        monthToInt.pop('Jun', None)
        monthToInt['June'] = 5
        visitorType = {"New_Visitor": 0, "Returning_Visitor": 1, "Other": 0}
        csvBool = {"FALSE": 0, "TRUE": 1}
        for row in reader:
            rowEvidence = []
            rowEvidence.append(int(row[0])) # administrative, int
            rowEvidence.append(float(row[1])) # administrative_duration, float
            rowEvidence.append(int(row[2])) # informational, int
            rowEvidence.append(float(row[3])) # informational duration, flaot
            rowEvidence.append(int(row[4])) # productRelated, int
            rowEvidence.append(float(row[5])) # productRelated, float
            rowEvidence.append(float(row[6])) # bounceRates, float
            rowEvidence.append(float(row[7])) # exitRates, float
            rowEvidence.append(float(row[8])) # pagerow, float
            rowEvidence.append(float(row[9])) # specialDay, float
            rowEvidence.append(monthToInt[row[10]]) # dict for month name to int
            rowEvidence.append(int(row[11])) # OperatingSystems, int
            rowEvidence.append(int(row[12])) # browser, int
            rowEvidence.append(int(row[13])) # region, int
            rowEvidence.append(int(row[14])) # trafficType, int
            rowEvidence.append(visitorType[row[15]]) # visitorType, int
            rowEvidence.append(csvBool[row[16]]) # weekend, int

            evidence.append(rowEvidence)

            labels.append(csvBool[row[17]])
        


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    raise NotImplementedError


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    raise NotImplementedError


if __name__ == "__main__":
    load_data("shopping.csv")
