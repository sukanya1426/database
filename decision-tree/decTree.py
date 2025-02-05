import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold

# Load the iris dataset
iris = sns.load_dataset("iris")
# print(iris)

# Function to check purity
def check_purity(data):
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)
    if len(unique_classes) == 1:
        return True
    return False

# Function to classify data
def classify_data(data):
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
    classification = unique_classes[counts_unique_classes.argmax()]
    return classification

# Function to get potential splits
def get_potential_splits(data):
    potential_splits = {}
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):
        potential_splits[column_index] = []
        values = data[:, column_index]
        unique_values = np.unique(values)
        for index in range(len(unique_values)):
            if index != 0:
                current_value = unique_values[index]
                previous_value = unique_values[index - 1]
                potential_split = (current_value + previous_value) / 2
                potential_splits[column_index].append(potential_split)
    return potential_splits

# Function to split data
def split_data(data, split_column, split_value):
    split_column_values = data[:, split_column]
    data_below = data[split_column_values <= split_value]
    data_above = data[split_column_values > split_value]
    return data_below, data_above

# Function to calculate entropy
def calculate_entropy(data):
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)
    probabilities = counts / counts.sum()
    # exit(0)
    entropy = sum(probabilities * -np.log2(probabilities))
    return entropy

# Function to calculate overall entropy
def calculate_overall_entropy(data_below, data_above):
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n
    overall_entropy = (p_data_below * calculate_entropy(data_below) + p_data_above * calculate_entropy(data_above))
    # exit(0)
    return overall_entropy

# Function to calculate F-statistic
def calculate_f_statistic(data, split_column, split_value):
    try:
        data_below, data_above = split_data(data, split_column, split_value)
        overall_mean = np.mean(data[:, split_column])

        n_below = len(data_below)
        n_above = len(data_above)

        mean_below = np.mean(data_below[:, split_column])
        mean_above = np.mean(data_above[:, split_column])

        ss_between = n_below * (mean_below - overall_mean) ** 2 + n_above * (mean_above - overall_mean) ** 2
        ss_within = np.sum((data_below[:, split_column] - mean_below) ** 2) + np.sum((data_above[:, split_column] - mean_above) ** 2)

        # Calculate F-statistic and handle potential division by zero
        if ss_within == 0:
            raise ZeroDivisionError # Raise ZeroDivisionError to be caught by the except block

        f_statistic = ss_between / (ss_within / (n_below + n_above - 2))
        return f_statistic

    except ZeroDivisionError:
        # Handle the ZeroDivisionError by skipping this split and returning None
        return 999.999999999999999

# Function to determine the best split
def determine_best_split(data, potential_splits):
    overall_metric = float('inf')
    best_split_column = None
    best_split_value = None

    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            # F-statistic calculation
            # current_metric = calculate_f_statistic(data, column_index, value)

            # Uncomment the following lines to use entropy instead:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_metric = calculate_overall_entropy(data_below, data_above)

            if current_metric < overall_metric:
                overall_metric = current_metric
                best_split_column = column_index
                best_split_value = value

    return best_split_column, best_split_value

# Decision tree algorithm
def decision_tree_algorithm(df, counter=0, min_samples=2, max_depth=5):
    if counter == 0:
        global COLUMN_HEADERS
        COLUMN_HEADERS = df.columns
        data = df.values
    else:
        data = df

    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = classify_data(data)
        return classification
    else:
        counter += 1
        potential_splits = get_potential_splits(data)
        split_column, split_value = determine_best_split(data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)
        feature_name = COLUMN_HEADERS[split_column]
        question = "{} <= {}".format(feature_name, split_value)
        sub_tree = {question: []}

        yes_answer = decision_tree_algorithm(data_below, counter, min_samples, max_depth)
        no_answer = decision_tree_algorithm(data_above, counter, min_samples, max_depth)

        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)

        return sub_tree

# Function to classify examples
def classify_example(example, tree):
    # Check if tree is a dictionary (not a leaf node)
    if isinstance(tree, dict):
        question = list(tree.keys())[0]
        feature_name, comparison_operator, value = question.split()
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
        if not isinstance(answer, dict):
            return answer
        else:
            residual_tree = answer
            return classify_example(example, residual_tree)
    else:  # If tree is a string (leaf node), directly return the classification
        return tree

# Updated function to calculate accuracy and F1, F2 measures
def calculate_metrics(df, tree):
    df = df.copy()
    df["classification"] = df.apply(classify_example, axis=1, args=(tree,))
    df["classification_correct"] = (df["classification"] == df["species"])

    y_true = df["species"]
    y_pred = df["classification"]

    accuracy = df["classification_correct"].mean()
    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')

    f2 = (1 + 2**2) * (precision * recall) / ((2**2 * precision) + recall)

    return accuracy, f1, f2

# k-Fold Cross-Validation
def k_fold_cross_validation(dataframe, k):
    kf = KFold(n_splits=k, shuffle=True)
    accuracies = []
    f1_scores = []
    f2_scores = []

    for train_index, test_index in kf.split(dataframe):
        train_df = dataframe.iloc[train_index]
        test_df = dataframe.iloc[test_index]

        tree = decision_tree_algorithm(train_df, max_depth=3)
        accuracy, f1, f2 = calculate_metrics(test_df, tree)

        accuracies.append(accuracy)
        f1_scores.append(f1)
        f2_scores.append(f2)

        print(f"Accuracy: {accuracy:.2f}\nF1 Score: {f1:.2f}\nF2 Score: {f2:.2f}\n")

    avg_accuracy = np.mean(accuracies)
    avg_f1 = np.mean(f1_scores)
    avg_f2 = np.mean(f2_scores)

    print(f"Average Accuracy over {k} folds: {avg_accuracy * 100:.2f}%")
    print(f"Average F1 Score over {k} folds: {avg_f1 * 100:.2f}%")
    print(f"Average F2 Score over {k} folds: {avg_f2 * 100:.2f}%")

k = 5
k_fold_cross_validation(iris, k)