import numpy as np


class Node():
    def __init__(self, feature_index=None, children=None,
                 info_gain=None, value=None):
        ''' constructor '''

        # for decision node
        self.feature_index = feature_index
        self.children = {}
        self.info_gain = info_gain

        # for leaf node
        self.value = value


class DecisionTree():

    def __init__(self, min_samples_split=2, max_depth=2):
        ''' constructor '''

        # initialize the root of the tree
        self.root = None

        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree '''

        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)

        # split until stopping conditions are met
        if num_samples >= self.min_samples_split and (
                                                      curr_depth <=
                                                      self.max_depth):
            # find the best split
            best_split = self.get_best_split(dataset,
                                             num_samples,
                                             num_features)

            # check if information gain is positive
            if best_split["info_gain"] > 0:
                # Create a new node with the best feature index
                node = Node(best_split["feature_index"],
                            info_gain=best_split["info_gain"])

                # Recursively build the children for the node
                for value, child_dataset in (best_split["split_datasets"]
                                             .items()):
                    node.children[value] = self.build_tree(child_dataset,
                                                           curr_depth + 1)

                return node

        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split '''

        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")

        # loop over all the features
        for feature_index in range(num_features):
            # Split the dataset based on the feature values
            split_datasets = self.split(dataset, feature_index)

            # Check if all the split datasets are not empty
            if all(len(split) > 0 for split in split_datasets.values()):
                # Calculate the information gain
                y = dataset[:, -1]
                curr_info_gain = self.information_gain(y,
                                                       {value:
                                                        split_datasets[value]
                                                        [:, -1]
                                                        for value in
                                                        split_datasets.keys()
                                                        },
                                                       mode="entropy")

                # Update the best split if needed
                if curr_info_gain > max_info_gain:
                    best_split["feature_index"] = feature_index
                    best_split["split_datasets"] = split_datasets
                    best_split["info_gain"] = curr_info_gain
                    max_info_gain = curr_info_gain

        # return best split
        return best_split

    def split(self, dataset, feature_index):
        ''' function to split the data '''

        feature_values = dataset[:, feature_index]
        unique_values = np.unique(feature_values)

        # Create a dictionary to store the split datasets for each unique value
        split_datasets = {}

        # Split the dataset based on the unique feature values
        for value in unique_values:
            split_datasets[value] = dataset[feature_values == value]

        return split_datasets

    def information_gain(self, parent, children, mode="entropy"):
        ''' function to compute information gain '''

        weights = [len(child) / len(parent) for child in children.values()]
        if mode == "gini":
            parent_impurity = self.gini_index(parent)
            child_impurities = [self.gini_index(child) for child
                                in children.values()]
            gain = parent_impurity - sum(weight * impurity for weight,
                                         impurity in zip(weights,
                                                         child_impurities))
        else:
            parent_entropy = self.entropy(parent)
            child_entropies = [self.entropy(child) for child
                               in children.values()]
            gain = parent_entropy - sum(weight * entropy for weight, entropy
                                        in zip(weights, child_entropies))

        return gain

    def entropy(self, y):
        ''' function to compute entropy '''

        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy

    def gini_index(self, y):
        ''' function to compute gini index '''

        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini

    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''

        Y = list(Y)
        return max(Y, key=Y.count)

    def print_tree(self, tree=None, indent=""):
        ''' function to print the tree '''

        if not tree:
            tree = self.root

        if tree.value is not None:
            print(indent + str(tree.value))

        else:
            print(indent + "X_" + str(tree.feature_index), "?", tree.info_gain)
            for value, child in tree.children.items():
                print(indent + "├─ Value =", value)
                self.print_tree(child, indent + "│  ")
            print(indent + "└─ ")

    def fit(self, X, Y):
        ''' function to train the tree '''

        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, X):
        ''' function to predict new dataset '''

        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions

    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''

        if tree.value is not None:
            return tree.value

        feature_val = x[tree.feature_index]
        if feature_val in tree.children:
            return self.make_prediction(x, tree.children[feature_val])
        else:
            # If the feature value is not present in the children,
            # return the majority class
            return self.calculate_leaf_value(np.array(
                [child.value for child
                 in tree.children.values()]))
