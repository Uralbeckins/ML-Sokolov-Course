import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    feature_vector = np.array(feature_vector)
    target_vector = np.array(target_vector)

    idx_sorted = np.argsort(feature_vector)
    feature_vector = feature_vector[idx_sorted]
    target_vector = target_vector[idx_sorted]
    tresholds_list = []
    ginies_list = []
    cum_sum = np.cumsum(target_vector)
    length = len(feature_vector)
    
    for i in range(len(feature_vector)-1):

        if feature_vector[i] == feature_vector[i+1]:
            continue

        tresholds_list.append(np.mean([feature_vector[i], feature_vector[i+1]]))

        p_r = cum_sum[i] / (i+1)
        p_l = (cum_sum[-1] - cum_sum[i]) / (length - i - 1)

        H_r = 1 - p_r ** 2 - (1 - p_r) ** 2
        H_l = 1 - p_l ** 2 - (1 - p_l) ** 2
        ginies_list.append(-(length-i-1) / length * H_l - (i+1) / length * H_r)

    if not ginies_list:
        return None, None, None, None
    best_idx = np.argmax(ginies_list)

    return tresholds_list, ginies_list, tresholds_list[best_idx], ginies_list[best_idx]


class DecisionTree:
    def __init__(self, feature_types, max_depth=1, min_samples_split=10, min_samples_leaf=1):
        if np.any(list(map(lambda x: x != "num" and x != "cat", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._depth = 0
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
        
        if len(sub_y) < self._min_samples_split or depth >= self._max_depth:
            node['type'] = 'terminal'
            node['class'] = Counter(sub_y).most_common(1)[0][0]
            return
        
        self._depth += 1

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(0, sub_X.shape[1]):
            
            feature_type = self._feature_types[feature]
            categories_map = {}
            best_split = None

            if feature_type == "num":
                feature_vector = np.array(sub_X.iloc[:, feature])
            elif feature_type == "cat":
                counts = Counter(sub_X.iloc[:, feature])
                clicks = Counter([sub_X.iloc[i, feature] for i in range(len(sub_X)) if sub_y[i] == 1])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = sorted(ratio.items(), key=lambda x: x[1])
                categories_map = {category: rank for rank, (category, _) in enumerate(sorted_categories)}
                feature_vector = np.array(sub_X.iloc[:, feature].map(categories_map))
            else:
                raise ValueError

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            
            if threshold is None:
                continue

            split = feature_vector < threshold
            if np.sum(split) <= self._min_samples_leaf or np.sum(np.logical_not(split)) <= self._min_samples_leaf:
                    continue
            
            if gini_best is None or gini < gini_best:
                feature_best = feature
                gini_best = gini
                
                if feature_type == "num":
                    threshold_best = threshold
                elif feature_type == "cat":
                    threshold_best = list(map(lambda x: x[0], filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError
                


        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "num":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "cat":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[best_split], sub_y[best_split], node["left_child"], depth+1)
        self._fit_node(sub_X[np.logical_not(best_split)], sub_y[np.logical_not(best_split)], node["right_child"], depth+1)

    def _predict_node(self, x, node):
        # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
        pass

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, depth=0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

class LinearRegressionTree():
    def __init__(self, feature_types, base_model_type=None, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        pass


# class DecisionTree:
#     def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
#         if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
#             raise ValueError("There is unknown feature type")

#         self._tree = {}
#         self._feature_types = feature_types
#         self._max_depth = max_depth
#         self._min_samples_split = min_samples_split
#         self._min_samples_leaf = min_samples_leaf

#     def _fit_node(self, sub_X, sub_y, node):
#         if np.all(sub_y != sub_y[0]):
#             node["type"] = "terminal"
#             node["class"] = sub_y[0]
#             return

#         feature_best, threshold_best, gini_best, split = None, None, None, None
#         for feature in range(1, sub_X.shape[1]):
#             feature_type = self._feature_types[feature]
#             categories_map = {}

#             if feature_type == "real":
#                 feature_vector = sub_X[:, feature]
#             elif feature_type == "categorical":
#                 counts = Counter(sub_X[:, feature])
#                 clicks = Counter(sub_X[sub_y == 1, feature])
#                 ratio = {}
#                 for key, current_count in counts.items():
#                     if key in clicks:
#                         current_click = clicks[key]
#                     else:
#                         current_click = 0
#                     ratio[key] = current_count / current_click
#                 sorted_categories = list(map(lambda x: x[1], sorted(ratio.items(), key=lambda x: x[1])))
#                 categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

#                 feature_vector = np.array(map(lambda x: categories_map[x], sub_X[:, feature]))
#             else:
#                 raise ValueError

#             if len(feature_vector) == 3:
#                 continue

#             _, _, threshold, gini = find_best_split(feature_vector, sub_y)
#             if gini_best is None or gini > gini_best:
#                 feature_best = feature
#                 gini_best = gini
#                 split = feature_vector < threshold

#                 if feature_type == "real":
#                     threshold_best = threshold
#                 elif feature_type == "Categorical":
#                     threshold_best = list(map(lambda x: x[0],
#                                               filter(lambda x: x[1] < threshold, categories_map.items())))
#                 else:
#                     raise ValueError

#         if feature_best is None:
#             node["type"] = "terminal"
#             node["class"] = Counter(sub_y).most_common(1)
#             return

#         node["type"] = "nonterminal"

#         node["feature_split"] = feature_best
#         if self._feature_types[feature_best] == "real":
#             node["threshold"] = threshold_best
#         elif self._feature_types[feature_best] == "categorical":
#             node["categories_split"] = threshold_best
#         else:
#             raise ValueError
#         node["left_child"], node["right_child"] = {}, {}
#         self._fit_node(sub_X[split], sub_y[split], node["left_child"])
#         self._fit_node(sub_X[np.logical_not(split)], sub_y[split], node["right_child"])