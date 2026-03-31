import numpy as np
from collections import Counter
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression


# def find_best_split(feature_vector, target_vector):
#     feature_vector = np.array(feature_vector)
#     target_vector = np.array(target_vector)

#     idx_sorted = np.argsort(feature_vector)
#     feature_vector = feature_vector[idx_sorted]
#     target_vector = target_vector[idx_sorted]
#     tresholds_list = []
#     ginies_list = []
#     cum_sum = np.cumsum(target_vector)
#     length = len(feature_vector)
    
#     for i in range(len(feature_vector)-1):

#         if feature_vector[i] == feature_vector[i+1]:
#             continue

#         tresholds_list.append(np.mean([feature_vector[i], feature_vector[i+1]]))

#         # Левая часть: индексы 0..i (всего i+1 элементов)
#         p_left = cum_sum[i] / (i+1)
#         # Правая часть: индексы i+1..length-1 (всего length-i-1 элементов)
#         p_right = (cum_sum[-1] - cum_sum[i]) / (length - i - 1)

#         H_left = 1 - p_left ** 2 - (1 - p_left) ** 2
#         H_right = 1 - p_right ** 2 - (1 - p_right) ** 2
#         gini = -((i+1) / length) * H_left - ((length - i - 1) / length) * H_right
#         ginies_list.append(gini)

#     if not ginies_list:
#         return None, None, None, None
#     best_idx = np.argmax(ginies_list)

#     return tresholds_list, ginies_list, tresholds_list[best_idx], ginies_list[best_idx]


# class DecisionTree:
#     def __init__(self, feature_types, max_depth=5, min_samples_split=10, min_samples_leaf=2):
#         if np.any(list(map(lambda x: x != "num" and x != "cat", feature_types))):
#             raise ValueError("There is unknown feature type")

#         self._tree = {}
#         self._max_reached_depth = 0
#         self._feature_types = feature_types
#         self._max_depth = max_depth
#         self._min_samples_split = min_samples_split
#         self._min_samples_leaf = min_samples_leaf

#     def _fit_node(self, sub_X, sub_y, node, depth):

#         self._max_reached_depth = max(self._max_reached_depth, depth)
#         # Преобразуем в DataFrame, если это необходимо
#         if not isinstance(sub_X, pd.DataFrame):
#             sub_X = pd.DataFrame(sub_X)
        
#         sub_y = np.array(sub_y)
        
#         if np.all(sub_y == sub_y[0]):
#             node["type"] = "terminal"
#             node["class"] = sub_y[0]
#             return
        
#         if len(sub_y) < self._min_samples_split or depth >= self._max_depth:
#             node['type'] = 'terminal'
#             node['class'] = Counter(sub_y).most_common(1)[0][0]
#             return

#         feature_best, threshold_best, gini_best, best_split = None, None, None, None
#         for feature in range(sub_X.shape[1]):
#             feature_type = self._feature_types[feature]
#             current_split = None

#             if feature_type == "num":
#                 feature_vector = np.array(sub_X.iloc[:, feature])
#             elif feature_type == "cat":
#                 counts = Counter(sub_X.iloc[:, feature])
#                 clicks = Counter([sub_X.iloc[i, feature] for i in range(len(sub_X)) if sub_y[i] == 1])
#                 ratio = {}
#                 for key, current_count in counts.items():
#                     if key in clicks:
#                         current_click = clicks[key]
#                     else:
#                         current_click = 0
#                     ratio[key] = current_click / current_count
#                 sorted_categories = sorted(ratio.items(), key=lambda x: x[1])
#                 categories_map = {category: rank for rank, (category, _) in enumerate(sorted_categories)}
#                 feature_vector = np.array(sub_X.iloc[:, feature].map(categories_map))
#             else:
#                 raise ValueError

#             _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            
#             if threshold is None:
#                 continue

#             # Создаем булев массив для разделения
#             current_split = feature_vector < threshold
                
#             # Проверяем min_samples_leaf
#             if np.sum(current_split) < self._min_samples_leaf or np.sum(~current_split) < self._min_samples_leaf:
#                 continue
            
#             # Выбираем лучший сплит (максимум Gini)
#             if gini_best is None or gini > gini_best:
#                 feature_best = feature
#                 gini_best = gini
#                 threshold_best = threshold
#                 best_split = current_split

#         if feature_best is None or best_split is None:
#             node["type"] = "terminal"
#             node["class"] = Counter(sub_y).most_common(1)[0][0]
#             return

#         node["type"] = "nonterminal"
#         node["feature_split"] = feature_best
        
#         if self._feature_types[feature_best] == "num":
#             node["threshold"] = threshold_best
#         elif self._feature_types[feature_best] == "cat":
#             # Для категориальных признаков сохраняем список категорий, которые идут влево
#             categories_left = []
#             feature_values = sub_X.iloc[:, feature_best]
#             for i in range(len(feature_values)):
#                 if best_split[i]:
#                     categories_left.append(feature_values.iloc[i])
#             # Убираем дубликаты
#             node["categories_split"] = list(set(categories_left))
#         else:
#             raise ValueError
        
#         node["left_child"], node["right_child"] = {}, {}
        
#         # Рекурсивно строим левое и правое поддеревья
#         left_indices = np.where(best_split)[0]
#         right_indices = np.where(~best_split)[0]
        
#         self._fit_node(sub_X.iloc[left_indices], sub_y[left_indices], node["left_child"], depth + 1)
#         self._fit_node(sub_X.iloc[right_indices], sub_y[right_indices], node["right_child"], depth + 1)


#     def _predict_node(self, x, node):
#         if node["type"] == "terminal":
#             return node["class"]
        
#         # Проверяем, что feature_split существует и не равен None
#         if "feature_split" not in node or node["feature_split"] is None:
#             return node.get("class", 0)  # Возвращаем класс по умолчанию, если есть
        
#         feature = node["feature_split"]
#         feature_type = self._feature_types[feature]

#         if feature_type == "num":
#             if "threshold" not in node:
#                 return node.get("class", 0)
#             if x.iloc[feature] < node["threshold"]:
#                 return self._predict_node(x, node["left_child"])
#             else:
#                 return self._predict_node(x, node["right_child"])

#         elif feature_type == "cat":
#             if "categories_split" not in node:
#                 return node.get("class", 0)
#             if x.iloc[feature] in node["categories_split"]:
#                 return self._predict_node(x, node["left_child"])
#             else:
#                 return self._predict_node(x, node["right_child"])
#         else:
#             return node.get("class", 0)

#     def fit(self, X, y):
#         # Преобразуем X в DataFrame, если это необходимо
#         if not isinstance(X, pd.DataFrame):
#             X = pd.DataFrame(X)
#         self._max_reached_depth = 0
#         self._fit_node(X, y, self._tree, depth=0)
#         return self

#     def predict(self, X):
#         # Преобразуем X в DataFrame, если это необходимо
#         if not isinstance(X, pd.DataFrame):
#             X = pd.DataFrame(X)
        
#         predicted = []
#         for i in range(X.shape[0]):
#             x = X.iloc[i]
#             predicted.append(self._predict_node(x, self._tree))
#         return np.array(predicted)


def find_best_split_mod(f_v, t_v, base_model=LinearRegression()):
    f_v = np.array(f_v)
    t_v = np.array(t_v)

    idx_sorted = np.argsort(f_v)
    f_v = f_v[idx_sorted]
    t_v = t_v[idx_sorted]
    thresholds_list = []
    losses_list = []
    length = len(f_v)

    for i in range(length-1):

        if f_v[i] == f_v[i+1]:
            continue

        thresholds_list.append(np.mean([f_v[i], f_v[i+1]]))

        # ЛЕВАЯ часть: индексы 0..i (всего i+1 элементов)
        f_left = f_v[:i+1].reshape(-1, 1)
        t_left = t_v[:i+1]
        
        # ПРАВАЯ часть: индексы i+1..length-1 (всего length-i-1 элементов)
        f_right = f_v[i+1:].reshape(-1, 1)
        t_right = t_v[i+1:]

        # Создаем копии модели с теми же параметрами
        model_left = type(base_model)(**base_model.get_params())
        model_right = type(base_model)(**base_model.get_params())
        
        model_left.fit(f_left, t_left)
        model_right.fit(f_right, t_right)

        loss_left = mean_squared_error(t_left, model_left.predict(f_left))
        loss_right = mean_squared_error(t_right, model_right.predict(f_right))

        # Взвешенная средняя MSE
        weighted_loss = ((i+1) * loss_left + (length - i - 1) * loss_right) / length
        losses_list.append(weighted_loss)

    if not losses_list:
        return None, None, None, None
    
    best_idx = np.argmin(losses_list)
    return thresholds_list, losses_list, thresholds_list[best_idx], losses_list[best_idx]

class LinearRegressionTree():
    def __init__(self, feature_types, base_model_type=LinearRegression, max_depth=5, min_samples_split=10, min_samples_leaf=2):
        if np.any(list(map(lambda x: x != "num" and x != "cat", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._base_model_type = base_model_type
        self._max_reached_depth = 0
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth):
        self._max_reached_depth = max(self._max_reached_depth, depth)
        
        if not isinstance(sub_X, pd.DataFrame):
            sub_X = pd.DataFrame(sub_X)
        
        sub_y = np.array(sub_y)
        
        # Условия остановки
        if len(sub_y) < self._min_samples_split or depth >= self._max_depth:
            model = self._base_model_type()
            model.fit(sub_X, sub_y)
            node['type'] = 'terminal'
            node['model'] = model
            return
        
        # Поиск лучшего разделения
        feature_best, threshold_best, loss_best, best_split = None, None, np.inf, None
        
        for feature in range(sub_X.shape[1]):
            if self._feature_types[feature] != "num":
                continue  # Пока поддерживаем только числовые признаки
                
            feature_vector = np.array(sub_X.iloc[:, feature])
            
            # Используем базовую модель для оценки разделения
            base_model = self._base_model_type()
            _, _, threshold, loss = find_best_split_mod(feature_vector, sub_y, base_model)
            
            if threshold is None or loss is None:
                continue
            
            # Разделение: левая часть <= threshold
            current_split = feature_vector <= threshold
            
            # Проверка min_samples_leaf
            left_count = np.sum(current_split)
            right_count = len(current_split) - left_count
            
            if left_count < self._min_samples_leaf or right_count < self._min_samples_leaf:
                continue
            
            # Выбор лучшего разделения (минимальная loss)
            if loss < loss_best:
                feature_best = feature
                threshold_best = threshold
                loss_best = loss
                best_split = current_split
        
        # Если не нашли подходящего разделения или сплит не улучшает loss
        if feature_best is None or best_split is None:
            model = self._base_model_type()
            model.fit(sub_X, sub_y)
            node['type'] = 'terminal'
            node['model'] = model
            return
        
        # Создаем нелистовой узел
        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        node["threshold"] = threshold_best
        node["left_child"], node["right_child"] = {}, {}
        
        # Рекурсивное построение поддеревьев
        left_indices = np.where(best_split)[0]
        right_indices = np.where(~best_split)[0]
        
        self._fit_node(sub_X.iloc[left_indices], sub_y[left_indices], 
                      node["left_child"], depth + 1)
        self._fit_node(sub_X.iloc[right_indices], sub_y[right_indices], 
                      node["right_child"], depth + 1)


    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["model"].predict(x.values.reshape(1, -1))[0]
        
        feature = node["feature_split"]
        feature_type = self._feature_types[feature]

        if feature_type == "num":

            if x.iloc[feature] <= node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])


    def fit(self, X, y):
        # Преобразуем X в DataFrame, если это необходимо
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self._max_reached_depth = 0
        self._fit_node(X, y, self._tree, depth=0)
        return self

    def predict(self, X):
        # Преобразуем X в DataFrame, если это необходимо
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        predicted = []
        for i in range(X.shape[0]):
            x = X.iloc[i]
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)