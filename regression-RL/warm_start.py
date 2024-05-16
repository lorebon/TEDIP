from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np
from collections import Counter
from sklearn.metrics import mean_squared_error, r2_score


def findParents(tree):
    nodes = len(tree.children_right)
    parents = np.full(nodes, np.nan)
    for i in range(nodes):
        for j in range(nodes):
            if tree.children_left[j] == i:
                if j == 0:
                    parents[i] = 0.1
                else:
                    parents[i] = j

            if tree.children_right[j] == i:
                if j == 0:
                    parents[i] = -0.1
                else:
                    parents[i] = -j
    return parents


def computePaths(X, y, n_estimators, max_depth, seed):
    print("computing paths...")
    #min_samples_leaf=np.ceil(0.05*X.shape[0]).astype(int)
    clf = RandomForestRegressor(n_estimators=n_estimators, random_state=seed,
                                 max_depth=max_depth)

    #clf = GradientBoostingRegressor(n_estimators=n_estimators, random_state=seed,
    #                                min_samples_leaf=np.ceil(0.05*X.shape[0]).astype(int),
    #                                max_depth=max_depth)

    clf.fit(X, y)
    paths = []

    for idx in range(n_estimators):
        #tree = clf.estimators_[idx][0].tree_
        tree = clf.estimators_[idx].tree_
        leaf_nodes = [t.item() for t in np.argwhere(tree.children_left == -1)]
        parents = findParents(tree)
        for leaf in leaf_nodes:
            path = []
            parent = parents[leaf]
            while not np.isnan(parent):
                if parent > 0:
                    parent = round(abs(parent))
                    path.append((tree.feature[parent], tree.threshold[parent], 'L'))
                else:
                    parent = round(abs(parent))
                    path.append((tree.feature[parent], tree.threshold[parent], 'R'))
                parent = parents[parent]

            # check for max depth
            paths.append(list(reversed(path)))

    return paths, clf


def computeSample(data, path, idx):
    if idx >= len(path):
        return True

    node = path[idx]
    if node[2] == 'L' and data[node[0]] <= node[1]:
            return computeSample(data, path, idx+1)

    elif node[2] == 'R' and data[node[0]] > node[1]:
            return computeSample(data, path, idx+1)

    return False


def computeLoss(X, y, paths, eps=None):
    n = X.shape[0]
    samples = []
    path_indices = list(range(len(paths)))

    # compute samples
    print("assigning samples...")
    for p, path in enumerate(paths):
        sample = []
        for i in range(n):
            if computeSample(X[i, :], path, 0):
                sample.append(i)
        samples.append(sample)

    number_of_paths = len(paths)
    loss = np.zeros(number_of_paths)
    labels = np.zeros(number_of_paths)

    print("computing loss...")
    for i, sample in enumerate(samples):
        y_sampled = y[sample]
        labels[i] = np.mean(y_sampled)
        loss[i] = mean_squared_error(y_sampled, np.repeat(labels[i], len(y_sampled)))


    print("check for fusions")

    fusion = []
    for i, path_i in enumerate(paths):
        count = 0
        for j, path_j in enumerate(paths):
            if j != i:
                shared = 0
                for k in range(min(len(path_i), len(path_j))):
                    if path_i[k][:-1] == path_j[k][:-1]:
                        shared += 1
                count += (2*shared)/(len(path_i)+len(path_j))
        #fusion.append(count/len(paths))
        fusion.append(count)

    print("fusions:", sorted(fusion, reverse=True))

    paths = [paths[idx] for idx in path_indices]
    samples = [samples[idx] for idx in path_indices]
    loss = [loss[idx] for idx in path_indices]
    labels = [labels[idx] for idx in path_indices]

    return loss, samples, labels, paths, fusion


def computeScore(X, y, paths, labels):
    n = X.shape[0]

    y_pred = np.zeros(len(y))
    for p, path in enumerate(paths):
        for i in range(n):
            if computeSample(X[i, :], path, 0):
                y_pred[i] = labels[p]

    return mean_squared_error(y, y_pred)


def sorensenDice(Z1, Z2):
    numerator = 2 * len(set(Z1) & set(Z2))
    denominator = len(Z1) + len(Z2)
    return round(numerator/denominator, 2)



