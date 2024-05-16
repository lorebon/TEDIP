from wildboar.ensemble import ShapeletForestClassifier
import numpy as np
from collections import Counter

def minDist(shapelet, data, metric):
    min_distance = float('inf')

    for i in range(len(data) - len(shapelet) + 1):
        distance = metric(shapelet, data[i:i + len(shapelet)])
        min_distance = min(min_distance, distance)

    return min_distance


def findParents(tree):
    nodes = len(tree.right)
    parents = np.full(nodes, np.nan)
    for i in range(nodes):
        for j in range(nodes):
            if tree.left[j] == i:
                if j == 0:
                    parents[i] = 0.1
                else:
                    parents[i] = j

            if tree.right[j] == i:
                if j == 0:
                    parents[i] = -0.1
                else:
                    parents[i] = -j
    return parents


def computePaths(X, y, n_estimators, min_size, max_size, n_shap, max_depth, seed):
    print("computing paths...")
    clf = ShapeletForestClassifier(n_estimators=n_estimators, random_state=seed,
                                   min_samples_leaf=np.ceil(0.05*X.shape[0]).astype(int),
                                   min_shapelet_size=min_size, max_shapelet_size=max_size,
                                   bootstrap=True, n_shapelets=500, max_depth=3)

    # stavolta voglio estrarre TUTTI i path presenti nella foresta
    clf.fit(X, y)
    paths = []
    max_depth = 3

    for idx in range(n_estimators):
        tree = clf.estimators_[idx].tree_
        leaf_nodes = [t.item() for t in np.argwhere(tree.left == -1)]
        parents = findParents(tree)
        for leaf in leaf_nodes:
            path = []
            parent = parents[leaf]
            while not np.isnan(parent):
                if parent > 0:
                    parent = round(abs(parent))
                    path.append((tree.feature[parent][1][1], tree.threshold[parent], 'L'))
                else:
                    parent = round(abs(parent))
                    path.append((tree.feature[parent][1][1], tree.threshold[parent], 'R'))
                parent = parents[parent]

            # check for max depth
            if len(path) <= max_depth:
                paths.append(list(reversed(path)))

    return paths, clf


def computeSample(data, path, idx, metric):
    if idx >= len(path):
        return True

    node = path[idx]
    if node[2] == 'L' and minDist(node[0], data, metric) <= node[1]:
            return computeSample(data, path, idx+1, metric)

    elif node[2] == 'R' and minDist(node[0], data, metric) > node[1]:
            return computeSample(data, path, idx+1, metric)

    return False


def computeLoss(X, y, paths, metric, eps=None):
    n = X.shape[0]
    samples = []
    path_indices = list(range(len(paths)))

    # compute samples
    print("assigning samples...")
    for p, path in enumerate(paths):
        sample = []
        for i in range(n):
            if computeSample(X[i, :], path, 0, metric):
                sample.append(i)
        samples.append(sample)

    number_of_paths = len(paths)
    loss = np.zeros(number_of_paths)
    labels = np.zeros(number_of_paths)

    print("computing loss...")
    for i, sample in enumerate(samples):
        y_sampled = y[sample]
        labels[i] = Counter(y_sampled).most_common(1)[0][0]
        loss[i] = len(y_sampled) - Counter(y_sampled).most_common(1)[0][1]

    print("check for fusions")
    fusion = []
    for i, path_i in enumerate(paths):
        count = 0
        for j, path_j in enumerate(paths):
            if j != i:
                shared = 0
                for k in range(min(len(path_i), len(path_j))):
                    if len(path_i[k][0]) == len(path_j[k][0]):
                        match = np.all(path_i[k][0] == path_j[k][0]) and path_i[k][1] == path_j[k][1]
                        if match:
                            shared += 1
                count += (2*shared)/(len(path_i)+len(path_j))
        #fusion.append(count/len(paths))
        fusion.append(count)

    print("fusions:", sorted(fusion, reverse=True))

    baseline = np.unique(y, return_counts=True)[1].max()

    paths = [paths[idx] for idx in path_indices]
    samples = [samples[idx] for idx in path_indices]
    loss = [loss[idx]/baseline for idx in path_indices]
    labels = [labels[idx] for idx in path_indices]

    return loss, samples, labels, paths, fusion


def computeScore(X, y, paths, labels, metric):
    n = X.shape[0]

    loss = 0
    samples = []
    for p, path in enumerate(paths):
        sample = []
        for i in range(n):
            if computeSample(X[i, :], path, 0, metric):
                sample.append(i)
        samples.append(sample)

    for i, sample in enumerate(samples):
        y_sampled = y[sample]
        loss += len(y_sampled) - len([x for x in y_sampled if x == labels[i]])

    return (n - loss)/n * 100


def sorensenDice(Z1, Z2):
    numerator = 2 * len(list(set(Z1) & set(Z2)))
    denominator = len(Z1) + len(Z2)
    return round(numerator/denominator, 2)



