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
                                   min_shapelet_size=min_size, max_shapelet_size=max_size,
                                   n_shapelets=n_shap, max_depth=3)

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


def computeSample(data, path, metric, idx=0):
    if idx >= len(path):
        return True

    node = path[idx]
    if node[2] == 'L' and minDist(node[0], data, metric) <= node[1]:
            return computeSample(data, path, metric, idx+1)

    elif node[2] == 'R' and minDist(node[0], data, metric) > node[1]:
            return computeSample(data, path, metric, idx+1)

    return False


def computeLoss(X, y, paths, metric, Nmin=None):
    n = X.shape[0]
    samples = []

    # compute samples
    print("assigning samples...")
    for p, path in enumerate(paths):
        sample = []
        for i in range(n):
            if computeSample(X[i, :], path, metric):
                sample.append(i)
        samples.append(sample)

    if Nmin is not None:
        paths_new = [path for (path, sample) in zip(paths, samples) if len(sample) >= np.ceil(Nmin*X.shape[0]).astype(int)]
        samples_new = [sample for (path, sample) in zip(paths, samples) if len(sample) >= np.ceil(Nmin*X.shape[0]).astype(int)]
        paths = paths_new
        samples = samples_new
        if len(paths) == 0:
            return None, None, None, None, None, None

    path_indices = list(range(len(paths)))
    number_of_paths = len(paths)
    loss = np.zeros(number_of_paths)
    labels = np.zeros(number_of_paths)
    weights = np.zeros(number_of_paths)

    baseline = np.unique(y, return_counts=True)[1].max()

    print("computing loss...")
    for i, sample in enumerate(samples):
        y_sampled = y[sample]
        labels[i] = Counter(y_sampled).most_common(1)[0][0]
        loss[i] = len(y_sampled) - Counter(y_sampled).most_common(1)[0][1]
        loss[i] = loss[i]/baseline
        weights[i] = len(y_sampled)
        #weights[i] = -loss[i]

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

    if len(fusion) == 0:
        return None, None, None, None, None, None

    if max(fusion) != min(fusion):
        fusion = [(float(i) - min(fusion)) / (max(fusion) - min(fusion)) for i in fusion]
    if max(loss) != min(loss):
        loss = [(float(i) - min(loss)) / (max(loss) - min(loss)) for i in loss]

    print("loss:", sorted(loss))
    print("fusions:", sorted(fusion, reverse=True))

    paths = [paths[idx] for idx in path_indices]
    samples = [samples[idx] for idx in path_indices]
    loss = [loss[idx] for idx in path_indices]
    labels = [labels[idx] for idx in path_indices]
    weights = [weights[idx] for idx in path_indices]

    return loss, samples, labels, paths, fusion, weights


def computeScore(X, y, paths, labels, metric, weights, mode):
    n = X.shape[0]

    y_pred = np.zeros(len(y))
    for i in range(n):
        covered_labels = []
        covered_leaves = []
        for p, path in enumerate(paths):
            if computeSample(X[i, :], path, metric):
                covered_labels.append(labels[p])
                covered_leaves.append(p)

        # in case or 2+ or 0 rules
        if len(covered_leaves) > 1:
            covered_weights = [weight for idx, weight in enumerate(weights) if idx in covered_leaves]
            y_pred[i] = covered_labels[np.argmax(covered_weights)]
        elif len(covered_leaves) == 1:
            y_pred[i] = covered_labels[0]
        else:
            y_pred[i] = mode

    return accuracy_score(y, y_pred)


def sorensenDice(Z1, Z2):
    numerator = 2 * len(list(set(Z1) & set(Z2)))
    denominator = len(Z1) + len(Z2)
    return round(numerator/denominator, 2)



