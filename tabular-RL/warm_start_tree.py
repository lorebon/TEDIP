from sklearn.ensemble import RandomForestClassifier
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


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
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed,
                                 max_depth=max_depth)

    clf.fit(X, y)
    paths = []
    trees_pathed = []
    trees_noded = []

    for idx in range(n_estimators):
        #tree = clf.estimators_[idx][0].tree_
        tree = clf.estimators_[idx].tree_
        leaf_nodes = [t.item() for t in np.argwhere(tree.children_left == -1)]
        parents = findParents(tree)
        tree_nodes = set()
        tree_paths = []
        for leaf in leaf_nodes:
            path = []
            parent = parents[leaf]
            while not np.isnan(parent):
                if parent > 0:
                    parent = round(abs(parent))
                    path.append((tree.feature[parent], tree.threshold[parent], 'L'))
                    tree_nodes.add((tree.feature[parent], tree.threshold[parent], 'L'))
                else:
                    parent = round(abs(parent))
                    path.append((tree.feature[parent], tree.threshold[parent], 'R'))
                    tree_nodes.add((tree.feature[parent], tree.threshold[parent], 'R'))
                parent = parents[parent]

            # check for max depth
            if 0 < len(path) <= max_depth:
                paths.append(list(reversed(path)))
                tree_paths.append(list(reversed(path)))
                #while len(path) > 1:
                #    path = path[1:]
                #    tree_paths.append(list(reversed(path)))

        trees_pathed.append(tree_paths)
        trees_noded.append(tree_nodes)

    #test
    #print("unique paths:", len(set(tuple(item) for item in paths)))
    #paths = [list(item) for item in set(tuple(item) for item in paths)]

    return paths, clf, trees_pathed, trees_noded


def computeSample(data, path, idx=0):
    if idx >= len(path):
        return True

    node = path[idx]
    if node[2] == 'L' and data[node[0]] <= node[1]:
            return computeSample(data, path, idx+1)

    elif node[2] == 'R' and data[node[0]] > node[1]:
            return computeSample(data, path, idx+1)

    return False


def computeLoss(X, y, paths, Nmin=None):
    n = X.shape[0]
    samples = []

    # compute samples
    print("assigning samples...")
    for p, path in enumerate(paths):
        sample = []
        for i in range(n):
            if computeSample(X[i, :], path, 0):
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
        #count /= len(path_i)
        #fusion.append(count/len(paths))
        fusion.append(count)

    if len(fusion) == 0:
        return None, None, None, None, None, None

    if max(fusion) != min(fusion):
        fusion = [(float(i) - min(fusion)) / (max(fusion) - min(fusion)) for i in fusion]
    if max(loss) != min(loss):
        loss = [(float(i) - min(loss)) / (max(loss) - min(loss)) for i in loss]
    #print("fusions:", sorted(fusion, reverse=True))

    paths = [paths[idx] for idx in path_indices]
    samples = [samples[idx] for idx in path_indices]
    loss = [loss[idx] for idx in path_indices]
    labels = [labels[idx] for idx in path_indices]
    #weights = weights/np.sum(weights)
    weights = [weights[idx] for idx in path_indices]


    return loss, samples, labels, paths, fusion, weights


def computeScore(X, y, paths, labels, weights, mode):
    n = X.shape[0]

    y_pred = np.zeros(len(y))
    for i in range(n):
        covered_labels = []
        covered_leaves = []
        for p, path in enumerate(paths):
            if computeSample(X[i, :], path):
                covered_labels.append(labels[p])
                covered_leaves.append(p)

        if len(covered_leaves) > 1:
            covered_weights = [weight for idx, weight in enumerate(weights) if idx in covered_leaves]
            y_pred[i] = covered_labels[np.argmax(covered_weights)]
        elif len(covered_leaves) == 1:
            y_pred[i] = covered_labels[0]
        else:
            y_pred[i] = mode

    return accuracy_score(y, y_pred)


def sorensenDice(Z1, Z2):
    numerator = 2 * len(set(Z1) & set(Z2))
    denominator = len(Z1) + len(Z2)
    return round(numerator/denominator, 2)


def checkTreePaths(paths, trees):
    represented_trees = 0
    for tree in trees:
        represented = 0
        for tree_path in tree:
            if represented == 0:
                for path in paths:
                    if tree_path == path:
                        represented = 1
                        break
        represented_trees += represented

    return represented_trees/len(trees)


def findBestTree(X, y, trees, lambd=1):
    max = np.inf
    max_index = None
    max_labels = None

    for idx, tree in enumerate(trees):
        loss, _, labels, _, fusion, _ = computeLoss(X, y, tree, None)
        obj = sum(loss) - lambd*sum(fusion)
        if obj < max:
            max = obj
            max_index = idx
            max_labels = labels

    return trees[max_index], max_labels


def checkTrees(paths, trees):
    represented_trees = 0
    for tree in trees:
        represented = 0
        for node in tree:
            if represented == 0:
                for path in paths:
                    if node in path:
                        represented = 1
                        break
        represented_trees += represented

    return represented_trees/len(trees)


def computeScoreTree(X, y, paths, labels):
    n = X.shape[0]

    y_pred = np.zeros(len(y))
    for i in range(n):
        for p, path in enumerate(paths):
            if computeSample(X[i, :], path, 0):
                y_pred[i] = labels[p]

    return accuracy_score(y, y_pred)


def plotPaths(loss, fusion, labels, leaves):
    # Create a scatter plot
    plt.scatter(loss, fusion, color='blue', marker='o', s=10)

    # Mark specific points in red
    plt.scatter([loss[i] for i in leaves],
                [fusion[i] for i in leaves],
                color='red', marker='o', s=10)
    plt.xlabel('Loss')
    plt.ylabel('Fusions')
    plt.title('2D Scatter Plot')

    # Show the plot
    plt.show()
