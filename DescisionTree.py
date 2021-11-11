import numpy as np
from numpy.lib.polynomial import roots
from collections import Counter

# Calculate the entropy = sum(prob.*surprise) => sum(prob.*log(1/prob))
def entropy(y): # y=samples
    hist = np.bincount(y)   # Histogram
    ps = hist/len(y)
    return -np.sum([p * np.log2(p) for p in ps if p>0])

class Node:
    def __init__(self, features=None, threshold=None, left=None, right=None, *, value=None):
        self.features=features
        self.threshold=threshold
        self.left=left
        self.right=right
        self.value=value
        
    def is_leaf_node(self):
        return self.value is not None
    
class DescisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None
        
    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X,y)
    
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique())
        
        # Stopping Criteria - max depth, min samples req, class distributions
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_labels(y)
            return Node(value=leaf_value)
        
        feat_idxs = np.random.choice(n_features, self.n_feats, replace = False)
        
        # Greedy Search
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        
    def _most_common_labels(self, y):
        counter = Counter(y)    # No. of occurances
        most_common = counter.most_common(1)[0][0]
        return most_common
    
    # Go over all feat values and calculate information gain
    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
                
        return split_idx, split_thresh
    
    def _information_gain(self, y, X_column, split_thresh):
        # parent E
        parent_entropy = entropy(y)
        
        # Generate Split
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        
        # Weighted avg of child L
        
        # return ig
    
    
    def predict(self, X):
        pass