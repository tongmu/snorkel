import numpy as np
import xgboost as xgb
from .disc_learning import NoiseAwareModel
from .utils import print_scores
from .utils import marginals_to_labels, MentionScorer
import cPickle
from scipy.sparse import csr_matrix, issparse
from time import time
from utils import LabelBalancer

class XGBoost_NoiseAware(NoiseAwareModel):

    def __init__(self, save_file=None, name='XGBoost', n_threads=None):
        """Noise-aware XGBoost"""
        #parameters
        self.name = name
        self.representation = False
        
        self.eta = .3
        self.min_child_weight = 1
        self.max_depth = 6
        self.gamma = 0
        self.subsample = 1
        self.colsample_bytree = 1
        self.lambda_val = 1
        self.alpha_val = 0
        self.num_rounds = 10
        self.which = 0
        
        #Data
        self.X = None
        
        #Trained Tree
        self.trained_tree = None
        self.seed = None

    def train(self, X, training_marginals, num_rounds=10, eta=0.3,
        min_child_weight=1, max_depth=6, gamma=0, subsample=1, colsample_bytree = 1, lambda_val = 1, alpha_val = 0, 
        verbose = False, seed=None, rebalance=False):
        
        if verbose:
            print("verbose")
        self.eta = eta
        self.min_child_weight = min_child_weight
        self.max_depth = max_depth
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.lambda_val = lambda_val
        self.alpha_val = alpha_val
        self.num_rounds = num_rounds
        self.seed = seed
        self.X = X
        train_idxs = LabelBalancer(training_marginals).get_train_idxs(rebalance)
        
        X_train = X[train_idxs, :]
        y_train = np.ravel(training_marginals)[train_idxs]
        # Run mini-batch SGD
        n = X_train.shape[0]
        dtrain = xgb.DMatrix( X_train, label=y_train)
        watchlist = [(dtrain, 'train')]
        param = {'eta': eta, 'min_child_weight': min_child_weight, 'max_depth': max_depth, 'gamma': gamma, 
                 'subsample': subsample, 'colsample_bytree': colsample_bytree, 'lambda': lambda_val, 'alpha': alpha_val,
                 'objective':'binary:logistic', 'eval_metric': 'logloss', 'silent':not(verbose)}
        self.trained_tree = xgb.train(param, dtrain, num_rounds, watchlist, verbose_eval = verbose)
        
    def save(self, model_name):
        self.trained_tree.save_model(model_name+'.model')
    def load(self, model_name):
        self.trained_tree.load_model(model_name+'.model')
    def marginals(self, X_test):
        dtest = xgb.DMatrix(X_test)
        return self.trained_tree.predict(dtest)