import numpy as np
import xgboost as xgb
from .utils import print_scores
from .utils import marginals_to_labels, MentionScorer

class XGBoost_NoiseAware(object):

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
        self.marginals = None
        
        #Trained Tree
        self.trained_tree = None
        self.seed = None
      
    def _logregobj(preds, dtrain):
        #Logistic Regression Objective
        labels = dtrain.get_label()
        preds = 1.0 / (1.0 + np.exp(-preds))
        grad = preds - labels
        hess = preds * (1.0-preds)
        return grad, hess

    def _evalerror(preds, dtrain):
        #Log Likelihood Error
        labels = dtrain.get_label()
        errors = np.mean(labels*np.log(1+np.exp(-preds))+(1-labels)*np.log(1+np.exp(preds)))
        return 'error', errors
    
    def _check_input(self, num_rounds, eta, min_child_weight):
        #TODO: Check all parameters are within their valid range
        if issparse(X):
            raise Exception("Sparse input matrix. Use SparseLogisticRegression")
        return X

    def train(self, X, train_marginals, num_rounds=10, eta=0.3,
        min_child_weight=1, max_depth=6, gamma=0, subsample=1, colsample_bytree = 1, lambda_val = 1, alpha_val = 0, 
        verbose = False, which = 0, seed=None):
        
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
        self.which = which
        self.seed = seed
        self.X = X
        self.marginals = train_marginals
        
        dtrain = xgb.DMatrix( self.X, label=self.marginals)
        watchlist = [(dtrain, 'train')]
        if which == 0:
            param = {'eta': eta, 'min_child_weight': min_child_weight, 'max_depth': max_depth, 'gamma': gamma, 
                     'subsample': subsample, 'colsample_bytree': colsample_bytree, 'lambda': lambda_val, 'alpha': alpha_val,
                     'objective':'binary:logistic', 'eval_metric': 'logloss', 'silent':not(verbose)}
            self.trained_tree = xgb.train(param, dtrain, num_rounds, watchlist, verbose_eval = verbose)
        else:
            param = {'eta': eta, 'min_child_weight': min_child_weight, 'max_depth': max_depth, 'gamma': gamma, 
                     'subsample': subsample, 'colsample_bytree': colsample_bytree, 'lambda': lambda_val, 'alpha': alpha_val, 'silent':not(verbose)}
            self.trained_tree = xgb.train(param, dtrain, num_round, watchlist, self._logregobj, self._evalerror, verbose_eval = verbose)
        
    def save(self, model_name):
        self.trained_tree.save_model(model_name+'.model')
    def load(self, model_name):
        self.trained_tree.load_model(model_name+'.model')
    
    def score(self, session, X_test, test_labels, gold_candidate_set=None, 
        b=0.5, set_unlabeled_as_neg=False, display=True, scorer=None,
        **kwargs):
        dtest = xgb.DMatrix( X_test, label=test_labels)
        
        preds_prob = self.trained_tree.predict(dtest)
        if self.which == 0:
            preds = np.round(preds_prob)
        else:
            preds[preds_prob > 0] = 1
            preds[preds_prob < 0] = 0

        TP = np.nonzero(np.logical_and(preds,test_labels))[0]
        TN = np.nonzero(np.logical_and(np.logical_not(preds),np.logical_not(test_labels)))[0]
        FP = np.nonzero(np.logical_and(preds,np.logical_not(test_labels)))[0]
        FN = np.nonzero(np.logical_and(np.logical_not(preds),test_labels))[0]
        print_scores(len(TP), len(FP), len(TN), len(FN))
        return TP, FP, TN, FN
       