import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

from imbalanced_loss.loss import immax_logistic_obj, immax_logloss_metric


class _XGBoost(BaseEstimator):
    def __init__(
        self,
        params: dict,
        valid_frac: float = 0.1,
        seed: int = 1234,
        num_boost_round: int = 10000,
        early_stopping_rounds: int = 100,
        verbose_eval: int = 1000,
    ) -> None:
        self.params = params
        self.valid_frac = valid_frac
        self.seed = seed
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval

        self.eval_result = dict()

    def fit(self, X, y, **model_kwargs) -> '_XGBoost':
        n_iter: int = self._estimate_best_iteration(X, y, **model_kwargs)
        dtrain = xgb.DMatrix(X, label=y)
        self.estimator = xgb.train(self.params, dtrain, num_boost_round=n_iter, **model_kwargs)
        return self

    def _estimate_best_iteration(self, X, y, **model_kwargs) -> int:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.valid_frac, random_state=self.seed)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_val, label=y_val)
        estimator_with_val: xgb.Booster = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=[(dtrain, 'train'), (dvalid, 'valid')],
            evals_result=self.eval_result,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=self.verbose_eval,
            **model_kwargs,
        )
        return estimator_with_val.best_iteration

    def _predict(self, X, **kwargs) -> np.ndarray:
        y_pred = self.estimator.predict(xgb.DMatrix(X), **kwargs)
        return y_pred

    def plot_importance(self, **kwargs):
        return xgb.plot_importance(self.estimator, **kwargs)

    def plot_metric(self, **kwargs):
        fig, ax = plt.subplots(**kwargs)
        for key in self.eval_result.keys():
            for metric in self.eval_result[key].keys():
                ax.plot(self.eval_result[key][metric], label=f'{key} ({metric})')
        ax.grid()
        ax.legend()


class XGBClassifier(_XGBoost, ClassifierMixin):
    def predict(self, X, threshold: float = 0.5, **kwargs) -> np.ndarray:
        return (self.predict_proba(X, **kwargs)[:, 1] > threshold).astype(int)

    def predict_proba(self, X, **kwargs) -> np.ndarray:
        y_pred = self._predict(X, **kwargs)
        return np.c_[1 - y_pred, y_pred]


class IMMAXClassifier(XGBClassifier):
    def __init__(
        self,
        params: dict,
        valid_frac: float = 0.1,
        seed: int = 1234,
        num_boost_round: int = 10000,
        early_stopping_rounds: int = 100,
        verbose_eval: int = 1000,
        rho_plus: float = 1.0,
        rho_minus: float = 1.0,
    ) -> None:
        super().__init__(
            params,
            valid_frac,
            seed,
            num_boost_round,
            early_stopping_rounds,
            verbose_eval,
        )
        self.rho_plus = rho_plus
        self.rho_minus = rho_minus

        self.obj = immax_logistic_obj(self.rho_plus, self.rho_minus)
        self.feval = immax_logloss_metric(self.rho_plus, self.rho_minus)

    def fit(self, X, y, **model_kwargs) -> 'IMMAXClassifier':
        n_iter: int = self._estimate_best_iteration(X, y, **model_kwargs)
        dtrain = xgb.DMatrix(X, label=y)
        self.estimator = xgb.train(
            self.params,
            dtrain,
            num_boost_round=n_iter,
            obj=self.obj,
            **model_kwargs,
        )
        return self

    def _estimate_best_iteration(self, X, y, **model_kwargs) -> int:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self.valid_frac,
            random_state=self.seed,
        )
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_val, label=y_val)
        estimator_with_val: xgb.Booster = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=[(dtrain, 'train'), (dvalid, 'valid')],
            evals_result=self.eval_result,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=self.verbose_eval,
            obj=self.obj,
            custom_metric=self.feval,
            **model_kwargs,
        )
        return estimator_with_val.best_iteration
