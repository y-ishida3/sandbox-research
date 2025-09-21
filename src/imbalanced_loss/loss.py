import numpy as np
import xgboost as xgb


def immax_logistic_obj(rho_plus: float, rho_minus: float):
    """IMMAX（二値）ロジスティック版のカスタム目的関数。
    rho_plus:  正例クラスのマージン係数 ρ+
    rho_minus: 負例クラスのマージン係数 ρ−
    """

    def _obj(preds: np.ndarray, dtrain: xgb.DMatrix):
        # y in {0,1} を {−1,+1} に変換
        y01 = dtrain.get_label().astype(np.float64)
        y = 2.0 * y01 - 1.0  # -> {-1, +1}

        rho = np.where(y > 0, rho_plus, rho_minus)  # サンプルごとの ρ_y
        u = (y * preds) / rho  # u = y f / ρ_y

        # σ(u) = 1 / (1 + exp(-u))
        sigma = 1.0 / (1.0 + np.exp(-u))

        grad = (sigma - 1.0) * (y / rho)
        hess = (sigma * (1.0 - sigma)) / (rho * rho)

        return grad, hess

    return _obj


def immax_logloss_metric(rho_plus: float, rho_minus: float):
    """学習中にモニタするための損失（平均）。"""

    def _feval(preds: np.ndarray, dtrain: xgb.DMatrix):
        y01 = dtrain.get_label().astype(np.float64)
        y = 2.0 * y01 - 1.0
        rho = np.where(y > 0, rho_plus, rho_minus)
        u = (y * preds) / rho
        # natural log のロジ損
        # loss = np.log1p(np.exp(-u))
        loss = np.logaddexp(0, -u)
        return 'immax_logloss', float(np.mean(loss))

    return _feval
