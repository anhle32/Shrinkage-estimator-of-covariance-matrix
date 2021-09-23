import numpy as np
import pandas as pd
import scipy.optimize as sco
from zipline.api import record, order
from xquant.api import (
    eod_cancel,
    get_universe,
    history,
    is_end_date,
    pip_install,
    rebalance_portfolio,
    schedule_rebalance,
    short_nonexist_assets,
)

pip_install('PyPortfolioOpt')
from pypfopt import risk_models

WINDOW = 2
WEEKLY_DATA = True
WEEKLY_REBALANCE = True
METHODS = [
    'sample',
    'single_index',
    'shrinkage_single_index',
]
METHOD = 'shrinkage_single_index'
TOP_N = 9999


def initialize(context):
    context.frequency = 250
    context.window_length = WINDOW * context.frequency
    context.weights = dict()
    context.shrinkage = None


def handle_data(context, data):
    if WEEKLY_REBALANCE:
        if not schedule_rebalance(context, data, date_rule='week_end'):
            if len(context._miss_shares_df):
                _df = context._miss_shares_df.loc[context.datetime.strftime(
                    '%Y-%m-%d')]
                _df = _df[(_df != 0) & (_df.notnull())]
                for asset in _df.index:
                    order(asset, _df.loc[asset])
            if not context._is_production_mode:
                return None

    universe = get_universe(context, data)
    df = data.history(universe, 'close', context.window_length, '1d')
    record(Len_Original_Universe=len(universe))

    df = df.dropna(axis=1, thresh=250)
    universe = df.columns
    record(Len_Universe=len(universe))

    if TOP_N < len(universe):
        shares = data.history(universe, 'shares_outstanding',
                              context.window_length, '1d')
        market_cap = (shares * df).mean().sort_values()
        universe = market_cap.tail(TOP_N).index
        df = df[universe]

    volume = data.history(universe, 'volume', 1, '1d')
    total_trading_market = (df.iloc[-1] * volume.iloc[-1]).sum() / 1e6
    record(Total_Trading_Market_in_billion=total_trading_market)

    volume_df = data.history(universe, 'raw_volume', 20, '1d')
    liquid_df = np.minimum(volume_df.mean(), volume_df.median())
    liquid_df = liquid_df.sort_values(ascending=False)
    record(Liquidity_Minimum=liquid_df.min())

    if WEEKLY_DATA:
        context.frequency = 52
        df['yearweek'] = df.index.map(lambda x: x.strftime('%Y%W'))
        df = df.groupby('yearweek').tail(1).drop('yearweek', axis=1)

    df = df.loc[:, df.apply(pd.Series.nunique) > 1]
    universe = df.columns

    if 0 in df.shape:
        context.weights = {}
        rebalance_portfolio(context, data, context.weights)
        return None

    if METHOD == 'sample':
        cov = risk_models.sample_cov(prices=df, frequency=context.frequency)
    elif METHOD == 'single_index':
        cov = shrinkage_single_index(x=df,
                                     shrink=1,
                                     frequency=context.frequency)
    elif METHOD == 'shrinkage_single_index':
        cov, context.shrinkage = shrinkage_single_index(
            x=df, frequency=context.frequency)

    record(p_div_n=df.shape[1] / len(df))
    if context.shrinkage is not None:
        record(Shrinkage=context.shrinkage)

    if df.shape[1] > 10:
        max_weights = pd.Series(0.1, index=universe)

        max_weights[max_weights > 0.1] = 0.1
        max_weights = max_weights.fillna(0)
        record(Total_Max_Weight=max_weights.sum())
        record(Len_Max_Weight_as_10per=len(max_weights[max_weights >= 0.1]))

        weight_bounds = tuple([(0, max_weight) for max_weight in max_weights])
        weights = min_volatility(cov, weight_bounds)
        weights = pd.Series(weights, index=universe)
        weights = weights[weights > 1e-6]
        context.weights = weights.to_dict()

        record(Len_Optimization=len(universe))
        record(Liquidity_Min_in_Portfolio=liquid_df.loc[weights.index].min())
    else:
        context.weights = {
            k: v
            for k, v in context.weights.items() if k in universe
        }
    record(Total_Weight=sum(context.weights.values()))
    record(Len_Portfolio=len(context.weights))

    if WEEKLY_REBALANCE:
        if not schedule_rebalance(context, data, date_rule='week_end'):
            return None
    rebalance_portfolio(context, data, context.weights)


def shrinkage_single_index(x, shrink=None, frequency=252):
    """
    This estimator is a weighted average of the sample  covariance matrix and a "prior" or "shrinkage target".
    Here, the prior is given by a one-factor model.
    The factor is equal to the cross-sectional average   of all the random variables.
    The notation follows Ledoit and Wolf (2003), version: 04/2014
    NOTE: use (pairwise) covariance on raw returns
    Parameters
    ----------
    x : T x N stock returns
    shrink : given shrinkage intensity factor if none, code calculates
    Returns
    -------
    tuple : np.ndarray which contains the shrunk covariance matrix
          : float shrinkage intensity factor

    """
    x = x.pct_change().dropna(how='all')

    t, n = np.shape(x)
    meanx = x.mean(axis=0)
    x = x - np.tile(meanx, (t, 1))
    xmkt = x.mean(axis=1).reshape(t, 1)

    sample = pd.DataFrame(np.append(x, xmkt, axis=1)).cov() * (t - 1) / t
    sample = sample.as_matrix()
    covmkt = sample[0:n, n].reshape(n, 1)
    varmkt = sample[n, n]
    sample = sample[:n, :n]
    prior = np.dot(covmkt, covmkt.T) * varmkt
    prior[np.eye(n) == 1] = np.diag(sample)

    if shrink == 1:
        return prior

    x = x.as_matrix()
    x = np.nan_to_num(x)

    if shrink is None:
        c = np.linalg.norm(sample - prior, "fro")**2
        y = x**2
        p = 1 / t * np.sum(np.dot(y.T, y)) - np.sum(sample**2)

        rdiag = 1 / t * np.sum(y**2) - sum(np.diag(sample)**2)
        z = x * np.tile(xmkt, (n, ))
        v1 = 1 / t * np.dot(y.T, z) - np.tile(covmkt, (n, )) * sample
        roff1 = (np.sum(v1 * np.tile(covmkt, (n, )).T) / varmkt -
                 np.sum(np.diag(v1) * covmkt.T) / varmkt)
        v3 = 1 / t * np.dot(z.T, z) - varmkt * sample
        roff3 = (np.sum(v3 * np.dot(covmkt, covmkt.T)) / varmkt**2 -
                 np.sum(np.diag(v3).reshape(-1, 1) * covmkt**2) / varmkt**2)
        roff = 2 * roff1 - roff3
        r = rdiag + roff

        k = (p - r) / c
        shrinkage = max(0, min(1, k / t))
    else:
        shrinkage = shrink

    sigma = shrinkage * prior + (1 - shrinkage) * sample
    sigma = sigma * frequency
    return sigma, shrinkage


def volatility(weights, cov_matrix, gamma=0):
    L2_reg = gamma * (weights**2).sum()
    portfolio_volatility = np.dot(weights.T, np.dot(cov_matrix, weights))
    return portfolio_volatility + L2_reg


def min_volatility(cov, weight_bounds):
    initial_guess = [0] * len(cov)
    constraints = [
        {
            "type": "eq",
            "fun": lambda x: np.sum(x) - 1,
        },
    ]
    args = (cov, 0)
    result = sco.minimize(
        volatility,
        x0=initial_guess,
        args=args,
        method="SLSQP",
        bounds=weight_bounds,
        constraints=constraints,
    )
    if not result['success']:
        raise ValueError
    weights = result["x"]
    return weights
