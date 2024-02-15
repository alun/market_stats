import numpy as np
import pandas as pd
from IPython.display import display


def get_log_returns(portfolio_history):
    return np.log(portfolio_history).diff().dropna()


def _get_pct_drawdown(log_returns):
    cum_returns = log_returns.cumsum()
    last_peak = cum_returns.cummax()
    log_dd = (cum_returns - last_peak).dropna()
    return (np.exp(log_dd) - 1) * 100


def plot_dd(returns, is_logreturns=False):
    log_returns = returns if is_logreturns else get_log_returns(returns)
    pct_dd = _get_pct_drawdown(log_returns)
    ax = pct_dd.plot(figsize=(15, 8), color="r", alpha=0.7)
    ax.grid(axis="both")
    ax.set_ylabel("Drawdown, %")


def _DrawdownStreaks__to_years(days):
    return days / 365.25


def _DrawdownStreaks__to_months(days):
    return 12 * _DrawdownStreaks__to_years(days)


class DrawdownStreaks:
    def __init__(self, returns, is_logreturns=False, default_display_limit=10):
        log_returns = returns if is_logreturns else get_log_returns(returns)
        self.__default_display_limit = default_display_limit
        self.__pct_dd = _get_pct_drawdown(log_returns)
        self.__streaks = DrawdownStreaks.__find_true_streaks(self.__pct_dd.values != 0)

    def __find_true_streaks(array):
        # finds all "True" streaks in array from longest to shortest

        masked = np.concatenate(([False], array, [False]))
        true_streaks = np.flatnonzero(masked[1:] != masked[:-1]).reshape(-1, 2)
        ends = true_streaks[:, 1]
        true_streaks_descending = (true_streaks[:, 1] - true_streaks[:, 0]).argsort()[
            ::-1
        ]
        return true_streaks[true_streaks_descending]

    def __get_streak_details(self, streak):
        pct_dd = self.__pct_dd

        start, end = streak
        if end == len(pct_dd.index):
            end = end - 1

        days_total = (pct_dd.index[end] - pct_dd.index[start]).days
        max_dd_index = start + np.argmin(pct_dd.iloc[start:end])
        max_depth = pct_dd.iloc[max_dd_index]

        days_to_maxdd = (pct_dd.index[max_dd_index] - pct_dd.index[start]).days

        return dict(
            days=days_total,
            months=__to_months(days_total),
            years=__to_years(days_total),
            from_date=pct_dd.index[start].date(),
            to_date=pct_dd.index[end].date(),
            max_depth=max_depth,
            max_dd_date=pct_dd.index[max_dd_index].date(),
            days_d=days_to_maxdd,
            months_d=__to_months(days_to_maxdd),
            years_d=__to_years(days_to_maxdd),
        )

    def to_dataframe(self):
        dd_streaks = self.__streaks

        streaks_info = []
        for streak in dd_streaks:
            streaks_info.append(self.__get_streak_details(streak))

        return pd.DataFrame(streaks_info)

    def display(self, limit=10):
        df = self.to_dataframe()
        if df.empty:
            print("No drawdown found")
        else:
            pd.options.display.float_format = "{:,.2f}".format
            print("Longest drawdowns")
            display(df.head(limit))
            print("Deepest drawdowns")
            display(
                df.sort_values(by=["max_depth"])
                .reset_index()
                .drop(["index"], axis=1)
                .head(limit)
            )
            pd.options.display.float_format = None

    def _ipython_display_(self):
        self.display(self.__default_display_limit)


class PortfolioStats:

    def __init__(
        self, returns, is_logreturns=False, trading_days_yearly=252, risk_free_rate=0.00
    ):
        log_returns = returns if is_logreturns else get_log_returns(returns)
        trading_days_yearly_sqrt = np.sqrt(trading_days_yearly)
        self.risk_free_rate = risk_free_rate

        annual_return = np.exp(log_returns.mean() * trading_days_yearly) - 1
        self.annual_return = annual_return
        annual_volatility = log_returns.std() * trading_days_yearly_sqrt
        self.annual_volatility = annual_volatility = annual_volatility

        self.sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility

        annual_downside = (
            log_returns.loc[log_returns < 0].std() * trading_days_yearly_sqrt
        )
        self.annual_downside = annual_downside
        self.sortino_ratio = (annual_return - risk_free_rate) / annual_downside

    def _ipython_display_(self):
        print(
            f"""Annualized return {(self.annual_return * 100):.2f}%
Annualized volatility {(self.annual_volatility * 100):.2f}%
Sharpe ratio {self.sharpe_ratio:.2f} 
Annualized downside volatility {(self.annual_downside * 100):.2f}%
Sortino ratio {self.sortino_ratio:.2f}
(on the basis of {self.risk_free_rate * 100:.2f}% risk free rate)
""".lstrip()
        )
