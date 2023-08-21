from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8")


class SMABacktester:
    """Class for the vectorized backtesting of SMA-based trading strategies."""

    def __init__(self, symbol, datapath, SMA_S, SMA_L, start, end, tc):
        """ Contruct a new SMABacktester.

        Args:
        ----
        symbol: str
            ticker symbol (instrument) to be backtested
        datapath: str
            the path to the csv dataset
        SMA_S: int
            moving window in bars (e.g. days) for shorter SMA
        SMA_L: int
            moving window in bars (e.g. days) for longer SMA
        start: str
            start date for data import
        end: str
            end date for data import
        tc: float
            costs per trade
        """
        self.symbol = symbol
        self.path = datapath
        self.SMA_S = SMA_S
        self.SMA_L = SMA_L
        self.start = start
        self.end = end
        self.results = None
        self.tc = tc
        self.get_data()
        self.prepare_data()

    def __repr__(self):
        # trunk-ignore(flake8/E501)
        return "SMABacktester(symbol={}, SMA_S={}, SMA_L={}, start={}, end={})".format(
            self.symbol, self.SMA_S, self.SMA_L, self.start, self.end
        )

    def get_data(self):
        """Import the data from forex_pairs.csv (source can be changed)."""

        raw = pd.read_csv(self.path,
                          parse_dates=["Date"],
                          index_col="Date")

        raw = raw[self.symbol].to_frame().dropna()  # type: ignore
        raw = raw.loc[self.start:self.end].copy()
        raw.rename(columns={self.symbol: "price"}, inplace=True)
        raw["returns"] = np.log(raw / raw.shift(1))
        self.data = raw

    def prepare_data(self):
        """Prepare the data for strategy backtesting (strategy-specific)."""

        data = self.data.copy()
        data["SMA_S"] = data["price"].rolling(self.SMA_S).mean()
        data["SMA_L"] = data["price"].rolling(self.SMA_L).mean()
        self.data = data

    def set_parameters(self, SMA_S=None, SMA_L=None):
        """Update SMA parameters and the prepared dataset."""

        if SMA_S is not None:
            self.SMA_S = SMA_S
            self.data["SMA_S"] = self.data["price"].rolling(self.SMA_S).mean()
        if SMA_L is not None:
            self.SMA_L = SMA_L
            self.data["SMA_L"] = self.data["price"].rolling(self.SMA_L).mean()

    def test_strategy(self):
        """Backtests the SMA-based trading strategy."""
        data = self.data.copy().dropna()
        data["position"] = np.where(data["SMA_S"] > data["SMA_L"], 1, -1)
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data.dropna(inplace=True)

        data["trades"] = data.position.diff().fillna(0).abs()

        # Subtract trading costs
        data.strategy = data.strategy - data.trades * self.tc

        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data

        # absolute performance of the strategy
        perf = data["cstrategy"].iloc[-1]

        # out-/underperformance of strategy
        outperf = perf - data["creturns"].iloc[-1]

        return round(perf, 6), round(outperf, 6)

    def plot_results(self):
        """Plot the performance of the trading strategy and
        compares to "buy and hold"."""

        if self.results is None:
            print("Run test_strategy() first.")
        else:
            title = "{} | SMA_S = {} | SMA_L = {}".format(
                self.symbol, self.SMA_S, self.SMA_L
            )
            self.results[["creturns", "cstrategy"]].plot(title=title,
                                                         figsize=(12, 8))

    def optimize_parameters(self, SMA_S_range, SMA_L_range):
        """Find the optimal strategy (global maximum) given the
        SMA parameter ranges.

        Args:
        ----
        SMA_S_range: tuple
            tuples of the form (start, end, step size)
        SMA_L_range: tuple
            tuples of the form (start, end, step size)
        """

        combinations = list(product(range(*SMA_S_range), range(*SMA_L_range)))

        # test all combinations
        results = []
        for comb in combinations:
            self.set_parameters(comb[0], comb[1])
            results.append(self.test_strategy()[0])

        best_perf = np.max(results)  # best performance
        opt = combinations[np.argmax(results)]  # optimal parameters

        # run/set the optimal strategy
        self.set_parameters(opt[0], opt[1])
        self.test_strategy()

        # create a df with many results
        many_results = pd.DataFrame(data=combinations,
                                    columns=["SMA_S", "SMA_L"])
        many_results["performance"] = results
        self.results_overview = many_results

        return opt, best_perf
