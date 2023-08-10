"""Backtest mean reversion strategies."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from itertools import product

plt.style.use("seaborn-v0_8")


class BollingerBacktester:
    """Class for backtesting the bollinger bands strategy."""

    def __init__(self, symbol, datapath, SMA_m, deviation, start, end, tcost):
        """Construct a new BollingerBacktester object.

        Args:
        ----
        symbol : str
            The ticker symbol
        datapath : str
            The path to the csv data file
        SMA_m : int
            The SMA for the midline
        deviation : int
            Distance from the SMA_m
        start : str
            The starting date
        end : str
            The ending date
        tcost : float
            The estimated cost per trade
        """
        self.symbol = symbol
        self.path = datapath
        self.SMA_m = SMA_m
        self.deviation = deviation
        self.start = start
        self.end = end
        self.ptc = tcost
        self.results = None
        self.get_data()
        self.prepare_data()

    def __repr__(self):
        return (
            "BollingerBacktester(symbol={}, SMA_m={}, dev={}, start={}, end={})".format(
                self.symbol, self.SMA_m, self.deviation, self.start, self.end
            )
        )

    def get_data(self):
        """Import the data from forex_pairs.csv (source can be changed)."""

        raw = pd.read_csv(self.path, parse_dates=["time"], index_col="time")

        raw = raw[self.symbol].to_frame().dropna()  # type: ignore
        raw = raw.loc[self.start:self.end].copy()
        raw.rename(columns={self.symbol: "price"}, inplace=True)
        raw["Returns"] = np.log(raw / raw.shift())
        self.data = raw

    def prepare_data(self):
        """Prepare the data for strategy backtesting."""

        data = self.data.copy()
        data["SMA"] = data.price.rolling(self.SMA_m).mean()
        data["Lower"] = data.SMA - (
            data.price.rolling(self.SMA_m).std() * self.deviation)
        data["Upper"] = data.SMA + (
            data.price.rolling(self.SMA_m).std() * self.deviation)
        data["Distance"] = data.price - data.SMA
        self.data = data

    def set_parameters(self, SMA_m=None, dev=None):
        """Update the SMA parameters and prepared dataset."""

        if SMA_m is not None:
            self.SMA_m = SMA_m
            self.data["SMA"] = self.data.price.rolling(self.SMA_m).mean()
            self.data["Lower"] = (
                self.data.SMA - (
                    self.data.price.rolling(self.SMA_m).std() * self.deviation
                )
            )
            self.data["Upper"] = (
                self.data.SMA + (
                    self.data.price.rolling(self.SMA_m).std() * self.deviation
                )
            )

        if dev is not None:
            self.deviation = dev
            self.data["Lower"] = (
                self.data.SMA - (
                    self.data.price.rolling(self.SMA_m).std() * self.deviation
                )
            )
            self.data["Upper"] = (
                self.data.SMA + (
                    self.data.price.rolling(self.SMA_m).std() * self.deviation
                )
            )

    def test_strategy(self):
        """Backtest the Bollinger Band strategy."""

        data = self.data.copy().dropna()
        data['Distance'] = data.price - data.SMA

        # Establish long/short position
        data['Position'] = np.where(data.price < data.Lower, 1, np.nan)
        data['Position'] = np.where(data.price > data.Upper, -1, data.Position)

        # Establish neutral position
        data['Position'] = np.where(
            data.Distance * data.Distance.shift() < 0, 0, data.Position)

        # Establish hold position where theres no long/short/neutral
        data['Position'] = data.Position.ffill().fillna(0)

        # Vectorize strategy
        data['Strategy'] = data.Position.shift() * data.Returns
        data.dropna(inplace=True)
        data['Trades'] = data.Position.diff().fillna(0).abs()

        # Subtract transaction costs
        data['Strategy'] = data.Strategy - data.Trades * self.ptc

        data['CReturns'] = data.Returns.cumsum().apply(np.exp)
        data['CStrategy'] = data.Strategy.cumsum().apply(np.exp)
        self.results = data

        # Performance of the strategy
        perf = data.CStrategy.iloc[-1]
        outperf = perf - data.CReturns.iloc[-1]

        return round(perf, 6), round(outperf, 6)

    def plot_results(self):
        """Plot the performance of the Bollinger Bands trading strategy
        compared to buy and hold."""

        if self.results is None:
            print("Run test_strategy() before plotting results.")
        else:
            title = f"{self.symbol} | SMA = {self.SMA_m} | Dev = {self.deviation}"
            self.results[['CReturns', 'CStrategy']].plot(title=title,
                                                         figsize=(12, 8))

    def optimize_parameters(self, SMA_range, dev_range):
        """Find the optimal strategy give the SMA and deviation parameters.

        Args:
        ----
        SMA_range : tuple
            Tuple of the form (SMA, deviation, step size)
        dev_range : tuple
            Tuple of the form (SMA, deviation, step size)
        """
        combinations = list(product(range(*SMA_range), range(*dev_range)))

        results = []
        for comb in combinations:
            self.set_parameters(comb[0], comb[1])
            results.append(self.test_strategy()[0])

        best_perf = np.max(results)
        opt = combinations[np.argmax(results)]

        self.set_parameters(opt[0], opt[1])
        self.test_strategy()

        many_results = pd.DataFrame(data=combinations,
                                    columns=["SMA", "Dev"])
        many_results["performance"] = results
        self.results_overview = many_results

        return opt, best_perf
