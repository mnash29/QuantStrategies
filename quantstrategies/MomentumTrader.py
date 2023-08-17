"""Momentum trading strategy class."""
import time
import pandas as pd
import numpy as np

from datetime import datetime, timedelta
from operator import itemgetter

import tpqoa


class MomentumTrader(tpqoa.tpqoa):
    """Create a basic momentum trader class.

    Args:
    ----
    conf_file : str
        The file path to the OANDA cfg file
    instrument : str
        The ticker symbol of the instrument being analyzed
    bar_length : str
        The time interval to view the tick data, e.g. '1min'
    window: int
        The number of point in the tick data for establishing a moving average
    units: int
        The number of units to use when entering/exiting a position
    features: list
        The list of feature columns for use in model prediction
    model: model
        The trained machine learning model
    dev: int
        The number of deviations away from the trendline for use in the Bollinger Band
        strategy, default=2
    SMA: tuple
        The tuple of window lengths for calculating smooth moving averages. The first two
        values are used for evaluating a SMA cross over strategy and the third is used for
        the Bollinger Band middle trend line, default=(50, 200, 20)
    EMA: tuple
        The typle of window lengths for calculating stacked EMA's, default=(8, 21, 34, 55, 89)
    """

    def __init__(self, conf_file, instrument,
                 bar_length, window, units,
                 dev=2, SMA=(50, 200, 20)):
        super().__init__(conf_file)

        self.instrument = instrument
        self.bar_length = pd.to_timedelta(bar_length)
        self.tick_data = pd.DataFrame()
        self.raw_data = pd.DataFrame()
        self.data = pd.DataFrame()
        self.last_bar = None

        # SMA and Bollinger band strategy attributes
        self.SMA = SMA
        self.deviation = dev

        # Contrarian strategy attributes
        self.window = window
        self.units = units

    def get_most_recent(self, days=5):
        """Retreive the most recent trading tick data.

        Kwargs:
        ------
        **days : int, optional
            The number trading days, by default 5
        """
        while True:
            time.sleep(2)
            utcnow = datetime.utcnow()
            now = utcnow - timedelta(microseconds=utcnow.microsecond)
            past = now - timedelta(days=days)

            df = self.get_history(instrument=self.instrument,
                                  start=past,
                                  end=now,
                                  granularity="S5",
                                  price="M",
                                  localize=False).c.dropna().to_frame()
            df.rename(columns={"c": self.instrument}, inplace=True)
            df = df.resample(
                self.bar_length, label="right").last().dropna().iloc[:-1]

            self.raw_data = df.copy()
            self.data = df.copy()
            self.last_bar = self.raw_data.index[-1]

            if pd.to_datetime(datetime.utcnow()).tz_localize("UTC") - self.last_bar < self.bar_length:
                break

    def on_success(self, time, bid, ask):
        super().on_success(time, bid, ask)

        recent_tick = pd.to_datetime(time)
        df = pd.DataFrame({self.instrument: (ask + bid)/2}, index=[recent_tick])
        self.tick_data = pd.concat([self.tick_data, df])

        if recent_tick - self.last_bar >= self.bar_length:
            self.resample_and_join()

    def resample_and_join(self):
        """Resample the tick data and append to the raw data."""
        self.raw_data = pd.concat([self.raw_data, self.tick_data.resample(
            self.bar_length, label="right").last().ffill().iloc[:-1]])

        self.tick_data = self.tick_data[-1:]
        self.last_bar = self.raw_data.index[-1]

    def merge_dataframes(self, df: pd.DataFrame):
        """Merge the strategy data with the current dataset.

        Args:
        ----
        df : pd.DataFrame
            The new DataFrame with strategy specific columns
        """
        df.drop(self.instrument, axis=1, inplace=True)
        self.data = pd.merge(self.data, df, on='time')

    def eval_con_strategy(self):
        """Evaluate tick data using the basic contrarian trading strategy."""
        if self.data is None:
            raise ValueError("Make call to get_recent_data() or stream_data() before evaluating.")

        df = self.raw_data.copy()

        df['Returns'] = np.log(df[self.instrument] / df[self.instrument].shift())
        df['Position_Con'] = -np.sign(df.Returns.rolling(self.window).mean())

        self.merge_dataframes(df)

    def eval_sma_stategy(self):
        """Evaluate tick data using a basic SMA crossover strategy."""
        if len(self.SMA) < 3:
            raise ValueError("Invalid values set for SMA attribute")

        if self.data is None:
            raise ValueError("Make call to get_recent_data() or stream_data() before evaluating.")

        df = self.raw_data.copy()

        df['SMA_S'] = df[self.instrument].rolling(self.SMA[0]).mean()
        df['SMA_L'] = df[self.instrument].rolling(self.SMA[1]).mean()

        df['Position_SMA'] = np.where(df.SMA_S > df.SMA_L, 1, -1)

        self.merge_dataframes(df)

    def eval_bollband_strategy(self):
        """Evaluate tick data using a Bollinger Band strategy."""
        if len(self.SMA) < 3:
            raise ValueError("Invalid values set for SMA attribute")

        if self.data is None:
            raise ValueError("Make call to get_recent_data() or stream_data() before evaluating.")

        df = self.raw_data.copy()

        df['BB_SMA'] = df[self.instrument].rolling(self.SMA[2]).mean()
        df['BB_Lower'] = df.BB_SMA - (
            df[self.instrument].rolling(self.SMA[2]).std() * self.deviation)
        df['BB_Upper'] = df.BB_SMA + (
            df[self.instrument].rolling(self.SMA[2]).std() * self.deviation)
        df['Distance'] = df[self.instrument] - df.BB_SMA

        df['Position_BB'] = np.where(df[self.instrument] < df.BB_Lower, 1, np.nan)
        df['Position_BB'] = np.where(df[self.instrument] > df.BB_Lower, -1, df.Position_BB)
        df['Position_BB'] = np.where(
            df.Distance * df.Distance.shift() < 0, 0, df.Position_BB)
        df['Position_BB'] = df.Position_BB.ffill().fillna(0)

        self.merge_dataframes(df)

    # def log_reg_predict(self, features: list, model: LogisticRegression):
    #     """Output a target market direction given current tick data.

    #     Args:
    #     ----
    #     features : list
    #         The list of feature columns for use in class label prediction
    #     model : LogisticRegression
    #         The pre-trained machine learning model
    #     """
    #     df = self.raw_data.copy()
    #     df = df.append(self.tick_data)

    #     df['Returns'] = np.log(df[self.instrument] / df[self.instrument].shift())
    #     df['Position_ML'] = model.predict(df[features])

    #     return df


    def report_trade(self, order, action):
        """Display trade information.

        TODO: Ideally this should log to a file or database

        Args:
        ----
        order : dict
            The executed order information
        action : str
            The trade action taken, e.g. 'GOING LONG, GOING SHORT'
        """
        time, units, price, pl = itemgetter("time", "units", "price", "pl")(order)

        if self.profits is None:
            self.profits = [pl]
        else:
            self.profits.append(pl)

        cumpl = sum(self.profits)
        print("\n" + 100 * "-")
        print(f"{time} | {action}")
        print(f"{time} | units={units} | price={price} | P&L={pl} | Cum. P&L={cumpl}")
        print(100 * "-" + "\n")
