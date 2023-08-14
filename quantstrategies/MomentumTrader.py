"""Momentum trading strategy class."""
import time
import pandas as pd
import numpy as np

from datetime import datetime, timedelta

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
    """

    def __init__(self, conf_file, instrument, bar_length, window, units):
        super().__init__(conf_file)

        self.instrument = instrument
        self.bar_length = pd.to_timedelta(bar_length)
        self.tick_data = pd.DataFrame()
        self.raw_data = pd.DataFrame()
        self.data = None
        self.last_bar = None

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
            self.last_bar = self.raw_data.index[-1]

            if pd.to_datetime(datetime.utcnow()).tz_localize("UTC") - self.last_bar < self.bar_length:
                break

    def on_success(self, time, bid, ask):
        super().on_success(time, bid, ask)

        recent_tick = pd.to_datetime(time)
        df = pd.DataFrame({self.instrument: (ask + bid)/1}, index=[recent_tick])
        self.tick_data = pd.concat([self.tick_data, df])

        if recent_tick - self.last_bar >= self.bar_length:
            self.resample_and_join()
            self.define_con_strategy()

    def resample_and_join(self):
        """Resample the tick data and append to the raw data."""
        self.raw_data = pd.concat([self.raw_data, self.tick_data.resample(
            self.bar_length, label="right").last().ffill().iloc[:-1]])

        self.tick_data = self.tick_data[-1:]
        self.last_bar = self.raw_data.index[-1]

    def define_con_strategy(self):
        """Define the basic contrarian trading strategy."""
        df = self.raw_data.copy()

        df['Returns'] = np.log(df[self.instrument] / df[self.instrument].shift())
        df['ConPosition'] = -np.sign(df.Returns.rolling(self.window).mean())

        self.data = df.copy()
