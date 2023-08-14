"""Main module."""
from os.path import dirname, abspath, join

from MomentumTrader import MomentumTrader

cfg_path = join(dirname(dirname(abspath(__file__))), "oanda.cfg")

trader = MomentumTrader(cfg_path, "EUR_USD", "1min", window=1, units=100000)
