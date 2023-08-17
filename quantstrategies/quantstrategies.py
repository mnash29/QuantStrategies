"""Main module."""
import pickle
import numpy as np
from os.path import dirname, abspath, join

from MomentumTrader import MomentumTrader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

cfg_path = join(dirname(dirname(abspath(__file__))), "oanda.cfg")

trader = MomentumTrader(cfg_path, "EUR_USD", "1min", window=1, units=100000, dev=1, SMA=(50, 200, 47))
trader.get_most_recent()
trader.eval_con_strategy()
trader.eval_sma_stategy()
trader.eval_bollband_strategy()

trader.data['Direction'] = np.sign(trader.data.Returns.shift(-1))

lags = 3
cols = ['Position_Con', 'Position_SMA', 'Position_BB']
# cols = []
# for lag in range(1, lags + 1):
#     col = "lag{}".format(lag)
#     trader.data[col] = trader.data.Returns.shift(lag)
#     cols.append(col)

trader.data.dropna(inplace=True)

model = LogisticRegression(C=1e6, max_iter=100000, multi_class="ovr")

X_train, X_test, y_train, y_test = train_test_split(trader.data, trader.data.Direction, test_size=0.30, random_state=42, shuffle=False)

model.fit(X_train[cols], y_train)
scores = model.score(X_test[cols], y_test)
print(scores)

X_test['Pred'] = model.predict(X_test[cols])
X_test.to_csv('model_preds.csv')