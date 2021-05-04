"""
Description:
    This file is used to plot the trading results during the testing window
"""
import os
import pdb
import config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

nlstm = config.MlpLstmPolicy_PARAMS["n_lstm"]
RESULT_DIR = f"./results/start_balance_{config.STARTING_BALANCE}_nlstm_{nlstm}"
INTERVAL = config.INTERVAL

for f in os.listdir(RESULT_DIR):
    # get balance data
    if("balance" in f and INTERVAL in f):
        data = pd.read_csv(os.path.join(RESULT_DIR, f))
        f_split_name = f.split("_")
        label = f"{f_split_name[1].upper()} ({f_split_name[2]})"
        balance = data["account_value"].to_numpy()
        date_list = data["date"].values
        # plot it
        plt.plot(date_list, balance, label=label)

plt.xlabel("Date & Time")
plt.xticks(np.arange(0, len(date_list), len(date_list)/8))
plt.ylabel("Balance")
plt.title(f"Trading performance for interval = {INTERVAL}")
plt.legend()
plt.show()



