"""
Description:
    This file is used to plot the trading results during 2021-01-01 to 2021-04-30.
"""
import os
import pdb
import config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

nlstm = config.MlpLstmPolicy_PARAMS["n_lstm"]
RESULT_DIR = f"./results/start_balance_{config.INITIAL_BALANCE}_nlstm_{nlstm}"
INTERVAL = config.INTERVAL
start_date = config.START_DATE_TEST
end_date = config.END_DATE_TEST

sorted_f = sorted(os.listdir(RESULT_DIR))
label_map = {"MlpPolicy": "MLP only", }

# extract date list
data = pd.read_csv(f"./results/start_balance_{config.INITIAL_BALANCE}_nlstm_{nlstm}/balance_a2c_MlpPolicy_1000000_1d.csv")
date_list = data["date"].values

for f in sorted_f:
    # get balance data
    if("balance" in f and INTERVAL in f):
        if("reddit" in f):
            data = pd.read_csv(os.path.join(RESULT_DIR, f), sep=',',header=None).to_numpy()[0][:-1]
            f_split_name = f.split("_")
            label = f"{f_split_name[2].upper()} (with sentiment)"
            plt.plot(date_list, data, label=label)
        else:
            data = pd.read_csv(os.path.join(RESULT_DIR, f))
            f_split_name = f.split("_")
            p_type = label_map[f_split_name[2]]
            label = f"{f_split_name[1].upper()}"
            balance = data["account_value"].to_numpy()
            # plot it
            plt.plot(date_list, balance, label=label)

plt.xlabel("Date & Time", size = 15)
plt.xticks(np.arange(0, len(date_list), len(date_list)/7))
plt.ylabel("Balance", size = 15)
#plt.title(f"Trading performance for interval = {INTERVAL}")
plt.legend(prop={'size': 12})
plt.show()



