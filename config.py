# Note: Similar to config.py for 
# https://github.com/AI4Finance-LLC/Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020
# and 
# https://github.com/AI4Finance-LLC/FinRL
# though it has been modified to suit our needs (we mostly just needed the tickers)

# For model
N_FEATURES = 2
N_STOCKS = 30
INITIAL_BALANCE = 10000.00 # Start out with an initial balance of INITIAL_BALANCE USD
MAX_TRANSACTION = 500 # Can trade at most 500 shares (both buy and sell)
MAX_BALANCE = 100000000.00 # 100 million (unlikely to reach this)

# 10% short-term capital gains according to https://www.investopedia.com/articles/personal-finance/101515/comparing-longterm-vs-shortterm-capital-gain-tax-rates.asp for single up to $ 9,875
SELL_PENALTY = 0.05 # Chose 0.05 since we don't actually track the gains and losses 

START_DATE = "2009-01-01"
END_DATE = "2020-12-31"
START_DATE_TEST = "2021-01-01"
END_DATE_TEST = "2021-04-30"
INTERVAL = "1d"

REDDIT_START_DATE = "2016-01-01"
REDDIT_END_DATE = "2020-12-31"
REDDIT_START_DATE_TEST = "2021-01-01"
REDDIT_END_DATE_TEST = "2021-04-30"

# For Reddit
CLIENT_ID="hV8SIg2IoW5WNw"
CLIENT_SECRET="mpo4k-ZWCOauRw8rlkGcAICNaVWGRQ"
USER_AGENT="class project script"
SUBREDDIT="wallstreetbets"
N_POSTS=10
TITLE_POINTS=10
BODY_POINTS=5
COMMENT_POINTS=1

# For data preprocesseing
DEFAULT_COLUMNS = ['date', 'tic', 'open']
TECHNICAL_INDICATORS = ['macd']

# Model parameters
# Our own parameters
MlpPolicy_PARAMS = None
MlpLstmPolicy_PARAMS = {"n_lstm": 16}

# Parameters from https://github.com/AI4Finance-LLC/FinRL (FinRL)
A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007}
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "nminibatches": 1,
}
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "actor_lr": 0.001, "critic_lr": 0.01}
TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
SAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "batch_size": 64,
    "ent_coef": "auto_0.1",
}

# Custom ticker
FAANG = [
    "FB",
    "AMZN",
    "AAPL",
    "NFLX",
    "GOOGL"
]

################
# From https://github.com/AI4Finance-LLC/FinRL (FinRL)
################
# DOW 30 components as of 2019/1
DOW_30_TICKER = [
    "AAPL",
    "MSFT",
    "JPM",
    "V",
    "RTX",
    "PG",
    "GS",
    "NKE",
    "DIS",
    "AXP",
    "HD",
    "INTC",
    "WMT",
    "IBM",
    "MRK",
    "UNH",
    "KO",
    "CAT",
    "TRV",
    "JNJ",
    "CVX",
    "MCD",
    "VZ",
    "CSCO",
    "XOM",
    "BA",
    "MMM",
    "PFE",
    "WBA",
    "DD",
]

################
# From https://github.com/AI4Finance-LLC/FinRL (FinRL)
################
# NASDAQ 100 components as of 2019/1
NAS_100_TICKER = [
    "AMGN",
    "AAPL",
    "AMAT",
    "INTC",
    "PCAR",
    "PAYX",
    "MSFT",
    "ADBE",
    "CSCO",
    "XLNX",
    "QCOM",
    "COST",
    "SBUX",
    "FISV",
    "CTXS",
    "INTU",
    "AMZN",
    "EBAY",
    "BIIB",
    "CHKP",
    "GILD",
    "NLOK",
    "CMCSA",
    "FAST",
    "ADSK",
    "CTSH",
    "NVDA",
    "GOOGL",
    "ISRG",
    "VRTX",
    "HSIC",
    "BIDU",
    "ATVI",
    "ADP",
    "ROST",
    "ORLY",
    "CERN",
    "BKNG",
    "MYL",
    "MU",
    "DLTR",
    "ALXN",
    "SIRI",
    "MNST",
    "AVGO",
    "TXN",
    "MDLZ",
    "FB",
    "ADI",
    "WDC",
    "REGN",
    "LBTYK",
    "VRSK",
    "NFLX",
    "TSLA",
    "CHTR",
    "MAR",
    "ILMN",
    "LRCX",
    "EA",
    "AAL",
    "WBA",
    "KHC",
    "BMRN",
    "JD",
    "SWKS",
    "INCY",
    "PYPL",
    "CDW",
    "FOXA",
    "MXIM",
    "TMUS",
    "EXPE",
    "TCOM",
    "ULTA",
    "CSX",
    "NTES",
    "MCHP",
    "CTAS",
    "KLAC",
    "HAS",
    "JBHT",
    "IDXX",
    "WYNN",
    "MELI",
    "ALGN",
    "CDNS",
    "WDAY",
    "SNPS",
    "ASML",
    "TTWO",
    "PEP",
    "NXPI",
    "XEL",
    "AMD",
    "NTAP",
    "VRSN",
    "LULU",
    "WLTW",
    "UAL",
]
