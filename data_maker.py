import pandas as pd
import numpy as np
import yfinance as yf
import requests
import json
from io import StringIO

# get labels from wikipedia
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
headers = {"User-Agent": "Mozilla/5.0"}
html = requests.get(url, headers=headers, timeout=20)
html.raise_for_status()
sp500_df = pd.read_html(StringIO(html.text))[0]
tickers = sp500_df["Symbol"].str.replace(".", "-", regex=False).tolist() # for some reason there are 503 labels, figure out why!!!

# I think it is better to ignore the index
# tickers.insert(0,'^GSPC') # add S&P500 index, which we try to predict

# Download all S&P 500 data given the previous labels
data = yf.download(tickers=tickers,start="2015-01-01",end="2023-01-01",group_by='ticker',threads=True)

# just keep opening prices
open_prices = pd.concat({ticker: data[ticker]["Close"] for ticker in data.columns.levels[0]},axis=1)

# remove firms with nan-values, remove around 50 firms
open_prices_interp = open_prices.interpolate(method='linear', axis=0, limit=1) # if nan-values at one point then average between before and after, for instance, if uneven opening days for some reason
open_prices_interp = open_prices_interp.dropna(axis=1) # this should drop columns containing nan-values

# remove firms without sector or industry information for graphs, only removes 1 firm
tickers_with_data = list(open_prices_interp.columns)
firm_industry_dict = {}
removed_labels = []

sector_map   = dict(zip(sp500_df["Symbol"].str.replace(".", "-", regex=False), sp500_df["GICS Sector"]))
industry_map = dict(zip(sp500_df["Symbol"].str.replace(".", "-", regex=False), sp500_df["GICS Sub-Industry"]))

firm_industry_dict = {}
removed_labels = []

for firm in tickers_with_data:
    if firm in sector_map and firm in industry_map:
        firm_industry_dict[firm] = (sector_map[firm], industry_map[firm])
    else:
        removed_labels.append(firm)

tickers_with_data = [f for f in tickers_with_data if f not in removed_labels]

# remove firms without graph data
open_prices_interp = open_prices_interp.drop(columns=removed_labels)

### Save the data
with open("data_folder/firm_industry.json", "w") as f:
    json.dump(firm_industry_dict, f, indent=4)
open_prices_interp.to_csv('data_folder/open_prices_interp.csv')

set_of_sector = set()
set_of_industry = set()
for val in firm_industry_dict.values():
  sector, industry = val
  set_of_sector.add(sector)
  set_of_industry.add(industry)
# Make data more stationary by using log and diff, so we get the relative changes  (I thought that an LSTM model could predict non-stationary time-series, but it was a bit tough for it)
x = open_prices_interp.to_numpy()
x_returns = np.diff(np.log(x),axis=0)
# calculate correlation
corr_ = np.corrcoef(x_returns.T)

# make adjacency matrix
firm_tickers = tickers_with_data # I have removed  '^GSPC', so no difference now
N = len(firm_tickers)
A = np.zeros((N, N, 3)) # 2 types of relation, industry and sector

for i, ti in enumerate(firm_tickers):
    for j, tj in enumerate(firm_tickers):
        if ti == tj:
            continue
        if firm_industry_dict[ti][1] == firm_industry_dict[tj][1]:
            A[i, j, 0] = 1   # connection type 1
        elif firm_industry_dict[ti][0] == firm_industry_dict[tj][0]:
            A[i, j, 1] = 1   # connection type 2
# add the correlation matrix to adjacency matrix
A[:,:,2] = corr_  # this correlation is on the entire period, perhaps just do it for training period

np.save("data_folder/adjacency_matrix.npy", A)