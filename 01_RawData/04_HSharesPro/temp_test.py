import akshare as ak

stock_hk_hist_df = ak.stock_hk_hist(symbol="01810", period="daily", start_date="20250101", end_date="20250516", adjust="qfq")
print(stock_hk_hist_df)