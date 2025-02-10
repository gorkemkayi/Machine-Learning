import yfinance as yf
import csv

# Bitcoin sembolü (BTC-USD)
bitcoin = yf.Ticker("BTC-USD")

# Veriyi çekme
data = bitcoin.history(start="2016-09-11", end="2021-09-10")

# CSV dosyasına kaydetme
data.to_csv("bitcoin_prices.csv")
print("Bitcoin fiyatları 'bitcoin_prices.csv' dosyasına kaydedildi.")
