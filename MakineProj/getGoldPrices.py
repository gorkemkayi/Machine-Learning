import yfinance as yf
import csv

# Altın sembolü (GC=F)
gold = yf.Ticker("GC=F")

# Veriyi çekme
data = gold.history(start="2016-09-11", end="2021-09-10")

# CSV dosyasına kaydetme
data.to_csv("gold_prices.csv")
print("Altın fiyatları 'gold_prices.csv' dosyasına kaydedildi.")