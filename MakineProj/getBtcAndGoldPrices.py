import pandas as pd

# CSV'leri oku
gold_df = pd.read_csv('gold_prices.csv')
btc_df = pd.read_csv('bitcoin_prices.csv')

# Tarihleri önce datetime'a çevir, sonra string'e formatla
gold_df['Date'] = gold_df['Date'].str.split(' ').str[0]
btc_df['Date'] = btc_df['Date'].str.split(' ').str[0]

# Sütunları seç ve yeniden adlandır
gold_df = gold_df[['Date', 'Close']].rename(columns={'Close': 'Gold_Close'})
btc_df = btc_df[['Date', 'Close']].rename(columns={'Close': 'BTC_Close'})

# Birleştir
merged_df = pd.merge(gold_df, btc_df, on='Date', how='inner')

# Kaydet
merged_df.to_csv('merged_prices.csv', index=False)

print("Birleştirme tamamlandı")
print(f"Toplam satır sayısı: {len(merged_df)}")
print("\nİlk 5 satır:")
print(merged_df.head())