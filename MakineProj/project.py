# Import necessary libraries for data analysis and modeling
import pandas as pd
import numpy as np
from arch import arch_model
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import cross_val_score
import torch
import torch.nn as nn
import torch.optim as optim
from statsmodels.tsa.arima.model import ARIMA

# Import visualization libraries
import matplotlib
matplotlib.use('Agg')  # GUI olmayan backend'i kullan
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_data(df):
    """
    Veri setini hazırlar ve teknik göstergeleri hesaplar.
    
    Parametreler:
    df: Tarih, Gold_Close, BTC_Close sütunlarına sahip DataFrame
    
    Döndürür:
    - Teknik göstergeleri içeren işlenmiş veri seti
    """
    # Tarihi indeks olarak ayarla.
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Her iki varlık için getirileri hesapla.
    df['BTC_Returns'] = df['BTC_Close'].pct_change()
    df['Gold_Returns'] = df['Gold_Close'].pct_change()
    
    # Her bir varlık için teknik göstergeleri hesapla.
    for asset in ['BTC', 'Gold']:
        price_col = f'{asset}_Close'
        
        # RSI'yi (5 ve 10 günlük periyotlar için) hesapla.
        for period in [5, 10]:
            df[f'{asset}_RSI_{period}'] = calculate_rsi(df[price_col], period)
        
        # Momentum göstergelerini hesapla.
        for period in [5, 10]:
            # ROC (Değişim Oranı) hesapla.
            df[f'{asset}_ROC_{period}'] = df[price_col].pct_change(period)
            
            # Momentum
            df[f'{asset}_Momentum_{period}'] = df[price_col].diff(period)
        
        # 20 günlük volatiliteyi hesapla.
        df[f'{asset}_Volatility'] = df[f'{asset}_Returns'].rolling(20).std()
    
    # Hedef değişkenleri oluştur (1: yukarı, 0: aşağı).
    df['BTC_Target'] = (df['BTC_Returns'] > 0).astype(int)
    df['Gold_Target'] = (df['Gold_Returns'] > 0).astype(int)
    
    # NaN (boş) değerlerini temizle.
    df.dropna(inplace=True)
    
    return df

def calculate_rsi(prices, period=14):
    """
    Relative Strength Index (RSI) hesaplar.
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    return 100 - (100 / (1 + rs))

class TemporalAttention(nn.Module):
    """
    Zamansal Dikkat Mekanizması
    
    Bu sınıf, zaman serisi verilerindeki önemli noktaları öğrenir.
    """
    def __init__(self, input_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)
        attended = torch.sum(x * attention_weights, dim=1)
        return attended

class DualTCAN(nn.Module):
    def __init__(self, feature_dim, sequence_length, hidden_dim, num_layers=2, dropout_rate=0.2):
        super(DualTCAN, self).__init__()
        # Model parametreleri
        self.feature_dim = feature_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        # Bitcoin için evrişim katmanları
        self.btc_conv = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.feature_dim if i == 0 else self.hidden_dim,
                out_channels=self.hidden_dim,
                kernel_size=3,
                padding=1
            ) for i in range(self.num_layers)
        ])
        
        # Altın için evrişim katmanları
        self.gold_conv = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.feature_dim if i == 0 else self.hidden_dim,
                out_channels=self.hidden_dim,
                kernel_size=3,
                padding=1
            ) for i in range(self.num_layers)
        ])
        
        # Dikkat mekanizmaları
        self.btc_attention = TemporalAttention(self.hidden_dim)
        self.gold_attention = TemporalAttention(self.hidden_dim)
        
        # Tam bağlantılı katmanlar
        self.btc_fc = nn.Linear(self.hidden_dim, 1)
        self.gold_fc = nn.Linear(self.hidden_dim, 1)
        
        # Aktivasyon fonksiyonları
        self.dropout = nn.Dropout(self.dropout_rate)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        İleri beslemeli ağ işlemi
        
        Parametreler:
        x: Tensor, boyut: (batch_size, sequence_length, feature_dim * 2)
           İlk feature_dim özellikler Bitcoin için, sonrakiler Altın için
        
        Dönüş:
        tuple: (btc_predictions, gold_predictions)
        """
        batch_size = x.shape[0]
        
        # Veriyi Bitcoin ve Altın özelliklerine ayır
        btc_features = x[:, :, :self.feature_dim]  # İlk feature_dim özellikler
        gold_features = x[:, :, self.feature_dim:]  # Sonraki feature_dim özellikler
        
        # Evrişim katmanları için boyut düzenlemesi (batch, channels, sequence)
        btc_features = btc_features.transpose(1, 2)
        gold_features = gold_features.transpose(1, 2)
        
        # Bitcoin özellikleri işleme
        btc_x = btc_features
        for conv in self.btc_conv:
            btc_x = self.relu(conv(btc_x))
            btc_x = self.dropout(btc_x)
        
        # Altın özellikleri işleme
        gold_x = gold_features
        for conv in self.gold_conv:
            gold_x = self.relu(conv(gold_x))
            gold_x = self.dropout(gold_x)
        
        # Boyut düzenlemesi (batch, sequence, channels)
        btc_x = btc_x.transpose(1, 2)
        gold_x = gold_x.transpose(1, 2)
        
        # Dikkat mekanizması uygulama
        btc_attended = self.btc_attention(btc_x)
        gold_attended = self.gold_attention(gold_x)
        
        # Son tahminler
        btc_out = self.sigmoid(self.btc_fc(btc_attended))
        gold_out = self.sigmoid(self.gold_fc(gold_attended))
        
        return btc_out, gold_out

class ModelTrainer:
    """
    DualTCAN modeli için hiperparametre optimizasyonu ve eğitim yönetimi
    """
    def __init__(self):
        self.param_grid = {
            'feature_dim': [4, 6, 8, 10, 12],          # Özellik boyutu.
            'sequence_length': [5, 10, 15, 20, 25],    # Girdi zaman serisi uzunluğu.
            'hidden_dim': [32, 64, 128, 256, 512],     # Gizli katman boyutu
            'num_layers': [1, 2, 3, 4, 5],             # Evrişim katmanlarının sayısı(CNN).
            'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5]  # Dropout oranı.
        }
        self.best_params = None
        self.best_model = None

    def _get_param_combinations(self):
        """
        Parameter kombinasyonlarını oluşturur
        """
        from itertools import product
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        return [dict(zip(keys, v)) for v in product(*values)]

    def grid_search(self, train_loader, val_loader, num_epochs=50):
        """
        En iyi hiperparametreleri bulmak için grid search yapar
        """
        best_val_loss = float('inf')
        best_params = None
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for params in self._get_param_combinations():
            model = DualTCAN(params).to(device)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters())
            
            # Her parametre seti için kısa bir eğitim yapılır
            try:
                val_loss = self.train_model(
                    model, train_loader, val_loader,
                    criterion, optimizer, num_epochs=num_epochs,
                    early_stopping=True
                )
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = params
                    
            except Exception as e:
                continue
                
        print(f"Best DualTCAN parameters: {best_params}")
        print(f"Best validation loss: {best_val_loss}")
        return best_params

    def train_model(self, model, train_loader, val_loader, criterion,
                   optimizer, num_epochs, early_stopping=True):
        """
        Modeli eğitir ve validation loss değerini döndürür
        """
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for epoch in range(num_epochs):
            # Eğitim
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                btc_out, gold_out = model(batch['features'].to(device))
                loss = criterion(btc_out, batch['btc_target'].to(device)) + \
                       criterion(gold_out, batch['gold_target'].to(device))
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    btc_out, gold_out = model(batch['features'].to(device))
                    val_loss += criterion(btc_out, batch['btc_target'].to(device)) + \
                               criterion(gold_out, batch['gold_target'].to(device))
            
            val_loss /= len(val_loader)
            
            # Early stopping kontrolü
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
        
        return best_val_loss

class XGBoostModel:
    """
   Teknik göstergeler kullanarak fiyat yönü tahmini için XGBoost model sınıfı ve 
   hiperparametre optimizasyonu için grid search yeteneği
    """
    def __init__(self, params=None):
        # Arama için parametre grid tanımla
        self.param_grid = {
            'max_depth': [3, 4, 5, 6, 7],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'n_estimators': [50, 100, 150, 200, 250],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0, 0.1, 0.2, 0.3, 0.4]
        }
        # Modelleri None olarak başlatın - grid search'den sonra oluşturulacaklar
        self.btc_model = None
        self.gold_model = None
        self.best_params_btc = None
        self.best_params_gold = None

    def _get_param_combinations(self):
        """
        param_grid içinde tanımlanan hiperparametreler için tüm olası kombinasyonları oluşturur. 
        itertools.product fonksiyonu kullanılarak her parametre kombinasyonu üretilir ve bunlar bir liste halinde döndürülür.
        """
        from itertools import product
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        param_combinations = [dict(zip(keys, v)) for v in product(*values)]
        return param_combinations

    def grid_search(self, X_train, y_train, asset_type='btc'):
        """
        Her varlık için en iyi parametreleri bulmak amacıyla grid search yapın.
        XGBoost modelinin hiperparametrelerini optimize etmek için Grid Search işlemi yapar.
        Verilen eğitim verisi (X_train, y_train) ile 5 katlı çapraz doğrulama kullanarak her bir parametre kombinasyonu 
        için doğruluk puanlarını hesaplar.
        
        Parameters:
        X_train: Eğitim özellikleri
        y_train: Eğitim etiketleri
        asset_type: Hangi modeli optimize edeceğinizi belirtmek için 'btc' veya 'gold' kullanın
        """
        best_params = None
        best_score = float('-inf')
        
        # Her parametre kombinasyonunu deneyin.
        for params in self._get_param_combinations():
            model = xgb.XGBClassifier(**params)
            # 5 katlı çapraz doğrulama yapın.
            scores = cross_val_score(model, X_train, y_train, cv=5)
            avg_score = scores.mean()
            
            if avg_score > best_score:
                best_score = avg_score
                best_params = params
        
        # Belirli varlık için en iyi parametreleri saklayın
        if asset_type == 'btc':
            self.best_params_btc = best_params
        else:
            self.best_params_gold = best_params
            
        print(f"Best parameters for {asset_type}: {best_params}")
        print(f"Best cross-validation score: {best_score}")
        return best_params
    
    def fit(self, X_btc, y_btc, X_gold, y_gold):
        """
        Her iki varlık için optimize edilmiş parametreleri kullanarak modelleri eğitin.
        """
        # İlk olarak, her iki varlık için grid search yapın.
        self.best_params_btc = self.grid_search(X_btc, y_btc, 'btc')
        self.best_params_gold = self.grid_search(X_gold, y_gold, 'gold')
        
        # En iyi parametrelerle modelleri başlatın ve eğitin.
        self.btc_model = xgb.XGBClassifier(**self.best_params_btc)
        self.gold_model = xgb.XGBClassifier(**self.best_params_gold)
        
        # Modelleri eğitin(fit).
        self.btc_model.fit(X_btc, y_btc)
        self.gold_model.fit(X_gold, y_gold)
    
    def predict(self, X_btc, X_gold):
        """
        Optimize edilmiş modelleri kullanarak tahminlerde bulunun.
        """
        if self.btc_model is None or self.gold_model is None:
            raise ValueError("Models must be trained before making predictions")
            
        btc_pred = self.btc_model.predict(X_btc)
        gold_pred = self.gold_model.predict(X_gold)
        return btc_pred, gold_pred

class GARCHModel:
    """
    Grid search yeteneğiyle volatilite kümelenmesi analizi için GARCH modeli
    """
    def __init__(self):
        # GARCH modeli için arama yapılacak parametre grid'i
        self.param_grid = {
            'p': [1, 2, 3, 4, 5],  # AR parametreleri
            'q': [1, 2, 3, 4, 5],  # MA parametreleri
            'mean': ['Zero', 'AR', 'Constant'],  # Ortalama modeli
            'vol': ['GARCH', 'EGARCH', 'TARCH'],  # Volatilite modeli tipi
            'dist': ['normal', 'studentst', 't']  # Hata dağılımı
        }
        self.btc_model = None
        self.gold_model = None
        self.best_params_btc = None
        self.best_params_gold = None

    def _get_param_combinations(self):
        """
        Tüm parametre kombinasyonlarını oluşturur
        """
        from itertools import product
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        return [dict(zip(keys, v)) for v in product(*values)]

    def grid_search(self, returns, asset_type='btc'):
        """
        En iyi GARCH parametrelerini bulmak için grid search yapar
        
        Parametre kombinasyonları üzerinde AIC (Akaike Information Criterion) skorlarını hesaplar 
        ve en düşük AIC skoruna sahip parametreyi seçer. AIC değeri, modelin uygunluğunu değerlendirirken kullanılan 
        bir kriterdir ve modelin karmaşıklığını cezalandırır; düşük AIC daha iyi model uyumu anlamına gelir.
        """
        best_params = None
        best_aic = float('inf')  # AIC değeri düşük olmalı
        
        for params in self._get_param_combinations():
            try:
                model = arch_model(
                    returns,
                    p=params['p'],
                    q=params['q'],
                    mean=params['mean'],
                    vol=params['vol'],
                    dist=params['dist']
                )
                result = model.fit(disp='off', show_warning=False)
                
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_params = params
                    
            except Exception as e:
                continue  # Bazı parametre kombinasyonları uyumsuz olabilir
        
        if asset_type == 'btc':
            self.best_params_btc = best_params
        else:
            self.best_params_gold = best_params
            
        print(f"Best GARCH parameters for {asset_type}: {best_params}")
        print(f"Best AIC score: {best_aic}")
        return best_params

    def fit(self, btc_returns, gold_returns):
        """
        Her iki varlık için en iyi parametrelerle modeli eğitir
        """
        self.best_params_btc = self.grid_search(btc_returns, 'btc')
        self.best_params_gold = self.grid_search(gold_returns, 'gold')
        
        self.btc_model = arch_model(btc_returns, **self.best_params_btc)
        self.gold_model = arch_model(gold_returns, **self.best_params_gold)
        
        self.btc_fit = self.btc_model.fit(disp='off')
        self.gold_fit = self.gold_model.fit(disp='off')

def plot_price_analysis(df):
    """
    Fiyat analizi grafiklerini oluşturur ve kaydeder.
    
    Bu fonksiyon dört grafik oluşturur:
    1. Bitcoin ve Altın fiyatlarının zaman serisi
    2. Getirilerin dağılımı
    3. Volatilite karşılaştırması
    4. Korelasyon ısı haritası
    """
    plt.style.use('classic')  # Klasik matplotlib stili kullan
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Fiyat Zaman Serisi
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(df.index, df['BTC_Close'], label='Bitcoin', color='orange')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df.index, df['Gold_Close'], label='Altın', color='gold')
    ax1.set_title('Bitcoin ve Altın Fiyat Hareketleri', fontsize=12)
    ax1.set_xlabel('Tarih', fontsize=10)
    ax1.set_ylabel('Bitcoin Fiyatı ($)', fontsize=10)
    ax1_twin.set_ylabel('Altın Fiyatı ($)', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Çift y-ekseni için lejantları birleştir
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 2. Getirilerin Dağılımı
    ax2 = plt.subplot(2, 2, 2)
    ax2.hist(df['BTC_Returns'].dropna(), bins=50, alpha=0.5, label='Bitcoin', color='orange')
    ax2.hist(df['Gold_Returns'].dropna(), bins=50, alpha=0.5, label='Altın', color='gold')
    ax2.set_title('Günlük Getirilerin Dağılımı', fontsize=12)
    ax2.set_xlabel('Getiri', fontsize=10)
    ax2.set_ylabel('Frekans', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Volatilite Karşılaştırması
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(df.index, df['BTC_Volatility'], label='Bitcoin', color='orange')
    ax3.plot(df.index, df['Gold_Volatility'], label='Altın', color='gold')
    ax3.set_title('20 Günlük Volatilite Karşılaştırması', fontsize=12)
    ax3.set_xlabel('Tarih', fontsize=10)
    ax3.set_ylabel('Volatilite', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Korelasyon Isı Haritası
    ax4 = plt.subplot(2, 2, 4)
    correlation_features = ['BTC_Close', 'Gold_Close', 'BTC_Returns', 'Gold_Returns',
                          'BTC_Volatility', 'Gold_Volatility']
    correlation_matrix = df[correlation_features].corr()
    
    im = ax4.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar(im)
    
    # Korelasyon değerlerini matris üzerine yaz
    for i in range(len(correlation_features)):
        for j in range(len(correlation_features)):
            text = ax4.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black")
    
    ax4.set_xticks(range(len(correlation_features)))
    ax4.set_yticks(range(len(correlation_features)))
    ax4.set_xticklabels(correlation_features, rotation=45)
    ax4.set_yticklabels(correlation_features)
    ax4.set_title('Korelasyon Isı Haritası', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('price_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_technical_indicators(df):
    """
    Teknik gösterge grafiklerini oluşturur ve kaydeder.
    
    Bu fonksiyon üç grafik oluşturur:
    1. RSI göstergeleri
    2. ROC göstergeleri
    3. Momentum göstergeleri
    """
    plt.style.use('classic')
    fig = plt.figure(figsize=(20, 15))
    
    # 1. RSI Göstergeleri
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(df.index, df['BTC_RSI_5'], label='BTC RSI (5)', color='orange')
    ax1.plot(df.index, df['BTC_RSI_10'], label='BTC RSI (10)', color='red')
    ax1.plot(df.index, df['Gold_RSI_5'], label='Gold RSI (5)', color='gold')
    ax1.plot(df.index, df['Gold_RSI_10'], label='Gold RSI (10)', color='brown')
    ax1.axhline(y=70, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(y=30, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('RSI Göstergeleri', fontsize=12)
    ax1.set_ylabel('RSI Değeri', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. ROC Göstergeleri
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(df.index, df['BTC_ROC_5'], label='BTC ROC (5)', color='orange')
    ax2.plot(df.index, df['BTC_ROC_10'], label='BTC ROC (10)', color='red')
    ax2.plot(df.index, df['Gold_ROC_5'], label='Gold ROC (5)', color='gold')
    ax2.plot(df.index, df['Gold_ROC_10'], label='Gold ROC (10)', color='brown')
    ax2.set_title('ROC (Rate of Change) Göstergeleri', fontsize=12)
    ax2.set_ylabel('ROC Değeri', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Momentum Göstergeleri
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(df.index, df['BTC_Momentum_5'], label='BTC Momentum (5)', color='orange')
    ax3.plot(df.index, df['BTC_Momentum_10'], label='BTC Momentum (10)', color='red')
    ax3.plot(df.index, df['Gold_Momentum_5'], label='Gold Momentum (5)', color='gold')
    ax3.plot(df.index, df['Gold_Momentum_10'], label='Gold Momentum (10)', color='brown')
    ax3.set_title('Momentum Göstergeleri', fontsize=12)
    ax3.set_xlabel('Tarih', fontsize=10)
    ax3.set_ylabel('Momentum Değeri', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('technical_indicators.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_performance(portfolio_values, btc_metrics, gold_metrics, df_aligned):
    """
    Model performans grafiklerini oluşturur ve kaydeder.
    
    Bu fonksiyon üç grafik oluşturur:
    1. Portföy değeri değişimi
    2. Model tahmin doğruluğu karşılaştırması
    3. Kümülatif getiri karşılaştırması
    """
    plt.style.use('classic')
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Portföy Değeri Değişimi
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(range(len(portfolio_values)), portfolio_values, 
             label='Portföy Değeri', color='blue', linewidth=2)
    ax1.set_title('Portföy Değeri Değişimi', fontsize=12)
    ax1.set_ylabel('Portföy Değeri ($)', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Model Tahmin Doğruluğu Karşılaştırması
    ax2 = plt.subplot(3, 1, 2)
    x = np.arange(len(btc_metrics))
    width = 0.35
    
    ax2.bar(x - width/2, list(btc_metrics.values()), width, 
            label='Bitcoin', color='orange', alpha=0.7)
    ax2.bar(x + width/2, list(gold_metrics.values()), width, 
            label='Altın', color='gold', alpha=0.7)
    
    ax2.set_ylabel('Skor', fontsize=10)
    ax2.set_title('Model Performans Metrikleri', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(list(btc_metrics.keys()))
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Kümülatif Getiri Karşılaştırması
    ax3 = plt.subplot(3, 1, 3)
    btc_cumret = (1 + df_aligned['BTC_Returns']).cumprod()
    gold_cumret = (1 + df_aligned['Gold_Returns']).cumprod()
    portfolio_cumret = np.array(portfolio_values) / 1000
    
    ax3.plot(range(len(btc_cumret)), btc_cumret, 
             label='Bitcoin B&H', color='orange', linewidth=2)
    ax3.plot(range(len(gold_cumret)), gold_cumret, 
             label='Altın B&H', color='gold', linewidth=2)
    ax3.plot(range(len(portfolio_cumret)), portfolio_cumret, 
             label='Model Portföyü', color='blue', linewidth=2)
    
    ax3.set_title('Kümülatif Getiri Karşılaştırması', fontsize=12)
    ax3.set_xlabel('Gün', fontsize=10)
    ax3.set_ylabel('Kümülatif Getiri', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

class ARIMAModel:
    """
    İlk 4 aylık dönem analizi için ARIMA modeli uygulaması
    """
    def __init__(self):
        self.btc_model = None
        self.gold_model = None
        self.param_grid = {
            'p': [0, 1, 2],  # AR parametreleri
            'd': [0, 1],     # Fark derecesi
            'q': [0, 1, 2]   # MA parametreleri
        }
        
    def _get_param_combinations(self):
        """Tüm parametre kombinasyonlarını oluşturur"""
        from itertools import product
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        return [dict(zip(keys, v)) for v in product(*values)]
    
    def grid_search(self, data, asset_type='btc'):
        """
        En iyi ARIMA parametrelerini bulmak için grid search yapar
        """
        best_params = None
        best_aic = float('inf')
        
        for params in self._get_param_combinations():
            try:
                model = ARIMA(
                    data,
                    order=(params['p'], params['d'], params['q'])
                )
                result = model.fit()
                
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_params = params
                    
            except Exception as e:
                continue
                
        print(f"Best ARIMA parameters for {asset_type}: {best_params}")
        print(f"Best AIC score: {best_aic}")
        return best_params
    
    def fit(self, btc_data, gold_data):
        """Her iki varlık için en iyi parametrelerle modeli eğitir"""
        # İlk 4 aylık veriyi seç (yaklaşık 120 gün)
        btc_initial = btc_data[:120]
        gold_initial = gold_data[:120]
        
        # En iyi parametreleri bul
        btc_params = self.grid_search(btc_initial, 'btc')
        gold_params = self.grid_search(gold_initial, 'gold')
        
        # Modelleri eğit
        self.btc_model = ARIMA(
            btc_initial,
            order=(btc_params['p'], btc_params['d'], btc_params['q'])
        ).fit()
        
        self.gold_model = ARIMA(
            gold_initial,
            order=(gold_params['p'], gold_params['d'], gold_params['q'])
        ).fit()
    
    def predict(self, steps=1):
        """Her iki varlık için tahmin yapar"""
        if self.btc_model is None or self.gold_model is None:
            raise ValueError("Models must be trained before making predictions")
            
        btc_forecast = self.btc_model.forecast(steps=steps)
        gold_forecast = self.gold_model.forecast(steps=steps)
        
        return btc_forecast, gold_forecast

def prepare_train_test_split(df, test_size=0.2):
    """
    Veri setini eğitim ve test olarak ayırır.
    
    Parameters:
    df: DataFrame - İşlenmiş veri seti
    test_size: float - Test veri seti oranı (varsayılan: 0.2)
    
    Returns:
    tuple: (train_df, test_df, split_index)
    """
    # Zaman serisi olduğu için karıştırmadan(shuffle) kronolojik olarak böl
    split_index = int(len(df) * (1 - test_size))
    
    # Veriyi böl
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    
    print(f"Eğitim seti boyutu: {len(train_df)} gün")
    print(f"Test seti boyutu: {len(test_df)} gün")
    
    return train_df, test_df, split_index

class CombinedModel:
    """
    GARCH, XGBoost, ARIMA ve TCAN modellerini kapsamlı analiz için birleştiren sınıf
    """
    def __init__(self):
        self.garch_model = GARCHModel()
        self.xgb_model = XGBoostModel()
        self.arima_model = ARIMAModel()  
        self.tcan_model = DualTCAN(
            feature_dim=6,
            sequence_length=10,
            hidden_dim=64
        )
        self.scaler = StandardScaler() # StandardScaler verilerin ölçeklenmesi için kullanılır. 
                                       # Bu, özellikle derin öğrenme ve makine öğrenme modellerinde verilerin uygun 
                                       # şekilde normalize edilmesi amacıyla gereklidir.
        self.transition_period = 120  # 4 aylık geçiş süresi
        self.transition_window = 30   # 1 aylık yumuşak geçiş penceresi

    def calculate_transition_weights(self, day):
        """Yumuşak geçiş için ağırlıkları hesaplar"""
        """Bu fonksiyon, geçiş dönemindeki ağırlıkları hesaplar. day parametresi, 
        günün hangi aşamada olduğunu belirler ve buna göre ARIMA modelinin etkisini azaltarak diğer modellerin etkisini artırır."""
        
        if day <= self.transition_period - self.transition_window:
            return 1.0  # Tamamen ARIMA
        elif day >= self.transition_period:
            return 0.0  # Tamamen diğer modeller
        else:
            progress = (day - (self.transition_period - self.transition_window)) / self.transition_window
            return 1.0 - progress

    def prepare_features(self, df):
        """
        Tüm modeller için özellikleri hazırlar

        Bu fonksiyon:

        Özellik sütunlarını seçer (RSI, ROC, Momentum)
        Bitcoin ve Altın özelliklerini ayrı ayrı gruplar
        Veriyi ölçeklendirir
        Zamansal diziler oluşturur
        
        Returns:
        tuple: (X_tensor, padded_length)
        """
        # Bitcoin ve Altın özellik sütunlarını ayırın.
        btc_features = [col for col in df.columns if 'BTC' in col and 
                       ('RSI' in col or 'ROC' in col or 'Momentum' in col)]
        gold_features = [col for col in df.columns if 'Gold' in col and 
                        ('RSI' in col or 'ROC' in col or 'Momentum' in col)]
        
        X_btc = df[btc_features].values
        X_gold = df[gold_features].values
        
        # Veriyi ölçeklendirin.
        X_btc_scaled = self.scaler.fit_transform(X_btc)
        X_gold_scaled = self.scaler.fit_transform(X_gold)
        
        # Zamansal diziler oluşturun.
        sequence_length = 10
        sequences = []
        padded_length = len(X_btc_scaled) - sequence_length + 1
        
        for i in range(padded_length):
            btc_seq = X_btc_scaled[i:i+sequence_length]
            gold_seq = X_gold_scaled[i:i+sequence_length]
            combined_seq = np.concatenate([btc_seq, gold_seq], axis=1)
            sequences.append(combined_seq)
        
        sequences_array = np.array(sequences)
        X_tensor = torch.FloatTensor(sequences_array)
        return X_tensor, padded_length
    
    def train(self, df):
        """
        Tüm modelleri eğitir
        Bu fonksiyon, tüm modelleri eğitir. İlk olarak, ARIMA modelini eğitir. 
        Ardından, TCAN modelini kullanarak Bitcoin ve Altın için hedef değerler tahmin edilir.
        """
        # Önce özellikleri hazırlıyoruz
        X_tensor, padded_length = self.prepare_features(df)
        
        # ARIMA modeli için ilk 4 aylık veriyi kullan
        self.arima_model.fit(
            df['BTC_Close'].values,
            df['Gold_Close'].values
        )
        
        # Toplam veri uzunluğunun %80'ini eğitim verisi olarak ayırıyoruz
        train_size = int(padded_length * 0.8)
        sequence_length = 10
        
        # Eğitim ve test verilerini ayırıyoruz
        train_data = df[sequence_length-1:padded_length+sequence_length-1]
        
        # Test verisindeki etiketler otomatik olarak test_data içinde geliyor
        test_data = df[train_size+sequence_length-1:padded_length+sequence_length-1]
        
        # Eğitim verilerini hazırlıyoruz
        X_train_tensor = X_tensor[:train_size]
        
        # Eğitim verisindeki etiketleri al
        y_train_btc = torch.FloatTensor(train_data['BTC_Target'].values[:train_size])
        y_train_gold = torch.FloatTensor(train_data['Gold_Target'].values[:train_size])
        
        # Set up training components
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.tcan_model.parameters())
        
        # Modeli eğitin
        for epoch in range(100):
            optimizer.zero_grad()
            btc_out, gold_out = self.tcan_model(X_train_tensor)
            
            loss_btc = criterion(btc_out.squeeze(), y_train_btc)
            loss_gold = criterion(gold_out.squeeze(), y_train_gold)
            total_loss = loss_btc + loss_gold
            
            total_loss.backward()
            optimizer.step()
        
        return test_data
    
    def predict(self, df):
        """Yumuşak geçişli tahmin yapar"""
        """Bu fonksiyon, yumuşak geçiş ağırlıkları kullanarak Bitcoin ve Altın için nihai tahminleri yapar. 
           ARIMA, TCAN ve diğer modellerin tahminlerini birleştirir."""
        X_tensor, padded_length = self.prepare_features(df)
        sequence_length = 10
        current_day = len(df)
        
        # ARIMA tahminleri
        btc_forecast, gold_forecast = self.arima_model.predict()
        arima_btc_pred = (btc_forecast > df['BTC_Close'].iloc[-1]).astype(int)
        arima_gold_pred = (gold_forecast > df['Gold_Close'].iloc[-1]).astype(int)
        
        # Diğer modellerin tahminleri
        with torch.no_grad():
            tcan_btc_prob, tcan_gold_prob = self.tcan_model(X_tensor)
            tcan_btc_prob = tcan_btc_prob.squeeze().numpy()
            tcan_gold_prob = tcan_gold_prob.squeeze().numpy()
            combined_btc = (tcan_btc_prob > 0.5).astype(int)
            combined_gold = (tcan_gold_prob > 0.5).astype(int)
        
        # Geçiş ağırlığını hesapla
        arima_weight = self.calculate_transition_weights(current_day)
        
        # Tahminleri birleştir
        final_btc_pred = []
        final_gold_pred = []
        final_btc_prob = []
        final_gold_prob = []
        
        for i in range(len(combined_btc)):
            btc_prob = arima_weight + (1 - arima_weight) * tcan_btc_prob[i]
            gold_prob = arima_weight + (1 - arima_weight) * tcan_gold_prob[i]
            
            final_btc_pred.append(1 if btc_prob > 0.5 else 0)
            final_gold_pred.append(1 if gold_prob > 0.5 else 0)
            final_btc_prob.append(btc_prob)
            final_gold_prob.append(gold_prob)
        
        df_aligned = df[sequence_length-1:padded_length+sequence_length-1]
        
        # Geçiş metriklerini logla
        if current_day <= self.transition_period:
            print(f"\nTransition Metrics (Day {current_day}):")
            print(f"ARIMA Weight: {arima_weight:.2f}")
            print(f"Other Models Weight: {1-arima_weight:.2f}")
        
        return (
            np.array(final_btc_pred),
            np.array(final_gold_pred),
            np.array(final_btc_prob),
            np.array(final_gold_prob),
            df_aligned
        )

def simulate_trading(data, predictions, initial_balance=1000): # initial_balance: Kullanıcı tarafından sağlanan başlangıç bakiyesi (varsayılan değer 1000 $).
    """ARIMA geçiş dönemini dikkate alan trading simülasyonu"""
    """ """
    btc_balance = initial_balance / 2  # Başlangıçta, 1000 $'lık toplam bakiyenin yarısı Bitcoin'e (500 $) ve     
    gold_balance = initial_balance / 2 # diğer yarısı Altın'a (500 $) yatırılır.
    transition_period = 120  # 4 ay olarak belirlendi ve bu dönemde daha temkinli bir strateji uygulanacaktır.
    
    btc_holdings = 0
    gold_holdings = 0
    
    btc_prices = data['BTC_Close'].values
    gold_prices = data['Gold_Close'].values
    
    btc_pred, gold_pred = predictions
    
    portfolio_values = []
    daily_returns = []
    
    for i in range(len(btc_pred)):
        current_day = i + 1
        
        # ARIMA dönemi için daha konservatif strateji
        if current_day <= transition_period:
            confidence_threshold = 0.65
            position_size = 0.3 #  her işlemde yalnızca portföyün %30'u kadar alım-satım yapılır. Bu strateji daha konservatif bir yaklaşım sergiler.
        else: #Geçiş dönemi bittikten sonra (120 gün sonrasında) daha agresif bir strateji izlenir:
            confidence_threshold = 0.5
            position_size = 0.5 # 0.5, her işlemde portföyün %50'si kadar alım-satım yapılır.
        
        # Bitcoin trading
        if btc_pred[i] == 1 and btc_balance > 0: #Eğer tahmin (btc_pred[i] == 1 veya gold_pred[i] == 1) alım sinyali veriyorsa ve portföyde yeterli bakiye varsa, alım yapılır.
            trade_amount = btc_balance * position_size
            btc_holdings += (trade_amount * 0.98) / btc_prices[i]
            btc_balance -= trade_amount
        elif btc_pred[i] == 0 and btc_holdings > 0:
            sold_amount = btc_holdings * position_size
            btc_balance += sold_amount * btc_prices[i] * 0.98
            btc_holdings -= sold_amount
        
        # Gold trading
        if gold_pred[i] == 1 and gold_balance > 0:
            trade_amount = gold_balance * position_size
            gold_holdings += (trade_amount * 0.99) / gold_prices[i]
            gold_balance -= trade_amount
        elif gold_pred[i] == 0 and gold_holdings > 0:
            sold_amount = gold_holdings * position_size
            gold_balance += sold_amount * gold_prices[i] * 0.99
            gold_holdings -= sold_amount
        
        # Portfolio değeri hesapla
        portfolio_value = (
            btc_balance +
            (btc_holdings * btc_prices[i]) +
            gold_balance +
            (gold_holdings * gold_prices[i])
        )
        portfolio_values.append(portfolio_value)
        
        # Günlük getiri hesapla
        if i > 0:
            daily_return = (portfolio_value - portfolio_values[i-1]) / portfolio_values[i-1]
            daily_returns.append(daily_return)
        
        # Geçiş dönemi metriklerini logla
        if current_day == transition_period:
            print("\nTransition Period Performance:")
            print(f"Initial Balance: ${initial_balance:,.2f}")
            print(f"Balance after 4 months: ${portfolio_value:,.2f}")
            print(f"Return: {((portfolio_value-initial_balance)/initial_balance*100):,.2f}%")
            if daily_returns:
                print(f"Volatility: {np.std(daily_returns)*100:,.2f}%")
    
    return portfolio_values

def calculate_metrics(y_true, y_pred):
    """
    Model performans metriklerini hesaplar.
    
    Returns dictionary of:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
def plot_roc_curves(y_true_btc, y_pred_proba_btc, y_true_gold, y_pred_proba_gold):
    """
    Bitcoin ve Altın tahminleri için ROC eğrilerini çizer.
    
    Parametreler:
    y_true_btc: Gerçek Bitcoin etiketleri (0 veya 1)
    y_pred_proba_btc: Bitcoin için tahmin olasılıkları (0-1 arası)
    y_true_gold: Gerçek Altın etiketleri (0 veya 1)
    y_pred_proba_gold: Altın için tahmin olasılıkları (0-1 arası)
    """
    
    # Bitcoin için ROC eğrisi hesaplama
    fpr_btc, tpr_btc, _ = roc_curve(y_true_btc, y_pred_proba_btc)
    roc_auc_btc = auc(fpr_btc, tpr_btc)
    
    # Altın için ROC eğrisi hesaplama
    fpr_gold, tpr_gold, _ = roc_curve(y_true_gold, y_pred_proba_gold)
    roc_auc_gold = auc(fpr_gold, tpr_gold)
    
    # Görselleştirme
    plt.figure(figsize=(10, 8))
    
    # Bitcoin ROC eğrisi
    plt.plot(fpr_btc, tpr_btc, color='orange', lw=2,
             label=f'Bitcoin ROC Eğrisi (AUC = {roc_auc_btc:.2f})')
    
    # Altın ROC eğrisi
    plt.plot(fpr_gold, tpr_gold, color='gold', lw=2,
             label=f'Altın ROC Eğrisi (AUC = {roc_auc_gold:.2f})')
    
    # Referans çizgisi (rastgele tahmin)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Rastgele Tahmin')
    
    # Grafik düzenlemeleri
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Yanlış Pozitif Oranı (False Positive Rate)', fontsize=12)
    plt.ylabel('Doğru Pozitif Oranı (True Positive Rate)', fontsize=12)
    plt.title('Bitcoin ve Altın İçin ROC Eğrileri', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Grafiği kaydet
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'Bitcoin AUC': roc_auc_btc,
        'Gold AUC': roc_auc_gold
    }

if __name__ == "__main__":
    try:
        # Data loading and preparation phase
        print("1. Step: Loading data...")
        df = pd.read_csv('merged_prices.csv')
        
        print("2. Step: Preparing data...")
        df = prepare_data(df)
        
        print("2.1 Step: Creating price analysis plots...")
        plot_price_analysis(df)
        
        print("2.2 Step: Creating technical indicator plots...")
        plot_technical_indicators(df)
        
        # Model training phase
        print("3. Step: Creating model...")
        model = CombinedModel()
        
        print("4. Step: Training model...")
        test_data = model.train(df)
        
        # Prediction phase
        print("5. Step: Making predictions...")
        predictions = model.predict(test_data)
        btc_pred = predictions[0]
        gold_pred = predictions[1]
        btc_prob = predictions[2]
        gold_prob = predictions[3]
        aligned_data = predictions[4]
        
        # ROC eğrilerini çiz ve AUC değerlerini al
        roc_metrics = plot_roc_curves(
            aligned_data['BTC_Target'], btc_prob,
            aligned_data['Gold_Target'], gold_prob
        )
        
        # ROC metriklerini yazdır
        print("\nROC Analizi Sonuçları:")
        for metric, value in roc_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Simulation phase
        print("6. Step: Starting trading simulation...")
        portfolio_values = simulate_trading(
            aligned_data,
            (btc_pred, gold_pred),
            initial_balance=1000
        )
        
        # Performance evaluation phase
        print("7. Step: Calculating performance metrics...")
        btc_metrics = calculate_metrics(aligned_data['BTC_Target'], btc_pred)
        gold_metrics = calculate_metrics(aligned_data['Gold_Target'], gold_pred)
        
        # Display results
        print("\n=== RESULTS ===")
        print("\nBitcoin Prediction Metrics:")
        for metric, value in btc_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        print("\nGold Prediction Metrics:")
        for metric, value in gold_metrics.items():
            print(f"{metric}: {value:.4f}")
            
        print("\nROC Curve Analysis:")
        print("------------------")
        print(f"Bitcoin AUC Score: {roc_metrics['Bitcoin AUC']:.4f}")
        print(f"Gold AUC Score: {roc_metrics['Gold AUC']:.4f}")
        
        final_return = (portfolio_values[-1] - 1000) / 1000 * 100
        print(f"\nTotal Return: %{final_return:.2f}")
        
        print("\n8. Step: Creating performance plots...")
        plot_model_performance(portfolio_values, btc_metrics, gold_metrics, aligned_data)
        
    except Exception as e:
        print(f"\nERROR OCCURRED: {str(e)}")
        print("Error details:")
        import traceback
        traceback.print_exc()