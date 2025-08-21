import asyncio
import aiohttp
import pandas as pd
import numpy as np
import sqlite3
import telegram
import logging
import yfinance as yf
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from pytrends.request import TrendReq
import praw
import xgboost as xgb
from transformers import pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
from io import StringIO
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================== #
#         הגדרות כלליות          #
# ============================== #

DB_FILE = "penny_stocks.db"
LOG_FILE = "bot.log"
OUTPUT_FILE = "top_stocks.csv"
TELEGRAM_TOKEN = "8453354058:AAGG0v0zLWTe1NJE7ttfaUZvoutf5XNGU7s"
CHAT_ID = "6387878532"
ALPHA_VANTAGE_API_KEY = "PQT4IGSHW87JP58H"
REDDIT_CLIENT_ID = "ZOa0YjqoW-H_-aFXhIXrLw"
REDDIT_CLIENT_SECRET = "7v6s4PJr2kdbvtfNDq7khltKXVkCrw"
REDDIT_USER_AGENT = "_bot_v1"
GAIN_THRESHOLD = 0.05
RSI_COLD_THRESHOLD = 40
VOLUME_THRESHOLD = 500_000
MARKET_CAP_THRESHOLD = 50_000_000
FLOAT_THRESHOLD = 50_000_000
MAX_API_RETRIES = 3
RETRY_DELAY = 15
MAX_TICKERS = 50
RATE_LIMIT_PER_MINUTE = 60

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

bot = telegram.Bot(token=TELEGRAM_TOKEN)

# ============================== #
#          DB + Logging           #
# ============================== #

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                entry_price REAL,
                target_price REAL,
                stop_loss REAL,
                score REAL,
                predicted_gain REAL,
                days_in_trade INTEGER,
                position_size REAL,
                timestamp TEXT,
                google_trend REAL,
                reddit_sentiment REAL,
                short_interest REAL,
                feature_importance TEXT,
                UNIQUE(ticker, timestamp)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                error_message TEXT,
                timestamp TEXT
            )
        ''')
        conn.commit()

def log_error(msg):
    logging.error(msg)
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO errors (error_message, timestamp) VALUES (?, ?)", 
                      (msg, datetime.utcnow().isoformat()))
        conn.commit()

async def send_telegram(msg):
    for attempt in range(MAX_API_RETRIES):
        try:
            await bot.send_message(chat_id=CHAT_ID, text=msg)
            return
        except Exception as e:
            log_error(f"Telegram error (attempt {attempt + 1}): {e}")
            if attempt < MAX_API_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)
    log_error("Failed to send Telegram message after retries")
    await send_telegram(f"⚠️ שגיאה: לא הצלחתי לשלוח התראה, בדוק את {LOG_FILE}")

# ============================== #
#         משיכת טיקרים           #
# ============================== #

async def fetch_tickers():
    tickers = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    async with aiohttp.ClientSession(headers=headers) as session:
        try:
            exchanges = ['^IXIC', '^NYA']
            for exchange in exchanges:
                tickers_data = yf.Tickers(exchange)
                for ticker_symbol in tickers_data.tickers:
                    try:
                        ticker = tickers_data.tickers[ticker_symbol]
                        info = ticker.info
                        if (info.get('regularMarketPrice', 0) <= 5.0 and
                            info.get('regularMarketVolume', 0) >= VOLUME_THRESHOLD and
                            info.get('exchange') in ['NAS', 'NYQ']):
                            tickers.append(ticker_symbol)
                    except Exception as e:
                        log_error(f"Yahoo Finance ticker {ticker_symbol} error: {e}")
            tickers = list(set(tickers))[:MAX_TICKERS]
        except Exception as e:
            log_error(f"Yahoo Finance fetch error: {e}")

        if len(tickers) < MAX_TICKERS:
            try:
                url = f"https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={ALPHA_VANTAGE_API_KEY}"
                async with session.get(url, timeout=10) as response:
                    response.raise_for_status()
                    text = await response.text()
                    csv_data = StringIO(text)
                    data = pd.read_csv(csv_data)
                    data = data[(data['status'] == 'Active') & 
                               (data['exchange'].isin(['NASDAQ', 'NYSE'])) & 
                               (data['assetType'] == 'Stock')]
                    tickers.extend(data['symbol'].tolist())
                    tickers = list(set(tickers))[:MAX_TICKERS]
            except Exception as e:
                log_error(f"Alpha Vantage fetch error: {e}")

        if len(tickers) < 10:
            try:
                fallback_tickers = ['AACB','AACG','AACI','AACT','AAM','AAME','AAMI','AAOI','AARD','AAT',
                                    'AAUC','ABAT','ABCL','ABEO','ABL','ABLV','ABOS','ABP','ABSI','ABTS']
                tickers.extend([t for t in fallback_tickers if t not in tickers])
                tickers = list(set(tickers))[:MAX_TICKERS]
            except Exception as e:
                log_error(f"Fallback Finviz document error: {e}")

    return tickers

# ============================== #
#         פונקציות עזר           #
# ============================== #

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs)).iloc[-1]

def calculate_atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period, min_periods=1).mean().iloc[-1]

def kelly_criterion(win_prob=0.6, win_loss_ratio=2.0, max_fraction=0.15):
    f_star = win_prob - (1 - win_prob) / win_loss_ratio
    return max(0, min(f_star, max_fraction))

# ============================== #
#      Google Trends + Reddit     #
# ============================== #

async def get_google_trends(ticker):
    for attempt in range(MAX_API_RETRIES):
        try:
            pytrens = TrendReq(hl='en-US', tz=360)
            pytrens.build_payload([ticker], timeframe='now 7-d')
            data = pytrens.interest_over_time()
            if not data.empty:
                return data[ticker].mean() / 100
        except Exception as e:
            log_error(f"Google Trends error for {ticker} (attempt {attempt + 1}): {e}")
            if attempt < MAX_API_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)
    return 0

async def analyze_reddit_sentiment(ticker):
    try:
        reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET, 
                             user_agent=REDDIT_USER_AGENT)
        sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
        sentiment = 0
        count = 0
        for post in reddit.subreddit("wallstreetbets+pennystocks").search(ticker, limit=5):
            text = post.title + ' ' + (post.selftext or '')
            result = sentiment_analyzer(text[:512])[0]
            sentiment += result['score'] if result['label'] == 'POSITIVE' else -result['score']
            count += 1
        return sentiment / count if count > 0 else 0
    except Exception as e:
        log_error(f"Reddit sentiment error for {ticker}: {e}")
        return 0

# ============================== #
#         ניתוח ML מתקדם         #
# ============================== #

def prepare_features(df, info, vix_data=None):
    df = df.copy()
    df['returns'] = df['Close'].pct_change()
    df['ma_10'] = df['Close'].rolling(10).mean()
    df['ma_50'] = df['Close'].rolling(50).mean()
    df['rsi'] = calculate_rsi(df['Close'])
    df['atr'] = calculate_atr(df)
    df['short_interest'] = info.get('shortPercentOfFloat', 0)
    df['float'] = info.get('floatShares', 0)
    df['vix'] = vix_data['Close'].iloc[-1] if vix_data is not None else 0
    df = df.dropna()
    return df[['Close', 'ma_10', 'ma_50', 'rsi', 'atr', 'returns', 'short_interest', 'float', 'vix']]

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(30, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

async def analyze_ticker(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        for attempt in range(MAX_API_RETRIES):
            try:
                info = ticker_obj.info
                df = yf.download(ticker, period="6mo", interval="1d", progress=False)
                vix_data = yf.download("^VIX", period="1d", interval="1d", progress=False)
                break
            except Exception as e:
                log_error(f"Yahoo Finance error for {ticker} (attempt {attempt + 1}): {e}")
                if attempt < MAX_API_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    return None

        if df.empty or df['Volume'].iloc[-1] < VOLUME_THRESHOLD or info.get('marketCap', 0) < MARKET_CAP_THRESHOLD or info.get('floatShares', 0) > FLOAT_THRESHOLD:
            return None

        features_df = prepare_features(df, info, vix_data)
        if features_df.empty:
            return None

        X = features_df[['ma_10', 'ma_50', 'rsi', 'atr', 'returns', 'short_interest', 'float', 'vix']]
        y = features_df['Close']

        xgb_model = xgb.XGBRegressor(n_estimators=100, objective='reg:squarederror')
        voting_model = VotingRegressor([
            ('rf', RandomForestRegressor(n_estimators=50)),
            ('gb', GradientBoostingRegressor(n_estimators=50)),
            ('xgb', xgb_model)
        ])
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            voting_model.fit(X_train, y_train)
            pred = voting_model.predict(X_test)
            scores.append(mean_squared_error(y_test, pred, squared=False))
        avg_rmse = np.mean(scores)
        if avg_rmse > 0.1 * y.iloc[-1]:
            return None

        xgb_model.fit(X, y)
        importance = {feature: score for feature, score in zip(X.columns, xgb_model.feature_importances_)}
        importance_str = ", ".join(f"{k}: {v:.3f}" for k, v in importance.items())

        X_lstm = X.values.reshape((X.shape[0], 1, X.shape[1]))
        lstm_model = build_lstm_model((1, X.shape[1]))
        lstm_model.fit(X_lstm[:-1], y.values[1:], epochs=5, batch_size=32, verbose=0)
        voting_pred = voting_model.predict(X.iloc[[-1]])[0]
        lstm_pred = lstm_model.predict(X_lstm[-1:], verbose=0)[0][0]
        predicted_price = 0.7 * voting_pred + 0.3 * lstm_pred
        predicted_gain = max((predicted_price - y.iloc[-1]) / y.iloc[-1], 0.05)

        last_close = df['Close'].iloc[-1]
        atr = calculate_atr(df)
        rsi = calculate_rsi(df['Close'])
        ma_50 = df['Close'].rolling(50).mean().iloc[-1]
        if rsi > RSI_COLD_THRESHOLD:
            return None

        google_trend = await get_google_trends(ticker)
        reddit_sentiment = await analyze_reddit_sentiment(ticker)

        score = min(1.0, 0.4 * (RSI_COLD_THRESHOLD - rsi) / RSI_COLD_THRESHOLD +
                    0.3 * (1 - atr / last_close) + 0.2 * (last_close / ma_50) +
                    0.05 * google_trend + 0.05 * reddit_sentiment +
                    0.1 * (info.get('shortPercentOfFloat', 0) / 100))

        days_in_trade = max(3, min(14, int(10 / max(atr / last_close, 0.01))))
        win_prob = 0.6 + 0.1 * (info.get('shortPercentOfFloat', 0) > 20) + 0.1 * (rsi < 20)
        position_size = kelly_criterion(win_prob=win_prob)

        return {
            "ticker": ticker,
            "entry_price": last_close,
            "target_price": last_close * (1 + predicted_gain),
            "stop_loss": last_close - 2 * atr,
            "score": score,
            "predicted_gain": predicted_gain,
            "days_in_trade": days_in_trade,
            "position_size": position_size,
            "google_trend": google_trend,
            "reddit_sentiment": reddit_sentiment,
            "short_interest": info.get('shortPercentOfFloat', 0),
            "feature_importance": importance_str
        }
    except Exception as e:
        log_error(f"Analyze ticker {ticker} failed: {e}")
        return None

# ============================== #
#       סריקת מניות קרות        #
# ============================== #

async def scan_stocks():
    try:
        results = []
        cold_list = await fetch_tickers()
        if not cold_list:
            await send_telegram("⚠️ שגיאה: לא נמצאו טיקרים מתאימים")
            return []

        tasks = [analyze_ticker(ticker) for ticker in cold_list]
        analyses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for analysis in analyses:
            if isinstance(analysis, Exception):
                log_error(f"Analysis failed: {analysis}")
                continue
            if analysis and analysis['predicted_gain'] > GAIN_THRESHOLD:
                results.append(analysis)

        top_results = sorted(results, key=lambda x: x['score'], reverse=True)[:3]
        
        if not top_results:
            await send_telegram("⚠️ לא נמצאו מניות עם פוטנציאל מספיק גבוה")
            return []

        output_data = []
        msg = "🏆 3 המניות המובילות:\n\n"
        for analysis in top_results:
            timestamp = datetime.utcnow().isoformat()
            with sqlite3.connect(DB_FILE) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM trades WHERE ticker=? AND timestamp=?", 
                             (analysis['ticker'], timestamp))
                if cursor.fetchone()[0] == 0:
                    cursor.execute('''
                        INSERT INTO trades 
                        (ticker, entry_price, target_price, stop_loss, score, predicted_gain, 
                         days_in_trade, position_size, timestamp, google_trend, reddit_sentiment, short_interest, feature_importance)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (analysis['ticker'], analysis['entry_price'], analysis['target_price'],
                          analysis['stop_loss'], analysis['score'], analysis['predicted_gain'],
                          analysis['days_in_trade'], analysis['position_size'], timestamp,
                          analysis['google_trend'], analysis['reddit_sentiment'], analysis['short_interest'],
                          analysis['feature_importance']))
                    conn.commit()
            
            msg += (f"❄️ {analysis['ticker']} | כניסה: ${analysis['entry_price']:.2f} | "
                    f"יעד: ${analysis['target_price']:.2f} | סטופ: ${analysis['stop_loss']:.2f} | "
                    f"ציון: {analysis['score']:.2f} | תחזית עלייה: {analysis['predicted_gain']*100:.2f}% | "
                    f"מאפיינים: {analysis['feature_importance']}\n\n")
            
            output_data.append({
                'ticker': analysis['ticker'],
                'entry_price': analysis['entry_price'],
                'target_price': analysis['target_price'],
                'stop_loss': analysis['stop_loss'],
                'score': analysis['score'],
                'predicted_gain': analysis['predicted_gain'],
                'timestamp': timestamp
            })

        pd.DataFrame
