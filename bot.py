import asyncio
import aiohttp
import pandas as pd
import numpy as np
import sqlite3
import telegram
import logging
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
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================== #
#         הגדרות כלליות          #
# ============================== #

DB_FILE = "penny_stocks.db"
LOG_FILE = "bot.log"
OUTPUT_FILE = "top_stocks.csv"
TELEGRAM_TOKEN = "8453354058:AAGG0v0zLWTe1NJE7ttfaUZvoutf5XNGU7s"
CHAT_ID = "6387878532"
FMP_API_KEY = "5nhxZGIiFnjG8JxcdSKljx0eZRuqwELX"
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
MAX_TICKERS = 10
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
#         בדיקת FMP API           #
# ============================== #

async def check_fmp_api():
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://financialmodelingprep.com/api/v3/quote/AAPL?apikey={FMP_API_KEY}"
            async with session.get(url, timeout=10) as response:
                response.raise_for_status()
                data = await response.json()
                if not data or len(data) == 0:
                    raise Exception("FMP API returned empty data for AAPL")
                return True
    except Exception as e:
        log_error(f"FMP API check failed: {e}")
        return False

# ============================== #
#         משיכת טיקרים           #
# ============================== #

async def fetch_tickers():
    tickers = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    async with aiohttp.ClientSession(headers=headers) as session:
        if not await check_fmp_api():
            log_error("FMP API is down, falling back to Yahoo Finance")
            try:
                import yfinance as yf
                exchanges = ['^IXIC', '^NYA']
                for exchange in exchanges:
                    tickers_data = yf.Tickers(exchange)
                    for ticker_symbol in tickers_data.tickers:
                        if any(suffix in ticker_symbol for suffix in ['-WS', '-U', '-R', '-P-']):
                            continue
                        try:
                            ticker = tickers_data.tickers[ticker_symbol]
                            info = ticker.info
                            if (info.get('regularMarketPrice', 0) <= 5.0 and
                                info.get('regularMarketVolume', 0) >= VOLUME_THRESHOLD and
                                info.get('exchange') in ['NAS', 'NYQ']):
                                df = yf.download(ticker_symbol, period="1d", interval="1d", progress=False)
                                if not df.empty:
                                    tickers.append(ticker_symbol)
                            await asyncio.sleep(1 / RATE_LIMIT_PER_MINUTE)
                        except Exception as e:
                            log_error(f"Yahoo Finance ticker {ticker_symbol} error: {e}")
                    tickers = list(set(tickers))[:MAX_TICKERS]
            except Exception as e:
                log_error(f"Yahoo Finance fetch error: {e}")
        else:
            try:
                # משיכת רשימת טיקרים פעילים מ-FMP
                url = f"https://financialmodelingprep.com/api/v3/stock/list?apikey={FMP_API_KEY}"
                async with session.get(url, timeout=10) as response:
                    response.raise_for_status()
                    data = await response.json()
                    df = pd.DataFrame(data)
                    df['symbol'] = df['symbol'].astype(str).fillna('')
                    # בדיקה אם העמודה 'volume' קיימת, אחרת נשתמש ב-'avgVolume' או נדלג
                    volume_col = 'volume' if 'volume' in df.columns else 'avgVolume' if 'avgVolume' in df.columns else None
                    if volume_col is None:
                        log_error("No volume column found in FMP data")
                        raise Exception("No volume column found in FMP data")
                    df = df[
                        (df['price'] <= 5.0) &
                        (df[volume_col] >= VOLUME_THRESHOLD) &
                        (df['exchangeShortName'].isin(['NASDAQ', 'NYSE'])) &
                        (~df['symbol'].str.contains('-WS|-U|-R|-P-', na=False))
                    ]
                    tickers = df['symbol'].tolist()[:MAX_TICKERS]
                    logging.info(f"Fetched {len(tickers)} tickers from FMP")
            except Exception as e:
                log_error(f"FMP fetch error: {e}")

        if len(tickers) < 5:
            try:
                # רשימת טיקרים חלופית מעודכנת
                fallback_tickers = ['AACG', 'AAOI', 'AAME', 'AATC', 'ABAT', 'ABCB', 'ABSI', 'ABVC', 'ACAD', 'ACET']
                for ticker_symbol in fallback_tickers:
                    if any(suffix in ticker_symbol for suffix in ['-WS', '-U', '-R', '-P-']):
                        continue
                    try:
                        async with session.get(
                            f"https://financialmodelingprep.com/api/v3/quote/{ticker_symbol}?apikey={FMP_API_KEY}",
                            timeout=10
                        ) as response:
                            response.raise_for_status()
                            data = await response.json()
                            if data and len(data) > 0 and ticker_symbol not in tickers:
                                tickers.append(ticker_symbol)
                        await asyncio.sleep(1 / RATE_LIMIT_PER_MINUTE)
                    except Exception as e:
                        log_error(f"Fallback ticker {ticker_symbol} error: {e}")
                tickers = list(set(tickers))[:MAX_TICKERS]
            except Exception as e:
                log_error(f"Fallback tickers error: {e}")

    logging.info(f"Fetched {len(tickers)} valid tickers")
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

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ============================== #
#      Google Trends + Reddit     #
# ============================== #

async def get_google_trends(ticker):
    for attempt in range(MAX_API_RETRIES):
        try:
            pytrends = TrendReq(hl='en-US', tz=360)
            pytrends.build_payload([ticker], timeframe='now 7-d')
            data = pytrends.interest_over_time()
            if not data.empty:
                return data[ticker].mean() / 100
            await asyncio.sleep(1 / RATE_LIMIT_PER_MINUTE)
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
            await asyncio.sleep(1 / RATE_LIMIT_PER_MINUTE)
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
    df['vix'] = vix_data['Close'].iloc[-1] if vix_data is not None and not vix_data.empty else 0
    df = df.dropna()
    return df[['Close', 'ma_10', 'ma_50', 'rsi', 'atr', 'returns', 'short_interest', 'float', 'vix']]

async def analyze_ticker(ticker):
    if not isinstance(ticker, str) or ticker.lower() == 'nan':
        log_error(f"Invalid ticker: {ticker}")
        return None
    try:
        async with aiohttp.ClientSession() as session:
            result = await asyncio.wait_for(analyze_ticker_inner(ticker, session), timeout=30)
            return result
    except asyncio.TimeoutError:
        log_error(f"Timeout analyzing ticker {ticker}")
        return None
    except Exception as e:
        log_error(f"Analyze ticker {ticker} failed: {str(e)}")
        return None

async def analyze_ticker_inner(ticker, session):
    try:
        # משיכת נתונים היסטוריים מ-FMP
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?timeseries=252&apikey={FMP_API_KEY}"
        async with session.get(url, timeout=10) as response:
            response.raise_for_status()
            data = await response.json()
            if not data.get('historical'):
                log_error(f"No historical data for {ticker}")
                return None
            df = pd.DataFrame(data['historical'])
            df = df.rename(columns={'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
            if len(df) < 50:
                log_error(f"Insufficient data for {ticker} - Data shape: {df.shape}")
                return None

        # משיכת נתונים פונדמנטליים
        url_info = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={FMP_API_KEY}"
        async with session.get(url_info, timeout=10) as response:
            response.raise_for_status()
            info_data = await response.json()
            info = info_data[0] if info_data else {}

        # משיכת נתוני VIX
        try:
            import yfinance as yf
            vix_data = yf.download("^VIX", period="1d", interval="1d", progress=False)
            if vix_data.empty:
                log_error(f"VIX data is empty for {ticker}")
                vix_data = None
        except Exception as e:
            log_error(f"VIX fetch error for {ticker}: {e}")
            vix_data = None

        google_trend = await get_google_trends(ticker)
        reddit_sentiment = await analyze_reddit_sentiment(ticker)

        df_features = prepare_features(df, info, vix_data)
        if df_features.empty:
            log_error(f"No features for {ticker}")
            return None

        X = df_features.drop(columns=['Close'])
        y = df_features['Close']
        tscv = TimeSeriesSplit(n_splits=5)
        voting_model = VotingRegressor([
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
            ('xgb', xgb.XGBRegressor(n_estimators=100, random_state=42))
        ])

        mse_scores = []
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            voting_model.fit(X_train, y_train)
            y_pred = voting_model.predict(X_test)
            mse_scores.append(mean_squared_error(y_test, y_pred))

        # LSTM
        X_lstm = X.values.reshape((X.shape[0], 1, X.shape[1]))
        lstm_model = build_lstm_model((1, X.shape[1]))
        lstm_model.fit(X_lstm[:-1], y.values[1:], epochs=5, batch_size=32, verbose=0)

        voting_pred = voting_model.predict(X.iloc[[-1]])[0]
        lstm_pred = lstm_model.predict(X_lstm[-1:], verbose=0)[0][0]
        predicted_price = 0.7 * voting_pred + 0.3 * lstm_pred
        current_price = df['Close'].iloc[-1]
        predicted_gain = (predicted_price - current_price) / current_price

        atr = calculate_atr(df)
        target_price = current_price + atr
        stop_loss = current_price - atr
        position_size = kelly_criterion()

        return {
            'ticker': ticker,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'predicted_gain': predicted_gain,
            'target_price': target_price,
            'stop_loss': stop_loss,
            'score': np.mean(mse_scores) ** -0.5,
            'position_size': position_size,
            'google_trend': google_trend,
            'reddit_sentiment': reddit_sentiment,
            'short_interest': info.get('shortPercentOfFloat', 0),
            'feature_importance': str(voting_model.estimators_[0].feature_importances_)
        }
    except Exception as e:
        log_error(f"Analyze ticker {ticker} inner failed: {str(e)} - Data shape: {df.shape if 'df' in locals() and not df.empty else 'empty'}")
        return None

async def scan_stocks():
    try:
        results = []
        cold_list = await fetch_tickers()
        if not cold_list:
            await send_telegram("⚠️ שגיאה: לא נמצאו טיקרים מתאימים")
            return []
        logging.info(f"Starting analysis of {len(cold_list)} tickers")
        tasks = [analyze_ticker(ticker) for ticker in cold_list]
        analyses = await asyncio.gather(*tasks, return_exceptions=True)
        for i, analysis in enumerate(analyses):
            logging.info(f"Processed ticker {cold_list[i]}: {'Success' if analysis and not isinstance(analysis, Exception) else 'Failed'}")
            if isinstance(analysis, Exception):
                log_error(f"Analysis failed for {cold_list[i]}: {analysis}")
                continue
            if analysis and analysis['predicted_gain'] > GAIN_THRESHOLD:
                results.append(analysis)
        if results:
            df_results = pd.DataFrame(results)
            df_results.to_csv(OUTPUT_FILE, index=False)
            msg = f"📊 נמצאו {len(results)} מניות מבטיחות:\n"
            for _, row in df_results.iterrows():
                msg += (f"מניה: {row['ticker']}\n"
                        f"מחיר נוכחי: ${row['current_price']:.2f}\n"
                        f"תחזית רווח: {row['predicted_gain']*100:.2f}%\n"
                        f"גודל פוזיציה: {row['position_size']*100:.2f}%\n\n")
            await send_telegram(msg)
        else:
            await send_telegram("😕 לא נמצאו מניות עם פוטנציאל רווח מעל הסף")
        return results
    except Exception as e:
        log_error(f"Scan stocks error: {e}")
        return []

# ============================== #
#           הרצת הבוט            #
# ============================== #

if __name__ == "__main__":
    init_db()
    asyncio.run(scan_stocks())
