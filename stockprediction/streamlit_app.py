import os
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.layers import LSTM
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="IRMTP", page_icon="./logo-removebg-preview-Edited.png", layout="wide")
tab1, tab2, tab3 = st.tabs(["Prediction", "Insights", "About Us"])

info_multi = ''' IRMTP is your go-to platform for exploring AI-powered stock forecasting and analysis using real-time stock values via Yahoo Finance.      
Whether you're a data science enthusiast or a market observer, this app blends cutting-edge deep learning with intuitive tools to bring you actionable insights.'''

with tab1: 
    st.header('IRMTP Web Application')
    st.info(info_multi)

# ---------------- CUSTOM LSTM ----------------
class CustomLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop("time_major", None)  # Remove unrecognized arg
        super().__init__(*args, **kwargs)

# ---------------- LOAD MODELS ----------------
model_paths = {
    "Apple": "./models/Apple_Model.h5",
    "Google": "./models/Google_Model.h5",
    "Tesla": "./models/3rd-Tesla-LSTM-Model.h5",
    "Amazon": "./models/Amazon-LSTM-Model.h5",
    "Intel": "./models/Intel-2nd-LSTM-Model.h5",
    "Meta": "./models/Meta-LSTM-Model.h5",
    "Microsoft": "./models/Microsoft-LSTM-Model.h5",
}

models = {}
for name, path in model_paths.items():
    if os.path.exists(path):
        try:
            models[name] = load_model(path, custom_objects={"LSTM": CustomLSTM}, compile=False)
        except Exception as e:
            st.warning(f"⚠️ Could not load {name} model: {e}")
    else:
        st.warning(f"⚠️ Model file missing: {path}")

# ---------------- DATA FETCH ----------------
def On_Balance_Volume(Close, Volume):
    change = Close.diff()
    OBV = np.cumsum(np.where(change > 0, Volume, np.where(change < 0, -Volume, 0)))
    return OBV

@st.cache_data
def df_process(ticker):
    end = datetime.now()
    start = end - relativedelta(months=3)
    df = yf.download(ticker, start=start, end=end)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.columns.name = None
    df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low',
                            'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'})
    df['garman_klass_volatility'] = ((np.log(df['high']) - np.log(df['low'])) ** 2) / 2 - \
                                    (2 * np.log(2) - 1) * ((np.log(df['close']) - np.log(df['open'])) ** 2)
    df['dollar_volume'] = (df['close'] * df['volume']) / 1e6
    df['ema'] = df['close'].ewm(span=14, adjust=False).mean()
    df['obv'] = On_Balance_Volume(df['close'], df['volume'])
    df['macd'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['ma_3_days'] = df['close'].rolling(3).mean()
    return df

# ---------------- PREDICTION HELPERS ----------------
def create_feed_dset(df_processed, feature_list, n_past, model):
    dset = df_processed.filter(feature_list).dropna(axis=0)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(dset)

    dataX = []
    for i in range(n_past, len(df_scaled)):
        dataX.append(df_scaled[i - n_past:i, 0:df_scaled.shape[1]])
    dataX = np.array(dataX)

    prediction = model.predict(dataX) if model else np.zeros((5, 1))
    return prediction, scaler

def inverse_transform_predictions(prediction_init, scaler, features):
    prediction_array = np.repeat(prediction_init, features, axis=-1)
    pred = scaler.inverse_transform(np.reshape(prediction_array, (len(prediction_init), features)))[:5, 0]
    return pred

def prediction_table(pred_list):
    pred_df = pd.DataFrame({'Predicted Day': ['Tomorrow', '2nd Day', '3rd Day', '4th Day', '5th Day'], 
                            'Adj. Closing Price($)': [ '%.2f' % elem for elem in pred_list]})
    pred_df.set_index('Predicted Day', inplace=True)
    return pred_df

def generate_insight(df_processed, pred_list):
    actual_values = df_processed['close'].values.tolist()
    if actual_values and pred_list:
        last_actual_price = actual_values[-1]
        next_predicted_price = pred_list[0]
        percent_change = (next_predicted_price - last_actual_price) / last_actual_price * 100
        return f"""
        <div style="font-family: Arial; font-size: 16px;">
            <strong>Next predicted price:</strong> <span style="color: #4CAF50;">${next_predicted_price:.2f}</span><br>
            <strong>Last actual price:</strong> <span style="color: #FF5722;">${last_actual_price:.2f}</span><br>
            <strong>Change:</strong> 
            <span style="color: {'#4CAF50' if percent_change >= 0 else '#FF5722'};">{percent_change:+.2f}%</span>
        </div>
        """
    return "<div style='font-family: Arial;'>Not enough data.</div>"

# ---------------- TAB1 UI ----------------
stock_selection = tab1.selectbox("Select Stock:", options=list(model_paths.keys()))

if stock_selection in models:
    df_processed = df_process(stock_selection[:4].upper())  # crude ticker mapping
    # Simplified feature list for demo
    features = ['close', 'obv', 'ema']
    pred_init, scaler = create_feed_dset(df_processed, features, 15, models[stock_selection])
    pred_list = inverse_transform_predictions(pred_init, scaler, len(features)).tolist()
    pred_df = prediction_table(pred_list)
    insight = generate_insight(df_processed, pred_list)

    col1, col2 = tab1.columns(2)
    with col1:
        st.subheader(f"{stock_selection} - Predictions for 5 Days")
        st.dataframe(pred_df)
    with col2:
        st.markdown(insight, unsafe_allow_html=True)

tab1.warning("Disclaimer: Research & educational purposes only.", icon="❗")

# ---------------- TAB2 (Insights) ----------------
with tab2:
    st.header("IRMTP: Interactive Stock Insights")
    st.write("📊 Explore trends with SMA, EMA, RSI, OBV")

# ---------------- TAB3 (About Us) ----------------
with tab3:
    st.header("About Us", divider=True)
    st.info("Created as a final year project by passionate students.")
