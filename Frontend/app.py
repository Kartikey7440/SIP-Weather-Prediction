import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from statsmodels.tsa.statespace.sarimax import SARIMAX

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


st.set_page_config(page_title="Weather Forecasting Dashboard", layout="wide")
st.title("🌦 Weather Forecasting Using Multiple Models")


@st.cache_data
def load_data():
    df = pd.read_csv("Refined_DaliyDelhiClimateData.csv")
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

  
    df['temp_lag1'] = df['meantemp'].shift(1)
    df['temp_lag7'] = df['meantemp'].shift(7)
    df['temp_rolling7'] = df['meantemp'].rolling(7).mean()
    df['temp_rolling30'] = df['meantemp'].rolling(30).mean()

    df.dropna(inplace=True)
    return df

df = load_data()
st.subheader("📊 Dataset Preview")
st.dataframe(df)


features = [
    'humidity', 'wind_speed', 'meanpressure',
    'temp_lag1', 'temp_lag7', 'temp_rolling7', 'temp_rolling30'
]

X = df[features]
y = df['meantemp']


scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1,1))

split = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y_scaled[:split], y_scaled[split:]



def plot_last30_and_future(actual_series, predicted_series, future_forecast, model_name):
    last_30_actual = actual_series[-30:]
    last_30_pred = predicted_series[-30:]

    future_dates = pd.date_range(
        start=actual_series.index[-1] + pd.Timedelta(days=1),
        periods=7
    )

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(last_30_actual.index, last_30_actual.values,
            label="Actual (Last 30 Days)", color="blue")

    ax.plot(last_30_pred.index, last_30_pred.values,
            label="Predicted (Last 30 Days)", color="green")

    ax.plot(future_dates, future_forecast,
            label="Forecast (Next 7 Days)", color="orange", marker='o')

    ax.set_title(f"{model_name}: Last 30 Days + Next 7 Days Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)



model_name = st.sidebar.selectbox(
    "Select Forecasting Model",
    ["SARIMA", "Linear Regression", "SVR", "LSTM"]
)


# SARIMA MODEL
if model_name == "SARIMA":
    st.header("📈 SARIMA Forecast")

    model = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,7))
    sarima_fit = model.fit(disp=False)

    forecast = sarima_fit.forecast(steps=7)

    # fig, ax = plt.subplots(figsize=(12,5))
    # ax.plot(y.index, y, label="Actual")
    # ax.plot(
    #     pd.date_range(y.index[-1], periods=8, freq='D')[1:],
    #     forecast, label="Next 7 Days Forecast", color='red'
    # )
    # ax.legend()
    # st.pyplot(fig)

   
    full_pred = sarima_fit.predict(
        start=len(y) - 300,
        end=len(y) - 1
    )

    
    full_pred = pd.Series(
            full_pred,
            index=y.index[-300:]
        )

       
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(y.index[-300:], y[-300:], label="Actual", color="blue")
    ax.plot(y.index[-300:], full_pred, label="Predicted", color="orange")
    ax.set_title("SARIMA – Last 300 Days Prediction")
    ax.legend()
    st.pyplot(fig)


    test_actual = y[-7:]
    test_pred = full_pred[-7:]

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(test_actual.index, test_actual, label="Actual")
    ax.plot(test_actual.index, test_pred, label="Predicted")
    ax.set_title("SARIMA – Last 7 Days (Test)")
    ax.legend()
    st.pyplot(fig)


    sarima_pred = sarima_fit.predict(start=y.index[0], end=y.index[-1])

    last_30_actual = y[-30:]
    last_30_pred = sarima_pred[-30:]

    future_forecast = sarima_fit.forecast(steps=7)

    future_dates = pd.date_range(
        start=y.index[-1] + pd.Timedelta(days=1),
        periods=7,
        freq='D'
    )


    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        last_30_actual.index,
        last_30_actual,
        label="Actual (Last 30 Days)",
        color="blue"
    )

    ax.plot(
        last_30_pred.index,
        last_30_pred,
        label="Predicted (Last 30 Days)",
        color="green"
    )

   
    ax.plot(
        future_dates,
        future_forecast,
        label="Forecast (Next 7 Days)",
        color="orange",
        marker="o"
    )

    ax.set_title("SARIMA Forecast: Last 30 Days + Next 7 Days")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


    mae = mean_absolute_error(test_actual, test_pred)
    rmse = np.sqrt(mean_squared_error(test_actual, test_pred))
    r2 = r2_score(test_actual, test_pred)

    st.subheader("📊 Performance (Last 7 Days)")
    st.write("MAE:", mae)
    st.write("RMSE:", rmse)
    st.write("R²:", r2)


    

# LINEAR REGRESSION
elif model_name == "Linear Regression":
    st.header("📈 Linear Regression")

    lr = LinearRegression()
    lr.fit(X_train, y_train)

   
    full_pred_scaled = lr.predict(X_scaled)
    full_pred = scaler_y.inverse_transform(
        full_pred_scaled.reshape(-1, 1)
    ).flatten()

    full_pred_last300 = pd.Series(
        full_pred[-300:],
        index=y.index[-300:]
    )


    
    test_pred_scaled = lr.predict(X_test)
    test_pred = scaler_y.inverse_transform(
        test_pred_scaled.reshape(-1, 1)
    ).flatten()

    
    y_test_actual = scaler_y.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()

    
    test_index = y.index[-len(y_test):]

    
    test_pred_series = pd.Series(
        test_pred,
        index=test_index
    )

   
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(test_index, y_test_actual, label="Actual (Test)", color="blue")
    ax.plot(test_index, test_pred_series, label="Predicted (Test)", color="orange")
    ax.set_title("Linear Regression – Test Data Prediction")
    ax.legend()
    st.pyplot(fig)


   
    last_row = X.iloc[-1].values.copy()
    future_preds = []

    for _ in range(7):
        scaled_row = scaler_X.transform(last_row.reshape(1, -1))
        pred_scaled = lr.predict(scaled_row)
        pred = scaler_y.inverse_transform(
            pred_scaled.reshape(-1, 1)
        )[0, 0]

        future_preds.append(pred)

        
        last_row = np.roll(last_row, -1)
        last_row[0] = pred

    
    plot_last30_and_future(
        actual_series=y,
        predicted_series=full_pred_last300,
        future_forecast=future_preds,
        model_name="Linear Regression"
    )

    
    st.subheader("📊 Performance (Test Set)")
    st.write("MAE:", mean_absolute_error(y_test_actual, test_pred))
    st.write("RMSE:", np.sqrt(mean_squared_error(y_test_actual, test_pred)))
    st.write("R²:", r2_score(y_test_actual, test_pred))



# SVR MODEL
elif model_name == "SVR":
    st.header("📈 Support Vector Regression")

    svr = SVR(C=50, gamma=0.01, epsilon=0.01)
    svr.fit(X_train, y_train.ravel())

   
    full_svr_pred_scaled = svr.predict(X_scaled)
    full_svr_pred = scaler_y.inverse_transform(
        full_svr_pred_scaled.reshape(-1,1)
    ).flatten()

    full_svr_pred_series = pd.Series(full_svr_pred, index=y.index)

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(y.index, y, label="Actual", color="blue")
    ax.plot(y.index, full_svr_pred_series, label="Predicted", color="orange")
    ax.set_title("SVR: Full Dataset Prediction")
    ax.legend()
    st.pyplot(fig)

    
    last_row = X.iloc[-1].values.copy()
    future_preds = []

    for _ in range(7):
        scaled_row = scaler_X.transform(last_row.reshape(1, -1))
        pred_scaled = svr.predict(scaled_row)
        pred = scaler_y.inverse_transform(
            pred_scaled.reshape(-1,1)
        )[0][0]

        future_preds.append(pred)

        last_row = np.roll(last_row, -1)
        last_row[0] = pred

    
    plot_last30_and_future(
        actual_series=y,
        predicted_series=full_svr_pred_series,
        future_forecast=future_preds,
        model_name="SVR"
    )

    
    y_pred_test = scaler_y.inverse_transform(
        svr.predict(X_test).reshape(-1,1)
    )
    y_actual_test = scaler_y.inverse_transform(y_test)

    st.subheader("📊 Performance (Test Set)")
    st.write("MAE:", mean_absolute_error(y_actual_test, y_pred_test))
    st.write("RMSE:", np.sqrt(mean_squared_error(y_actual_test, y_pred_test)))
    st.write("R²:", r2_score(y_actual_test, y_pred_test))



# LSTM MODEL
elif model_name == "LSTM":
    st.header("📈 LSTM (Deep Learning Model)")

    X_lstm = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    X_train_l, X_test_l = X_lstm[:split], X_lstm[split:]

    model = Sequential([
        LSTM(50, activation='tanh', input_shape=(X_train_l.shape[1],1)),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train_l, y_train, epochs=10, batch_size=32, verbose=0)

    y_pred = scaler_y.inverse_transform(model.predict(X_test_l))
    y_actual = scaler_y.inverse_transform(y_test)

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(y_actual, label="Actual")
    ax.plot(y_pred, label="Predicted")
    ax.legend()
    st.pyplot(fig)



    full_lstm_pred = scaler_y.inverse_transform(
        model.predict(X_lstm)
    ).flatten()

    full_lstm_series = pd.Series(full_lstm_pred, index=y.index)

    last_seq = X_lstm[-1].copy()
    future_preds = []

    for _ in range(7):
        pred_scaled = model.predict(last_seq.reshape(1, last_seq.shape[0], 1))
        pred = scaler_y.inverse_transform(pred_scaled)[0][0]
        future_preds.append(pred)

        
        last_seq = np.roll(last_seq, -1)
        last_seq[-1] = pred_scaled

 
  
    last_30_actual = y.iloc[-30:]
    last_30_pred = full_lstm_series.iloc[-30:]

    future_dates = pd.date_range(
        start=y.index[-1] + pd.Timedelta(days=1),
        periods=7
    )

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(last_30_actual.index, last_30_actual, label="Actual (Last 30 Days)")
    ax.plot(last_30_pred.index, last_30_pred, label="Predicted (Last 30 Days)", color="orange")
    ax.plot(future_dates, future_preds, label="Future 7 Days Forecast", color="red", marker="o")
    ax.set_title("LSTM – Last 30 Days & Future 7 Days Forecast")
    ax.legend()
    st.pyplot(fig)



    st.subheader("📊 Performance")
    st.write("MAE:", mean_absolute_error(y_actual, y_pred))
    st.write("RMSE:", np.sqrt(mean_squared_error(y_actual, y_pred)))
    st.write("R²:", r2_score(y_actual, y_pred))


st.markdown("---")
st.markdown("**SIP Second Year Project | Weather Forecasting System**")
