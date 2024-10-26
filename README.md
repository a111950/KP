# KP
# importing libraries
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# choosing file path
file = 'activity.txt'
data = pd.read_csv(file, delim_whitespace=True, header=None)

# Naming rows/columns
data.columns = ['date', 'time', 'jd', 'unknown1', 'flux_observed', 'flux_adjusted', 'flux_absolute']

# changing to datetime
data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')

# daily flux calculation
daily_flux = data.groupby('date')['flux_observed'].mean().reset_index()
daily_flux.columns = ['ds', 'y']  # Prophet이 요구하는 열 이름 (ds: 날짜, y: 예측할 값)

# Prophet model creation and learning
model = Prophet()
model.fit(daily_flux)

# model data creation (3개월, 약 90일)
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# observation graph
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=daily_flux['ds'], y=daily_flux['y'], mode='lines+markers', name='Observed Daily Flux'))
fig1.update_layout(
    title='Current Solar Radio Flux Status',
    xaxis_title='Date',
    yaxis_title='Solar Flux (Observed)',
    template='plotly_dark'
)

# prediction model graph
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted Daily Flux'))
fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dot')))
fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dot')))
fig2.update_layout(
    title='3-Month Solar Radio Flux Prediction',
    xaxis_title='Date',
    yaxis_title='Solar Flux (Predicted)',
    template='plotly_dark'
)

# Graph Display
fig1.show()
fig2.show()

fig1.write_html("observed_daily_flux.html")
fig2.write_html("predicted_daily_flux.html")
