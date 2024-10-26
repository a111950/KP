import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# Load and prepare the monthly sunspot data
file = 'monthly_sunspot_1749.txt'
data = pd.read_csv(file, sep=r'\s+', header=None, on_bad_lines='skip', engine='python')

# Set column names
data.columns = ['Year', 'Month', 'Date_Fraction', 'Sunspot', 'ID1', 'ID2']

# Create a date column and convert it to datetime format
data['ds'] = pd.to_datetime(data[['Year', 'Month']].assign(DAY=1))

# Prepare the data for Prophet (only using 'ds' and 'y' columns)
data = data[['ds', 'Sunspot']].rename(columns={'Sunspot': 'y'})

# 1. Historical Sunspot Data Graph
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=data['ds'], y=data['y'], mode='markers', name='Observed Sunspot Number'))
fig1.update_layout(
    title='Monthly Sunspot Number Progression from 1749 to Now',
    xaxis_title='Date',
    yaxis_title='Sunspot Number (Observed)',
    template='plotly_dark'
)

# 2. Sunspot 12-Month Forecast
model = Prophet()
model.fit(data)
future = model.make_future_dataframe(periods=12, freq='M')
forecast = model.predict(future)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='markers', name='Predicted Sunspot Number'))
fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dot')))
fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dot')))
fig2.update_layout(
    title='12-Month Sunspot Number Prediction',
    xaxis_title='Date',
    yaxis_title='Sunspot Number (Predicted)',
    template='plotly_dark'
)

# Display graphs
fig1.show()
fig2.show()

# Save graphs to HTML files
fig1.write_html("observed_monthly_sunspot_progression.html")
fig2.write_html("predicted_monthly_sunspot_forecast.html")