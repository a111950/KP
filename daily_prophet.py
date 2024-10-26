# 필요한 라이브러리 임포트
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 파일 경로 지정
file = 'activity.txt'
data = pd.read_csv(file, delim_whitespace=True, header=None)

# 열 이름 지정
data.columns = ['date', 'time', 'jd', 'unknown1', 'flux_observed', 'flux_adjusted', 'flux_absolute']

# 날짜 열을 datetime 형식으로 변환
data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')

# 일별 평균 flux 관측값 계산
daily_flux = data.groupby('date')['flux_observed'].mean().reset_index()
daily_flux.columns = ['ds', 'y']  # Prophet이 요구하는 열 이름 (ds: 날짜, y: 예측할 값)

# Prophet 모델 생성 및 학습
model = Prophet()
model.fit(daily_flux)

# 예측 데이터 생성 (3개월, 약 90일)
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# 관측 데이터 그래프
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=daily_flux['ds'], y=daily_flux['y'], mode='lines+markers', name='Observed Daily Flux'))
fig1.update_layout(
    title='Current Solar Radio Flux Status',
    xaxis_title='Date',
    yaxis_title='Solar Flux (Observed)',
    template='plotly_dark'
)

# 예측 데이터 그래프
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

# 그래프 표시
fig1.show()
fig2.show()

fig1.write_html("observed_daily_flux.html")
fig2.write_html("predicted_daily_flux.html")