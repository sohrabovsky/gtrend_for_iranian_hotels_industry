from pytrends.request import TrendReq
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from dash import Dash ,dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go

path1= r'/home/sohrab-salehin/Documents/python_scripts/hotel_invst/'
path2= r'/home/sohrab-salehin/Documents/python_scripts/decompose/hotel/'
city_ids = pd.read_csv(path1 + "hotel_id_and_names.csv")
city_sessions = pd.read_csv(path1 + "top_sessions_hotels.csv").rename(  # should be imported everytime for updating
    columns={"city_ids": "city_id"}
)
city_name_fa_en = pd.read_csv(path1 + "city_name_fa_en.csv").dropna()
city_name_fa_en.rename(columns={"City Name Fa": "city_name_fa"}, inplace=True)

top_ten = city_sessions[:10]
cities = pd.merge(left=city_ids, right=city_name_fa_en, on="city_name_fa", how="inner")
cities = cities[["city_id", "city_name_fa"]].drop_duplicates(subset=["city_name_fa"])
cities = cities.merge(right=top_ten, on="city_id", how="inner").sort_values(
    by="Number_of_session", ascending=False
)
hotel_city_sales = pd.read_csv(path2 + "hotel_city_sales.csv", parse_dates=["Registered Date"])
hotel_city_sales = hotel_city_sales.pivot_table(
    index=["Registered Date"],
    columns=["City Name Fa"],
    values=["Sum of New Etl Room Nights"],
    aggfunc="sum",
).fillna(0)
hotel_city_sales["All"] = hotel_city_sales.sum(axis=1)

# Set up the TrendReq object
pytrends = TrendReq(hl="fa-IR", tz=330, timeout=10)
def google_trend_frame(keyword):
    frame = pd.DataFrame()
    newframe = frame.copy()
    pytrends.build_payload(kw_list=[keyword], timeframe="today 5-y")
    related_queries = pytrends.related_queries()
    top_df = related_queries[keyword]["top"]
    splitted_keyword = keyword.split()
    global top_keywords_list
    top_keywords_list = (
        top_df[
            (top_df["query"].str.contains(splitted_keyword[0]))
            | (top_df["query"].str.contains(splitted_keyword[1]))
        ]
        .sort_values(by="value", ascending=False)
        .reset_index(drop=True)
    )
    i = 1
    while i <= (len(top_keywords_list) - 4):
        serie = np.arange(i, i + 4)
        i = i + 4
        kw_list = top_keywords_list["query"].values[
            np.concatenate((0, serie), axis=None)
        ]
        pytrends.build_payload(kw_list=kw_list, timeframe="today 5-y")
        for column in pytrends.interest_over_time().columns[0:5]:
            newframe = pytrends.interest_over_time()[[column]]
            frame = pd.concat([frame, newframe], axis=1)
    frame = frame.loc[:, ~frame.columns.duplicated(keep="first")]
    frame["total"] = frame.sum(axis=1)
    return frame

### Decompose - Seasonality
df= pd.DataFrame()


for keyword in cities['city_name_fa'].values:
    dff= pd.DataFrame(columns= ['date', 'g-trend', 'sales-trend', 'g-seasonal', 'sales-seasonal'])
    sales = hotel_city_sales["Sum of New Etl Room Nights", keyword]
    sales = sales.resample(rule="W-Sun").sum()
    keyword= keyword + ' هتل'
    gtrend = google_trend_frame(keyword)
    gtrend.index.freq = pd.infer_freq(gtrend.index)
    gtrend = gtrend["total"]
    decomp_gtrend = seasonal_decompose(gtrend, model="additive")  ### Additive Model
    decomp_sales = seasonal_decompose(sales, model="additive")  ### Additive Model
    scaler = MinMaxScaler()
    df_trends = pd.merge(
        left=decomp_gtrend.trend,
        right=decomp_sales.trend,
        left_index=True,
        right_index=True,
        how="inner",
    )
    df_seasonal= pd.merge(
        left=decomp_gtrend.seasonal,
        right=decomp_sales.seasonal,
        left_index=True,
        right_index=True,
        how="inner",
    )
    df_trends_scaled = pd.DataFrame(
        scaler.fit_transform(df_trends), columns=df_trends.columns
    )
    df_seasonal_scaled = pd.DataFrame(
        scaler.fit_transform(df_seasonal), columns=df_seasonal.columns
    )
    date= pd.date_range(start= min(np.min(df_trends.index), np.min(df_seasonal.index)),
                    end= max(np.max(df_trends.index), np.max(df_seasonal.index)),
                    freq= 'W-Sun')
    google_trend_scaled= df_trends_scaled['trend_x']
    sales_trend_scaled= df_trends_scaled['trend_y']
    google_seasonal_scaled= df_seasonal_scaled['seasonal_x']
    sales_seasonal_scaled= df_seasonal_scaled['seasonal_y']
    dff['date']= date
    dff['g-trend']= google_trend_scaled
    dff['sales-trend']= sales_trend_scaled
    dff['g-seasonal']= google_seasonal_scaled
    dff['sales-seasonal']= sales_seasonal_scaled
    melted_dff= dff.melt(id_vars= ['date'], value_vars= ['g-trend', 'sales-trend', 'g-seasonal', 'sales-seasonal'])
    melted_dff['city']= keyword
    df= df.append(melted_dff)

# Initialize the app
app = Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("City Data"),
    html.Label("Select a city:"),
    dcc.Dropdown(
        id="city-dropdown",
        options=[
            {"label": city, "value": city} for city in df["city"].unique()
        ],
        value=df["city"].unique()[0],
    ),
    html.Label("Select data type:"),
    dcc.RadioItems(
        id="data-type-radio",
        options=[
            {"label": "Trend", "value": "trend"},
            {"label": "Seasonal", "value": "seasonal"}
        ],
        value="trend",
    ),
    dcc.Graph(id="city-data-graph"),
])

# Define the callback function
@app.callback(
    Output("city-data-graph", "figure"),
    Input("city-dropdown", "value"),
    Input("data-type-radio", "value")
)
def update_graph(city, data_type):
    filtered_df = df[(df["city"] == city) & (df["variable"].str.contains(data_type))]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=filtered_df[filtered_df["variable"].str.contains("sales")]["date"],
        y=filtered_df[filtered_df["variable"].str.contains("sales")]["value"],
        name="Sales"
    ))
    
    fig.add_trace(go.Scatter(
        x=filtered_df[filtered_df["variable"].str.contains("g")]["date"],
        y=filtered_df[filtered_df["variable"].str.contains("g")]["value"],
        name="Google Trend"
    ))
    
    fig.update_layout(title=f"{city} {data_type.capitalize()} Data")
    
    return fig

# Run the app
if __name__ == "__main__":
    app.run_server()