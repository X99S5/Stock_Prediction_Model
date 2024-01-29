# Stock Prediction Model

The issue this project addresses relates to the difficult problem of predicting stock prices on a day-to-day
basis. The reason why stocks are so hard to predict stems from them reacting instantly to any shifts in market
information and to the fact that they can react to many obtuse factors which are hard to quantify. There are
however universal economic factors which reflect the trend in stock prices, somewhat accurately. These factors
are used by experts, more so to predict long term trends rather than give them an advantage in short term trading,
due to the high unpredictability of such. Hence the difficulty also lies in obtaining relevant data which can predict
the stocks movements accurately in the short term.

In this project, I attempted to chip away at this problem by applying time series analysis machine learning
models such as ARIMA (Auto Regressive Integrated Moving Average). However, to make the model as accurate
as possible we first have to pick relevant data . We then need to obtain and store this data in a reliable way.
After obtaining and storing the data, we need to preprocess the data into a format which allows us to easily
visualize and explore it as well as use in our models. Once the data is preprocessed and graphed, we can explore it and 
use this insight to infer decisions  in how to approach building our model. We can then build our models and apply the
relevant data to them and hence analyze the results and performance of these models. 

A brief summary of the methodologies used in this project, were to obtain relevant economic data (Consumer
Price index, Federal Reserve Discount Rate, Brent Crude Oil Prices), preprocess / clean / analyze the data, and
apply the individual data sets to train the optimal respective Arima model to predict stock price of Apple (AAPL)
for a 1month time period (2022 1st May – 1st June). The data was also concatenated together and applied to the
Arima model to test its predictive power with multiple exogenous factors. The predictive power of these models
were then compared to the baseline model which was just applying the stock data to the Arima model and it
predicting solely based on previous values.

Graphs/plots related to the results of this project can be seen in the src/graphs folder.

All the code with related graphs can also be seen in Assignment.ipynb

The best preforming model's predictive accuracy was approximatly 3%. 
