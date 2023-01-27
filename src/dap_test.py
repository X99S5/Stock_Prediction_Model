"""Fits processed data to ARIMA models and
    provides plots of results for insight."""
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def data_test(stock_data_resample, exogenous_df):
    """Fits processed data to ARIMA models and
    provides plots of results for insight."""

    plt.style.use('Solarize_Light2')
    warnings.filterwarnings("ignore")

    # Section 1 ############################
    # Close price rolling prediction of Apple stock using SARIMAX and only previous stock values
    train_data = stock_data_resample['Close'][0:1854]
    test_data = pd.DataFrame(stock_data_resample['Close'][1854:])
    test_data['Predictions'] = 0
    for i in range(len(test_data)):
        model = SARIMAX(train_data, order=(0, 1, 1),)
        model_fit = model.fit(disp=False)

        future_forcast = model_fit.forecast()
        test_data['Predictions'][i] = future_forcast

        train_data = train_data.append(test_data['Close'][i:i + 1])
        train_data = train_data[1:]

    # plot for close price prediction
    plt.figure(figsize=(25, 10))
    plt.plot(stock_data_resample['Close'][1835:], '-bo', label='Natural Logged StockPrice')
    plt.plot(test_data['Predictions'], '-ro', label='Predictions')

    plt.errorbar(x=test_data.index, y=test_data['Predictions'],
                 yerr=0.015,
                 fmt='ro')
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Natural Logged StockPrice")
    plt.title("Close price rolling prediction of Apple stock using SARIMAX(0,1,1) and only previous stock values")
    plt.show(block=False)
    err_1 = np.sqrt(np.sum(np.square(test_data['Close'] - test_data['Predictions'])))
    print("The root mean squared error for just the stock price is : ", err_1)
    pred_save1 = test_data
    pred_save1.columns = ['Close', 'No Exogenous Variables']



    # Section 2 ############################
    # Close price rolling prediction of Apple stock using SARIMAX and previous stock values
    # + exogenous variable (CrudeOil Price)

    train_data = stock_data_resample['Close'][0:1854]
    train_data_exog = exogenous_df['CrudeOil Price'][0:1854]
    test_data = pd.DataFrame(stock_data_resample['Close'][1854:])
    test_data_exog = pd.DataFrame(exogenous_df['CrudeOil Price'][1854:])
    test_data['Predictions'] = 0

    for i in range(len(test_data)):
        model = SARIMAX(endog=train_data, exog=train_data_exog, order=(0, 1, 1))
        model_fit = model.fit(disp=False)

        future_forcast = model_fit.forecast(exog=test_data_exog['CrudeOil Price'][i:i + 1])
        test_data['Predictions'][i] = future_forcast

        train_data = train_data.append(test_data['Close'][i:i + 1])
        train_data = train_data[1:]
        train_data_exog = train_data_exog.append(test_data_exog['CrudeOil Price'][i:i + 1])
        train_data_exog = train_data_exog[1:]

        # plot for close price prediction
    plt.figure(figsize=(25, 10))
    plt.plot(stock_data_resample['Close'][1835:], '-bo', label='Natural Logged StockPrice')
    plt.plot(test_data['Predictions'], '-ro', label='Predictions')

    plt.errorbar(x=test_data.index, y=test_data['Predictions'],
                 yerr=0.016,
                 fmt='ro')
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Natural Logged StockPrice")
    plt.title(
        "Close price rolling prediction of Apple stock using SARIMAX(0,1,1) and previous stock values + exogenous variable (CrudeOil Price)")
    plt.show(block=False)
    err_2 = np.sqrt(np.sum(np.square(test_data['Close'] - test_data['Predictions'])))
    print("The root mean squared error for stock price + CrudeOil Price is: ", err_2)
    pred_save2 = test_data

    # Section 3 ############################
    # Close price rolling prediction of Apple stock using SARIMAX and previous stock values
    # + exogenous variable (CPI Index)

    train_data = stock_data_resample['Close'][0:1854]
    train_data_exog = exogenous_df['CPI Index'][0:1854]
    test_data = pd.DataFrame(stock_data_resample['Close'][1854:])
    test_data_exog = pd.DataFrame(exogenous_df['CPI Index'][1854:])
    test_data['Predictions'] = 0

    for i in range(len(test_data)):
        model = SARIMAX(endog=train_data, exog=train_data_exog, order=(5, 2, 1))
        model_fit = model.fit(disp=False)

        future_forcast = model_fit.forecast(exog=test_data_exog['CPI Index'][i:i + 1])
        test_data['Predictions'][i] = future_forcast

        train_data = train_data.append(test_data['Close'][i:i + 1])
        train_data = train_data[1:]
        train_data_exog = train_data_exog.append(test_data_exog['CPI Index'][i:i + 1])
        train_data_exog = train_data_exog[1:]

        # plot for close price prediction
    plt.figure(figsize=(25, 10))
    plt.plot(stock_data_resample['Close'][1835:], '-bo', label='Natural Logged StockPrice')
    plt.plot(test_data['Predictions'], '-ro', label='Predictions')

    plt.errorbar(x=test_data.index, y=test_data['Predictions'],
                 yerr=0.04,
                 fmt='ro')
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Natural Logged StockPrice")
    plt.title(
        "Close price rolling prediction of Apple stock using SARIMAX(5,2,1) and previous stock values + exogenous variable (CPI Index)")
    plt.show(block=False)
    err_3 = np.sqrt(np.sum(np.square(test_data['Close'] - test_data['Predictions'])))
    print("The root mean squared error for stock price + CPI Index is: ", err_3)
    pred_save3 = test_data

    # Section 4 ############################
    # Close price rolling prediction of Apple stock using SARIMAX and previous stock values
    # + exogenous variable (Discount Rate)

    train_data = stock_data_resample['Close'][0:1854]
    train_data_exog = exogenous_df['Discount Rate'][0:1854]
    test_data = pd.DataFrame(stock_data_resample['Close'][1854:])
    test_data_exog = pd.DataFrame(exogenous_df['Discount Rate'][1854:])
    test_data['Predictions'] = 0

    for i in range(len(test_data)):
        model = SARIMAX(endog=train_data, exog=train_data_exog, order=(0, 1, 1))
        model_fit = model.fit(disp=False)

        future_forcast = model_fit.forecast(exog=test_data_exog['Discount Rate'][i:i + 1])
        test_data['Predictions'][i] = future_forcast

        train_data = train_data.append(test_data['Close'][i:i + 1])
        train_data = train_data[1:]
        train_data_exog = train_data_exog.append(test_data_exog['Discount Rate'][i:i + 1])
        train_data_exog = train_data_exog[1:]

        # plot for close price prediction
    plt.figure(figsize=(25, 10))
    plt.plot(stock_data_resample['Close'][1835:], '-bo', label='Natural Logged StockPrice')
    plt.plot(test_data['Predictions'], '-ro', label='Predictions')

    plt.errorbar(x=test_data.index, y=test_data['Predictions'],
                 yerr=0.016,
                 fmt='ro')
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Natural Logged StockPrice")
    plt.title(
        "Close price rolling prediction of Apple stock using SARIMAX(0,1,1) and previous stock values + exogenous variable (Discount Rate)")
    plt.show(block=False)
    err_4 = np.sqrt(np.sum(np.square(test_data['Close'] - test_data['Predictions'])))
    print("The root mean squared error for stock price + FED Discount Rate is: ", err_4)
    pred_save4 = test_data

    # Section 5 ############################
    # Close price rolling prediction of Apple stock using SARIMAX and previous stock values
    # + All exogenous variables

    train_data = stock_data_resample['Close'][0:1854]
    train_data_exog = exogenous_df[0:1854]
    test_data = pd.DataFrame(stock_data_resample['Close'][1854:])
    test_data_exog = pd.DataFrame(exogenous_df[1854:])
    test_data['Predictions'] = 0

    for i in range(len(test_data)):
        model = SARIMAX(endog=train_data, exog=train_data_exog, order=(0, 1, 1))
        model_fit = model.fit(disp=False)

        future_forcast = model_fit.forecast(exog=test_data_exog[i:i + 1])
        test_data['Predictions'][i] = future_forcast

        train_data = train_data.append(test_data['Close'][i:i + 1])
        train_data = train_data[1:]
        train_data_exog = train_data_exog.append(test_data_exog[i:i + 1])
        train_data_exog = train_data_exog[1:]

        # plot for close price prediction
    plt.figure(figsize=(25, 10))
    plt.plot(stock_data_resample['Close'][1835:], '-bo', label='Natural Logged StockPrice')
    plt.plot(test_data['Predictions'], '-ro', label='Predictions')

    plt.errorbar(x=test_data.index, y=test_data['Predictions'],
                 yerr=0.017,
                 fmt='ro')
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Natural Logged StockPrice")
    plt.title(
        "Close price rolling prediction of Apple stock using SARIMAX(0,1,1) and previous stock values + All exogenous variables")
    plt.show(block=False)
    err_5 = np.sqrt(np.sum(np.square(test_data['Close'] - test_data['Predictions'])))
    print("The root mean squared error for stock price + All Exogenous variables is: ", err_5)
    pred_save5 = test_data

    # Section 6 ############################
    #plots the results
    print("\n========Results========\n")
    preformance_errors = pd.DataFrame(np.array(
        [err_1, err_2, err_3, err_4,err_5]).reshape(1, -1),
                                     columns=['No Exogenous Variables', 'Crude Oil Prices', 'CPI', 'Discount Rate',
                                            'All Exogenous Variables'])

    print("\n****Root Mean Squared Error of Models****\n")
    print(preformance_errors)

    plt.figure()
    _ = sns.scatterplot(preformance_errors)
    plt.ylabel("Root Mean Squared Error of Models")
    plt.title("Model Comparision based on Root Mean Squared Error", fontsize=11)
    plt.show(block=False)

    plt.figure()
    _ = sns.scatterplot(-100 * (preformance_errors - 0.13914122395486347) / 0.13914122395486347)
    plt.ylabel("Performance Improvement compared to base case (no exogenous variables) %", fontsize=7)
    plt.title("Model Comparision based on percentage improvement", fontsize=11)
    plt.show(block=False)

    full_data = pred_save1
    full_data['Crude Oil Prices'] = pred_save2['Predictions']
    full_data['CPI'] = pred_save3['Predictions']
    full_data['Discount Rate'] = pred_save4['Predictions']
    full_data['All Exogenous Variables'] = pred_save5['Predictions']
    plt.figure()
    sns.lineplot(full_data)
    plt.title("All predictions plotted together", fontsize=11)
