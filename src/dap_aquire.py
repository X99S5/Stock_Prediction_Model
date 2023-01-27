"""Retrieves all the datasets from storage and preforms initial cleaning.
Also provides visualisation of cleaned raw data."""
import pandas as pd
from matplotlib import pyplot as plt

def data_aquire():
    """Retrieves all the datasets from storage and preforms initial cleaning.
    Also provides visualisation of cleaned raw data."""

    # Section 1 ############################
    # Handles stock_Data
    # Stage 1: Aquiring historical stock data of Apple
    # #+ initial cleaning from yfinance api
    stock_data = pd.read_csv('src/stockData.csv')
    stock_data = stock_data.set_index('Date')
    stock_data.index = pd.to_datetime(stock_data.index)



    # Section 2 ############################
    # Handles Federal Reserve Discount Rate
    # Stage 1: Aquiring historical rates from federal reserve
    # #+ initial cleaning
    fr_data = pd.read_csv('src/FRB_H15.csv')
    fr_data.columns = ['Date', 'Federal funds Rate', 'Prime Rate', 'Discount Rate']
    fr_data = fr_data[5:]
    fr_data = fr_data.reset_index(drop=True)
    fr_data = fr_data.set_index('Date')
    fr_data.index = pd.to_datetime(fr_data.index)
    fr_data = fr_data.astype({'Discount Rate': 'float'
                                 , 'Prime Rate': 'float'
                                 , 'Federal funds Rate': 'float'})




    # Section 3 ############################
    # Handles CPI Index Data
    # Stage 1: Retrieving stored Historical CPI data originally obtained from Fred Api
    # #+ initial cleaning
    conpi_data = pd.read_csv('src/USA_CPI_Index.csv')
    conpi_data = conpi_data.reset_index(drop=True)
    conpi_data = conpi_data.set_index('Date')
    conpi_data.index = pd.to_datetime(conpi_data.index)
    conpi_data = conpi_data.astype({'CPI Index': 'float'})




    # Section 4 ############################
    # Handles Brent Crude Oil prices Data
    # Stage 1: Retrieving stored Historical Brent Crude Oil data originally obtained from Fred Api
    # #+ initial cleaning
    oil_data = pd.read_csv('src/USA_BrentCrudeOil.csv')
    oil_data = oil_data.reset_index(drop=True)
    oil_data = oil_data.set_index('Date')
    oil_data.index = pd.to_datetime(oil_data.index)
    oil_data = oil_data.astype({'CrudeOil Price': 'float'})


    # Section 5 ############################
    #plots cleaned raw data + printing and returning data
    plt.style.use('Solarize_Light2')
    plt.figure(figsize=(15, 9))
    plt.plot(stock_data['Close']['2017-04-03':'2022-05-31'], label='Apple Stock price in USD')
    plt.plot(oil_data['2017-04-03':'2022-05-31'], label='Brent Crude Oil Commodity price in USD/Barrel')
    plt.plot(conpi_data['2017-04-03':'2022-05-31'] - 200, label='Shifted(-200) U.S Consumer Price Index')
    plt.plot(fr_data['Discount Rate']['2017-04-03':'2022-05-31'] * 10,
             label='Scaled(*10) U.S Federal Reserve Discount Rate %')

    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("")
    plt.title("Cleaned Raw Data Visualisation")
    plt.show(block=False)


    print("\n ============ Acquisition Section ============ \n")
    print("\n*Stock data* \n",stock_data)
    print("\n*FED rates data* \n", fr_data)
    print("\n*CPI data* \n", conpi_data)
    print("\n*Brent Crude Oil data* \n ", oil_data)

    return stock_data,fr_data,conpi_data,oil_data
