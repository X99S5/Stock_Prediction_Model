"""Preforms further processing/transformations on datasets."""
import pandas as pd
import numpy as np

def data_preprocess(stock_data,fr_data,conpi_data,oil_data):
    """Preforms further processing/transformations on datasets."""


    # Section 1 ############################
    # Handles stock_Data
    # Stage 2: resampling data and interpolating missing values
    stock_data_resample = pd.DataFrame(stock_data.resample('1d').sum())
    stock_data_resample[stock_data_resample == 0] = None
    stock_data_resample = stock_data_resample.interpolate(method='pad')
    stock_data_resample = np.log(stock_data_resample)

    # Stage 3: Working out first difference
    stock_data_resample['FirstDifference'] = stock_data_resample['Close'].diff().fillna(0)
    stock_data_resample = stock_data_resample.fillna(0)

    # Section 2 ############################
    # Handles Federal Reserve Discount Rate
    # Stage 2: Resampling data and interpolating missing values
    fr_data_resample = pd.DataFrame(fr_data.resample('1d').sum())
    fr_data_resample[fr_data_resample == 0] = None
    fr_data_resample = fr_data_resample.interpolate(method='time')
    fr_data_resample = pd.DataFrame(fr_data_resample['2017-04-03':'2022-05-31'])


    # Section 3 ############################
    # Handles CPI Index Data
    # Stage 2: Resampling data and interpolating missing values for cpi data
    conpi_data_resample = pd.DataFrame(conpi_data.resample('1d').sum())
    conpi_data_resample[conpi_data_resample == 0] = None
    conpi_data_resample = conpi_data_resample.interpolate(method='time')
    conpi_data_resample = pd.DataFrame(conpi_data_resample['2017-04-03':'2022-05-31'])


    # Section 4 ############################
    # Handles Brent Crude Oil prices Data
    # Stage 2: resampling data and interpolating missing values
    oil_data_resample = pd.DataFrame(oil_data.resample('1d').sum())
    oil_data_resample[oil_data_resample == 0] = None
    oil_data_resample = oil_data_resample.interpolate(method='time')
    oil_data_resample = pd.DataFrame(oil_data_resample['2017-04-03':'2022-05-31'])

    # Section 5 ############################
    # concatanates exogenous variables
    exogenous_df = pd.concat([np.log(conpi_data_resample[['CPI Index']]), fr_data_resample[['Discount Rate']]], axis=1)
    exogenous_df['CrudeOil Price'] = np.log(oil_data_resample['CrudeOil Price'])
    merged_df = exogenous_df.copy()
    merged_df['Close'] = stock_data_resample['Close']

    # Section 6 ############################
    # prints and returns data
    print("\n ============ Preprocessing Section ============ \n")
    print("\n**All Preprocessed Variables** \n", merged_df)


    return stock_data_resample,exogenous_df,merged_df