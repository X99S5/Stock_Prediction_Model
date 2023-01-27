"""Runs all src modules together. Also controls Pause Timer."""
import src

def main():
    """Runs all src modules together. Also controls Pause Timer."""

    stock_data,fr_data,conpi_data,oil_data = src.data_aquire()
    stock_data_resample,exogenous_df,merged_df = src.data_preprocess(stock_data,fr_data,
                                                                     conpi_data,oil_data)
    src.data_eda(stock_data_resample,exogenous_df,merged_df)
    src.data_test(stock_data_resample, exogenous_df)

    #Note the blank graphs appearing as the code runs do not stop the code !
    #When the code reaches the end these graphs will be plotted appropriately.
    #Then the code will be paused for 15 seconds then it will close automatically.
    #This is to allow visualisation of plots.Increase time below as you see fit.
    plt.pause(15)

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    main()
