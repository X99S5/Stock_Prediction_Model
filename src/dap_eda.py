"""Displays useful plots of the processed datasets for insight.
 Also displays hypothesis tests and model fit optimisations."""
import warnings
from pmdarima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib import pyplot as plt
import seaborn as sns

def data_eda(stock_data_resample,exogenous_df,merged_df):
    """Displays useful plots of the processed datasets for insight.
    Also displays hypothesis tests and model fit optimisations."""

    plt.style.use('Solarize_Light2')
    warnings.filterwarnings("ignore")

    # Section 1 ############################
    # Plots pacf and acf
    _ = plot_acf(stock_data_resample['FirstDifference'])
    plt.xlabel("Lags")
    _ = plot_pacf(stock_data_resample['FirstDifference'])
    plt.xlabel("Lags")
    plt.show(block=False)

    # Section 2 ############################
    # Heatmap of correlations of all variables
    c_c = merged_df.corr()
    sns.heatmap(c_c, annot=True)
    plt.title("HeatMap of Correlations between All Variables", fontsize=11)
    plt.show(block=False)

    # Section 3 ############################
    # Pair plot of all variables
    sns.pairplot(merged_df)
    plt.show(block=False)

    # Section 3 ############################
    # Histogram distribution plots of all variables
    sns.displot(exogenous_df['CPI Index'], kde=True).set(
        title='Histogram Distribution Plot For Natural Logged CPI Index')
    sns.displot(exogenous_df['Discount Rate'], kde=True).set(title='Histogram Distribution Plot For Discount Rate')
    sns.displot(exogenous_df['CrudeOil Price'], kde=True).set(
        title='Histogram Distribution Plot For Natural Logged Crude Oil price')
    plt.show(block=False)

    # Section 4 ############################
    # Hypothesis Tests
    print("\n============ EDA/Hypothesis Testing Section ============ \n")
    print("\n***Auto Arima applied only on Stock Price*** \n")
    stepwise_fit = auto_arima(stock_data_resample['Close'], trace=False, suppress_warnings=True)
    print("\n",stepwise_fit.summary())

    print("\n***Auto Arima applied on Stock Price + Discount Rate*** \n")
    stepwise_fit = auto_arima(stock_data_resample['Close'], X=exogenous_df[['Discount Rate']], trace=False,
                              suppress_warnings=True)
    print("\n",stepwise_fit.summary())

    print("\n***Auto Arima applied on Stock Price + CPI Index*** \n")
    stepwise_fit = auto_arima(stock_data_resample[['Close']], X=exogenous_df[['CPI Index']], trace=False,
                              suppress_warnings=True)
    print("\n",stepwise_fit.summary())

    print("\n***Auto Arima applied on Stock Price + CrudeOil Price*** \n")
    stepwise_fit = auto_arima(stock_data_resample['Close'], X=exogenous_df[['CrudeOil Price']], trace=False,
                              suppress_warnings=True)
    print("\n",stepwise_fit.summary())

    print("\n***Auto Arima applied on Stock Price + All Exogenous variables*** \n")
    stepwise_fit = auto_arima(stock_data_resample['Close'], X=exogenous_df, trace=False, suppress_warnings=True)
    print("\n",stepwise_fit.summary())
