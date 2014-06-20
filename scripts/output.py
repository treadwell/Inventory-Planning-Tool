


def plot_forecast(month, forecast):
    # plot forecast
    plt.plot(month,forecast, linewidth=2.0, label='demand forecast')
    plt.ylabel('Units')
    plt.xlabel('Month')

    plt.title('Forecasted Demand', y=1.05, weight = "bold")
    plt.legend()
    plt.savefig(path + 'output/' + '01_forecast.png',dpi=300)
    plt.draw()