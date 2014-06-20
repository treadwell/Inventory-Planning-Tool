import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy as np


# utility functions

def dot_sum(*args):
    # element wise sum of list of tuples
    ans = [0]*len(args[0])
    for arg in args:
        for i, point in enumerate(arg):
            ans[i] += point
    return tuple(ans)


def plot_forecast(month, forecast):
    # plot forecast
    plt.plot(month,forecast, linewidth=2.0, label='demand forecast')
    plt.ylabel('Units')
    plt.xlabel('Month')

    plt.title('Forecasted Demand', y=1.05, weight = "bold")
    plt.legend()
    plt.savefig('../output/' + '01_forecast.png',dpi=300)
    plt.draw()

def plot_returns(month, forecast, returns):
    plt.plot(month,forecast, linewidth=2.0, label='demand forecast')
    plt.plot(month,returns, linewidth=2.0, label='returns forecast')
    plt.ylabel('Units')
    plt.xlabel('Month')

    plt.title('Forecasted Demand and Returns', y=1.05, weight = "bold")
    plt.legend()
    plt.savefig('../output/' + '02_returns.png', dpi=300)
    plt.draw()

def plot_end_inv(month, end_inv):
    plt.plot(month,end_inv, linewidth=2.0, color = "g",label='ending inventory')
    d = np.array([0]*len(month))
    plt.fill_between(month, d, end_inv, where=end_inv>=d, interpolate=True, facecolor='green')
    plt.ylabel('Units')
    plt.xlabel('Month')

    plt.title('Planned Ending Inventory Position', y=1.05, weight = "bold")
    plt.legend()
    plt.savefig('../output/' + '03_inventory.png', dpi=300)

    plt.draw()

def plot_cost_bars(FMC, VMC, POD_VMC, carry_stg_cost):

    N = 4
    FMC_plot   = (sum(FMC), 0, 0, 0)
    VMC_plot = (sum(VMC), 0, 0, 0)
    PODVMC_plot     = (sum(POD_VMC), 0, 0, 0)
    carry_storage_plot   = (sum(carry_stg_cost), 0, 0, 0)
    lost_sales_plot = (0, 0, 0, 0)
               
    POD_plot = (0,0,0,0)
               
    ind = np.arange(N)    # the x locations for the groups
    width = 0.45       # the width of the bars: can also be len(x) sequence



    p1 = plt.barh(ind, FMC_plot, width, color='b')
    p2 = plt.barh(ind, VMC_plot, width, color='g', left=FMC_plot)
    p3 = plt.barh(ind, PODVMC_plot, width, color = 'r', left = dot_sum(VMC_plot, FMC_plot))
    p4 = plt.barh(ind, carry_storage_plot, width, color='c', left =
                 dot_sum(VMC_plot, FMC_plot, PODVMC_plot))
    p5 = plt.barh(ind, lost_sales_plot, width, color='m',
                 left = dot_sum(VMC_plot, FMC_plot, PODVMC_plot, carry_storage_plot))
    p6 = plt.barh(ind, POD_plot, width, color='y',
                 left = dot_sum(VMC_plot, FMC_plot, PODVMC_plot, carry_storage_plot, lost_sales_plot))

    plt.xlabel('Cost ($)')
    plt.title('Expected Lifetime Cost', y=1.05, weight = "bold")
    plt.yticks(ind+width/2., ('Plan', '', '', '') )

    lgd = plt.legend( (p1[0], p2[0],p3[0],p4[0], p5[0], p6[0]), 
               ('Fixed Mfg', 'Variable Mfg', 'POD Variable', 'Carry/Storage', 'Lost Sales', 
                'POD Safety'), loc='upper center', bbox_to_anchor=(0.95, 1.05), 
               ncol=1, fancybox=True, shadow=True)
    plt.grid()

    plt.savefig('../output/' + '04_plan_cost.png', dpi=300, 
            bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.draw()

def plot_demand(month, forecast, demand, lower_CI, upper_CI):
    plt.plot(month,forecast, linewidth=2.0, label='demand forecast')
    plt.plot(month,demand, linewidth=2.0, label='actual demand')
    plt.plot(month,upper_CI, linewidth=0.5, label='95% Conf Interval', color="blue")
    plt.plot(month,lower_CI, linewidth=0.5, color="blue")
    plt.ylabel('Units')
    plt.xlabel('Month')
    plt.title('Demand: Actual vs. Forecast', y=1.05, weight = "bold")
    plt.legend()

    plt.savefig('../output/' + '05_forecast_error.png', dpi=300)

    plt.draw()

def plot_end_inv_posn_act(month, end_inv_posn_act):
    #print inventory_plot

    ''' combine this into the previous inventory position plot function'''

    plt.plot(month,end_inv_posn_act, linewidth=2.0, 
             label='end inventory position', color='green')
    d = np.array([0]*len(month))
    plt.fill_between(month, d, end_inv_posn_act, where=end_inv_posn_act<=d, interpolate=True, facecolor='red')
    plt.fill_between(month, d, end_inv_posn_act, where=end_inv_posn_act>=d, interpolate=True, facecolor='green')
    plt.ylabel('Units')
    plt.xlabel('Month')
    plt.title('Actual Ending Inventory Position', y=1.05, weight = "bold")
    plt.legend()

    plt.savefig('../output/' + '06_inventory_posn.png', dpi=300)
    plt.draw()

def plot_cost_bars_2(FMC, VMC, POD_VMC, carry_stg_cost, exp_lost_sales_cost):

    ''' Combine this into the previous plot_cost_bars function'''
    N = 4
    FMC_plot   = (sum(FMC), sum(FMC), 0,0)
    VMC_plot = (sum(VMC), sum(VMC), 0, 0)
    PODVMC_plot     = (sum(POD_VMC), sum(POD_VMC), 0, 0)
    carry_storage_plot   = (sum(carry_stg_cost), sum(carry_stg_cost), 0, 0)
    lost_sales_plot = (0, sum(exp_lost_sales_cost), 0, 0)
               
    POD_plot = (0,0,0, 0)
               
    ind = np.arange(N)    # the x locations for the groups
    width = 0.45       # the width of the bars: can also be len(x) sequence



    p1 = plt.barh(ind, FMC_plot, width, color='b')
    p2 = plt.barh(ind, VMC_plot, width, color='g', left=FMC_plot)
    p3 = plt.barh(ind, PODVMC_plot, width, color = 'r', left = dot_sum(VMC_plot, FMC_plot))
    p4 = plt.barh(ind, carry_storage_plot, width, color='c', left =
                 dot_sum(VMC_plot, FMC_plot, PODVMC_plot))
    p5 = plt.barh(ind, lost_sales_plot, width, color='m',
                 left = dot_sum(VMC_plot, FMC_plot, PODVMC_plot, carry_storage_plot))
    p6 = plt.barh(ind, POD_plot, width, color='y',
                 left = dot_sum(VMC_plot, FMC_plot, PODVMC_plot, carry_storage_plot, lost_sales_plot))

    plt.xlabel('Cost ($)')

    plt.title('Expected Lifetime Cost', y=1.05, weight = "bold")
    plt.yticks(ind+width/2., ('Plan', 'Act w\nLost Sales', '', '') )

    lgd = plt.legend( (p1[0], p2[0],p3[0],p4[0], p5[0], p6[0]), 
               ('Fixed Mfg', 'Variable Mfg', 'POD Variable', 'Carry/Storage', 'Lost Sales', 
                'POD Safety'), loc='upper center', bbox_to_anchor=(0.95, 1.05), ncol=1, fancybox=True, shadow=True)
    plt.grid()

    plt.savefig('../output/' + '07_lost_sale_cost.png', dpi=300, 
            bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.draw()

def plot_end_inv_2(month, end_inv, end_inv_ss):

    '''Combine this with other end_inv plot functions'''

    plt.plot(month,end_inv, linewidth=2.0, label='end inventory position', 
             color ='green')
    d = np.array([0]*len(month))
    plt.fill_between(month, d, end_inv_ss, where=end_inv<=d, 
                     interpolate=True, facecolor='red')
    plt.fill_between(month, d, end_inv_ss, where=end_inv>=d, 
                     interpolate=True, facecolor='green')
    plt.ylabel('Units')
    plt.xlabel('Month')
    plt.title('Expected Ending Inventory Position', y=1.05, weight = "bold")
    plt.legend()

    plt.savefig('../output/' + '08_safety_stock.png', dpi=300)
    plt.draw()

def plot_end_inv_3(month, end_inv_posn_act):

    # this should be the same as the lost sale area chart

    plt.plot(month,end_inv_posn_act, linewidth=2.0, 
             label='end inventory position', color='green')
    d = np.array([0]*len(month))
    plt.fill_between(month, d, end_inv_posn_act, where=end_inv_posn_act<=d, interpolate=True, facecolor='blue')
    plt.fill_between(month, d, end_inv_posn_act, where=end_inv_posn_act>=d, interpolate=True, facecolor='green')
    plt.ylabel('Units')
    plt.xlabel('Month')
    plt.title('Actual Ending Inventory Position', y=1.05, weight = "bold")
    plt.legend()

    plt.savefig('../output/' + '09_POD_posn.png', dpi=300)
    plt.draw()

def plot_cost_bars_final(FMC, FMC_ss, VMC, VMC_ss, POD_VMC, POD_VMC_ss, carry_stg_cost, carry_stg_cost_ss, exp_lost_sales_cost, exp_lost_sales_cost_ss, lost_sales_as_POD):
    '''Combine with other bar charts'''


    N = 4
    FMC_plot   = (sum(FMC), sum(FMC), sum(FMC_ss), sum(FMC))
    VMC_plot = (sum(VMC), sum(VMC), sum(VMC_ss), sum(VMC))
    PODVMC_plot     = (sum(POD_VMC), sum(POD_VMC), sum(POD_VMC_ss), sum(POD_VMC))
    carry_storage_plot   = (sum(carry_stg_cost), sum(carry_stg_cost), 
                            sum(carry_stg_cost_ss), sum(carry_stg_cost))
    lost_sales_plot = (0, sum(exp_lost_sales_cost), sum(exp_lost_sales_cost_ss), 0)
               
    POD_plot = (0,0,0, sum(lost_sales_as_POD))
               
    ind = np.arange(N)    # the x locations for the groups
    width = 0.45       # the width of the bars: can also be len(x) sequence



    p1 = plt.barh(ind, FMC_plot, width, color='b')
    p2 = plt.barh(ind, VMC_plot, width, color='g', left=FMC_plot)
    p3 = plt.barh(ind, PODVMC_plot, width, color = 'r', left = dot_sum(VMC_plot, FMC_plot))
    p4 = plt.barh(ind, carry_storage_plot, width, color='c', left =
                 dot_sum(VMC_plot, FMC_plot, PODVMC_plot))
    p5 = plt.barh(ind, lost_sales_plot, width, color='m',
                 left = dot_sum(VMC_plot, FMC_plot, PODVMC_plot, carry_storage_plot))
    p6 = plt.barh(ind, POD_plot, width, color='y',
                 left = dot_sum(VMC_plot, FMC_plot, PODVMC_plot, carry_storage_plot, lost_sales_plot))

    plt.xlabel('Cost ($)')

    plt.title('Expected Lifetime Cost', y=1.05, weight = "bold")
    plt.yticks(ind+width/2., ('Plan', 'Act w\nLost Sales', 'Act w SS', 'Act w POD') )

    lgd = plt.legend( (p1[0], p2[0],p3[0],p4[0], p5[0], p6[0]), 
               ('Fixed Mfg', 'Variable Mfg', 'POD Variable', 'Carry/Storage', 'Lost Sales', 
                'POD Safety'), loc='upper center', bbox_to_anchor=(1.1, 1.05), 
               ncol=1, fancybox=True, shadow=True)
    plt.grid()

    plt.savefig('../output/' + '10_final_cost_comparison.png', dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.draw()

