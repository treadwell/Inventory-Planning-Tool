import pandas as pd
import numpy as np
from scipy import stats
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')



# <codecell>

# set path to working directory

path = '/Users/kbrooks/Documents/MH/Projects/Inventory Planning Tool/'

# set costs

cost = {'perOrder': 80.0, 'WACC': 0.12, 'POD_vmc': 5.0, 'fmc': 1000, 
        'vmc': 2.0, 'lost_margin': 10.00, 'allow_POD':True}

# set forecast parameters

starting_monthly_demand = 1000
number_months = 36
trendPerMonth = -0.05
seasonCoeffs = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
#seasonCoeffs = [0.959482026,0.692569699,0.487806875,0.543161208,0.848077745,0.936798779,1.596854431,1.618086981,1.433374588,0.949500605,0.828435702,1.105851362]

initial_cv = 0.15
per_period_cv = 0.015

# set returns parameters

returns_rate = 0.2
lag = 3

# set other parameters

order_n_months_supply = 9
inv_0 = 0  # starting inventory...not tested
replen_lead_time = 2
target_service_level = 0.99

# program flow

# <codecell>

# utility functions

def dot_sum(*args):
    # element wise sum of list of tuples
    ans = [0]*len(args[0])
    for arg in args:
        for i, point in enumerate(arg):
            ans[i] += point
    return tuple(ans)

# -----------------------
#   Expected lost sales
# -----------------------

def calc_expected_lost_sales(inventory, mean_demand, std_dev_demand):
    shortfall = integrate.quad(lambda x: (x-inventory)*stats.norm.pdf(x, loc=mean_demand, 
                                scale=std_dev_demand), inventory, np.inf)[0]
    return shortfall

#inventory, mean_demand, std_dev_demand = 900, 1000, 1
#print "Shortfall:", calc_expected_lost_sales(inventory, mean_demand, std_dev_demand)
#print "Shortfall:", calc_expected_lost_sales(900, 1000, 100)
#print "Shortfall:", calc_expected_lost_sales(1000, 900, 100)
# Note that this should never be negative.

print  "1. Start with a forecast of demand"

# Start with a forecast of demand

def calc_forecast(starting_monthly_demand, number_months, trendPerMonth, seasonCoeffs):
    month = [i+1 for i in xrange(number_months)]

    level = [starting_monthly_demand] * number_months

    trend = [int(x * (1+trendPerMonth)**i) for i, x in enumerate(level)]

    forecast = [int(x*seasonCoeffs[i % 12]) for i, x in enumerate(trend)]

    return month, forecast

month, forecast = calc_forecast(starting_monthly_demand, number_months, trendPerMonth, seasonCoeffs)


def calc_sd_forecast(initial_cv, per_period_cv, forecast):

    sd_forecast = [(initial_cv + i*per_period_cv) * monthly_forecast for i, monthly_forecast in enumerate(forecast)]

    return sd_forecast

sd_forecast = calc_sd_forecast(initial_cv, per_period_cv, forecast)

def plot_forecast(month, forecast):
    # plot forecast
    plt.plot(month,forecast, linewidth=2.0, label='demand forecast')
    plt.ylabel('Units')
    plt.xlabel('Month')

    plt.title('Forecasted Demand', y=1.05, weight = "bold")
    plt.legend()
    plt.savefig(path + 'output/' + '01_forecast.png',dpi=300)
    plt.draw()

plot_forecast(month, forecast)

print "2. ...and a forecast of returns"

def calc_returns_forecast(number_months, forecast):

    # returns forecast

    returns = [0]*number_months

    for i,x in enumerate(forecast):
        if i < lag:
            returns[i] = 0
        else:
            returns[i] = int(returns_rate*forecast[i-lag])
        #print i
        #print "forecast:", forecast[i]
        #print "returns:", returns[i]

    #print "forecast:", forecast
    return returns

returns = calc_returns_forecast(number_months, forecast)

def plot_returns(month, forecast, returns):
    plt.plot(month,forecast, linewidth=2.0, label='demand forecast')
    plt.plot(month,returns, linewidth=2.0, label='returns forecast')
    plt.ylabel('Units')
    plt.xlabel('Month')

    plt.title('Forecasted Demand and Returns', y=1.05, weight = "bold")
    plt.legend()
    plt.savefig(path + 'output/' + '02_returns.png', dpi=300)
    plt.draw()

plot_returns(month, forecast, returns)

print "3. use planned purchases, demand, and returns to calculate inventory position..."


# revised logic with ROP



def calc_POD_breakeven(cost):
    # note that this should really incorporate WACC
    return cost['fmc']/(cost['POD_vmc']-cost['vmc'])

POD_breakeven = calc_POD_breakeven(cost)

def determine_plan(month, forecast, returns, cost, reorder_point = None, inv_0 = 0):
    if reorder_point == None:
        reorder_point = [0] * len(month)

    starting_inventory = []
    ending_inventory = []
    average_inventory = []
    POD_orders = [0]*len(forecast)
    orders=[0]*len(forecast)

    for i, fcst in enumerate(forecast):
        # calculate starting inventory
        if i == 0:
            starting_inventory.append(inv_0)
        else:
            starting_inventory.append(ending_inventory[i-1])
        # calculate trial ending inventory
        trial_ending_inventory = starting_inventory[i] - fcst + returns[i]
        # if trial ending inventory < ROP, place order
        if trial_ending_inventory < reorder_point[i]:
            # determine order quantity
            # orders = current shortfall + order quantity
            n=order_n_months_supply
            terminal = min(i+n, len(reorder_point)-1)  # used to make sure index doesn't exceed length of list
            orders[i] = sum(forecast[i:i+n])-sum(returns[i:i+n])-starting_inventory[i]+reorder_point[terminal]
            # order POD if the size of the order is too small
            # POD order quantity will be just what is needed in the current month
            if orders[i] < calc_POD_breakeven(cost) and cost['allow_POD'] == True:
                POD_orders[i] = max(forecast[i]-starting_inventory[i]-returns[i],0)
                orders[i] = 0
        else:
            orders[i] = 0
        # calculate ending inventory from inventory balance equation
        ending_inventory.append(starting_inventory[i] - forecast[i] + returns[i]
                                    + orders[i] + POD_orders[i])
        #print i
        #print "orders:", orders[i]
        #print "POD_orders:", POD_orders[i]
        #print "start_inv:", starting_inventory[i]
        #print "end_inv:", ending_inventory[i]

        # calculate average inventory in order to calculate period carrying cost
        average_inventory.append((starting_inventory[i]+ending_inventory[i])/2)

    return orders, POD_orders, starting_inventory, ending_inventory, average_inventory

orders, POD_orders, start_inv, end_inv, avg_inv = determine_plan(month, forecast, returns, cost, [0]* number_months)


def plot_end_inv(month, end_inv):
    plt.plot(month,end_inv, linewidth=2.0, color = "g",label='ending inventory')
    d = np.array([0]*len(forecast))
    plt.fill_between(month, d, end_inv, where=end_inv>=d, interpolate=True, facecolor='green')
    plt.ylabel('Units')
    plt.xlabel('Month')

    plt.title('Planned Ending Inventory Position', y=1.05, weight = "bold")
    plt.legend()
    plt.savefig(path + 'output/' + '03_inventory.png', dpi=300)

    plt.draw()

plot_end_inv(month, end_inv)

print "4. ...yielding a lifetime expected cost."


# Fixed manufacturing cost - fmc
# per order cost - perOrder
# variable manufacturing cost - vmc
# variable manufacturing cost of POD - POD_vmc
# WACC for pricing inventory carrying cost - WACC

# print "Start Inventory", start_inv
# print "Forecast:", forecast
# print "Forecast SD:", sd_forecast

def calc_costs(forecast, sd_forecast, orders, POD_orders, avg_inv, start_inv, cost):
    FMC = [cost['fmc'] + cost['perOrder'] if round(order) else 0 for order in orders]
    VMC = [cost['vmc'] * order for order in orders]
    POD_VMC = [cost['POD_vmc'] * POD_order for POD_order in POD_orders]
    umc = (sum(FMC)+sum(VMC)+sum(POD_VMC)) / (sum(orders)+sum(POD_orders)) # approximation - should be a list
    carry_stg_cost = [float(cost['WACC']) / 12 * month_avg * umc for month_avg in avg_inv] 
    lost_sales_expected = [cost['lost_margin']*int(calc_expected_lost_sales(inv+ret+order+POD_order, dem, sd)) 
                           for inv, ret, order, POD_order, dem, sd in 
                           zip(start_inv, returns, orders, POD_orders, forecast, sd_forecast)]

    return FMC, VMC, POD_VMC, umc, carry_stg_cost, lost_sales_expected

FMC, VMC, POD_VMC, umc, carry_stg_cost, exp_lost_sales_cost = calc_costs(forecast, sd_forecast, orders, POD_orders, avg_inv, start_inv, cost)

print "Expected lost sales (units):", [int(lost_dollars/cost['lost_margin']) for lost_dollars in exp_lost_sales_cost]

print "------------------------------"
print "Total FMC:", locale.currency(sum(FMC), grouping=True )
print "Total VMC:", locale.currency(sum(VMC), grouping=True )
print "Total POD VMC:", locale.currency(sum(POD_VMC), grouping=True )
print "umc:", locale.currency(umc, grouping=True )
print "Total carrying / storage cost:", locale.currency(sum(carry_stg_cost), grouping=True )
print "Total expected lost sales:", locale.currency(sum(exp_lost_sales_cost), grouping=True )


# plot results

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

    plt.savefig(path + 'output/' + '04_plan_cost.png', dpi=300, 
            bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.draw()


plot_cost_bars(FMC, VMC, POD_VMC, carry_stg_cost)

print  "5. But forecasts are wrong..."


def calc_demand(forecast, sd_forecast):

    ##############################
    np.random.seed(5)
    ############################## 3, 4, 

    demand = [max(round(np.random.normal(fcst, sd)),0) for fcst, sd in zip(forecast, sd_forecast)]
    lower_CI = [fcst - 1.96 * sd for fcst, sd in zip(forecast, sd_forecast)]
    upper_CI = [fcst + 1.96 * sd for fcst, sd in zip(forecast, sd_forecast)]
    return demand, lower_CI, upper_CI

demand, lower_CI, upper_CI = calc_demand(forecast, sd_forecast)

def plot_demand(month, forecast, demand, lower_CI, upper_CI):
    plt.plot(month,forecast, linewidth=2.0, label='demand forecast')
    plt.plot(month,demand, linewidth=2.0, label='actual demand')
    plt.plot(month,upper_CI, linewidth=0.5, label='95% Conf Interval', color="blue")
    plt.plot(month,lower_CI, linewidth=0.5, color="blue")
    plt.ylabel('Units')
    plt.xlabel('Month')
    plt.title('Demand: Actual vs. Forecast', y=1.05, weight = "bold")
    plt.legend()

    plt.savefig(path + 'output/' + '05_forecast_error.png', dpi=300)

    plt.draw()


print  "6. ...leading to stockouts with lost sales and expediting..."


def inv_from_demand(demand, orders, POD_orders, returns):

    start_inv_act = []
    start_inv_posn_act = []
    end_inv_act = []
    end_inv_posn_act = []
    avg_inv_act = []

    for i, order in enumerate(orders):
        # calculate starting inventory
        if i == 0:
            start_inv_act.append(max(0,inv_0))  # eventually replace this with an optional input
            start_inv_posn_act.append(0)
        else:
            start_inv_act.append(end_inv_act[i-1])
            start_inv_posn_act.append(end_inv_posn_act[i-1])
        
        # calculate ending inventory from inventory balance equation
        end_inv_act.append(max(0,start_inv_act[i] - demand[i] + orders[i] + 
                               POD_orders[i] + returns[i]))
        end_inv_posn_act.append(start_inv_posn_act[i] - demand[i] + orders[i] + 
                                POD_orders[i] + returns[i])

        # calculate average inventory in order to calculate period carrying cost
        avg_inv_act.append((start_inv_act[i]+end_inv_act[i])/2)
    
    return end_inv_posn_act, avg_inv_act

end_inv_posn_act, avg_inv_act = inv_from_demand(demand, orders, POD_orders, returns)

#print month
#print "forecast:", forecast
#print "demand:", demand
#print "returns:", returns
#print "orders:", orders
#print "POD_orders:", POD_orders
#print "start_inv:", start_inv
#print "end_inv:", end_inv
#print "avg_inv:", avg_inv
#print end_inv_posn_act

def plot_end_inv_posn_act(month, end_inv_posn_act):
    #print inventory_plot

    ''' combine this into the previous inventory position plot function'''

    plt.plot(month,end_inv_posn_act, linewidth=2.0, 
             label='end inventory position', color='green')
    d = np.array([0]*len(forecast))
    plt.fill_between(month, d, end_inv_posn_act, where=end_inv_posn_act<=d, interpolate=True, facecolor='red')
    plt.fill_between(month, d, end_inv_posn_act, where=end_inv_posn_act>=d, interpolate=True, facecolor='green')
    plt.ylabel('Units')
    plt.xlabel('Month')
    plt.title('Actual Ending Inventory Position', y=1.05, weight = "bold")
    plt.legend()

    plt.savefig(path + 'output/' + '06_inventory_posn.png', dpi=300)
    plt.draw()

plot_end_inv_posn_act(month, end_inv_posn_act)

print "7. ...and additional costs."


print "Total actual FMC:", sum(FMC)
print "Total actual VMC:", sum(VMC)
print "Total actual POD VMC:", sum(POD_VMC)
print "actual umc:", umc
print "Total actual carrying / storage cost:", sum(carry_stg_cost)
print "Total expected lost sales:", sum(exp_lost_sales_cost)


# ----------------------

def plot_cost_bars_2(FMC, VMC, POD_VMC, carry_stg_cost):

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

    plt.savefig(path + 'output/' + '07_lost_sale_cost.png', dpi=300, 
            bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.draw()

plot_cost_bars_2(FMC, VMC, POD_VMC, carry_stg_cost)

# *	Customers may get stock from elsewhere
# *	Customers may cancel the order and order again
# *	Customers may backorder until you're back in stock
# *	The cost of the backorder can also vary significantly
#     *	proportional to lifetime value if it's an adoption
#     *	Loss of marginal cost early in the life
#     *	etc.
# *	most estimates put the loss at around 7% of the units
# 
# Note there are many things that will keep you from actually having to incur this cost:


print "8. The usual approach to avoid lost sales is to carry safety stock"



# develop plan with ROPs loaded

def calc_reorder_points(target_service_level, replen_lead_time, sd_forecast):
    # ROP = replen_lead_time * fcst + 1.96 *(replen_lead_time*sd**2)**0.5

    service_multiplier = stats.norm.ppf(target_service_level, loc=0, scale=1)

    #reorder_point = [int(replen_lead_time*fcst +service_multiplier*(replen_lead_time*sd**2)**0.5) 
    #                 for fcst, sd in zip(forecast,sd_forecast)]

    reorder_point = [int(service_multiplier*(replen_lead_time)**2*sd) for sd in sd_forecast]

    return reorder_point

reorder_point = calc_reorder_points(target_service_level, replen_lead_time, sd_forecast)
# FMC, VMC, POD_VMC, umc, carry_stg_cost, exp_lost_sales_cost = 
# calc_costs(forecast, sd_forecast, orders, POD_orders, avg_inv, end_inv, cost)
orders_ss, POD_orders_ss, start_inv_ss, end_inv_ss, avg_inv_ss = determine_plan(month, forecast, returns, cost, reorder_point)

FMC_ss, VMC_ss, POD_VMC_ss, umc_ss, carry_stg_cost_ss, exp_lost_sales_cost_ss = calc_costs(forecast, sd_forecast, orders_ss, POD_orders_ss, avg_inv_ss, start_inv_ss, cost)


determine_plan

print "Expected lost sales (units) no SS:", int(sum(exp_lost_sales_cost)/cost['lost_margin'])
print "Expected lost sales (units) with SS:", int(sum(exp_lost_sales_cost_ss)/cost['lost_margin'])




def plot_end_inv_2(month, end_inv):

    '''Combine this with other end_inv plot functions'''

    plt.plot(month,end_inv, linewidth=2.0, label='end inventory position', 
             color ='green')
    d = np.array([0]*number_months)
    plt.fill_between(month, d, end_inv_ss, where=end_inv<=d, 
                     interpolate=True, facecolor='red')
    plt.fill_between(month, d, end_inv_ss, where=end_inv>=d, 
                     interpolate=True, facecolor='green')
    plt.ylabel('Units')
    plt.xlabel('Month')
    plt.title('Expected Ending Inventory Position', y=1.05, weight = "bold")
    plt.legend()

    plt.savefig(path + 'output/' + '08_safety_stock.png', dpi=300)
    plt.draw()

plot_end_inv_2(month, end_inv_ss)

# <headingcell level=1>

print "9. Another is to use POD."

# <codecell>

	
#	 - it looks like lost sales, but is cheaper
#  put the lost sales numbers into the POD vector and recalc positions and costs
# Use the lost sales line chart with a note about the cost of a lost sale
#  revalued_lost_sales = [0]*len(forecast)
#lost_sales_as_POD = [-posn * cost['POD_vmc'] if posn < 0 else 0 for posn in end_inv_posn_act]

#lost_sales_as_POD = [cost['POD_vmc']*calc_expected_lost_sales(inv, dem, sd) for inv, dem, sd in zip(end_inv, forecast, sd_forecast)]
def calc_lost_sales_as_POD(cost, start_inv, returns, orders, POD_orders, forecast, sd_forecast):
    lost_sales_as_POD = [cost['POD_vmc']*int(calc_expected_lost_sales(inv+ret+order+POD_order, dem, sd)) 
                               for inv, ret, order, POD_order, dem, sd in 
                               zip(start_inv, returns, orders, POD_orders, forecast, sd_forecast)]
    return lost_sales_as_POD

lost_sales_as_POD = calc_lost_sales_as_POD(cost, start_inv, returns, orders, POD_orders, forecast, sd_forecast)

print "Total actual FMC:", sum(FMC)
print "Total actual VMC:", sum(VMC)
print "Total actual POD VMC:", sum(POD_VMC)
print "actual umc:", umc
print "Total actual carrying / storage cost:", sum(carry_stg_cost)
print "Total actual lost sales (as POD):", sum(lost_sales_as_POD)

def plot_end_inv_3(month, end_inv_posn_act):

    # this should be the same as the lost sale area chart

    plt.plot(month,end_inv_posn_act, linewidth=2.0, 
             label='end inventory position', color='green')
    d = np.array([0]*len(forecast))
    plt.fill_between(month, d, end_inv_posn_act, where=end_inv_posn_act<=d, interpolate=True, facecolor='blue')
    plt.fill_between(month, d, end_inv_posn_act, where=end_inv_posn_act>=d, interpolate=True, facecolor='green')
    plt.ylabel('Units')
    plt.xlabel('Month')
    plt.title('Actual Ending Inventory Position', y=1.05, weight = "bold")
    plt.legend()

    plt.savefig(path + 'output/' + '09_POD_posn.png', dpi=300)
    plt.draw()

plot_end_inv_3(month, end_inv_posn_act)

print "10. POD is best."


# Show stacked bars of the three alternatives or three panels


print  "\nLost sales alternative"

# <codecell>

print "Total actual FMC:", locale.currency( sum(FMC), grouping=True )
print "Total actual VMC:", locale.currency( sum(VMC), grouping=True )
print "Total actual POD VMC:", locale.currency( sum(POD_VMC), grouping=True )
print "actual umc:", locale.currency( umc, grouping=True )
print "Total actual carrying / storage cost:", locale.currency( sum(carry_stg_cost), grouping=True )
print "Total expected lost sales:", locale.currency( sum(exp_lost_sales_cost), grouping=True )
print "Grand total with lost sales:", locale.currency( sum(FMC)+sum(VMC)+sum(POD_VMC)+sum(carry_stg_cost)+
                                                      sum(exp_lost_sales_cost), grouping=True )

# <headingcell level=2>

print "\nSafety stock alternative"

# <codecell>

# change to capture new FMC, VMC etc w POD


print "Total actual FMC:", locale.currency( sum(FMC_ss), grouping=True )
print "Total actual VMC:", locale.currency( sum(VMC_ss), grouping=True )
print "Total actual POD VMC:", locale.currency( sum(POD_VMC_ss), grouping=True )
print "actual umc:", locale.currency( umc, grouping=True )
print "Total actual carrying / storage cost:", locale.currency( sum(carry_stg_cost_ss), grouping=True )
print "Total expected lost sales:", locale.currency( sum(exp_lost_sales_cost_ss), grouping=True )
print "Grand total with safety stock:", locale.currency( sum(FMC_ss)+sum(VMC_ss)+sum(POD_VMC_ss)+sum(carry_stg_cost_ss) + sum(exp_lost_sales_cost_ss), grouping=True )

# extra costs come from printing more



print "\nPOD alternative"


print "Total actual FMC:", locale.currency( sum(FMC), grouping=True )
print "Total actual VMC:", locale.currency( sum(VMC), grouping=True )
print "Total actual POD VMC:", locale.currency( sum(POD_VMC), grouping=True )
print "actual umc:", locale.currency( umc, grouping=True )
print "Total actual carrying / storage cost:", locale.currency( sum(carry_stg_cost), grouping=True )
print "Total actual lost sales (as POD):", locale.currency( sum(lost_sales_as_POD), grouping=True )
print "Grand total with lost sales (as POD):", locale.currency( sum(FMC)+sum(VMC)+sum(POD_VMC)+sum(carry_stg_cost)+sum(lost_sales_as_POD), grouping=True )

# <codecell>

#  Need a stacked bar here
#cost_plot = ggplot(aes(x='Cost Components'), data = plot_df) + \
#    geom_bar(weight = costs) + \
#    ggtitle("10. POD offers the best cost profile.")  + \
#    ylab("Lifetime Cost (dollars)") + \
#    scale_y_continuous(labels='comma')
    
#ggsave(cost_plot, "10_cost_comparison.png")

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

    plt.savefig(path + 'output/' + '10_final_cost_comparison.png', dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.draw()

plot_cost_bars_final(FMC, FMC_ss, VMC, VMC_ss, POD_VMC, POD_VMC_ss, carry_stg_cost, carry_stg_cost_ss, exp_lost_sales_cost, exp_lost_sales_cost_ss, lost_sales_as_POD)


# <codecell>

test_columns = ["FMC", "VMC", "VMC POD", "Carry/Stg", "Lost Sales", "LS as POD"]
test_rows = ["Planned", "Act w Lost Sales", "Act w SS", "Act w POD"]


test_df = pd.DataFrame(0,index=test_rows, columns=test_columns)
test_df = test_df.set_value("Planned", 'FMC', sum(FMC))
test_df = test_df.set_value("Planned", 'VMC', sum(VMC))
test_df = test_df.set_value("Planned", 'VMC POD', sum(POD_VMC))
test_df = test_df.set_value("Planned", 'Carry/Stg', sum(carry_stg_cost))
test_df = test_df.set_value("Planned", 'Lost Sales', 0)
test_df = test_df.set_value("Planned", 'LS as POD', 0.)

test_df = test_df.set_value("Act w Lost Sales", 'FMC', sum(FMC))
test_df = test_df.set_value("Act w Lost Sales", 'VMC', sum(VMC))
test_df = test_df.set_value("Act w Lost Sales", 'VMC POD', sum(POD_VMC))
test_df = test_df.set_value("Act w Lost Sales", 'Carry/Stg', sum(carry_stg_cost))
test_df = test_df.set_value("Act w Lost Sales", 'Lost Sales', sum(exp_lost_sales_cost))
test_df = test_df.set_value("Act w Lost Sales", 'LS as POD', 0.)

test_df = test_df.set_value("Act w SS", 'FMC', sum(FMC_ss))
test_df = test_df.set_value("Act w SS", 'VMC', sum(VMC_ss))
test_df = test_df.set_value("Act w SS", 'VMC POD', sum(POD_VMC_ss))
test_df = test_df.set_value("Act w SS", 'Carry/Stg', sum(carry_stg_cost_ss))
test_df = test_df.set_value("Act w SS", 'Lost Sales', sum(exp_lost_sales_cost_ss))
test_df = test_df.set_value("Act w SS", 'LS as POD', 0.)

test_df = test_df.set_value("Act w POD", 'FMC', sum(FMC))
test_df = test_df.set_value("Act w POD", 'VMC', sum(VMC))
test_df = test_df.set_value("Act w POD", 'VMC POD', sum(POD_VMC))
test_df = test_df.set_value("Act w POD", 'Carry/Stg', sum(carry_stg_cost))
test_df = test_df.set_value("Act w POD", 'Lost Sales', 0.)
test_df = test_df.set_value("Act w POD", 'LS as POD', sum(lost_sales_as_POD))

print test_df

test_df.plot(kind='barh', stacked=True, legend=False)
plt.legend(loc='upper center', bbox_to_anchor=(0.95, 1.05), ncol=1, fancybox=True, shadow=True)
plt.title('Expected Lifetime Cost', y=1.05, weight = "bold")

#ax = pandas.DataFrame.from_records(d,columns=h)
#ax.plot()
#fig = matplotlib.pyplot.gcf()
#fig.savefig('graph.png')

# <headingcell level=1>

print "11. What does a POD lifecycle look like?"

# <codecell>

# Inventory position

# <codecell>

# Lifetime cost versus base

# <markdowncell>

# *	low volume titles
# *	samples
# *	End-of-life
# *	Custom
# *	Illustration

# <headingcell level=1>

print "12. What does a title with dramatic seasonality look like?"

# <codecell>

# Demand forecast

# <codecell>

# Inventory position

# <headingcell level=1>

print "13. What happens with systematic over-forecasting?"

# <codecell>

# Actual versus plan with bias

# <codecell>

# lifetime costs versus base

# <headingcell level=1>

print "14. What if we're really terrible at forecasting?"

# <codecell>

# Inventory position with higher SS levels

# <codecell>

# Lifetime costs versus base

# <headingcell level=1>

print "15. What if we use Economic Order Quantity (EOQ) or other techniques?"

# <codecell>

# Lifetime costs versus base

# <headingcell level=1>

print "16. What if we dramatically reduce our print lead times?"

# <codecell>

# Inventory position with lower SS levels

# <codecell>

# Lifetime costs versus base

# <headingcell level=1>

print "15. To do this you need infrastructure:"

# *  file mgmt
# *  printers (conventional and POD)
# *  links between OP, MM, and S&OP systems
# *  make this an illustration

# <headingcell level=2>

# To Do list

# <markdowncell>

# * Fixes
#     * fix currency formatting on graphs
#     * fix stacked bar charts
#         *  turn them horizontal
#         *  make sure they actually work
# * Rationalize variables and functions
#     * plan vs actual
#     * lost sales vs. safety stock vs. POD
# * Sensitivity
#     * Book type (get different costs and demand profiles)
#     * Volumes (forecasts, returns)
#     * Bias on forecast
#     * Actual returns with a bias and a variance
#     * Return scrap rate
# * Additions
#     * Expected cost of lost sales to base costs
#         * probability that demand will exceed inventory in period i
#         * by how much?
#         * what's the value?
#     * Obso metric at the end
#     * Actual returns
#     * lifetime demand allocated across periods
#     * express CV as a function of horizon:  proportional to sqrt(hzn)
#     * dynamic planning - recalc of order quantities
#     * planning with expected cost of stockout

