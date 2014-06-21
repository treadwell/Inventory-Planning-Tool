import output as o
import pandas as pd
import numpy as np
from scipy import stats
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


# set path to working directory

path = '/Users/kbrooks/Documents/MH/Projects/Inventory Planning Tool/'


# -----------------------
#   Expected lost sales
# -----------------------

def calc_expected_lost_sales(inventory, mean_demand, std_dev_demand):
    '''Utilizes loss function to calculate lost sales'''
    shortfall = integrate.quad(lambda x: (x-inventory)*stats.norm.pdf(x, loc=mean_demand, 
                                scale=std_dev_demand), inventory, np.inf)[0]
    return shortfall

#inventory, mean_demand, std_dev_demand = 900, 1000, 1
#print "Shortfall:", calc_expected_lost_sales(inventory, mean_demand, std_dev_demand)
#print "Shortfall:", calc_expected_lost_sales(900, 1000, 100)
#print "Shortfall:", calc_expected_lost_sales(1000, 900, 100)
# Note that this should never be negative.



# Start with a forecast of demand

def calc_forecast(starting_monthly_demand, number_months, trendPerMonth, seasonCoeffs):
    '''Returns two lists holding a sequential month number and a second with
    monthly forecasts based on level, trend and seasonality.  This is not 
    normalized to equal a particular liftetime sales target.'''
    
    month = [i+1 for i in xrange(number_months)]

    level = [starting_monthly_demand] * number_months

    trend = [int(x * (1+trendPerMonth)**i) for i, x in enumerate(level)]

    forecast = [int(x*seasonCoeffs[i % 12]) for i, x in enumerate(trend)]

    return month, forecast



def calc_sd_forecast(initial_cv, per_period_cv, forecast):
    '''Returns a list with forecast standard deviation.  It is calculated
    based on a starting coefficient of variation with an incremental cv for 
    every successive month in the future.'''

    sd_forecast = [(initial_cv + i*per_period_cv) * monthly_forecast for i, monthly_forecast in enumerate(forecast)]

    return sd_forecast


def calc_returns_forecast(forecast, returns_rate, lag):

    '''Calculate a returns forecast based by applying a percentage and lag to demand forecast'''

    returns = [0]*len(forecast)

    for i,x in enumerate(forecast):
        if i < lag:
            returns[i] = 0
        else:
            returns[i] = int(returns_rate*forecast[i-lag])

    return returns


def calc_POD_breakeven(cost):
    # note that this should really incorporate WACC
    return cost['fmc']/(cost['POD_vmc']-cost['vmc'])


def determine_plan(month, forecast, returns, cost, order_n_months_supply, reorder_point = None, inv_0 = 0):
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





# Fixed manufacturing cost - fmc
# per order cost - perOrder
# variable manufacturing cost - vmc
# variable manufacturing cost of POD - POD_vmc
# WACC for pricing inventory carrying cost - WACC

# print "Start Inventory", start_inv
# print "Forecast:", forecast
# print "Forecast SD:", sd_forecast

def calc_costs(forecast, sd_forecast, returns, orders, POD_orders, avg_inv, start_inv, cost):
    FMC = [cost['fmc'] + cost['perOrder'] if round(order) else 0 for order in orders]
    VMC = [cost['vmc'] * order for order in orders]
    POD_VMC = [cost['POD_vmc'] * POD_order for POD_order in POD_orders]
    umc = (sum(FMC)+sum(VMC)+sum(POD_VMC)) / (sum(orders)+sum(POD_orders)) # approximation - should be a list
    carry_stg_cost = [float(cost['WACC']) / 12 * month_avg * umc for month_avg in avg_inv] 
    lost_sales_expected = [cost['lost_margin']*int(calc_expected_lost_sales(inv+ret+order+POD_order, dem, sd)) 
                           for inv, ret, order, POD_order, dem, sd in 
                           zip(start_inv, returns, orders, POD_orders, forecast, sd_forecast)]

    return FMC, VMC, POD_VMC, umc, carry_stg_cost, lost_sales_expected


def calc_demand(forecast, sd_forecast):

    ##############################
    np.random.seed(5)
    ############################## 3, 4, 

    demand = [max(round(np.random.normal(fcst, sd)),0) for fcst, sd in zip(forecast, sd_forecast)]
    lower_CI = [fcst - 1.96 * sd for fcst, sd in zip(forecast, sd_forecast)]
    upper_CI = [fcst + 1.96 * sd for fcst, sd in zip(forecast, sd_forecast)]
    return demand, lower_CI, upper_CI



def inv_from_demand(demand, orders, POD_orders, returns, inv_0 = 0):

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


# ----------------------



# develop plan with ROPs loaded

def calc_reorder_points(target_service_level, replen_lead_time, sd_forecast):
    # ROP = replen_lead_time * fcst + 1.96 *(replen_lead_time*sd**2)**0.5

    service_multiplier = stats.norm.ppf(target_service_level, loc=0, scale=1)

    #reorder_point = [int(replen_lead_time*fcst +service_multiplier*(replen_lead_time*sd**2)**0.5) 
    #                 for fcst, sd in zip(forecast,sd_forecast)]

    reorder_point = [int(service_multiplier*(replen_lead_time)**2*sd) for sd in sd_forecast]

    return reorder_point


def calc_lost_sales_as_POD(cost, start_inv, returns, orders, POD_orders, forecast, sd_forecast):
    lost_sales_as_POD = [cost['POD_vmc']*int(calc_expected_lost_sales(inv+ret+order+POD_order, dem, sd)) 
                               for inv, ret, order, POD_order, dem, sd in 
                               zip(start_inv, returns, orders, POD_orders, forecast, sd_forecast)]
    return lost_sales_as_POD

