import output as o
import pandas as pd
import numpy as np
from scipy import stats
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


class Title(object):
    def __init__(self, title_name, cost):
        self.title_name = title_name
        self.cost = cost

    def output(self):
        ''' Print variables and values'''  # there has to be a better way to do this!
        print "title_name :", self.title_name
        print "cost: ", self.cost


class Demand_Plan(object):
    def __init__(self, title, title_demand_plan, title_returns_plan):
        self.title = title
        self.title_demand_plan = title_demand_plan
        self.title_returns_plan = title_returns_plan
        forecast = self.title_demand_plan.calc_forecast()
        sd_forecast = self.title_demand_plan.calc_sd_forecast()
        returns_forecast = self.title_returns_plan.calc_returns_forecast()

    def output(self):
        ''' Print variables and values'''  # there has to be a better way to do this!
        print "title_name :", self.title_name
        print "cost: ", self.cost
        print "forecast", self.forecast


class Title_Demand_Plan(Demand_Plan):
    def __init__(self, title, starting_monthly_demand, number_months, trendPerMonth, seasonCoeffs, initial_cv, per_period_cv):
        self.title = title
        self.starting_monthly_demand = starting_monthly_demand
        self.number_months = number_months
        self.trendPerMonth = trendPerMonth
        self.seasonCoeffs = seasonCoeffs
        self.initial_cv = initial_cv
        self.per_period_cv = per_period_cv
        
    def output(self):
        ''' Print variables and values'''  # there has to be a better way to do this!
        print "title: ", self.title
        print "starting_monthly_demand: ", self.starting_monthly_demand
        print "number_months:", self.number_months

    def calc_forecast(self):
        '''Returns two lists holding a sequential month number and a second with
        monthly forecasts based on level, trend and seasonality.  This is not 
        normalized to equal a particular liftetime sales target.'''
        
        month = [i+1 for i in xrange(self.number_months)]

        level = [self.starting_monthly_demand] * self.number_months

        trend = [int(x * (1+self.trendPerMonth)**i) for i, x in enumerate(level)]

        forecast = [int(x*self.seasonCoeffs[i % 12]) for i, x in enumerate(trend)]

        return month, forecast


    def calc_sd_forecast(self):
        '''Returns a list with forecast standard deviation.  It is calculated
        based on a starting coefficient of variation with an incremental cv for 
        every successive month in the future.'''

        month, forecast = self.calc_forecast()

        sd_forecast = [int((self.initial_cv + i*self.per_period_cv) * monthly_forecast) for i, monthly_forecast in enumerate(forecast)]

        return sd_forecast



class Title_Returns_Plan(Demand_Plan):
    def __init__(self, title, returns_rate, lag):
        self.title = title
        self.returns_rate = returns_rate
        self.lag = lag
    
    def output(self):
        ''' Print variables and values'''  # there has to be a better way to do this!
        print "title: ", self.title
        print "returns_rate: ", self.returns_rate
        print "lag:", self.lag

    def calc_returns_forecast(self):
        '''Calculate a returns forecast based by applying a percentage and lag to demand forecast'''

        returns = [0]*len(self.forecast)
        for i,x in enumerate(self.forecast):
            if i < lag:
                returns[i] = 0
            else:
                returns[i] = int(self.returns_rate*self.forecast[i-self.lag])
        return returns


class Supply_Plan(object):
    def __init__(title, purchasing_plan, print_plan, ss_plan):
        self.title = title
        self.purchasing_plan = purchasing_plan
        self.print_plan = print_plan
        self.ss_plan = ss_plan

    def output(self):
        print self.summary_stats
        print self.dataframe
        print self.plots




class Purchasing(object):
    """Base class for purchasing strategies"""

    def __init__(self, title):
        self.title = title

    def subclassable_fn(self):
        raise NotImplementedError # fn that's going to be implemented in a subclass

    def plot(self):
        # plots relevant metrics
        pass


    def output(self):
        return self.month_supply()


class month_supply(Purchasing):
    def subclassable_fn(self):
        return 42

class EOQ(Purchasing):
    """Economic Order Quantity"""

    def subclassable_fn(self):
        import random
        return random.randint(1, 500)




# ### in scenario.py:
# for pp in plans.purchasing_plans:
#     for ss in plans.ss_plans:
#         for p in plans.print_plans:
#             plans.Plan(title, pp, ss, p).generate_graphs() # or w/e the output is


# ###



# set path to working directory

path = '/Users/kbrooks/Documents/MH/Projects/Inventory Planning Tool/'




if __name__ == '__main__':
    print "------------------- Unit tests -------------------"
    
    cost = cost = {'perOrder': 80.0, 'WACC': 0.12, 'POD_vmc': 5.0, 'fmc': 1000, 
        'vmc': 2.0, 'lost_margin': 10.00, 'allow_POD':True}

    print "\n-------------test class Title-------------"
    xyz = Title("xyz", cost)
    print "xyz:", xyz
    xyz.output()

    print "\n-------------test class Title_Demand_Plan-------------"
    starting_monthly_demand = 1000
    number_months = 36
    trendPerMonth = -0.05
    seasonCoeffs = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
    #seasonCoeffs = [0.959482026,0.692569699,0.487806875,0.543161208,0.848077745,0.936798779,1.596854431,1.618086981,1.433374588,0.949500605,0.828435702,1.105851362]
    initial_cv = 0.15
    per_period_cv = 0.015

    title_demand_plan_1 = Title_Demand_Plan("xyz", starting_monthly_demand, number_months, trendPerMonth, seasonCoeffs, initial_cv, per_period_cv)
    print "title_demand_plan_1:", title_demand_plan_1
    title_demand_plan_1.output()
    print title_demand_plan_1.calc_forecast()  # shouldn't this pull arguments from the object?
    print title_demand_plan_1.calc_sd_forecast()

    print  "\n-------------test class Title_Returns_Plan-------------"

    returns_rate = 0.2
    lag = 3
    title_returns_plan_1 = Title_Returns_Plan("xyz", returns_rate, lag)
    title_returns_plan_1.output()
    # title_returns_plan_1.calc_returns_forecast()

    print "\n-------------test class Demand_Plan-------------"

    demand_plan_1 = Demand_Plan(xyz, title_demand_plan_1, title_returns_plan_1)





