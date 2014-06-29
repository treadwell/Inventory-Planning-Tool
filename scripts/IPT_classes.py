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

    def __repr__(self):
        return str(self.__dict__)


class Demand_Plan(object):
    def __init__(self, title, starting_monthly_demand, number_months, trendPerMonth, seasonCoeffs, initial_cv, per_period_cv):
        self.title_name = title.title_name
        self.starting_monthly_demand = starting_monthly_demand
        self.number_months = number_months
        self.trendPerMonth = trendPerMonth
        self.seasonCoeffs = seasonCoeffs
        self.initial_cv = initial_cv
        self.per_period_cv = per_period_cv
        self.months, self.forecast = self.calc_forecast()
        self.sd_forecast = self.calc_sd_forecast()
        
    def __repr__(self):
        return str(self.__dict__)

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

class Aggressive_Demand_Plan(Demand_Plan):
     # self.starting_monthly_demand *= 2 # would return an error, there's no
     # starting_monthly_demand defined
 
    def __init__(self, *args):
        super(Aggressive_Demand_Plan, self).__init__(*args)
        #Demand_Plan.__init__(self, *args) # also an option
        #super().__init__(self, *args) ## Python 3
        self.starting_monthly_demand *= 2
        self.months, self.forecast = self.calc_forecast()
        self.sd_forecast = self.calc_sd_forecast()

class Conservative_Demand_Plan(Demand_Plan):
    def __init__(self, *args):
        super(Conservative_Demand_Plan, self).__init__(*args)
        self.starting_monthly_demand /= 2.0
        self.months, self.forecast = self.calc_forecast()
        self.sd_forecast = self.calc_sd_forecast()

class Returns_Plan(object):
    def __init__(self, title, demand_plan, returns_rate, lag):
        self.title_name = title.title_name
        self.returns_rate = returns_rate
        self.lag = lag
        self.months = demand_plan.months
        self.forecast = demand_plan.forecast
        self.returns = self.calc_returns_forecast()
    
    def __repr__(self):
        return str(self.__dict__)

    def calc_returns_forecast(self):
        '''Calculate a returns forecast based by applying a percentage and lag to demand forecast'''

        returns = [0]*len(self.forecast)

        for i, x in enumerate(self.forecast):
            if i < lag:
                returns[i] = 0
            else:
                returns[i] = int(self.returns_rate*self.forecast[i-self.lag])
        return returns


class Purchase_Plan(object):
    """Base class for purchasing strategies"""

    def __init__(self, title):
        self.title_name = title.title_name

    def __repr__(self):
        return str(self.__dict__)

    def subclassable_fn(self):
        raise NotImplementedError # fn that's going to be implemented in a subclass

    def plot(self):
        # plots relevant metrics
        pass


class month_supply(Purchase_Plan):
    def subclassable_fn(self):
        return 42

class EOQ(Purchase_Plan):
    """Economic Order Quantity"""

    def subclassable_fn(self):
        import random
        return random.randint(1, 500)

class Print_Plan(object):
    """Base class for print strategies"""

    def __init__(self, title):
        self.title_name = title.title_name

    def __repr__(self):
        return str(self.__dict__)

class SS_Plan(object):
    """Base class for safety stock strategies"""
    
    def __init__(self, title):
        self.title_name = title.title_name

    def __repr__(self):
        return str(self.__dict__)


def scenario(title, Demand_Plan, Returns_Plan, Print_Plan, Purchase_Plan, SS_Plan):
    # title should be an actual object
    # the other inputs should be classes

    # capture user input here (in the future), for now just enter the parameters by hand
    # Demand Plan parameters
    starting_monthly_demand = 1000
    number_months = 36
    trendPerMonth = -0.05
    seasonCoeffs = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
    initial_cv = 0.15
    per_period_cv = 0.015

    # Returns Plan parameters
    returns_rate = 0.2
    lag = 3


    d = Demand_Plan(title, starting_monthly_demand, number_months, trendPerMonth, seasonCoeffs, initial_cv, per_period_cv)
    r = Returns_Plan(title, d, returns_rate, lag)
    pr = Print_Plan(title)
    pu = Purchase_Plan(title)
    ss = SS_Plan(title)

    # put complete data frame and summary stats in here, but for now...

    stats = map(sum, [d.forecast, r.returns])

    return stats

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

    print "\n------------- test class Title -------------"
    xyz = Title("xyz", cost)
    print "object xyz:", xyz

    print "\n------------- test class Demand_Plan -------------"
    starting_monthly_demand = 1000
    number_months = 36
    trendPerMonth = -0.05
    seasonCoeffs = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
    #seasonCoeffs = [0.959482026,0.692569699,0.487806875,0.543161208,0.848077745,0.936798779,1.596854431,1.618086981,1.433374588,0.949500605,0.828435702,1.105851362]
    initial_cv = 0.15
    per_period_cv = 0.015

    demand_plan_1 = Demand_Plan(xyz, starting_monthly_demand, number_months, trendPerMonth, seasonCoeffs, initial_cv, per_period_cv)
    print "object demand_plan_1:", demand_plan_1
    
    print "months:", demand_plan_1.months
    print "forecast:", demand_plan_1.forecast
    print "sd_forecast:", demand_plan_1.sd_forecast

    print "\n------------- test class Aggressive_Demand_Plan -------------"
    demand_plan_2 = Aggressive_Demand_Plan(xyz, starting_monthly_demand, number_months, trendPerMonth, seasonCoeffs, initial_cv, per_period_cv)
    
    print "months:", demand_plan_2.months
    print "forecast:", demand_plan_2.forecast
    print "sd_forecast:", demand_plan_2.sd_forecast

    print "\n------------- test class Conservative_Demand_Plan -------------"
    demand_plan_3 = Conservative_Demand_Plan(xyz, starting_monthly_demand, number_months, trendPerMonth, seasonCoeffs, initial_cv, per_period_cv)
    
    print "months:", demand_plan_3.months
    print "forecast:", demand_plan_3.forecast
    print "sd_forecast:", demand_plan_3.sd_forecast


    print  "\n------------- test class Returns_Plan -------------"
    returns_rate = 0.2
    lag = 3
    returns_plan_1 = Returns_Plan(xyz, demand_plan_1, returns_rate, lag)
    print "object returns_plan_1:", returns_plan_1
    print "returns:", returns_plan_1.returns

    print  "\n------------- test class Print_Plan -------------"

    print_plan_1 = Print_Plan(xyz)
    print "object print_plan_1:", print_plan_1


    print  "\n------------- test class Purchase_Plan -------------"

    purchase_plan_1 = Purchase_Plan(xyz)
    print "object purchase_plan_1:", purchase_plan_1


    print  "\n------------- test class SS_Plan -------------"

    ss_plan_1 = SS_Plan(xyz)
    print "object ss_plan_1:", ss_plan_1

    print  "\n------------- test scenario function -------------"

    print scenario(xyz, Demand_Plan, Returns_Plan, Print_Plan, Purchase_Plan, SS_Plan)
    print scenario(xyz, Aggressive_Demand_Plan, Returns_Plan, Print_Plan, Purchase_Plan, SS_Plan)
    print scenario(xyz, Conservative_Demand_Plan, Returns_Plan, Print_Plan, Purchase_Plan, SS_Plan)


