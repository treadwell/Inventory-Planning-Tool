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

    def __init__(self, title, demand_plan, returns_plan, print_plan, ss_plan, order_n_months_supply, reorder_point = None, inv_0 = 0):
        self.title_name = title.title_name
        self.cost = title.cost
        self.months = demand_plan.months
        self.forecast = demand_plan.forecast
        self.sd_forecast = demand_plan.sd_forecast
        self.returns = returns_plan.returns
        self.POD_breakeven = self.calc_POD_breakeven()
        self.order_n_months_supply = order_n_months_supply
        self.reorder_point = reorder_point
        self.inv_0 = inv_0
        self.orders, self.POD_orders, self.starting_inventory, self.ending_inventory, self.average_inventory = self.determine_plan()

        self.FMC, self.VMC, self.POD_VMC, self.umc, self.carry_stg_cost, self.lost_sales_expected = self.calc_costs()

    def __repr__(self):
        return str(self.__dict__)

    def subclassable_fn(self):
        raise NotImplementedError # fn that's going to be implemented in a subclass

    def calc_POD_breakeven(self):
        # note that this should really incorporate WACC
        return self.cost['fmc']/(self.cost['POD_vmc']-self.cost['vmc'])


    def determine_plan(self):
        if self.reorder_point == None:
            self.reorder_point = [0] * len(self.months)

        starting_inventory = []
        ending_inventory = []
        average_inventory = []
        POD_orders = [0]*len(self.forecast)
        orders=[0]*len(self.forecast)

        for i, fcst in enumerate(self.forecast):
            # calculate starting inventory
            if i == 0:
                starting_inventory.append(self.inv_0)
            else:
                starting_inventory.append(ending_inventory[i-1])
            # calculate trial ending inventory
            trial_ending_inventory = starting_inventory[i] - fcst + self.returns[i]
            # if trial ending inventory < ROP, place order
            if trial_ending_inventory < self.reorder_point[i]:
                # determine order quantity
                # orders = current shortfall + order quantity
                n=self.order_n_months_supply
                terminal = min(i+n, len(self.reorder_point)-1)  # used to make sure index doesn't exceed length of list
                orders[i] = sum(self.forecast[i:i+n])-sum(self.returns[i:i+n])-starting_inventory[i]+self.reorder_point[terminal]
                # order POD if the size of the order is too small
                # POD order quantity will be just what is needed in the current month
                if orders[i] < self.POD_breakeven and self.cost['allow_POD'] == True:
                    POD_orders[i] = max(self.forecast[i]-starting_inventory[i]-self.returns[i],0)
                    orders[i] = 0
            else:
                orders[i] = 0
            # calculate ending inventory from inventory balance equation
            ending_inventory.append(starting_inventory[i] - self.forecast[i] + self.returns[i]
                                        + orders[i] + POD_orders[i])
            #print i
            #print "orders:", orders[i]
            #print "POD_orders:", POD_orders[i]
            #print "start_inv:", starting_inventory[i]
            #print "end_inv:", ending_inventory[i]

            # calculate average inventory in order to calculate period carrying cost
            average_inventory.append((starting_inventory[i]+ending_inventory[i])/2)

        return orders, POD_orders, starting_inventory, ending_inventory, average_inventory

    def calc_expected_lost_sales(self, inventory, mean_demand, std_dev_demand):
        '''Utilizes loss function to calculate lost sales'''
        shortfall = integrate.quad(lambda x: (x-inventory)*stats.norm.pdf(x, loc=mean_demand, 
                                scale=std_dev_demand), inventory, np.inf)[0]
        return shortfall


    def calc_costs(self):
        FMC = [self.cost['fmc'] + self.cost['perOrder'] if round(order) else 0 for order in self.orders]
        VMC = [self.cost['vmc'] * order for order in self.orders]
        POD_VMC = [self.cost['POD_vmc'] * POD_order for POD_order in self.POD_orders]
        umc = (sum(FMC)+sum(VMC)+sum(POD_VMC)) / (sum(self.orders)+sum(self.POD_orders)) # approximation - should be a list
        carry_stg_cost = [float(self.cost['WACC']) / 12 * month_avg * umc for month_avg in self.average_inventory] 
        lost_sales_expected = [self.cost['lost_margin']*int(self.calc_expected_lost_sales(inv+ret+order+POD_order, dem, sd)) 
                               for inv, ret, order, POD_order, dem, sd in 
                               zip(self.starting_inventory, self.returns, self.orders, self.POD_orders, self.forecast, self.sd_forecast)]

        return FMC, VMC, POD_VMC, umc, carry_stg_cost, lost_sales_expected



    def inv_from_demand(demand, orders, POD_orders, returns, inv_0 = 0):  # move to an "Actual class"

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

    def calc_lost_sales_as_POD(cost, start_inv, returns, orders, POD_orders, forecast, sd_forecast):  #  this goes in the SS Plan class
        lost_sales_as_POD = [cost['POD_vmc']*int(calc_expected_lost_sales(inv+ret+order+POD_order, dem, sd)) 
                                   for inv, ret, order, POD_order, dem, sd in 
                                   zip(start_inv, returns, orders, POD_orders, forecast, sd_forecast)]
        return lost_sales_as_POD


class Months_Supply(Purchase_Plan):
    def __init__(self, month_supply):
        self.month_supply = month_supply


    def subclassable_fn(self):
        raise NotImplementedError

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
    
    def __init__(self, title, demand_plan, target_service_level, replen_lead_time):
        self.title_name = title.title_name
        self.sd_forecast = demand_plan.sd_forecast
        self.target_service_level = target_service_level
        self.replen_lead_time = replen_lead_time
        self.reorder_point = self.calc_reorder_points()

    def __repr__(self):
        return str(self.__dict__)

        # develop plan with ROPs loaded

    def calc_reorder_points(self):
        # ROP = replen_lead_time * fcst + 1.96 *(replen_lead_time*sd**2)**0.5

        service_multiplier = stats.norm.ppf(self.target_service_level, loc=0, scale=1)

        #reorder_point = [int(replen_lead_time*fcst +service_multiplier*(replen_lead_time*sd**2)**0.5) 
        #                 for fcst, sd in zip(forecast,sd_forecast)]
        reorder_point = [int(service_multiplier*(self.replen_lead_time)**2*sd) for sd in self.sd_forecast]

        return reorder_point


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

    # Purchase plan parameters
    order_n_months_supply = 9

    # SS plan parameters
    replen_lead_time = 2
    target_service_level = 0.99


    d = Demand_Plan(title, starting_monthly_demand, number_months, trendPerMonth, seasonCoeffs, initial_cv, per_period_cv)
    r = Returns_Plan(title, d, returns_rate, lag)
    pr = Print_Plan(title)

    ss = SS_Plan(title, d, target_service_level, replen_lead_time)
    pu = Purchase_Plan(title, d, r, pr, ss, order_n_months_supply)

    # put complete data frame and summary stats in here, but for now...

    stats = map(sum, [d.forecast, r.returns, pu.orders])

    scenario_costs = map(sum, [pu.FMC, pu.VMC, pu.POD_VMC, pu.carry_stg_cost, pu.lost_sales_expected])

    return stats, scenario_costs, sum(scenario_costs)

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

    print  "\n------------- test class SS_Plan -------------"
    replen_lead_time = 2
    target_service_level = 0.99

    ss_plan_1 = SS_Plan(xyz, demand_plan_1, target_service_level, replen_lead_time)
    print "object ss_plan_1:", ss_plan_1
    print "object ss_plan_1 reorder points:", ss_plan_1.reorder_point


    print  "\n------------- test class Purchase_Plan -------------"

    order_n_months_supply = 9

    purchase_plan_1 = Purchase_Plan(xyz, demand_plan_1, returns_plan_1, print_plan_1, ss_plan_1, order_n_months_supply)
    print "object purchase_plan_1:", purchase_plan_1
    print "object purchase_plan_1 POD breakeven:", purchase_plan_1.POD_breakeven
    print "object purchase_plan_1 orders:", purchase_plan_1.orders
    print "object purchase_plan_1 POD orders:", purchase_plan_1.POD_orders


 

    print  "\n------------- test scenario function -------------"

    print scenario(xyz, Demand_Plan, Returns_Plan, Print_Plan, Purchase_Plan, SS_Plan)
    print scenario(xyz, Aggressive_Demand_Plan, Returns_Plan, Print_Plan, Purchase_Plan, SS_Plan)
    print scenario(xyz, Conservative_Demand_Plan, Returns_Plan, Print_Plan, Purchase_Plan, SS_Plan)


