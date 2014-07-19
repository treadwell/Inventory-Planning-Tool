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
        based on a coefficient of variation function. This presumes that a forecast has been
        generated within a month of a reorder being placed.'''

        month, forecast = self.calc_forecast()

        sd_forecast = [int(self.initial_cv * monthly_forecast) for monthly_forecast in forecast]

        return sd_forecast

    def plot(self):
        # plot forecast
        plt.plot(self.months,self.forecast, linewidth=2.0, label='demand forecast')
        plt.ylabel('Units')
        plt.xlabel('Month')

        plt.title(self.title_name + ' Forecasted Demand', y=1.05, weight = "bold")
        plt.legend()
        #plt.savefig('./output/' + '01_forecast.png',dpi=300)
        plt.show(1)

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

    def __init__(self, title, demand_plan, returns_plan, print_plan, ss_plan, order_n_months_supply = 9, inv_0 = 0):
        self.title_name = title.title_name
        self.cost = title.cost
        self.demand_plan = demand_plan
        # reference months, forecast, etc. on the fly/as needed 
        ## months = self.demand_plan.months
        self.months = demand_plan.months
        self.forecast = demand_plan.forecast
        self.sd_forecast = demand_plan.sd_forecast
        self.returns = returns_plan.returns
        self.POD_breakeven = self.calc_POD_breakeven()
        self.order_n_months_supply = order_n_months_supply
        self.reorder_point = ss_plan.reorder_point
        self.SS_as_POD_flag = ss_plan.SS_as_POD_flag
        self.inv_0 = inv_0

        self.technology_types = ['POD', 'Digital', 'Conventional', 'Offshore']
        #self.order_types = ['POD_orders', 'digital_orders', 'orders', 'offshore_orders']

        self.POD_orders, self.digital_orders, self.orders, self.offshore_orders, self.starting_inventory, \
                self.ending_inventory, self.average_inventory, = self.determine_plan()

        self.POD_FMC, self.digital_FMC, self.conv_FMC, self.offshore_FMC, self.POD_VMC,  \
                self.digital_VMC, self.conv_VMC, self.offshore_VMC, self.umc, \
                self.carry_stg_cost, self.lost_sales_expected = self.calc_costs(self.forecast, self.sd_forecast, self.returns, 
                    self.POD_orders, self.digital_orders, self.orders, self.offshore_orders, self.starting_inventory, self.average_inventory,)

        #self.POD_FMC = self.calc_costs()[0]
        #self.calculated_costs = self.calc_costs()

        self.total_cost =  sum(self.POD_FMC + self.conv_FMC + self.digital_FMC + self.offshore_FMC + self.POD_VMC + self.conv_VMC + \
                self.digital_VMC + self.offshore_VMC + self.carry_stg_cost + self.lost_sales_expected)

    def __repr__(self):
        return str(self.__dict__)

    def subclassable_fn(self):
        raise NotImplementedError # fn that's going to be implemented in a subclass

    def calc_POD_breakeven(self):
        # note that this should really incorporate WACC
        return self.cost['fmc']/(self.cost['POD_vmc']-self.cost['vmc'])

    

    def calc_order_qty(self, i, forecast, returns):
        # determine order quantity
        # orders = n months net demand
        n = self.order_n_months_supply

        qty = sum(forecast[i:i+n])-sum(returns[i:i+n])#-starting_inventory

        def get_umc(umc_type):
            a, b = self.cost['Printing'][umc_type] # can use a slice if tuple has len >2
            return a/qty + b

        umcs = map(get_umc, self.technology_types)

        orders = [0]*4 # POD_order, Digital_order, Conventional_order, Offshore_order
        mindex = umcs.index(min(umcs)) # get index of min umc
        if mindex == 0: # POD wins
            orders[0] = max(0, (forecast[i] - returns[i])) # only return one month of supply
            # print "POD wins", POD_order, POD_umc
        else:
            orders[mindex] = qty # set appropriate order to qty

        return orders

    def calc_reorder_point(self):
        '''Calculate the inventory level at which you would generate a replenishment.
        It is based on service level x variance in demand over the lead time + expected demand over lead time'''
        #  calculate the reorder point every period
        #  calculate the variance over the lead time (lead time is a parameter)
        #  use the forecast at 1, 2, 3 periods out from i depending on lead time w cv periods
        #  target service level
        #  look for other instances of sd_forecast (lost sales calculation!)
        pass


    def determine_plan(self):
        # pass stuff in to functions explicitly; don't use global variables
        # within the class
        if self.reorder_point == None:
            self.reorder_point = [0] * len(self.months)

        starting_inventory = []
        ending_inventory = []
        average_inventory = []
        start_inv, avg_inv, end_inv = [], [], []

        horizon = len(self.forecast)
        POD_orders, orders, digital_orders, offshore_orders = [0]*horizon, [0]*horizon, [0]*horizon, [0]*horizon

        for i, fcst in enumerate(self.forecast):
            # calculate starting inventory
            if i == 0:
                start_inv.append(self.inv_0)
            else:
                start_inv.append(end_inv[i-1])
            # calculate trial ending inventory
            trial_ending_inventory = start_inv[i] - fcst + self.returns[i]
            # if trial ending inventory < ROP, place order
            if trial_ending_inventory < self.reorder_point[i]:  # replace with "get reorder point function"
                # determine order quantity
                POD_orders[i], digital_orders[i], orders[i], offshore_orders[i] = self.calc_order_qty(i, self.forecast, self.returns)

            # calculate ending inventory from inventory balance equation
            end_inv.append(start_inv[i] - self.forecast[i] + self.returns[i]
                                        + orders[i] + POD_orders[i] + digital_orders[i] + offshore_orders[i])

            # calculate average inventory in order to calculate period carrying cost
            avg_inv.append((start_inv[i]+end_inv[i])/2)

        return POD_orders, digital_orders, orders, offshore_orders, start_inv, end_inv, avg_inv, 

    def calc_expected_lost_sales(self, inventory, mean_demand, std_dev_demand):
        '''Utilizes loss function to calculate lost sales.'''
        shortfall = integrate.quad(lambda x: (x-inventory)*stats.norm.pdf(x, loc=mean_demand, 
                                scale=std_dev_demand), inventory, np.inf)[0]
        return shortfall

    # def calc_lost_sales_as_POD(cost, start_inv, returns, orders, POD_orders, forecast, sd_forecast):  #  this goes in the SS Plan class
    #     lost_sales_as_POD = [cost['POD_vmc']*int(calc_expected_lost_sales(inv+ret+order+POD_order, dem, sd)) 
    #                                for inv, ret, order, POD_order, dem, sd in 
    #                                zip(start_inv, returns, orders, POD_orders, forecast, sd_forecast)]
    #     return lost_sales_as_POD


    def calc_costs(self, forecast, sd_forecast, returns, POD_orders, digital_orders, orders, offshore_orders, starting_inventory, average_inventory):
        
        agg_orders = {"POD": POD_orders, "Digital": digital_orders, "Conventional": orders, 
                "Offshore": offshore_orders}

        # Caclulate Fixed Manufacturing Cost (FMC)

        FMCs = [[self.cost['Printing'][tt][0] if round(order) else 0 for order in agg_orders[tt]]
                    for tt in self.technology_types]
       
        POD_FMC, digital_FMC, conv_FMC, offshore_FMC = FMCs[0], FMCs[1], FMCs[2], FMCs[3]

        sum_FMC = sum(map(sum, FMCs))
                
        # Calculate Variable Manufacturing Cost (VMC)

        VMCs = [[self.cost['Printing'][tt][1] * order for order in agg_orders[tt]]
            for tt in self.technology_types]

        POD_VMC, digital_VMC, conv_VMC, offshore_VMC = VMCs[0], VMCs[1], VMCs[2], VMCs[3]

        sum_VMC = sum(map(sum, VMCs))

        # def VMC_cost(order_type):
        #     [...]
        #     return foo

        # sum_VMC = sum(map(VMC_cost, order_type))

        sum_orders = sum(map(sum, agg_orders.values()))

        umc = (sum_FMC + sum_VMC) / sum_orders # approximation - should be a list

        carry_stg_cost = [float(self.cost['WACC']) / 12 * month_avg * umc for month_avg in average_inventory]

        if self.SS_as_POD_flag == False:
            loss = self.cost['lost_margin']
        else:
            loss = self.cost['Printing']["POD"][1]


        # the following needs to be updated to feed the correct expected demand over a replenishment lead time 
        # and accumulated variance.
        lost_sales_expected = [loss*int(self.calc_expected_lost_sales(inv+ret+POD_order+digital_order+conv_order+offshore_order, dem, sd)) 
                                for inv, ret, POD_order, digital_order, conv_order, offshore_order, dem, sd in 
                                    zip(starting_inventory, returns, agg_orders["POD"], agg_orders["Digital"], 
                                        agg_orders["Conventional"], agg_orders["Offshore"], forecast, sd_forecast)]

        return POD_FMC, digital_FMC, conv_FMC, offshore_FMC, POD_VMC, digital_VMC, conv_VMC, offshore_VMC, umc, carry_stg_cost, lost_sales_expected



    def calc_inv(self):  # move to an "Actual class"

        start_inv = []       # must be >= 0
        start_inv_posn = []  # can be negative
        end_inv = []         # must be >= 0
        end_inv_posn = []    # can be negative
        avg_inv = []         # must be >= 0

        for i, order in enumerate(orders):
            # calculate starting inventory
            if i == 0:
                start_inv.append(max(0,self.inv_0))  # eventually replace this with an optional input
                start_inv_posn.append(0)
            else:
                start_inv.append(end_inv[i-1])
                start_inv_posn.append(end_inv_posn[i-1])
            
            # calculate ending inventory from inventory balance equation
            end_inv.append(max(0,start_inv[i] - self.demand[i] + self.orders[i] + 
                                   self.POD_orders[i] + self.returns[i]))
            end_inv_posn.append(start_inv_posn[i] - self.demand[i] + self.orders[i] + 
                                    self.POD_orders[i] + self.returns[i])

            # calculate average inventory in order to calculate period carrying cost
            avg_inv.append((start_inv[i]+end_inv[i])/2)
        
        return end_inv_posn, avg_inv


    # ----------------------




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
        self.demand_plan = demand_plan
        self.target_service_level = target_service_level
        self.replen_lead_time = replen_lead_time
        self.reorder_point = self.calc_reorder_points(target_service_level, replen_lead_time, self.demand_plan.forecast, initial_cv, per_period_cv)
        self.SS_as_POD_flag = False

    def __repr__(self):
        return str(self.__dict__)

    

    def calc_leadtime_sd(self, r, forecast, initial_cv, per_period_cv): 
        ''' Calculates the cumulative standard deviation of a forecast made r periods ago,
        where r is the replenishment lead time.

        if i = 5 and r = 3, forecast was generated in period 1
        forecast for i = 5, horizon = 3
        forecast for i = 4, horizon = 2
        forecast for i = 2, horizon = 1'''

        def round_up(x):
            return int(int(x + 1) if int(x) != x else int(x))

        replen_sds = []
        for i, f in enumerate(forecast):
            result = []
            fract_period = r % 1
            number_periods = round_up(r)
            for j in range(max(0,int(i-r+1), i+1)):
                horizon = j-i+number_periods
                if fract_period and j == i:
                    result += [(fract_period * forecast[j]*(initial_cv + per_period_cv * horizon))**2]
                else:
                    result += [(forecast[j]*(initial_cv + per_period_cv * horizon))**2]
                #print j, horizon, forecast[j], result
                period_sd = sum(result)**0.5
            replen_sds += [period_sd]
        return replen_sds

    def calc_reorder_points(self, target_service_level, replen_lead_time, forecast, initial_cv, per_period_cv):
        '''While this is labeled calc_reorder_points, it actually calculates safety stock.  
        It takes the accumulated variance in demand over the replenshment lead time 
        x a service multiplier.  It can be enhanced to reflect variance in the
        lead time itself, but does not currently do that.

        Reorder points would just be the demand over the lead time plus the safety stock.  
        Instead we're calculating the safety stock at the point when the order 
        would normally have been placed and moving it out to when the replenshment should have come in.
        Investment-wise, only the safety stock matters and we can assume that the stock would have been 
        ordered to come in when needed.

        This works out to be the safety stock based on the previous months corresponding to the
        replenishment interval.  It is calculated using principle that variance of a sum of RVs is 
        equal to the sum of the variance of the RVs: 

        Note that SS(0) = 0.
        if i < r, then just use the variances for the months for which we have forecasts.

        If r = 2.5, then we use:
        var period i: (0.5 * sd_forecast(i))**2
        var period i-1: (sd_forecast(i-1)**2
        var period i-2: (sd_forecast(i-2)**2
        sd of r = (sum(variances)(**0.5 '''

        '''(target_service_level, replen_lead_time, forecast, initial_cv, per_period_cv) --> ROP list'''

        # ROP = replen_lead_time * fcst + 1.96 *(replen_lead_time*sd**2)**0.5

        service_multiplier = stats.norm.ppf(target_service_level, loc=0, scale=1)

        SDs = self.calc_leadtime_sd(replen_lead_time, forecast, initial_cv, per_period_cv)

        # this needs to be enhanced to accumulate demand variances over the replenishment lead time.  
        reorder_point = [int(service_multiplier*sd) for sd in SDs]

        return reorder_point

class SS_Plan_None(SS_Plan):
    def __init__(self, title, demand_plan, target_service_level, replen_lead_time):
        self.reorder_point = None
        self.SS_as_POD_flag = False

class SS_Plan_POD(SS_Plan):
    def __init__(self, title, demand_plan, target_service_level, replen_lead_time):
        self.reorder_point = None
        self.SS_as_POD_flag = True


def scenario(title, Demand_Plan, Returns_Plan, Print_Plan, Purchase_Plan, SS_Plan, order_n_months_supply = 9):
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
    # order_n_months_supply = 9  # input in function

    # SS plan parameters
    replen_lead_time = 2
    target_service_level = 0.99


    d = Demand_Plan(title, starting_monthly_demand, number_months, trendPerMonth, seasonCoeffs, initial_cv, per_period_cv)
    r = Returns_Plan(title, d, returns_rate, lag)
    pr = Print_Plan(title)

    ss = SS_Plan(title, d, target_service_level, replen_lead_time)
    pu = Purchase_Plan(title, d, r, pr, ss, order_n_months_supply)

    # put complete data frame and summary stats in here, but for now...

    # scenario_costs = map(sum, [pu.FMC, pu.VMC, pu.POD_VMC, pu.carry_stg_cost, pu.lost_sales_expected])

    stats = {'forecast' : sum(d.forecast), 'returns' : sum(r.returns), 'orders': sum(pu.orders), 
        'total cost': pu.total_cost, 'scrap': pu.ending_inventory[-1]}

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
    
    cost = {'perOrder': 80.0, 'WACC': 0.12, 'POD_vmc': 5.0, 'fmc': 1000, 
        'vmc': 2.0, 'lost_margin': 10.00, 'allow_POD':True, "Printing": {"POD": (0, 4.7), 
        "Digital": (100, 1.43), "Conventional": (1625, 1.00),
        "Offshore":(2000, 1.50)}}


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
    # print "object demand_plan_1:", demand_plan_1
    
    # print "months:", demand_plan_1.months
    # print "forecast:", demand_plan_1.forecast
    # print "sd_forecast:", demand_plan_1.sd_forecast
    # demand_plan_1.plot()  # this works


    print "\n------------- test class Aggressive_Demand_Plan -------------"
    
    # demand_plan_2 = Aggressive_Demand_Plan(xyz, starting_monthly_demand, number_months, trendPerMonth, seasonCoeffs, initial_cv, per_period_cv)
    
    # print "months:", demand_plan_2.months
    # print "forecast:", demand_plan_2.forecast
    # print "sd_forecast:", demand_plan_2.sd_forecast

    print "\n------------- test class Conservative_Demand_Plan -------------"
    
    # demand_plan_3 = Conservative_Demand_Plan(xyz, starting_monthly_demand, number_months, trendPerMonth, seasonCoeffs, initial_cv, per_period_cv)
    
    # print "months:", demand_plan_3.months
    # print "forecast:", demand_plan_3.forecast
    # print "sd_forecast:", demand_plan_3.sd_forecast


    print  "\n------------- test class Returns_Plan -------------"
    
    returns_rate = 0.2
    lag = 3
    returns_plan_1 = Returns_Plan(xyz, demand_plan_1, returns_rate, lag)
    # print "object returns_plan_1:", returns_plan_1
    # print "returns:", returns_plan_1.returns

    print  "\n------------- test class Print_Plan -------------"

    print_plan_1 = Print_Plan(xyz)
    print "object print_plan_1:", print_plan_1

    print  "\n------------- test class SS_Plan -------------"
    
    replen_lead_time = 2
    target_service_level = 0.99

    ss_plan_1 = SS_Plan(xyz, demand_plan_1, target_service_level, replen_lead_time)
    # print "object ss_plan_1:", ss_plan_1
    # print "object ss_plan_1 reorder points:", ss_plan_1.reorder_point


    print  "\n------------- test class Purchase_Plan -------------"

    starting_monthly_demand = 1000
    number_months = 36
    trendPerMonth = -0.00
    seasonCoeffs = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
    #seasonCoeffs = [0.959482026,0.692569699,0.487806875,0.543161208,0.848077745,0.936798779,1.596854431,1.618086981,1.433374588,0.949500605,0.828435702,1.105851362]
    initial_cv = 0.15
    per_period_cv = 0.015

    demand_plan_1 = Demand_Plan(xyz, starting_monthly_demand, number_months, trendPerMonth, seasonCoeffs, initial_cv, per_period_cv)
    
    returns_rate = 0.0
    lag = 3
    returns_plan_1 = Returns_Plan(xyz, demand_plan_1, returns_rate, lag)
    
    ss_plan_1 = SS_Plan_None(xyz, demand_plan_1, target_service_level, replen_lead_time)
    ss_plan_2 = SS_Plan(xyz, demand_plan_1, target_service_level, replen_lead_time)

    purchase_plan_1 = Purchase_Plan(xyz, demand_plan_1, returns_plan_1, print_plan_1, ss_plan_1)
    purchase_plan_2 = Purchase_Plan(xyz, demand_plan_1, returns_plan_1, print_plan_1, ss_plan_2)
    # print "object purchase_plan_1:", purchase_plan_1
    # print "object purchase_plan_1 POD breakeven:", purchase_plan_1.POD_breakeven
    print "Conv. orders, no ss:", purchase_plan_1.orders, sum(purchase_plan_1.orders)
    print "Digital orders, no ss:", purchase_plan_1.digital_orders, sum(purchase_plan_1.digital_orders)
    print "POD orders, no ss:", purchase_plan_1.POD_orders, sum(purchase_plan_1.POD_orders)
    print "Offshore orders, no ss:", purchase_plan_1.offshore_orders, sum(purchase_plan_1.offshore_orders)
    print "Total cost:", purchase_plan_1.total_cost
    #
    print "\nPurchase plan with safety stock"
    print "-------------------------------"

    print "Demand", demand_plan_1.forecast
    print "Returns", returns_plan_1.returns
    print "Reorder Points", ss_plan_2.reorder_point
    print "Starting Inventory:", purchase_plan_2.starting_inventory
    print "Conv. orders w ss", purchase_plan_2.orders, sum(purchase_plan_2.orders)
    print "Digital orders, w ss:", purchase_plan_2.digital_orders, sum(purchase_plan_2.digital_orders)
    print "POD orders, w ss:", purchase_plan_2.POD_orders, sum(purchase_plan_2.POD_orders)
    print "Offshore orders, w ss:", purchase_plan_2.offshore_orders, sum(purchase_plan_2.offshore_orders)
    print "Total cost:", purchase_plan_2.total_cost


    print  "\n------------- test scenario function -------------"

    print "Normal Demand:", scenario(xyz, Demand_Plan, Returns_Plan, Print_Plan, Purchase_Plan, SS_Plan)
    print "Aggressive Demand:", scenario(xyz, Aggressive_Demand_Plan, Returns_Plan, Print_Plan, Purchase_Plan, SS_Plan)
    print "Conservative Demand:", scenario(xyz, Conservative_Demand_Plan, Returns_Plan, Print_Plan, Purchase_Plan, SS_Plan)

    print "\nCompare various safety stock and months supply scenarios"
    for ss_scenario in [SS_Plan, SS_Plan_None, SS_Plan_POD]:
        print "\t", ss_scenario
        for i in [6, 9, 12, 18, 36]:
            print "\t\tTotal Cost for", i, "mo supply:", locale.currency(scenario(xyz, Demand_Plan, Returns_Plan, Print_Plan, 
                    Purchase_Plan, ss_scenario, i)['total cost'], grouping = True)

