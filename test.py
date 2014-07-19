import scripts.IPT_classes as IPT
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


'''Script tests classes of IPT_classes module.'''


path = '/Users/kbrooks/Documents/MH/Projects/Inventory Planning Tool/'

print "------------------- Unit tests -------------------"

cost = {'perOrder': 80.0, 'WACC': 0.12, 'POD_vmc': 5.0, 'fmc': 1000, 
    'vmc': 2.0, 'lost_margin': 10.00, 'allow_POD':True, "Printing": {"POD": (0, 4.7), 
    "Digital": (100, 1.43), "Conventional": (1625, 1.00),
    "Offshore":(2000, 1.50)}}


print "\n------------- test class Title -------------"

xyz = IPT.Title("xyz", cost)
print "object xyz:", xyz


print "\n------------- test class Demand_Plan -------------"

starting_monthly_demand = 1000
number_months = 36
trendPerMonth = -0.05
seasonCoeffs = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
#seasonCoeffs = [0.959482026,0.692569699,0.487806875,0.543161208,0.848077745,0.936798779,1.596854431,1.618086981,1.433374588,0.949500605,0.828435702,1.105851362]
initial_cv = 0.15
per_period_cv = 0.015

demand_plan_1 = IPT.Demand_Plan(xyz, starting_monthly_demand, number_months, trendPerMonth, seasonCoeffs, initial_cv, per_period_cv)
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
returns_plan_1 = IPT.Returns_Plan(xyz, demand_plan_1, returns_rate, lag)
# print "object returns_plan_1:", returns_plan_1
# print "returns:", returns_plan_1.returns

print  "\n------------- test class Print_Plan -------------"

print_plan_1 = IPT.Print_Plan(xyz)
print "object print_plan_1:", print_plan_1

print  "\n------------- test class SS_Plan -------------"

starting_monthly_demand = 100
number_months = 36
trendPerMonth = 0
seasonCoeffs = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
#seasonCoeffs = [0.959482026,0.692569699,0.487806875,0.543161208,0.848077745,0.936798779,1.596854431,1.618086981,1.433374588,0.949500605,0.828435702,1.105851362]
initial_cv = 1
per_period_cv = 0

demand_plan_1 = IPT.Demand_Plan(xyz, starting_monthly_demand, number_months, trendPerMonth, seasonCoeffs, initial_cv, per_period_cv)

replen_lead_time = 2
target_service_level = 0.99

ss_plan_1 = IPT.SS_Plan(xyz, demand_plan_1, target_service_level, replen_lead_time)
#print "object ss_plan_1:", ss_plan_1
print "forecast:", demand_plan_1.forecast
print "SDs:", ss_plan_1.calc_leadtime_sd(2, demand_plan_1.forecast, initial_cv, per_period_cv)
print "object ss_plan_1 reorder points:", ss_plan_1.reorder_point


print  "\n------------- test class Purchase_Plan -------------"

starting_monthly_demand = 1000
number_months = 36
trendPerMonth = -0.00
seasonCoeffs = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
#seasonCoeffs = [0.959482026,0.692569699,0.487806875,0.543161208,0.848077745,0.936798779,1.596854431,1.618086981,1.433374588,0.949500605,0.828435702,1.105851362]
initial_cv = 0.15
per_period_cv = 0.015

demand_plan_1 = IPT.Demand_Plan(xyz, starting_monthly_demand, number_months, trendPerMonth, seasonCoeffs, initial_cv, per_period_cv)

returns_rate = 0.0
lag = 3
returns_plan_1 = IPT.Returns_Plan(xyz, demand_plan_1, returns_rate, lag)

ss_plan_1 = IPT.SS_Plan_None(xyz, demand_plan_1, target_service_level, replen_lead_time)
ss_plan_2 = IPT.SS_Plan(xyz, demand_plan_1, target_service_level, replen_lead_time)

purchase_plan_1 = IPT.Purchase_Plan(xyz, demand_plan_1, returns_plan_1, print_plan_1, ss_plan_1)
purchase_plan_2 = IPT.Purchase_Plan(xyz, demand_plan_1, returns_plan_1, print_plan_1, ss_plan_2)

print "\nPurchase plan without safety stock"
print "-------------------------------"
print "Conv. orders, no ss:", purchase_plan_1.orders, sum(purchase_plan_1.orders)
print "Digital orders, no ss:", purchase_plan_1.digital_orders, sum(purchase_plan_1.digital_orders)
print "POD orders, no ss:", purchase_plan_1.POD_orders, sum(purchase_plan_1.POD_orders)
print "Offshore orders, no ss:", purchase_plan_1.offshore_orders, sum(purchase_plan_1.offshore_orders)
print "Ending Inventory:", purchase_plan_1.ending_inventory
print "Total cost:", purchase_plan_1.total_cost
#
print "\nPurchase plan with safety stock"
print "-------------------------------"

print "Demand", demand_plan_1.forecast
print "Returns", returns_plan_1.returns
print "Reorder Points", ss_plan_2.reorder_point
print "Starting Inventory:", purchase_plan_2.starting_inventory
print "Ending Inventory:", purchase_plan_2.ending_inventory
print "Conv. orders w ss", purchase_plan_2.orders, sum(purchase_plan_2.orders)
print "Digital orders, w ss:", purchase_plan_2.digital_orders, sum(purchase_plan_2.digital_orders)
print "POD orders, w ss:", purchase_plan_2.POD_orders, sum(purchase_plan_2.POD_orders)
print "Offshore orders, w ss:", purchase_plan_2.offshore_orders, sum(purchase_plan_2.offshore_orders)
print "Total cost:", purchase_plan_2.total_cost


print  "\n------------- test scenario function -------------"

print "Normal Demand:", IPT.scenario(xyz, IPT.Demand_Plan, IPT.Returns_Plan, IPT.Print_Plan, IPT.Purchase_Plan, IPT.SS_Plan)
print "Aggressive Demand:", IPT.scenario(xyz, IPT.Aggressive_Demand_Plan, IPT.Returns_Plan, IPT.Print_Plan, IPT.Purchase_Plan, IPT.SS_Plan)
print "Conservative Demand:", IPT.scenario(xyz, IPT.Conservative_Demand_Plan, IPT.Returns_Plan, IPT.Print_Plan, IPT.Purchase_Plan, IPT.SS_Plan)

print "\nCompare various safety stock and months supply scenarios"
for ss_scenario in [IPT.SS_Plan, IPT.SS_Plan_None, IPT.SS_Plan_POD]:
    print "\t", ss_scenario
    for i in [6, 9, 12, 18, 36]:
        print "\t\tTotal Cost for", i, "mo supply:", locale.currency(IPT.scenario(xyz, IPT.Demand_Plan, IPT.Returns_Plan, IPT.Print_Plan, 
                IPT.Purchase_Plan, ss_scenario, i)['total cost'], grouping = True)
