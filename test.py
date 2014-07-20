import scripts.IPT_classes as IPT
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


'''Script tests classes of IPT_classes module.'''


print "------------------- Unit tests -------------------"

cost = {'perOrder': 80.0, 'WACC': 0.12, 'lost_margin': 10.00, 'allow_POD':True, 
		"Printing": {"POD": (0, 4.7), 
    				"Digital": (100, 1.43), 
    				"Conventional": (1625, 1.00),
    				"Offshore":(2000, 1.50)}}


print "\n------------- test class Title -------------"

xyz = IPT.Title("xyz", cost)
assert xyz.cost['Printing']['POD']
assert xyz.cost['Printing']['Digital']
assert xyz.cost['Printing']['Conventional']
assert xyz.cost['Printing']['Offshore']


print "\n------------- test class Demand_Plan -------------"

starting_monthly_demand = 1000
number_months = 36
trendPerMonth = -0.05
seasonCoeffs = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
#seasonCoeffs = [0.959482026,0.692569699,0.487806875,0.543161208,0.848077745,0.936798779,1.596854431,1.618086981,1.433374588,0.949500605,0.828435702,1.105851362]
initial_cv = 0.15
per_period_cv = 0.015

demand_plan_1 = IPT.Demand_Plan(xyz, starting_monthly_demand, number_months, trendPerMonth, seasonCoeffs, initial_cv, per_period_cv)

assert len(IPT.Demand_Plan(xyz, 1000, 36, 0, 12 * [1], 0, 0).months) == 36
assert len(IPT.Demand_Plan(xyz, 1000, 36, 0, 12 * [1], 0, 0).forecast) == 36
assert len(IPT.Demand_Plan(xyz, 1000, 36, 0, 12 * [1], 0, 0).sd_forecast) == 36

assert sum(IPT.Demand_Plan(xyz, 1000, 36, 0, 12 * [1], 0, 0).forecast) == 36000
assert sum(IPT.Demand_Plan(xyz, 1000, 36, -1, 12 * [1], 0, 0).forecast) == 1000
assert sum(IPT.Demand_Plan(xyz, 1000, 36, -.5, 12 * [1], 0, 0).forecast)  == 1994

assert sum(IPT.Demand_Plan(xyz, 1000, 36, 0, 12 * [1], .5, 1).sd_forecast) == 18000

demand_plan_1.plot(xyz.title_name, demand_plan_1.months, demand_plan_1.forecast, outfile_name = "test_forecast.png", saveflag = True, showflag = False)  # this works


print "\n------------- test class Aggressive_Demand_Plan -------------"

# demand_plan_2 = Aggressive_Demand_Plan(xyz, starting_monthly_demand, number_months, trendPerMonth, seasonCoeffs, initial_cv, per_period_cv)

assert len(IPT.Aggressive_Demand_Plan(xyz, 1000, 36, 0, 12 * [1], 0, 0).months) == 36
assert len(IPT.Aggressive_Demand_Plan(xyz, 1000, 36, 0, 12 * [1], 0, 0).forecast) == 36
assert len(IPT.Aggressive_Demand_Plan(xyz, 1000, 36, 0, 12 * [1], 0, 0).sd_forecast) == 36

assert sum(IPT.Aggressive_Demand_Plan(xyz, 1000, 36, 0, 12 * [1], 0, 0).forecast) == 72000
assert sum(IPT.Aggressive_Demand_Plan(xyz, 1000, 36, -1, 12 * [1], 0, 0).forecast) == 2000
assert sum(IPT.Aggressive_Demand_Plan(xyz, 1000, 36, -.5, 12 * [1], 0, 0).forecast)  == 3994

print "\n------------- test class Conservative_Demand_Plan -------------"

# demand_plan_3 = Conservative_Demand_Plan(xyz, starting_monthly_demand, number_months, trendPerMonth, seasonCoeffs, initial_cv, per_period_cv)

assert len(IPT.Conservative_Demand_Plan(xyz, 1000, 36, 0, 12 * [1], 0, 0).months) == 36
assert len(IPT.Conservative_Demand_Plan(xyz, 1000, 36, 0, 12 * [1], 0, 0).forecast) == 36
assert len(IPT.Conservative_Demand_Plan(xyz, 1000, 36, 0, 12 * [1], 0, 0).sd_forecast) == 36

assert sum(IPT.Conservative_Demand_Plan(xyz, 1000, 36, 0, 12 * [1], 0, 0).forecast) == 18000
assert sum(IPT.Conservative_Demand_Plan(xyz, 1000, 36, -1, 12 * [1], 0, 0).forecast) == 500
assert sum(IPT.Conservative_Demand_Plan(xyz, 1000, 36, -.5, 12 * [1], 0, 0).forecast)  == 994


print  "\n------------- test class Returns_Plan -------------"

returns_rate = 0.2
lag = 3

title_retns_test = IPT.Title("xyz", {'perOrder': 80.0, 'WACC': 0.12, 'lost_margin': 10.00, 'allow_POD':True, "Printing": {"POD": (0, 4.7), 
    				"Digital": (100, 1.43), "Conventional": (1625, 1.00),"Offshore":(2000, 1.50)}})
d_plan_retns_test =  IPT.Demand_Plan(xyz, 1000, 36, 0, 12 * [1], 0, 0)

returns_plan_test = IPT.Returns_Plan(title_retns_test, d_plan_retns_test, returns_rate, lag)

assert sum(IPT.Returns_Plan(title_retns_test, d_plan_retns_test, 0, 0).returns) == 0
assert sum(IPT.Returns_Plan(title_retns_test, d_plan_retns_test, .2, 0).returns) == 7200
assert sum(IPT.Returns_Plan(title_retns_test, d_plan_retns_test, 0, 3).returns) == 0
assert sum(IPT.Returns_Plan(title_retns_test, d_plan_retns_test, 1, 3).returns) == 33000
assert sum(IPT.Returns_Plan(title_retns_test, d_plan_retns_test, 1, 0).returns) == 36000
assert sum(IPT.Returns_Plan(title_retns_test, d_plan_retns_test, 1, 36).returns) == 0
assert sum(IPT.Returns_Plan(title_retns_test, d_plan_retns_test, 1, 48).returns) == 0

#print "object returns_plan_1:", returns_plan_1
# print "returns:", returns_plan_1.returns

print  "\n------------- test class Print_Plan -------------"

print_plan_1 = IPT.Print_Plan(xyz)
#print "object print_plan_1:", print_plan_1

print  "\n------------- test class SS_Plan -------------"

title = IPT.Title("xyz", {'perOrder': 80.0, 'WACC': 0.12, 'lost_margin': 10.00, 'allow_POD':True, "Printing": {"POD": (0, 4.7), 
    				"Digital": (100, 1.43), "Conventional": (1625, 1.00),"Offshore":(2000, 1.50)}})
d_plan =  IPT.Demand_Plan(xyz, 100, 36, 0, 12 * [1], 1, 0)

r_plan = IPT.Returns_Plan(title, d_plan, 0.2, 3)

replen_lead_time = 2
target_service_level = 0.99

ss_plan_1 = IPT.SS_Plan(title, d_plan, target_service_level, replen_lead_time)
#print "object ss_plan_1:", ss_plan_1
#print "forecast:", d_plan.forecast
#print "SDs:", ss_plan_1.calc_leadtime_sd(0, d_plan.forecast, d_plan.initial_cv, d_plan.per_period_cv)
assert len(ss_plan_1.calc_leadtime_sd(replen_lead_time, d_plan.forecast, d_plan.initial_cv, d_plan.per_period_cv)) == 36
assert "SDs:", ss_plan_1.calc_leadtime_sd(0, d_plan.forecast, 1, 0)[35] == 0
assert "SDs:", ss_plan_1.calc_leadtime_sd(1, d_plan.forecast, 1, 0)[35] == (1 * 100**2)**0.5
assert "SDs:", ss_plan_1.calc_leadtime_sd(2, d_plan.forecast, 1, 0)[35] == (2 * 100**2)**0.5
assert "SDs:", ss_plan_1.calc_leadtime_sd(3, d_plan.forecast, 1, 0)[35] == (3 * 100**2)**0.5
assert "SDs:", ss_plan_1.calc_leadtime_sd(2.5, d_plan.forecast, 1, 0)[35] == (2 * 100**2+ (0.5*100)**2)**0.5

# test the multiplier
# test the rops

#print "object ss_plan_1 calc reorder points:", ss_plan_1.calc_reorder_points(d_plan, target_service_level, replen_lead_time)


assert "reorder points:", IPT.SS_Plan(title, d_plan, .99, 1).reorder_point[35] == 232
assert "reorder points:", IPT.SS_Plan(title, d_plan, .99, 2).reorder_point[35] == 328
assert "reorder points:", IPT.SS_Plan(title, d_plan, .99, 3).reorder_point[35]  == 402

assert "\nreorder points:", IPT.SS_Plan(title, d_plan, .95, 1).reorder_point[35] == 164
assert "reorder points:", IPT.SS_Plan(title, d_plan, .95, 2).reorder_point[35] == 232
assert "reorder points:", IPT.SS_Plan(title, d_plan, .95, 3).reorder_point[35] == 284

assert "\nreorder points:", IPT.SS_Plan(title, d_plan, .50, 1).reorder_point[35] == 0
assert "reorder points:", IPT.SS_Plan(title, d_plan, .50, 2).reorder_point[35] == 0
assert "reorder points:", IPT.SS_Plan(title, d_plan, .50, 3).reorder_point[35]  == 0


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
