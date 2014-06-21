import processing as p
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


print  "1. Start with a forecast of demand"

month, forecast = p.calc_forecast(starting_monthly_demand, number_months, trendPerMonth, seasonCoeffs)

sd_forecast = p.calc_sd_forecast(initial_cv, per_period_cv, forecast)

o.plot_forecast(month, forecast)

print "\n2. ...and a forecast of returns"

returns = p.calc_returns_forecast(forecast, returns_rate, lag)

o.plot_returns(month, forecast, returns)

print "\n3. use planned purchases, demand, and returns to calculate inventory position..."

POD_breakeven = p.calc_POD_breakeven(cost)

orders, POD_orders, start_inv, end_inv, avg_inv = p.determine_plan(month, forecast, returns, cost, order_n_months_supply, [0]* number_months)

o.plot_end_inv(month, end_inv)



print "\n4. ...yielding a lifetime expected cost."

FMC, VMC, POD_VMC, umc, carry_stg_cost, exp_lost_sales_cost = p.calc_costs(forecast, sd_forecast, returns, orders, POD_orders, avg_inv, start_inv, cost)

print "\tExpected cumulative lost sales (units):", sum([int(lost_dollars/cost['lost_margin']) for lost_dollars in exp_lost_sales_cost])

print "\t------------------------------"
print "\tTotal FMC:", locale.currency(sum(FMC), grouping=True )
print "\tTotal VMC:", locale.currency(sum(VMC), grouping=True )
print "\tTotal POD VMC:", locale.currency(sum(POD_VMC), grouping=True )
print "\tumc:", locale.currency(umc, grouping=True )
print "\tTotal carrying / storage cost:", locale.currency(sum(carry_stg_cost), grouping=True )
print "\tTotal expected lost sales:", locale.currency(sum(exp_lost_sales_cost), grouping=True )


o.plot_cost_bars(FMC, VMC, POD_VMC, carry_stg_cost)


print  "\n5. But forecasts are wrong..."

demand, lower_CI, upper_CI = p.calc_demand(forecast, sd_forecast)

o.plot_demand(month, forecast, demand, lower_CI, upper_CI)


print  "\n6. ...leading to stockouts with lost sales and expediting..."

end_inv_posn_act, avg_inv_act = p.inv_from_demand(demand, orders, POD_orders, returns)

o.plot_end_inv_posn_act(month, end_inv_posn_act)


print "\n7. ...and additional costs."

print "\tTotal actual FMC:", locale.currency(sum(FMC), grouping=True )
print "\tTotal actual VMC:", locale.currency(sum(VMC), grouping=True )
print "\tTotal actual POD VMC:", locale.currency(sum(POD_VMC), grouping=True )
print "\tactual umc:", locale.currency(umc, grouping=True )
print "\tTotal actual carrying / storage cost:", locale.currency(sum(carry_stg_cost), grouping=True )
print "\tTotal expected lost sales:", locale.currency(sum(exp_lost_sales_cost), grouping=True )

o.plot_cost_bars_2(FMC, VMC, POD_VMC, carry_stg_cost, exp_lost_sales_cost)

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


print "\n8. The usual approach to avoid lost sales is to carry safety stock"

reorder_point = p.calc_reorder_points(target_service_level, replen_lead_time, sd_forecast)

orders_ss, POD_orders_ss, start_inv_ss, end_inv_ss, avg_inv_ss = p.determine_plan(month, forecast, returns, cost, order_n_months_supply, reorder_point)

FMC_ss, VMC_ss, POD_VMC_ss, umc_ss, carry_stg_cost_ss, exp_lost_sales_cost_ss = p.calc_costs(forecast, sd_forecast, returns, orders_ss, POD_orders_ss, avg_inv_ss, start_inv_ss, cost)

print "\tExpected lost sales (units) no SS:", int(sum(exp_lost_sales_cost)/cost['lost_margin'])
print "\tExpected lost sales (units) with SS:", int(sum(exp_lost_sales_cost_ss)/cost['lost_margin'])

o.plot_end_inv_2(month, end_inv, end_inv_ss)


print "\n9. Another is to use POD."
	
#	 - it looks like lost sales, but is cheaper
#  put the lost sales numbers into the POD vector and recalc positions and costs
# Use the lost sales line chart with a note about the cost of a lost sale
#  revalued_lost_sales = [0]*len(forecast)

lost_sales_as_POD = p.calc_lost_sales_as_POD(cost, start_inv, returns, orders, POD_orders, forecast, sd_forecast)

print "\tTotal actual FMC:", sum(FMC)
print "\tTotal actual VMC:", sum(VMC)
print "\tTotal actual POD VMC:", sum(POD_VMC)
print "\tactual umc:", umc
print "\tTotal actual carrying / storage cost:", sum(carry_stg_cost)
print "\tTotal actual lost sales (as POD):", sum(lost_sales_as_POD)

o.plot_end_inv_3(month, end_inv_posn_act)

print "\n10. POD is best."

print  "\nLost sales alternative"

print "\tTotal actual FMC:", locale.currency( sum(FMC), grouping=True )
print "\tTotal actual VMC:", locale.currency( sum(VMC), grouping=True )
print "\tTotal actual POD VMC:", locale.currency( sum(POD_VMC), grouping=True )
print "\tactual umc:", locale.currency( umc, grouping=True )
print "\tTotal actual carrying / storage cost:", locale.currency( sum(carry_stg_cost), grouping=True )
print "\tTotal expected lost sales:", locale.currency( sum(exp_lost_sales_cost), grouping=True )
print "\tGrand total with lost sales:", locale.currency( sum(FMC)+sum(VMC)+sum(POD_VMC)+sum(carry_stg_cost)+
                                                      sum(exp_lost_sales_cost), grouping=True )

print "\nSafety stock alternative"

print "\tTotal actual FMC:", locale.currency( sum(FMC_ss), grouping=True )
print "\tTotal actual VMC:", locale.currency( sum(VMC_ss), grouping=True )
print "\tTotal actual POD VMC:", locale.currency( sum(POD_VMC_ss), grouping=True )
print "\tactual umc:", locale.currency( umc, grouping=True )
print "\tTotal actual carrying / storage cost:", locale.currency( sum(carry_stg_cost_ss), grouping=True )
print "\tTotal expected lost sales:", locale.currency( sum(exp_lost_sales_cost_ss), grouping=True )
print "\tGrand total with safety stock:", locale.currency( sum(FMC_ss)+sum(VMC_ss)+sum(POD_VMC_ss)+sum(carry_stg_cost_ss) + sum(exp_lost_sales_cost_ss), grouping=True )


print "\nPOD alternative"


print "\tTotal actual FMC:", locale.currency( sum(FMC), grouping=True )
print "\tTotal actual VMC:", locale.currency( sum(VMC), grouping=True )
print "\tTotal actual POD VMC:", locale.currency( sum(POD_VMC), grouping=True )
print "\tactual umc:", locale.currency( umc, grouping=True )
print "\tTotal actual carrying / storage cost:", locale.currency( sum(carry_stg_cost), grouping=True )
print "\tTotal actual lost sales (as POD):", locale.currency( sum(lost_sales_as_POD), grouping=True )
print "\tGrand total with lost sales (as POD):", locale.currency( sum(FMC)+sum(VMC)+sum(POD_VMC)+sum(carry_stg_cost)+sum(lost_sales_as_POD), grouping=True )


o.plot_cost_bars_final(FMC, FMC_ss, VMC, VMC_ss, POD_VMC, POD_VMC_ss, carry_stg_cost, carry_stg_cost_ss, exp_lost_sales_cost, exp_lost_sales_cost_ss, lost_sales_as_POD)

print "\ntest section with data frames\n"
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

print "\n11. What does a POD lifecycle look like?"


# Lifetime cost versus base

# *	low volume titles
# *	samples
# *	End-of-life
# *	Custom
# *	Illustration

# <headingcell level=1>

print "\n12. What does a title with dramatic seasonality look like?"

# <codecell>

# Demand forecast

# <codecell>

# Inventory position

# <headingcell level=1>

print "\n13. What happens with systematic over-forecasting?"

# <codecell>

# Actual versus plan with bias

# <codecell>

# lifetime costs versus base

# <headingcell level=1>

print "\n14. What if we're really terrible at forecasting?"

# <codecell>

# Inventory position with higher SS levels

# <codecell>

# Lifetime costs versus base

# <headingcell level=1>

print "\n15. What if we use Economic Order Quantity (EOQ) or other techniques?"

# <codecell>

# Lifetime costs versus base

# <headingcell level=1>

print "\n16. What if we dramatically reduce our print lead times?"

# <codecell>

# Inventory position with lower SS levels

# <codecell>

# Lifetime costs versus base

# <headingcell level=1>

print "\n15. To do this you need infrastructure:"

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

