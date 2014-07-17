

forecast = [1] * 10
sd = [1, 2, 3, 4, 5, 6, 7, 8]

initial_cv = 0.15
per_period_cv = 0.015

def calc_leadtime_sd(r, i, sd, result = []):  # pass original i along in function in order to get delta i for cv
	# because of this, an iterative approach might be better than recursive
	if r <= 0:
		return sum(result)
	else:
		fract_period = r % 1
		number_periods = int(r)
		if fract_period:
			result = result + [fract_period * sd[i]]
			r = number_periods + 1
		else:
			result = result + [sd[i]]
		print r, i, result
		return calc_leadtime_sd(r-1, i-1, sd, result)

print calc_leadtime_sd(2.5, 4, sd)

print sd

print "-------------------"

round_up = lambda num: int(int(num + 1) if int(num) != num else int(num))

def calc_leadtime_sd_iter(r, i, sd): 
	''' Calculates the cumulative standard deviation of a forecast made r periods ago,
	where r is the replenishment lead time.

	if i = 5 and r = 3, forecast was generated in period 1
	forecast for i = 5, horizon = 3
	forecast for i = 4, horizon = 2
	forecast for i = 2, horizon = 1'''
	result = []
	fract_period = r % 1
	number_periods = round_up(r)
	for j in range(int(i-r+1), i+1):
		horizon = j-i+number_periods
		if fract_period and j == i:
			result += [fract_period * sd[j]]
		else:
	 		result += [sd[j]]
	 	print j, horizon, sd[j], result
	return sum(result)

print calc_leadtime_sd_iter(2.5, 5, sd)
#  now fix the index overruns!
# reverse the partial r
# use the cv++ approach versus sd - longer forecasts.
