import numpy as np

exin_rates_lower_bound = 5.0
exin_rates_upper_bound = 7.5
distance_r_to_exin_bound = 2.0
cv_lower_bound = 0.85
cv_upper_bound = 1.15
distance_penalty = 6.0

data = np.loadtxt('brunel_results/brunel_array_results_15.csv')
# find all rows with acceptable E/I rates:
mean_exin = np.mean((data[:,8:9]),axis=1)
mean_exin_accept = np.all((
    mean_exin <= exin_rates_upper_bound, mean_exin >=exin_rates_lower_bound), axis=0)

# is R rate also acceptable?
dist_r_exin = np.abs(data[:,10] - mean_exin)
dist_r_exin_accept = np.abs(dist_r_exin) <= distance_r_to_exin_bound

# where do both criteria hold?
rates_accept = np.all((mean_exin_accept, dist_r_exin_accept), axis=0)
print('We have {0} results with acceptable rates.'.format(rates_accept.sum()))

# if the rates fit, what are the resulting CVs?
cvs = data[:, 14]
cvs_accept = np.all((cvs <= cv_upper_bound, cvs >= cv_lower_bound), axis=0)

all_accept = np.all((rates_accept, cvs_accept), axis=0)
# also acceptable rates?
print('{0} among these have CVs between {1} and {2}'.format(
    all_accept.sum(), cv_lower_bound, cv_upper_bound))


# of the remaining configurations, which has...
# ... the largest dV2?
testindices1 = data[ :, 6 ] == np.amax(data[ all_accept, 6 ])
testconfigs1 = data[ testindices1, : ]
# ... the largest total dV?
testvalue2 =  np.amax(data[ all_accept, 6 ] + data[ all_accept, 5 ])
testindices2 = (data[ :, 5 ] + data[ :, 6 ]) == testvalue2
testconfigs2 = data[ testindices2, : ]
# ... the lowest RMSE of rate difference and dV total?
# testvalue3 = np.sqrt((distance_penalty * np.ones_like(data[ 
#   all_accept, 6 ]) - data[ all_accept, 6 ])**2 + dist_r_exin[ all_accept ]**2)
# testindices3 = np.sqrt((distance_penalty * np.ones_like(data[ :, 6 ]) - data[ 
#   :, 6 ])**2 + dist_r_exin**2) == np.amin(testvalue3)
# testconfigs3 = data[ testindices3, : ]
# ... the lower RMSE of rate difference and dV2?
testvalue3 = np.sqrt((distance_penalty * np.ones_like(data[ 
    all_accept, 6 ]) - data[ all_accept, 6 ])**2 + dist_r_exin[ all_accept ]**2)
testindices3 = np.sqrt((distance_penalty * np.ones_like(data[ :, 6 ]) - data[ 
    :, 6 ])**2 + dist_r_exin**2) == np.amin(testvalue3)
testconfigs3 = data[ testindices3, : ]

def convert_to_paramdict(input):
    networkparamdict = {
        'p_rate':   0,
        'C_m':      input[ 1 ],
        'g':        input[ 2 ],
        'g_1':      input[ 3 ],
        'tau_1':    input[ 4 ],
        'V_dist':   input[ 5 ],
        'V_dist2':  input[ 6 ],
        'K_ext':   {'E': 2000, 'I': 1900, 'R': 2000},
        'bg_rate':  8., 
        'fraction': input[ 7 ]}
    return networkparamdict