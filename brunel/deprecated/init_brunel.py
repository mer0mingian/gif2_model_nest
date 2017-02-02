try: 
	reload(gif2_brunel_f)
except:
	from gif2_brunel_f import *

execfile('better_configuration_finder.py')

print('Importing complete.')
configure_nest_kernel()

print('Nest Kernel configures')
networkdict = convert_to_paramdict(testconfigs1[0,:])

print('Chosen network parameters:')
for i, j in networkdict.iteritems():
	print('{0} = {1}'.format(i,j))
print('\n')

print('Simulating')
resultlist, spikelists = run_brunel(networkdict)

print('Done. Congregating results. \n')
resultarray = construct_resultarray(resultlist, networkdict)

print(resultarray)