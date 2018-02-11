import matplotlib
matplotlib.use('Agg')
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np 
import statistics as st


def boostrap(statistic_func, iterations, data):
	samples  = np.random.choice(data,replace = True, size = [iterations, len(data)])
	#print samples.shape
	data_mean = statistic_func(data)
	vals = []
	for sample in samples:
		sta = statistic_func(sample)
		#print sta
		vals.append(sta)
	b = np.array(vals)
	#print b
	lower, upper = np.percentile(b, [2.5, 97.5])
	return data_mean,lower, upper


if __name__ == "__main__":

	dataset = pd.read_csv ('vehicles.csv')
	dataset["Count"] = list(range(0, len(dataset), 1)) # Add in index column for plotting facility
	#dataset1 = dataset.values[:, 0] # extract current fleet MPG
	dataset1 = dataset.iloc[0:, 0:1]
	dataset2=dataset.iloc[0:79,1:]
	print(dataset1)
	print(dataset2)
	#pd.set_option('display.max_rows',len(test)) # To display maximum rows in ouptput window

	#dataset2 = dataset.values[:, 1]
	#dataset2 = dataset[:79]  # extract New Fleet MPG
	#print(dataset2)


	# Scatter plot for current fleet

	sns_plotCurrentFleet = sns.lmplot(dataset.columns[0],dataset.columns[2] , data=dataset, fit_reg=False)

	sns_plotCurrentFleet.axes[0,0].set_ylim(0,)
	sns_plotCurrentFleet.axes[0,0].set_xlim(0,)

	sns_plotCurrentFleet.savefig("CurrentFleetScatterPlot.png",bbox_inches='tight')


	#plt.clf()#clean the figure
	#sns_plot2 = sns.distplot(data, bins=20, kde=False, rug=True).get_figure()#a che serve bins

	# New fleet Scatterplot

	sns_plotNewFleet = sns.lmplot(dataset.columns[1],dataset.columns[2] , data=dataset, fit_reg=False)

	sns_plotNewFleet.axes[0,0].set_ylim(0,)
	sns_plotNewFleet.axes[0,0].set_xlim(0,)

	sns_plotNewFleet.savefig("NewFleetScatterPlot.png",bbox_inches='tight')

	# Calculate Standard deviations both current and new fleet
	std1=np.std(dataset.values[:,0])
	std2=np.std(dataset2.values[:,0])
	print ('Standard deviation of current fleet',std1)
	print ('Standard deviation of New fleet',std2)

	data = dataset.values.T[0]# transpose current fleets
	#print (data)
	boots = []
	for i in range(10,50,1):
			boot = boostrap(np.std, i, data)
			boots.append([i,boot[0], "std"])
			boots.append([i,boot[1], "lower"])
			boots.append([i,boot[2], "upper"])
			#print (boot)

	
	df_boot = pd.DataFrame(boots,  columns=['Current Fleet STDV Boundaries','std',"Value"])
	sns_plot = sns.lmplot(df_boot.columns[0],df_boot.columns[1], data=df_boot, fit_reg=False,  hue="Value",legend=True)
	sns_plot.axes[0,0].set_ylim(0,)
	sns_plot.axes[0,0].set_xlim(0,50)


	sns_plot.savefig("bootstrap_confidence-CurrentFleet.png",bbox_inches='tight')



	data = dataset2.values.T[0]#traspone la matrice
	#print (data)
	boots = []
	for i in range(10,50,1):
			boot = boostrap(np.std, i, data)
			boots.append([i,boot[0], "std"])
			boots.append([i,boot[1], "lower"])
			boots.append([i,boot[2], "upper"])
			#print (boot)

	
	df_boot = pd.DataFrame(boots,  columns=['New Fleet STDV Boundary','std',"Value"])
	sns_plot = sns.lmplot(df_boot.columns[0],df_boot.columns[1], data=df_boot, fit_reg=False,  hue="Value")

	sns_plot.axes[0,0].set_ylim(0,)
	sns_plot.axes[0,0].set_xlim(0,50)


	sns_plot.savefig("bootstrap_confidence-NewFleet.png",bbox_inches='tight')


