import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats.mstats import mode

# The data describes the rate of high schoolers who take the SAT, and the
# average Math and Verbal scores by state.

# The data looks complete. The only obvious issue is some states are more likely
# to use the ACT over the SAT, so those states number may not be representative
# of the average high schooler.

# Load the data into a list of lists
scores = []
with open('sat_scores.csv','U') as f:
    reader = csv.reader(f)
    for row in reader:
        scores.append(row)

# Print the data
print scores

# Extract a list of the labels from the data, and remove them
header = scores[0]
print '\n', header, '\n'
del scores[0]

# Create a list of state names extracted from the data
states = []
for i in range(len(header)):
    if header[i] == 'State':
        col = i
for row in scores:
    states.append(row[col])

# Print the types of each column
for i in range(len(scores[0])):
    print "Column %d has type: %s" % (i,str(type(scores[0][i]))[7:10])

# Reassigning the types
for row in range(len(scores)):
    for i in range(1,4):
        scores[row][i] = int(scores[row][i])

# Create a dictionary for each column mapping the state to its
# respective value for that column
rate = {}
verbal = {}
math = {}
for row in scores:
    rate[row[0]] = row[1]
    verbal[row[0]] = row[2]
    math[row[0]] = row[3]

# Print the min and max of each column
ratemax = max(val for key,val in rate.iteritems())
ratemin = min(val for key,val in rate.iteritems())
verbalmax = max(val for key,val in verbal.iteritems())
verbalmin = min(val for key,val in verbal.iteritems())
mathmax = max(val for key,val in math.iteritems())
mathmin = min(val for key,val in math.iteritems())
print "\nRate: max = %d, min = %d" % (ratemax,ratemin)
print "Verbal: max = %d, min = %d" % (verbalmax,verbalmin)
print "Math: max = %d, min = %d" % (mathmax,mathmin)

# Write a function using only list comprehensions to compute the Std Deviation
def computeStd(d):
    return np.std(np.array([val for key,val in d.iteritems()]))

print "\nThe standard deviation of Rate is %f" % computeStd(rate)
print "The standard deviation of Verbal is %f" % computeStd(verbal)
print "The standard deviation of Math is %f" % computeStd(math)

# Computing some more summary statistics:
print "\nRate: mean = %d, median = %d, mode = %d" % (np.mean(rate.values()), np.median(rate.values()), mode(rate.values())[0])
print "Math: mean = %d, median = %d, mode = %d" % (np.mean(math.values()), np.median(math.values()), mode(math.values())[0])
print "Verbal: mean = %d, median = %d, mode = %d" % (np.mean(verbal.values()), np.median(verbal.values()), mode(verbal.values())[0])

# Using MatPlotLib and PyPlot, plot the distribution of the Rate using histograms.
plt.hist([val for val in rate.values()])
plt.xlabel('Percent of Students Taking SAT')
plt.ylabel('Number of States')
plt.title('Rate of Participation')
plt.axis([0,100,0,17])
plt.grid(True)
plt.show()

# Plot the Math distribution
plt.hist([val for val in math.values()])
plt.xlabel('Average Score on Math Portion')
plt.ylabel('Number of States')
plt.title('Math')
plt.axis([400,650,0,14])
plt.grid(True)
plt.show()

# Plot the Verbal distribution
plt.hist([val for val in verbal.values()])
plt.xlabel('Average Score on Verbal Portion')
plt.ylabel('Number of States')
plt.title('Verbal')
plt.axis([400,650,0,10])
plt.grid(True)
plt.show()

# What is the typical assumption for data distribution
#     Normal

# Does that distribution hold true for our data?
#     The math distribution looks normal, the verbal distribution doesn't

# Plot some scatterplots:
plt.scatter([val for val in math.values()],[val for val in verbal.values()])
plt.xlabel('Math')
plt.ylabel('Verbal')
plt.title('Math vs Verbal SAT Scores')
plt.show()

plt.scatter([val for val in rate.values()],[val for val in math.values()])
plt.xlabel('Rate')
plt.ylabel('Math')
plt.title('Rate vs Math SAT Scores')
plt.show()

plt.scatter([val for val in rate.values()],[val for val in verbal.values()])
plt.xlabel('Rate')
plt.ylabel('Verbal')
plt.title('Rate vs Verbal SAT Scores')
plt.show()

# Are there any interesting relationships to note?
    # There is a positive correlation between Math and Verbal SAT Scores
    # There is a negative correlation between Rate and Math/Verbal SAT Scores

# Create box plots for each variable
plt.figure()
plt.boxplot([val for val in rate.values()], vert=False)
plt.xlabel('Rate')
plt.title('Rate of SAT Participation')
plt.show()

plt.figure()
plt.boxplot([val for val in math.values()], vert=False)
plt.xlabel('Math Score')
plt.title('Average SAT Math Score')
plt.show()

plt.figure()
plt.boxplot([val for val in verbal.values()], vert=False)
plt.xlabel('Verbal Score')
plt.title('Average SAT Verbal Score')
plt.show()

# Making all the boxplots show on the same image
fig = plt.figure()
ax1 = fig.add_subplot(311)
ax1.boxplot([val for val in rate.values()], vert=False)
ax1.set_title('Rate')

ax2 = fig.add_subplot(312)
ax2.boxplot([val for val in math.values()], vert=False)
ax2.set_title('Average SAT Math Score')

ax3 = fig.add_subplot(313)
ax3.boxplot([val for val in verbal.values()], vert=False)
ax3.set_title('Average SAT Verbal Score')

fig.subplots_adjust(hspace = 1)
plt.show()

# Recreating the graphs in seaborn
sns.distplot([val for val in rate.values()], bins = [10*x for x in range(11)], kde = False, norm_hist = False)
plt.axis([0,100,0,20])
plt.xlabel('Percent of Students Taking SAT')
plt.ylabel('Number of States')
plt.title('Rate of Participation')
plt.show()

sns.distplot([val for val in math.values()], bins = [i for i in range(400,650,10)],kde = False, norm_hist = False)
plt.axis([400,650,0,11])
plt.xlabel('Average Score on Math Portion')
plt.ylabel('Number of States')
plt.title('Math')
plt.show()

sns.distplot([val for val in verbal.values()], bins=[i for i in range(450,600,10)], kde=False, norm_hist=False)
plt.axis([450,600,0,10])
plt.xlabel('Average Score on Verbal Portion')
plt.ylabel('Number of States')
plt.title('Verbal')
plt.show()

df = pd.DataFrame()
df['Rate'] = [val for val in rate.values()]
df['Math'] = [val for val in math.values()]
df['Verbal'] = [val for val in verbal.values()]

sns.regplot('Math','Verbal', data=df)
plt.title('Math vs Verbal SAT Scores')
plt.show()

sns.regplot('Rate','Math', data=df)
plt.title('Rate of Participation vs Math SAT Scores')
plt.show()

sns.regplot('Rate','Verbal', data=df)
plt.title('Rate of Participation vs Verbal SAT Scores')
plt.show()

sns.boxplot('Rate',data=df,width=.2)
plt.title('Rate of Participation')
plt.show()

sns.boxplot('Math',data=df,width=.2)
plt.title('Average SAT Math Score')
plt.show()

sns.boxplot('Verbal',data=df,width=.2)
plt.title('Average SAT Verbal Score')
plt.show()
