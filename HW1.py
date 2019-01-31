#Andrew Lang
#Linear Regression using RMSE algorithm


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import csv

plt.xlabel('Days Fished')
plt.ylabel('Sockeye Harvest')
plt.title('Days fished vs Harvest Count')

data = pd.read_csv('dip-har-eff.csv')

rows = data.shape [0]
cols = data.shape [1]

data = data.values
data = data[np.arange(0,rows),:]

X = data[:, 1]
Y = data[:, 2]
X_max = np.max(X)
Y_max = np.max(Y)
X = np.true_divide(X, X_max)
Y = np.true_divide(Y, Y_max)
plt.xlim(0,max(X))
plt.ylim(0,max(Y))

#learning rate is constant and b1 and b0 are set to start at 0
learning_rate = 0.1
b1 = 0
b0 = 0

plt.scatter(X,Y)
plt.plot (X, b1*X + b0)
print (" b0: " + str(b0) + " b1: " + str(b1) + " Error: " + str(0))
plt.pause(0.1)

#batch size, will need to be changed accordingly to what size is wanting to be run
batch_size = 20

#checks to see if the batch size is greater than the dataset, if so, then just does a batch instead of mini batch
if batch_size > len(X):
    batch_size = len(X)

ERROR = 0
b1_temp = 0
b0_temp = 0

#Separate graph that shows the line adjusting and redraws a new line every time
plt.ion()
fig  = plt.figure(figsize=(6, 6))
sub = fig.add_subplot(221)
X_max = np.max(X)
Y_max = np.max(Y)
X_income = np.true_divide (X, X_max)
Y_income = np.true_divide (Y, Y_max)
sub.set_xlim(0,max(X))
sub.set_ylim(0,max(Y))
sub.scatter(X, Y)

#equation that calculates the rmse used within the algorithm
def RMSE(predict, actual):
    return np.sqrt(((predict - actual) ** 2).mean())

errors = []
times = []

start_time = time.time()

#Epoch counter
for x in range(500):
    print("Epoch #", x)
    
    ERRORS = RMSE([(b1*x)+b0 for x in X], Y)
    print (" b0: " + str(b0) + " b1: " + str(b1) + " Error: " , RMSE([(b1*x)+b0 for x in X], Y))

    count = 0

    errors.append(ERRORS)
    times.append(time.time() - start_time)

    #Runs through the data points every epoch
    for i in range(len(X)):

        ERROR = ERROR + (b1*X[i] + b0 - Y[i])**2

        b1_temp = b1_temp + (1/2)*((1/len(X))*(b1*X[i] + b0 - Y[i])**2)**(-1/2) * (2/len(X))*(b1*X[i] + b0 - Y[i])*X[i]
        b0_temp = b0_temp + (1/2)*((1/len(X))*(b1*X[i] + b0 - Y[i])**2)**(-1/2) * (2/len(X))*(b1*X[i] + b0 - Y[i])

        count += 1  

        #Once batch size is hit, line is adjusted
        if count == batch_size or i == len(X)-1:
            ERROR = (ERROR / batch_size)**(1/2)

            b1_temp = b1_temp / batch_size
            b0_temp = b0_temp / batch_size

            b0 = b0 - learning_rate * b0_temp
            b1 = b1 - learning_rate * b1_temp
            
            b1_temp = 0
            b0_temp = 0
            count = 0

    #Exports the data of error versus time to a csv file so it can be graphed in Excel 
    data_out = []
    for x in range(len(errors)):
        data_out.append([times[x], errors[x]])
    outd = pd.DataFrame(data_out)
    #outd.to_csv('batch_20-DRIFT.csv') Commented out so it won't write create a file

    sub.clear()
    sub.set_xlim(0,max(X))
    sub.set_ylim(0,max(Y))
    sub.scatter(X, Y)
    X_test = np.arange(0,1,0.1)
    sub.plot (X_test,  b1*X_test + b0)
    plt.pause(0.01)

print("b1 " , b1)
print("b0 " , b0)

#calcuating the R^2 value at the end of the program to see how accurate line is
def squared_error(start,line):
    return sum((line - start) * (line - start))

def determination(start,line):
    mean_line = [np.mean(start) for y in start]
    squared_error_val = squared_error(start, line)
    squared_error_y = squared_error(start, mean_line)
    return 1 - (squared_error_val/squared_error_y)

regression = [(b1*x)+b0 for x in X]
r_squared_value = determination(Y,regression)
print(f'R^2: {r_squared_value}')

plt.show()