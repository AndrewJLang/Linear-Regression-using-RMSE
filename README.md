Andrew Lang
CS 491 - Machine Learning

Instructions: 

This program was compiled and ran on python 3.7.1. The libaries of matplotlib.pyplot, numpy, pandas, time and csv were used (may need to pip install).

Time and csv installation can be taken out if the user doesn't want need write data to csv file (comment out corresponding code).

The  code begins by reading in the data from a desired file, then initializes the learning rate to 0.1 and b1 and b0 values to 0 and 0.  These can be changed as desired, but recommended to stay as is.

The only changes to a variable the user can/should make is for the batch_size variable, which can be set as desired (currently set to 20).

The program is set to run through 500 epochs, and an inner for loop iterates through the data points and adjust the gradient optimization every time the batch size is hit (look for use of count variable and if statement inside this loop).

The graph is then updated accordingly.

The R-squared value is calculated at the end of the program to see how accurate the data is for the line.