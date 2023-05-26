# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the linear regression using gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. To implement the linear regression using the standard libraries in the python.

2. Use the .isnull() function to check the empty.

3. Use the default function.

4. Open the head() function.

5. Use the loop function for a linear equation.

6. Predict the value for the y.

7. Print the program.

8. Plot the graph by using scatters keyword.

9. End the program.

## Program:
```
Program to implement the linear regression using gradient descent. 
```
~~~
/*
Program to implement the linear regression using gradient descent.
Developed by: REVATHI
RegisterNumber: 212221240045 
*/
``

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data=pd.read_csv('ex1.txt',header=None)


plt.scatter(data[0],data[1],color="green")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of city (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit prediction")


def computeCost(x,y,theta):
  """
  Take in a numpy array x, y, theta and generate the cost function of function using theta as parameter 
  in a linear regression model
  """

  m=len(y)
  h=x.dot(theta) #length of the training data
  square_err=(h - y)**2
  return 1/(2*m) * np.sum(square_err) # returning 3
  
data_n=data.values
m=data_n[:,0].size
x=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1);
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(x,y,theta) # call the function


def gradientDescent(x,y,theta,alpha,num_iters):
  """
  Take in numy array x, y and theta and update theta by taking number_iters gradient steps with learning rate of alpha

  return theta and the list of the cost of theta during each iteration
  """

  m=len(y)
  J_history=[]

  for i in range(num_iters):
    predictions = x.dot(theta)
    error = np.dot(x.transpose(),(predictions -y))
    descent=alpha * 1/m * error
    theta -= descent
    J_history.append(computeCost(x,y,theta))

  return theta,J_history
  

theta,J_history = gradientDescent(x,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")


plt.plot(J_history,color="orange")
plt.xlabel("Iteration")
plt.ylabel("$J(\theta)$")
plt.title("Cost Function using Gradient Descent")


plt.scatter(data[0],data[1],color="green")
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="magenta")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of the city (10,000s)")
plt.ylabel("Profit($10,000)")
plt.title("Profit Prediction")


def predict(x, theta):
  """
  Takes in numpy array of x and theta and return the predicted value of y based on theta
  """

  predictions = np.dot(theta.transpose(),x)

  return predictions[0]
  
  
predict1 = predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))


predict2 = predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))

*/
~~~
    


## Output:

![image](https://github.com/Revathi-Dayalan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/96000574/c490180f-4184-482d-bfb7-c856d940d506)

![image](https://github.com/Revathi-Dayalan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/96000574/84b944d4-9e88-4f85-bcb7-53ab34449a8d)

![image](https://github.com/Revathi-Dayalan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/96000574/7a2adecb-f7aa-4d9b-8ea8-8aa067d0f841)


![image](https://github.com/Revathi-Dayalan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/96000574/525347b0-a3e8-4d34-923b-85a90f51f8b1)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
