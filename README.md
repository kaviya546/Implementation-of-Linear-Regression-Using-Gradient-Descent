# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Developed by:   KAVIYA SNEKA M
RegisterNumber:  212223040091
*/
```
```C
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("ex1.txt",header=None)
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
    m=len(y) 
    h=X.dot(theta) 
    square_err=(h-y)**2
    return 1/(2*m)*np.sum(square_err) 

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(X,y,theta) 

def gradientDescent(X,y,theta,alpha,num_iters):
    m=len(y)
    J_history=[] #empty list
    for i in range(num_iters):
        predictions=X.dot(theta)
        error=np.dot(X.transpose(),(predictions-y))
        descent=alpha*(1/m)*error
        theta-=descent
        J_history.append(computeCost(X,y,theta))
    return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
    predictions=np.dot(theta.transpose(),x)
    return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For Population = 35000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For Population = 70000, we predict a profit of $"+str(round(predict2,0)))

```


## Output:

profile prediction:

![image](https://github.com/kaviya546/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150368823/c3fe8bcd-7347-4e86-8b93-59708caff22f)

Function:

![image](https://github.com/kaviya546/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150368823/4dd8198f-12e0-4ef2-a523-f5c400182e85)

Gradient descent:

![image](https://github.com/kaviya546/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150368823/e439155e-56c0-43fe-aa19-0ac381bdc5f4)

cost function using gradient descent:

![image](https://github.com/kaviya546/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150368823/c76ddd5c-b52f-4384-a71f-b95e07423122)

linear regression using profile prediction:

![image](https://github.com/kaviya546/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150368823/6e733bd2-fa22-4f63-b88f-9c1e33464425)

profile prediction for the population of 35000 :

![image](https://github.com/kaviya546/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150368823/441ab910-317a-4309-be32-e880e409c512)

##profile prediction for the population of 70000:

![image](https://github.com/kaviya546/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150368823/7f580853-356e-4320-b049-74c2eede984c)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
