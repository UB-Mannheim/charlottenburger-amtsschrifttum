# import the required libraries
import random
import matplotlib.pyplot as plt

def thresholded_gauss(vmin, vmax, scale=4):
    vmin, vmax = (vmin, vmax) if vmin < vmax else (vmax, vmin)
    mu, sigma = (vmin+vmax)/2, abs(vmax-vmin)/scale
    value = random.gauss(mu, sigma)
    while not vmin <= value <= vmax:
        value = random.gauss(mu, sigma)
    return value

# store the random numbers in a list
nums = []
mu = 100
sigma = 50

for i in range(10000):
    temp =  thresholded_gauss(-20, 2)
    nums.append(temp)

# plotting a graph
plt.hist(nums, bins = 200)
plt.show()
