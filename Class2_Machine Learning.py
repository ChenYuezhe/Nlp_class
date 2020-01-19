{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Machine Learning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.datasets import load_boston\n",
    "import random\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset = load_boston()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "metadata": {},
   "outputs": [],
   "source": [
    "x,y=dataset['data'],dataset['target']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(506, 13)"
      ]
     },
     "execution_count": 92,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(506,)"
      ]
     },
     "execution_count": 93,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(13,)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([2.7310e-02, 0.0000e+00, 7.0700e+00, 0.0000e+00, 4.6900e-01,\n",
       "       6.4210e+00, 7.8900e+01, 4.9671e+00, 2.0000e+00, 2.4200e+02,\n",
       "       1.7800e+01, 3.9690e+02, 9.1400e+00])"
      ]
     },
     "execution_count": 94,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print(x[1].shape)\n",
    "x[1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',\n",
       "       'TAX', 'PTRATIO', 'B', 'LSTAT'], dtype='<U7')"
      ]
     },
     "execution_count": 95,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset.feature_names # 特征值名字"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\".. _boston_dataset:\\n\\nBoston house prices dataset\\n---------------------------\\n\\n**Data Set Characteristics:**  \\n\\n    :Number of Instances: 506 \\n\\n    :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.\\n\\n    :Attribute Information (in order):\\n        - CRIM     per capita crime rate by town\\n        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.\\n        - INDUS    proportion of non-retail business acres per town\\n        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)\\n        - NOX      nitric oxides concentration (parts per 10 million)\\n        - RM       average number of rooms per dwelling\\n        - AGE      proportion of owner-occupied units built prior to 1940\\n        - DIS      weighted distances to five Boston employment centres\\n        - RAD      index of accessibility to radial highways\\n        - TAX      full-value property-tax rate per $10,000\\n        - PTRATIO  pupil-teacher ratio by town\\n        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town\\n        - LSTAT    % lower status of the population\\n        - MEDV     Median value of owner-occupied homes in $1000's\\n\\n    :Missing Attribute Values: None\\n\\n    :Creator: Harrison, D. and Rubinfeld, D.L.\\n\\nThis is a copy of UCI ML housing dataset.\\nhttps://archive.ics.uci.edu/ml/machine-learning-databases/housing/\\n\\n\\nThis dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.\\n\\nThe Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic\\nprices and the demand for clean air', J. Environ. Economics & Management,\\nvol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics\\n...', Wiley, 1980.   N.B. Various transformations are used in the table on\\npages 244-261 of the latter.\\n\\nThe Boston house-price data has been used in many machine learning papers that address regression\\nproblems.   \\n     \\n.. topic:: References\\n\\n   - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.\\n   - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.\\n\""
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset['DESCR']  # 用这个可以查看特征值名字代表什么意思"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([6.575, 6.421, 7.185, 6.998, 7.147, 6.43 , 6.012, 6.172, 5.631,\n",
       "       6.004, 6.377, 6.009, 5.889, 5.949, 6.096, 5.834, 5.935, 5.99 ,\n",
       "       5.456, 5.727, 5.57 , 5.965, 6.142, 5.813, 5.924, 5.599, 5.813,\n",
       "       6.047, 6.495, 6.674, 5.713, 6.072, 5.95 , 5.701, 6.096, 5.933,\n",
       "       5.841, 5.85 , 5.966, 6.595, 7.024, 6.77 , 6.169, 6.211, 6.069,\n",
       "       5.682, 5.786, 6.03 , 5.399, 5.602, 5.963, 6.115, 6.511, 5.998,\n",
       "       5.888, 7.249, 6.383, 6.816, 6.145, 5.927, 5.741, 5.966, 6.456,\n",
       "       6.762, 7.104, 6.29 , 5.787, 5.878, 5.594, 5.885, 6.417, 5.961,\n",
       "       6.065, 6.245, 6.273, 6.286, 6.279, 6.14 , 6.232, 5.874, 6.727,\n",
       "       6.619, 6.302, 6.167, 6.389, 6.63 , 6.015, 6.121, 7.007, 7.079,\n",
       "       6.417, 6.405, 6.442, 6.211, 6.249, 6.625, 6.163, 8.069, 7.82 ,\n",
       "       7.416, 6.727, 6.781, 6.405, 6.137, 6.167, 5.851, 5.836, 6.127,\n",
       "       6.474, 6.229, 6.195, 6.715, 5.913, 6.092, 6.254, 5.928, 6.176,\n",
       "       6.021, 5.872, 5.731, 5.87 , 6.004, 5.961, 5.856, 5.879, 5.986,\n",
       "       5.613, 5.693, 6.431, 5.637, 6.458, 6.326, 6.372, 5.822, 5.757,\n",
       "       6.335, 5.942, 6.454, 5.857, 6.151, 6.174, 5.019, 5.403, 5.468,\n",
       "       4.903, 6.13 , 5.628, 4.926, 5.186, 5.597, 6.122, 5.404, 5.012,\n",
       "       5.709, 6.129, 6.152, 5.272, 6.943, 6.066, 6.51 , 6.25 , 7.489,\n",
       "       7.802, 8.375, 5.854, 6.101, 7.929, 5.877, 6.319, 6.402, 5.875,\n",
       "       5.88 , 5.572, 6.416, 5.859, 6.546, 6.02 , 6.315, 6.86 , 6.98 ,\n",
       "       7.765, 6.144, 7.155, 6.563, 5.604, 6.153, 7.831, 6.782, 6.556,\n",
       "       7.185, 6.951, 6.739, 7.178, 6.8  , 6.604, 7.875, 7.287, 7.107,\n",
       "       7.274, 6.975, 7.135, 6.162, 7.61 , 7.853, 8.034, 5.891, 6.326,\n",
       "       5.783, 6.064, 5.344, 5.96 , 5.404, 5.807, 6.375, 5.412, 6.182,\n",
       "       5.888, 6.642, 5.951, 6.373, 6.951, 6.164, 6.879, 6.618, 8.266,\n",
       "       8.725, 8.04 , 7.163, 7.686, 6.552, 5.981, 7.412, 8.337, 8.247,\n",
       "       6.726, 6.086, 6.631, 7.358, 6.481, 6.606, 6.897, 6.095, 6.358,\n",
       "       6.393, 5.593, 5.605, 6.108, 6.226, 6.433, 6.718, 6.487, 6.438,\n",
       "       6.957, 8.259, 6.108, 5.876, 7.454, 8.704, 7.333, 6.842, 7.203,\n",
       "       7.52 , 8.398, 7.327, 7.206, 5.56 , 7.014, 8.297, 7.47 , 5.92 ,\n",
       "       5.856, 6.24 , 6.538, 7.691, 6.758, 6.854, 7.267, 6.826, 6.482,\n",
       "       6.812, 7.82 , 6.968, 7.645, 7.923, 7.088, 6.453, 6.23 , 6.209,\n",
       "       6.315, 6.565, 6.861, 7.148, 6.63 , 6.127, 6.009, 6.678, 6.549,\n",
       "       5.79 , 6.345, 7.041, 6.871, 6.59 , 6.495, 6.982, 7.236, 6.616,\n",
       "       7.42 , 6.849, 6.635, 5.972, 4.973, 6.122, 6.023, 6.266, 6.567,\n",
       "       5.705, 5.914, 5.782, 6.382, 6.113, 6.426, 6.376, 6.041, 5.708,\n",
       "       6.415, 6.431, 6.312, 6.083, 5.868, 6.333, 6.144, 5.706, 6.031,\n",
       "       6.316, 6.31 , 6.037, 5.869, 5.895, 6.059, 5.985, 5.968, 7.241,\n",
       "       6.54 , 6.696, 6.874, 6.014, 5.898, 6.516, 6.635, 6.939, 6.49 ,\n",
       "       6.579, 5.884, 6.728, 5.663, 5.936, 6.212, 6.395, 6.127, 6.112,\n",
       "       6.398, 6.251, 5.362, 5.803, 8.78 , 3.561, 4.963, 3.863, 4.97 ,\n",
       "       6.683, 7.016, 6.216, 5.875, 4.906, 4.138, 7.313, 6.649, 6.794,\n",
       "       6.38 , 6.223, 6.968, 6.545, 5.536, 5.52 , 4.368, 5.277, 4.652,\n",
       "       5.   , 4.88 , 5.39 , 5.713, 6.051, 5.036, 6.193, 5.887, 6.471,\n",
       "       6.405, 5.747, 5.453, 5.852, 5.987, 6.343, 6.404, 5.349, 5.531,\n",
       "       5.683, 4.138, 5.608, 5.617, 6.852, 5.757, 6.657, 4.628, 5.155,\n",
       "       4.519, 6.434, 6.782, 5.304, 5.957, 6.824, 6.411, 6.006, 5.648,\n",
       "       6.103, 5.565, 5.896, 5.837, 6.202, 6.193, 6.38 , 6.348, 6.833,\n",
       "       6.425, 6.436, 6.208, 6.629, 6.461, 6.152, 5.935, 5.627, 5.818,\n",
       "       6.406, 6.219, 6.485, 5.854, 6.459, 6.341, 6.251, 6.185, 6.417,\n",
       "       6.749, 6.655, 6.297, 7.393, 6.728, 6.525, 5.976, 5.936, 6.301,\n",
       "       6.081, 6.701, 6.376, 6.317, 6.513, 6.209, 5.759, 5.952, 6.003,\n",
       "       5.926, 5.713, 6.167, 6.229, 6.437, 6.98 , 5.427, 6.162, 6.484,\n",
       "       5.304, 6.185, 6.229, 6.242, 6.75 , 7.061, 5.762, 5.871, 6.312,\n",
       "       6.114, 5.905, 5.454, 5.414, 5.093, 5.983, 5.983, 5.707, 5.926,\n",
       "       5.67 , 5.39 , 5.794, 6.019, 5.569, 6.027, 6.593, 6.12 , 6.976,\n",
       "       6.794, 6.03 ])"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_rm = x[:,5]    # 只选择第个5个特征 ，即RM\n",
    "X_rm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(506,)"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_rm.shape\n",
    "#initialized parameters\n",
    "\n",
    "k = random.random() * 200 - 100  # -100 100\n",
    "b = random.random() * 200 - 100  # -100 100\n",
    "\n",
    "learning_rate = 1e-3\n",
    "\n",
    "iteration_num = 200 # 迭代次数\n",
    "losses = []\n",
    "for i in range(iteration_num):\n",
    "    \n",
    "    price_use_current_parameters = [price(r, k, b) for r in X_rm]  # \\hat{y}\n",
    "    \n",
    "    current_loss = loss(y, price_use_current_parameters)\n",
    "    losses.append(current_loss)\n",
    "    print(\"Iteration {}, the loss is {}, parameters k is {} and b is {}\".format(i,current_loss,k,b))\n",
    "    \n",
    "    k_gradient = partial_derivative_k(X_rm, y, price_use_current_parameters)\n",
    "    b_gradient = partial_derivative_b(y, price_use_current_parameters)\n",
    "    \n",
    "    k = k + (-1 * k_gradient) * learning_rate\n",
    "    b = b + (-1 * b_gradient) * learning_rate\n",
    "best_k = k\n",
    "best_b = b"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.collections.PathCollection at 0x171d91b4d68>"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# plot the RM（房间数量） with respect to y\n",
    "plt.scatter(X_rm,y)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    ">[plt.scatter()散点图相关用法](https://blog.csdn.net/m0_37393514/article/details/81298503)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Gradient descent"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Assume that the target funciton is a linear function\n",
    "$$ y = k*rm + b$$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [],
   "source": [
    "#define target function\n",
    "def price(rm, k, b):\n",
    "    return k * rm + b"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Define partial derivatives(平均平方误差)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "$$ \\frac{\\partial{loss}}{\\partial{k}} = -\\frac{2}{n}\\sum(y_i - \\hat{y_i})x_i$$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "$$ \\frac{\\partial{loss}}{\\partial{b}} = -\\frac{2}{n}\\sum(y_i - \\hat{y_i})$$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [],
   "source": [
    "# define loss function \n",
    "def loss(y,y_hat):\n",
    "    return sum((y_i - y_hat_i)**2 for y_i, y_hat_i in zip(list(y),list(y_hat)))/len(list(y))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Define partial derivatives### Define partial derivatives"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "$$ \\frac{\\partial{loss}}{\\partial{k}} = -\\frac{2}{n}\\sum(y_i - \\hat{y_i})x_i$$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "$$ \\frac{\\partial{loss}}{\\partial{b}} = -\\frac{2}{n}\\sum(y_i - \\hat{y_i})$$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [],
   "source": [
    "# define partial derivative \n",
    "def partial_derivative_k(x, y, y_hat): \n",
    "    n = len(y)\n",
    "    gradient = 0\n",
    "    for x_i, y_i, y_hat_i in zip(list(x),list(y),list(y_hat)):\n",
    "        gradient += (y_i-y_hat_i) * x_i\n",
    "    return -2/n * gradient\n",
    "\n",
    "def partial_derivative_b(y, y_hat):\n",
    "    n = len(y)\n",
    "    gradient = 0\n",
    "    for y_i, y_hat_i in zip(list(y),list(y_hat)):\n",
    "        gradient += (y_i-y_hat_i)\n",
    "    return -2 / n * gradient"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Iteration 0, the loss is 212613.73100263256, parameters k is -62.041776623451895 and b is -45.898206955669174\n",
      "Iteration 1, the loss is 179199.26949375562, parameters k is -56.21066220574491 and b is -44.98152516346933\n",
      "Iteration 2, the loss is 151037.31564282757, parameters k is -50.85743446230998 and b is -44.13996957922609\n",
      "Iteration 3, the loss is 127302.21386687583, parameters k is -45.942928380969455 and b is -43.367383264470774\n",
      "Iteration 4, the loss is 107298.09556965683, parameters k is -41.431188702529894 and b is -42.65811387017625\n",
      "Iteration 5, the loss is 90438.47758164482, parameters k is -37.28920686675383 and b is -42.00697228333012\n",
      "Iteration 6, the loss is 76229.0675752455, parameters k is -33.486679516819706 and b is -41.40919466261149\n",
      "Iteration 7, the loss is 64253.272342309254, parameters k is -29.995786795452663 and b is -40.86040758541878\n",
      "Iteration 8, the loss is 54159.984063863514, parameters k is -26.79098881070771 and b is -40.35659605125897\n",
      "Iteration 9, the loss is 45653.286488437974, parameters k is -23.848838782318587 and b is -39.89407410740637\n",
      "Iteration 10, the loss is 38483.77922346843, parameters k is -21.147811501563275 and b is -39.469457881923795\n",
      "Iteration 11, the loss is 32441.265784299743, parameters k is -18.668145849632907 and b is -39.07964082675181\n",
      "Iteration 12, the loss is 27348.59102812322, parameters k is -16.391700222345104 and b is -38.72177098974068\n",
      "Iteration 13, the loss is 23056.44729798606, parameters k is -14.301819803467382 and b is -38.39323014934396\n",
      "Iteration 14, the loss is 19438.997002772783, parameters k is -12.383214715602486 and b is -38.09161465931989\n",
      "Iteration 15, the loss is 16390.183295427432, parameters k is -10.621848157169449 and b is -37.81471786329775\n",
      "Iteration 16, the loss is 13820.62068543192, parameters k is -9.00483370707382 and b is -37.56051395055138\n",
      "Iteration 17, the loss is 11654.97442414039, parameters k is -7.520341045732773 and b is -37.32714313486634\n",
      "Iteration 18, the loss is 9829.751831469526, parameters k is -6.157509402696095 and b is -37.11289804806714\n",
      "Iteration 19, the loss is 8291.440809793501, parameters k is -4.906368097632914 and b is -36.91621124865753\n",
      "Iteration 20, the loss is 6994.940969767347, parameters k is -3.7577635933502593 and b is -36.73564375418522\n",
      "Iteration 21, the loss is 5902.241371634991, parameters k is -2.7032925271524393 and b is -36.56987451343201\n",
      "Iteration 22, the loss is 4981.306115881688, parameters k is -1.735240230588709 and b is -36.4176907414065\n",
      "Iteration 23, the loss is 4205.13511084146, parameters k is -0.8465242877905446 and b is -36.27797904642852\n",
      "Iteration 24, the loss is 3550.9724807308353, parameters k is -0.030642719462923296 and b is -36.14971728438989\n",
      "Iteration 25, the loss is 2999.639406119155, parameters k is 0.7183735865640318 and b is -36.03196708059597\n",
      "Iteration 26, the loss is 2534.9718369714265, parameters k is 1.4060045462481923 and b is -35.92386696447673\n",
      "Iteration 27, the loss is 2143.346593066464, parameters k is 2.037280971216265 and b is -35.82462606693981\n",
      "Iteration 28, the loss is 1813.2819579447616, parameters k is 2.6168213749305 and b is -35.73351833425451\n",
      "Iteration 29, the loss is 1535.1010565510044, parameters k is 3.1488657624206455 and b is -35.64987721613467\n",
      "Iteration 30, the loss is 1300.6481474360398, parameters k is 3.637306650791885 and b is -35.573090789157504\n",
      "Iteration 31, the loss is 1103.0495117386456, parameters k is 4.085717547459377 and b is -35.502597279840785\n",
      "Iteration 32, the loss is 936.511928661223, parameters k is 4.497379094460433 and b is -35.43788095462442\n",
      "Iteration 33, the loss is 796.1528291189156, parameters k is 4.875303070119982 and b is -35.37846834668695\n",
      "Iteration 34, the loss is 677.8571479862477, parameters k is 5.222254423669112 and b is -35.32392479199183\n",
      "Iteration 35, the loss is 578.1566781180475, parameters k is 5.540771504025202 and b is -35.273851249220606\n",
      "Iteration 36, the loss is 494.1283890311878, parameters k is 5.833184630730447 and b is -35.227881380327254\n",
      "Iteration 37, the loss is 423.30872914190576, parameters k is 6.1016331429164925 and b is -35.1856788703545\n",
      "Iteration 38, the loss is 363.62139906130176, parameters k is 6.348081051027971 and b is -35.14693496690353\n",
      "Iteration 39, the loss is 313.3164783977105, parameters k is 6.574331405815255 and b is -35.11136622125547\n",
      "Iteration 40, the loss is 270.91912137814955, parameters k is 6.7820394897221385 and b is -35.07871241461841\n",
      "Iteration 41, the loss is 235.18631714084154, parameters k is 6.972724927178583 and b is -35.04873465432792\n",
      "Iteration 42, the loss is 205.07044699181748, parameters k is 7.147782802399248 and b is -35.02121362607283\n",
      "Iteration 43, the loss is 179.68857019282643, parameters k is 7.308493866027262 and b is -34.995947989359024\n",
      "Iteration 44, the loss is 158.29653779754747, parameters k is 7.456033905296573 and b is -34.97275290447245\n",
      "Iteration 45, the loss is 140.2671756024107, parameters k is 7.591482346266369 and b is -34.95145868016426\n",
      "Iteration 46, the loss is 125.07189657713649, parameters k is 7.715830151062798 and b is -34.93190953216435\n",
      "Iteration 47, the loss is 112.26520368598698, parameters k is 7.829987067905391 and b is -34.91396244344047\n",
      "Iteration 48, the loss is 101.47162875157358, parameters k is 7.934788286960453 and b is -34.897486117864325\n",
      "Iteration 49, the loss is 92.37472443323519, parameters k is 8.03100055071664 and b is -34.88236001962955\n",
      "Iteration 50, the loss is 84.70778658547525, parameters k is 8.119327763587137 and b is -34.86847349139382\n",
      "Iteration 51, the loss is 78.24603499343588, parameters k is 8.200416141779117 and b is -34.85572494469325\n",
      "Iteration 52, the loss is 72.80002323926182, parameters k is 8.27485894110771 and b is -34.84402111670608\n",
      "Iteration 53, the loss is 68.21008448901463, parameters k is 8.343200797343847 and b is -34.83327638792799\n",
      "Iteration 54, the loss is 64.34165036104656, parameters k is 8.405941710850637 and b is -34.823412155767045\n",
      "Iteration 55, the loss is 61.081305633844565, parameters k is 8.463540704660442 and b is -34.81435625947546\n",
      "Iteration 56, the loss is 58.33346312479324, parameters k is 8.516419182755712 and b is -34.80604245221081\n",
      "Iteration 57, the loss is 56.01756125354717, parameters k is 8.564964013123245 and b is -34.798409916364264\n",
      "Iteration 58, the loss is 54.06570212783434, parameters k is 8.609530358137967 and b is -34.791402818609924\n",
      "Iteration 59, the loss is 52.4206609048059, parameters k is 8.65044427298373 and b is -34.78496990141989\n",
      "Iteration 60, the loss is 51.03420806614301, parameters k is 8.688005091121576 and b is -34.77906410805656\n",
      "Iteration 61, the loss is 49.86569541918322, parameters k is 8.722487614257888 and b is -34.77364223829853\n",
      "Iteration 62, the loss is 48.88086436827016, parameters k is 8.754144122834582 and b is -34.76866463238134\n",
      "Iteration 63, the loss is 48.050841517087235, parameters k is 8.78320622175037 and b is -34.76409488084075\n",
      "Iteration 64, the loss is 47.351292154917125, parameters k is 8.809886534816679 and b is -34.75989955813568\n",
      "Iteration 65, the loss is 46.76170680863776, parameters k is 8.834380260345098 and b is -34.75604797810194\n",
      "Iteration 66, the loss is 46.264799943499234, parameters k is 8.856866599247285 and b is -34.752511969447724\n",
      "Iteration 67, the loss is 45.84600318371631, parameters k is 8.87751006609551 and b is -34.74926566964824\n",
      "Iteration 68, the loss is 45.493038195058574, parameters k is 8.896461692735755 and b is -34.74628533573161\n",
      "Iteration 69, the loss is 45.195556707155504, parameters k is 8.91386013325918 and b is -34.74354917057178\n",
      "Iteration 70, the loss is 44.94483712164993, parameters k is 8.929832678416089 and b is -34.74103716341747\n",
      "Iteration 71, the loss is 44.73352881131793, parameters k is 8.944496186894014 and b is -34.73873094349055\n",
      "Iteration 72, the loss is 44.555436613487, parameters k is 8.957957940273262 and b is -34.73661364558273\n",
      "Iteration 73, the loss is 44.40533919950489, parameters k is 8.970316427914927 and b is -34.734669786667126\n",
      "Iteration 74, the loss is 44.27883599519166, parameters k is 8.981662067523715 and b is -34.732885152622174\n",
      "Iteration 75, the loss is 44.17221816427198, parameters k is 8.992077866657334 and b is -34.731246694238976\n",
      "Iteration 76, the loss is 44.08235987226234, parameters k is 9.001640030022138 and b is -34.729742431751355\n",
      "Iteration 77, the loss is 44.00662664287631, parameters k is 9.010418516998103 and b is -34.72836136719011\n",
      "Iteration 78, the loss is 43.94279812012889, parameters k is 9.018477553472065 and b is -34.72709340392022\n",
      "Iteration 79, the loss is 43.88900297166982, parameters k is 9.02587610172387 and b is -34.72592927277238\n",
      "Iteration 80, the loss is 43.84366402483485, parameters k is 9.032668291803187 and b is -34.72486046422836\n",
      "Iteration 81, the loss is 43.805452026905805, parameters k is 9.038903817553008 and b is -34.723879166164096\n",
      "Iteration 82, the loss is 43.77324667391795, parameters k is 9.044628300177203 and b is -34.72297820669507\n",
      "Iteration 83, the loss is 43.74610376545259, parameters k is 9.049883622012038 and b is -34.722151001705676\n",
      "Iteration 84, the loss is 43.72322752245396, parameters k is 9.05470823294358 and b is -34.7213915066789\n",
      "Iteration 85, the loss is 43.703947256482905, parameters k is 9.05913743171279 and b is -34.720694172473706\n",
      "Iteration 86, the loss is 43.68769770639136, parameters k is 9.06320362416636 and b is -34.72005390472671\n",
      "Iteration 87, the loss is 43.67400246592746, parameters k is 9.066936560342715 and b is -34.71946602658105\n",
      "Iteration 88, the loss is 43.662460016399116, parameters k is 9.070363552127725 and b is -34.718926244469806\n",
      "Iteration 89, the loss is 43.65273195490045, parameters k is 9.073509673072522 and b is -34.718430616703614\n",
      "Iteration 90, the loss is 43.64453307297414, parameters k is 9.076397941835337 and b is -34.717975524632706\n",
      "Iteration 91, the loss is 43.63762299483515, parameters k is 9.079049490589426 and b is -34.71755764617232\n",
      "Iteration 92, the loss is 43.63179913000456, parameters k is 9.081483719629198 and b is -34.717173931497804\n",
      "Iteration 93, the loss is 43.62689073373699, parameters k is 9.083718439305652 and b is -34.716821580731704\n",
      "Iteration 94, the loss is 43.622753901105696, parameters k is 9.085770000329568 and b is -34.71649802345938\n",
      "Iteration 95, the loss is 43.619267347980205, parameters k is 9.087653413395724 and b is -34.71620089992352\n",
      "Iteration 96, the loss is 43.61632885520367, parameters k is 9.089382459003387 and b is -34.715928043759774\n",
      "Iteration 97, the loss is 43.613852271719004, parameters k is 9.090969788276489 and b is -34.715677466147326\n",
      "Iteration 98, the loss is 43.61176498878191, parameters k is 9.092427015521135 and b is -34.71544734125837\n",
      "Iteration 99, the loss is 43.6100058112087, parameters k is 9.093764803197601 and b is -34.71523599290009\n"
     ]
    }
   ],
   "source": [
    "#initialized parameters\n",
    "\n",
    "k = random.random() * 200 - 100  # -100 100\n",
    "b = random.random() * 200 - 100  # -100 100\n",
    "\n",
    "learning_rate = 1e-3\n",
    "\n",
    "iteration_num = 100 # 迭代次数\n",
    "losses = []\n",
    "for i in range(iteration_num):\n",
    "    \n",
    "    price_use_current_parameters = [price(r, k, b) for r in X_rm]  # \\hat{y}\n",
    "    \n",
    "    current_loss = loss(y, price_use_current_parameters)\n",
    "    losses.append(current_loss)\n",
    "    print(\"Iteration {}, the loss is {}, parameters k is {} and b is {}\".format(i,current_loss,k,b))\n",
    "    \n",
    "    k_gradient = partial_derivative_k(X_rm, y, price_use_current_parameters)\n",
    "    b_gradient = partial_derivative_b(y, price_use_current_parameters)\n",
    "    \n",
    "    k = k + (-1 * k_gradient) * learning_rate\n",
    "    b = b + (-1 * b_gradient) * learning_rate\n",
    "best_k = k\n",
    "best_b = b"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x171dbbecb70>]"
      ]
     },
     "execution_count": 82,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(list(range(iteration_num)),losses)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.collections.PathCollection at 0x171db7a7f28>"
      ]
     },
     "execution_count": 88,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "price_use_best_parameters = [price(r, best_k, best_b) for r in X_rm]\n",
    "\n",
    "plt.scatter(X_rm,y) \n",
    "plt.scatter(X_rm,price_use_current_parameters)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
