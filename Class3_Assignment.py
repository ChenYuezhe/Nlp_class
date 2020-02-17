{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Assignment-03 First Step of Machine Learning: Model and Evaluation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "同学们，今天我们的学习了基本的机器学习概念，相比你已经对机器学习的这些方法有一个基本的认识了。值得说明的是，机器学习不仅仅是一系列方法，更重要的是一种思维体系，即：依据以往的、现有的数据，构建某种方法来解决未见过的问题。而且决策树，贝叶斯只是实现这个目标的一个方法，包括之后的神经网络。很有可能有一天，神经网络也会被淘汰，但是重要的是我们要理解机器学习的目标，就是尽可能的自动化解决未知的问题。"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![](https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1571556399207&di=4a97dc15ad08dd49d3748d1edf6109b3&imgtype=0&src=http%3A%2F%2Fc.hiphotos.baidu.com%2Fzhidao%2Fwh%3D450%2C600%2Fsign%3Dae742c6aedcd7b89e93932873a146e91%2F5d6034a85edf8db1b16050c40223dd54574e74c7.jpg)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part-1 Programming Review 编程回顾"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 1. Re-code the Linear-Regression Model using scikit-learning(10 points)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<评阅点>： \n",
    "> + 是否完成线性回归模型 (4')\n",
    "+ 能够进行预测新数据(3')\n",
    "+ 能够进行可视化操作(3')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "%matplotlib inline\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import random\n",
    "from sklearn.linear_model import LinearRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "def assmuing_function(x):\n",
    "    return 13.4 * x + 5 + random.randint(-5, 5)\n",
    "\n",
    "def f(x):\n",
    "    return reg.coef_ * x + reg.intercept_\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "random_data = np.random.random((30, 2))\n",
    "X = random_data[:, 0]\n",
    "y = random_data[:, 1]\n",
    "y = [assmuing_function(x) for x in X]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "reg = LinearRegression().fit(X.reshape(-1, 1), y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x17a7f5c3550>]"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXAAAAD4CAYAAAD1jb0+AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy8li6FKAAAcCUlEQVR4nO3de5SU9Z3n8fc3gLG9kJbQuNrQAxjFGyqmZkdlZ8PoKsboSDiJa4iuoyScaEIUZ4i4STSZyRxbyXgbJ6MEDJ7RIcYEWo1RzAQMuwMx29AqeMH7OBRGUGyNpo1cvvvHUw1dl6ar6/Lc6vM6h0PXr6qpL0/hx19/n9/z/MzdERGR5PlI1AWIiEhlFOAiIgmlABcRSSgFuIhIQinARUQSamiYbzZy5EgfO3ZsmG8pIpJ4a9eufdPdWwrHQw3wsWPH0tnZGeZbiogknpn9R6lxtVBERBJKAS4iklAKcBGRhBowwM1sjJmtNLNnzexpM7s8Nz7CzH5pZi/kfj+o/uWKiEivcmbgO4C/dvejgJOAr5rZ0cA84Ffufjjwq9xjEREJyYCrUNz9deD13Ne/N7NngVbgXGBK7mV3AY8BV9WlShGRnI6uLPOXb2Rzdw+HNjcxd+oEpk1qjbqsSAxqGaGZjQUmAY8DB+fCHXd/3cxG9fM9s4BZAG1tbdXUKiINrqMry9VL19OzfScA2e4erl66HqAhQ7zsk5hmdgDwM+AKd3+33O9z9wXunnH3TEtL0Tp0EZGyzV++cXd49+rZvpP5yzdGVFG0ygpwMxtGEN73uPvS3PAbZnZI7vlDgC31KVFEJLC5u2dQ42lXzioUAxYBz7r7jX2eegC4KPf1RcD9tS9PROKioyvL5PYVjJv3EJPbV9DRlQ29hkObmwY1nnblzMAnAxcCp5rZE7lfZwHtwOlm9gJweu6xiKRQb+85292Ds6f3HHaIz506gaZhQ/LGmoYNYe7UCaHWERflrEL5v4D18/RptS1HROJob73nME8e9r6XVqEEQr2ZlYgkU5x6z9MmtTZsYBfSpfQiMiD1nuNJAS4iA1LvOZ7UQhGRAan3HE8KcBEpi3rP8aMWiohIQinARUQSSgEuIpJQ6oGLSKzp9rH9U4CLSGzp9rF7pxaKiMSWbh+7d5qBi0hsxekS/nKF2fLRDFxEYitpl/CHfddGBbiIxFbSLuEPu+WjFoqIhGaw7YWkXcIfdstnwAA3szuBs4Et7n5sbuwE4HZgX2AHcJm7/7YuFYpIKlS6oiRJl/Af2txENhfWQ3fuYMeQobvH66GcFspi4MyCsRuA77r7CcA1ucciIv1qhBUlc6dO4LLOZbx6/dm8+P1pNPe8W9eWTzk78qwys7GFw8Dw3NcfAzbXtiwRSZskrigZlJtvZtqcObsfrj/4MPY/5GC+U8eWT6U98CuA5Wb2fYJZ/Cm1K0lE0qhve6FwPNFuuw1mz97zeNQoWL+eiaNG8e91futKV6FcCsxx9zHAHIJd60sys1lm1mlmnVu3bq3w7UQk6ZK2omRAt98OZnvCe8QI+N3v4I03ghAPgbn7wC8KWig/73MS8x2g2d3dzAx4x92H7+WPACCTyXhnZ2d1FYtIYqXiviaLFsGXvrTn8fDh8NxzcMghdXtLM1vr7pnC8UpbKJuBTwGPAacCL1Remog0iiStKCmyeDFcfPGex01N8MIL0Brd36ecZYRLgCnASDPbBFwLfBm4xcyGAh8As+pZpIhIZO6+Gy68cM/joUPh5ZdhzJjoauotZaAXuPsX+nnqkzWuRUQkPpYsgRkz8sdeeQXGjo2knFJ0JaZIjaSivytw331w3nn5Yy+9BOPHR1PPXijARWpA961OgWXLYPr0/LEXXoBPfCKaesqgm1mJ1EAjXGWYWg8+GCwH7Bvezz0H7rEOb9AMXKQmUn+VYRr94hfwmc/kjz3zDBx1VDT1VEABLlIDqb3KsAqxPSfw6KMwdWr+2IYNcMwx0dRTBbVQRGogSVcZdnRlmdy+gnHzHmJy+4q6bDYQ9sYGZfnhD4NWSd/wfuKJoFWSwPAGzcBFaiIJ963u6MrynQeeprtn++6xep1s3ds5gdCPSeEFOADr1sGkSeHWUQcKcJEaieNVhr1tjGx3D0ZwG9FC9QjWWJwTKLwAB+CBB+Ccc8Kroc7UQhFJqb5tDCgd3r1qHayR7mV5771Bq6RveC9dGrRKUhTeoAAXSa1SbYz+1DpYIzknsHRpENznn79n7N57g+D+7Gfr974RUgtFJKXKnVXXI1hDPSfwwANw7rn5Y3ffDV/8Yu3fK2YU4CIp1d/Sxr4O2m8Y155zTF2Cte7nBEqt4168GC66qH7vGTNqoYikVKk2huV+b21u4ub/eQJd15wRuxOvA5o/P2iV9A3vH/4waJU0UHiDZuAiqZWEpY2DcuutcPnl+WM/+AFcemk09cSAAlwkxeK4tHHQ7rgDvvKV/LFPfzpooTQ4BbiIxNOPfgSXXJI/NmUKrFwZSTlxNGAP3MzuNLMtZrahYHy2mW00s6fN7Ib6lSgiDeWee4Ied9/wPumkoMet8M5TzknMxcCZfQfM7C+Ac4Hj3P0Y4Pu1L01EGsr11wfBfcEFe8aOOy4I7jVroqsrxsrZUm1Vblf6vi4F2t39j7nXbKl9aSLSEG65Ba64In9sxAh4661o6kmQSpcRHgH8uZk9bma/NrM/7e+FZjbLzDrNrHPr1q0Vvp2IpM4ddwQz7sLwdld4l6nSAB8KHAScBMwFfmJmVuqF7r7A3TPunmlpaanw7UQkNe66KwjuwpUl7sEvKVulAb4JWOqB3wK7gJG1K0tEUuemm4Lg/qu/yh9XcFes0gDvAE4FMLMjgH2AN2tVlIikyD//cxDcV16ZP67grlo5ywiXAGuACWa2ycxmAncC43NLC38MXOSuT0JE+rjhhiC4L7ssf1zBXTPlrEL5Qj9PXdDPuIg0sn/6J/ja14rHFdo1pysxRaQ27rwTZs4sHldw140CXESqs2QJzJhRPK7grjsFuIhUZtkymD69eFzBHRoFuIgMziOPBHcDLKTgDp0CXETKs3IlnHpq8fiuXcFqEwmdAlxE9m71apg8uXhcwR05BbiIlLZ2LWQyxeM7d8JHtBtjHCjARSTfhg0wcWLx+PbtMFSRESf6NEQk8PzzMGFC8fgf/wj77BN+PTIgBbhIo3vlFRg/vni8pwf23Tf8eqRsCnCRRrVpE4wZUzz+/vuw337h1yODpjMRIo0mmw1WjxSG9zvvBGu5Fd6JoQAXaRSbNwfBPXp0/vi2bUFwDx8eTV1SMQW4SNpt3RoEd2tr/vjmzUFwH3RQNHVJ1RTgImnV3R0E96hR+eMvvRQE9yGHRFOX1Ew5GzrcaWZbcps3FD73N2bmZqbt1ETi4r33guAunFk/80wQ3KVWnEgilTMDXwycWThoZmOA04HXalyTiFTigw+C4D7wwPzxdeuC4D7qqGjqkroZMMDdfRWwrcRTNwHfAHQLMpEobd8eBHdTU/746tVBcE+aFE1dUncV9cDN7C+BrLs/WeN6RKRcO3YEwV14leSqVUFwn3xyNHVJaAZ9IY+Z7Qd8EzijzNfPAmYBtLW1DfbtRKTQrl0wZEjx+KOPwumnh1+PRKaSGfhhwDjgSTN7FRgNrDOz/1Lqxe6+wN0z7p5paWmpvFKRRucezLgLw/v++4PnFN4NZ9AzcHdfD+xel5QL8Yy7v1nDukSkl3vp27cuWQLnnx9+PRIbAwa4mS0BpgAjzWwTcK27L6p3YSL11NGV5TsPPE13z3YADtpvGNeecwzTJrUO8J0hK7VhwqJFcMkl4dcSgo6uLPOXb2Rzdw+HNjcxd+qE+H0mMTJggLv7FwZ4fmzNqhEJQUdXlrn3Pcn2XXsWUL39h+3M/WlwTj4WgVEquG+7Db761fBrCUlHV5arl66nZ/tOALLdPVy9dD0Qk88khnQlpjSc+cs35oV3r+07nfnLN0ZQUR9mxeHd3h60UVIc3hB8Lr3h3atn+87oP5MYU4BLw9nc3VPRc3VVKri/+c0guK+6KpqaQtbfsY/sM0kABbg0nEObmyp6ri5KBfcVVwTB/b3vhVtLxPo79qF/JgmiAJeGM3fqBIZ9pLjHPGyIMXdqiS3F+ujoyjK5fQXj5j3E5PYVdHRlKyuiVHDPnBkE9003VfZnJtzcqRNoGpa/RLJp2JABP5NGph15pOH0nhAb7CqUmpxkK3Vy8vOfh5/8ZBB/g+rEdaVHbw1xrC2uzD28W5lkMhnv7OwM7f1Eamly+wqyJfqxrc1N/Pu8U/f+zaWCe+pUeOSRGlVXnsL/CUEwy71u+kQFZYyZ2Vp3zxSOq4UiUqaKTrKVapWMGRO0SkIOb9BKj7RRCyVCcf1RtpHt7TM5tLmp5Ay85Em2UjPuAw6A3/++1iUPilZ6pItm4BHp/VE2292Ds6efWvFJManaQJ9JWSfZSs24IZhxRxzeoJUeaaMAj4h+lI2fgT6TaZNauW76RFqbmzCC3vfu3vHegjvE80wD0UqPdFELJSL6UTZ+yvlMpk1qzW9zlQptiFVo96WVHumiAI/IoPqpEoqqe9wQ2+Duq+h/QpJYaqFERD/Kxk/VPe4EhHcjq9lFWDGiGXhE9KNs/Oz1M0nwjFvSe6dDXcgjsjcK7lSo6iKsGOjvQh7NwEVKUXCnSloXDQzYAzezO81si5lt6DM238yeM7OnzGyZmTXXt0yRkKjHnUppXf9ezknMxcCZBWO/BI519+OA54Gra1yXSLgU3KmW1kUDAwa4u68CthWMPeruO3IPf0OwM71I8ii4G8JeL8JKsFr0wC8B7u3vSTObBcwCaGtrq8HbidSAetwNJ43r36taB25m3wR2APf09xp3X+DuGXfPtLS0VPN2ItXTjFtSpOIZuJldBJwNnOZhrkUUqYRm3JJCFQW4mZ0JXAV8yt3/UNuSRGpo2DDYsaN4XMEtKVDOMsIlwBpggpltMrOZwG3AgcAvzewJM7u9znWKDE5LSzDrLgxvtUokRQacgbv7F0oML6pDLSLVO+wwePnl4nGFtqSQbmYl6TBpUjDjLgxvzbglxRTgkmxTpgTB/cQT+eMKbmkAuhdKhbSfZcTOOQd+/vPicYW2NBAFeAXSemvKRJgxA5YsKR5XcEsDUgulAtrPMgJf/nLQKikMb7VKpIEpwCuQ1ltTxtKcOUFwL1yYP67gFlGAVyKtt6aMlW9/Owjum2/OH1dwi+ymAK9AWm9NGQvt7UFwf+97+eMKbpEiOolZAe1nWQe33gqXX148rtAeNK2QahwK8Aql8daUkVi0CL70peJxBXdFtEKqsaiFItH4138NWiWF4a1WSVW0QqqxaAYu4Vq2DKZPLx5XaNeEVkg1Fs3AJRwPPxzMuAvDe9cuhXcNaYVUY1GAS32tXBkE91ln5Y/3Bnd/Gy1IRbRCqrGohSL1sWYNnHJK8fjOnfARzRvqRSukGsuAAW5mdxJsnbbF3Y/NjY0g2Mh4LPAqcJ67v12/MiUx1q2DT36yeHzHDhgypHhcak4rpBpHOVOhxcCZBWPzgF+5++HAr3KPpZFt2BC0QwrD+8MPg1aJwrshdXRlmdy+gnHzHmJy+wo6urJRl5QqAwa4u68CthUMnwvclfv6LmBajeuSpHjhhSC4J07MH+/pCYJ72LBo6pLI9a5Jz3b34OxZk64Qr51Km5EHu/vrALnfR9WuJEmEV18NgvuII/LH33svCO59942kLIkPrUmvv7qfTTKzWWbWaWadW7durffbSb1ls0FwjxuXP/7OO0Fw779/NHVJ7GhNev1VGuBvmNkhALnft/T3Qndf4O4Zd8+0tLRU+HYSuS1bguAePTp//K23guAePjyauiS2tCa9/ioN8AeAi3JfXwTcX5tyJHa2bQuC++CD88e3bAmCe8SIaOqS2NOa9PorZxnhEmAKMNLMNgHXAu3AT8xsJvAa8Pl6FikRePdd+NjHisc3bYJWLVGTgWlNev2Zh3gZcyaT8c7OztDeTyrw/vtwwAHF46+8AmPHhl6OiICZrXX3TOG4rsSUwAcfQFOJ3uTzz8Phh4dfT43pHtmSRgrwRvfhh/DRjxaPb9gAxxwTfj11oHtkS1rpphSNaufO4ORkYXivWxecnExJeIPWI0t6KcAbza5dQXAPLfjha/XqILgnTYqmrjrSemRJKwV4o+i9dWvhPUlWrgyeO/nkaOoKgdYjS1opwNOuN7gLb+H68MPBc1OmRFJWmLQeWdJKJzHTrNRmCcuWwbTGuveY1iNLWinA06hUcN9zD8yYEX4tMaF7ZEsaqYWSJmbF4b1wYdAqaeDwFkkrBXgalAruW28NgnvmzGhqEpG6U4AnWangvv76ILhnz46mJhEJjQI8iUoF97e+FQT3N74RTU0iEjqdxEySUicnr7wS/uEfwq9FRCKnAE+CUsE9axbccUf4tYhIbCjA46xUcM+YESwJFJGGpwCPo1LBffbZ8OCD4dciIrFV1UlMM5tjZk+b2QYzW2Jm2oq8GqVOTn7qU8HJSYW3iBSoOMDNrBX4OpBx92OBIcD5tSqsoZQK7hNPDIL7scciKUlE4q/aFspQoMnMtgP7AZurL6mBlGqVHHYYvPhi+LWISOJUHODunjWz7xNsatwDPOrujxa+zsxmAbMA2traKn07IEXbYpUK7paWYKd3EZEyVdNCOQg4FxgHHArsb2YXFL7O3Re4e8bdMy0tLRUX2rstVra7B2fPtlgdXdmK/8zQlWqVnHBC0CpReDeEjq4sk9tXMG7eQ0xuX5Gsf78SO9WcxPwfwCvuvtXdtwNLgVNqU1axRG+LVSq4zz03CO6urmhqktClYhIisVJNgL8GnGRm+5mZAacBz9amrGKJ3BarVHCfdloQ3B0d0dQkkUn0JERiqeIAd/fHgZ8C64D1uT9rQY3qKpKobbH22ac4uE8+OQjuf/u3aGqSyCVyEiKxVtU6cHe/1t2PdPdj3f1Cd/9jrQorlIhtsUaMCIJ7+/Y9Y8cdFwT36tXR1SWxkKhJiCRCYu5GOG1SK9dNn0hrcxMGtDY3cd30ifFYhdLWFgT322/vGRs/PgjuJ5+Mri6JlURMQiRREnUpfey2xTr+eHjqqbyhbQc0s2rVhnjVKbGgvTml1hIV4LHxj/8IX/963tAfhn2Uo6/8GQBNS9cD6D9MKRK7SYgkmgJ8MBYvhosvzht676P7c+wV9+aN9a4s0H+oIlJPiemBR+qee4Ied9/wvuoqcGdiQXj30soCEak3Bfje3HdfENwX9LnAdM6c4ORkezuglQUiEh0FeCkdHUFwn3fenrFLLw2C+8Yb816qlQUiEhX1wPt66KFg44S+Zs6EhQv7/RatLBCRqCjAAZYvhzPPzB+74AL4l38p69u1skBEotDYAb5yJZx6av7Y5z4X9L5FRGKuMQN81apgq7K+tOekiCRMYwX4mjVwSsEdb884I2ihiIgkTGMEeDbLjk98gqEffLB76M0T/4yRa38TYVEiItVJd4C//joceSS8++7uv+ivx53IRef9LU3DhnBdV1YnHwukZts6kQaQzgB/4w04+mjYtm330LfOuIy7J521+7Eudy/Wu2NM76YDvTvGgO7rIhJHVQW4mTUDC4FjAQcucfc1tSisIlu3wrHH5u8veeutjMuOx0u8vJLL3dM8Q93bjjFp+TuKpEm1V2LeAjzi7kcCx1PHLdX26q23YPRoGDVqT3jfeGNw5eTs2TW73D3texpqxxiRZKlmV/rhwH8HFgG4+4fu3l2rwsqybRuMGwcjR0I2F6I33BAE95w5u19Wq8vd076noe7rIpIs1czAxwNbgR+ZWZeZLTSz/WtU1951d8Phh8PHPw6vvhqM/f3fB8E9d27Ry2u1m0/aZ6i6r4tIslTTAx8KnAjMdvfHzewWYB7w7b4vMrNZwCyAtra2Kt4uZ+1ayGT2PP7ud+Gaawb8tlpc7n5ocxPZEmGdlhmq7usikizVBPgmYFNud3oIdqifV/gid19Abrf6TCZT6lzi4PSu5b7mmiC8QzR36oS8VRqQvhmq7usikhwVB7i7/87M/tPMJrj7RuA04JnaldaPyZODVkkENEMVkTipdh34bOAeM9sHeBm4eIDXJ55mqCISF1UFuLs/AWQGfKGIiNScduQREUkoBbiISEIpwEVEEkoBLiKSUApwEZGEUoCLiCSUAlxEJKEU4CIiCaUAFxFJKAW4iEhCKcBFRBJKAS4iklAKcBGRhKr2drKSQh1dWd3zXCQBFOCSp6Mrm7frULa7h6uXrgdQiIvEjFookmf+8o15W8YB9GzfyfzlGyOqSET6U3WAm9mQ3K70P69FQRKtzSU2bd7buIhEpxYz8MuBZ2vw50gMHNrcNKhxEYlOVQFuZqOBzwALa1OORG3u1Ak0DRuSN9Y0bAhzp06IqCIR6U+1JzFvBr4BHNjfC8xsFjALoK2trcq3k3rrPVGpVSgi8VdxgJvZ2cAWd19rZlP6e527LwAWAGQyGa/0/dIk7sv0pk1qjVU9IlJaNTPwycBfmtlZwL7AcDO7290vqE1p6aRleiJSKxX3wN39ancf7e5jgfOBFQrvgWmZnojUitaBh0zL9ESkVmoS4O7+mLufXYs/K+20TE9EakUz8JBpmZ6I1IruhRIyLdMTkVpRgEdAy/REpBbUQhERSSgFuIhIQinARUQSSgEuIpJQCnARkYQy9/DuL2VmW4H/CO0N420k8GbURcSMjkkxHZN8jXo8/sTdWwoHQw1w2cPMOt09E3UdcaJjUkzHJJ+ORz61UEREEkoBLiKSUArw6CyIuoAY0jEppmOST8ejD/XARUQSSjNwEZGEUoCLiCSUArzOzOxMM9toZi+a2bwSz3/RzJ7K/VptZsdHUWeYBjomfV73p2a208w+F2Z9YSvneJjZFDN7wsyeNrNfh11j2Mr47+ZjZvagmT2ZOyYXR1Fn5Nxdv+r0CxgCvASMB/YBngSOLnjNKcBBua8/DTwedd1RH5M+r1sB/AL4XNR1R/xvpBl4BmjLPR4Vdd0xOCb/G7g+93ULsA3YJ+raw/6lGXh9/VfgRXd/2d0/BH4MnNv3Be6+2t3fzj38DTA65BrDNuAxyZkN/AzYEmZxESjneMwAlrr7awDurmMCDhxoZgYcQBDgO8ItM3oK8PpqBf6zz+NNubH+zAQermtF0RvwmJhZK/BZ4PYQ64pKOf9GjgAOMrPHzGytmf2v0KqLRjnH5DbgKGAzsB643N13hVNefGhHnvqyEmMl122a2V8QBPh/q2tF0SvnmNwMXOXuO4MJVqqVczyGAp8ETgOagDVm9ht3f77exUWknGMyFXgCOBU4DPilmf0fd3+33sXFiQK8vjYBY/o8Hk0wY8hjZscBC4FPu/tbIdUWlXKOSQb4cS68RwJnmdkOd+8Ip8RQlXM8NgFvuvv7wPtmtgo4HkhrgJdzTC4G2j1ogr9oZq8ARwK/DafEeFALpb7+H3C4mY0zs32A84EH+r7AzNqApcCFKZ5R9TXgMXH3ce4+1t3HAj8FLktpeEMZxwO4H/hzMxtqZvsBfwY8G3KdYSrnmLxG8BMJZnYwMAF4OdQqY0Az8Dpy9x1m9jVgOcGZ9Tvd/Wkz+0ru+duBa4CPAz/IzTh3eIrvtlbmMWkY5RwPd3/WzB4BngJ2AQvdfUN0VddXmf9G/g5YbGbrCVouV7l7w91mVpfSi4gklFooIiIJpQAXEUkoBbiISEIpwEVEEkoBLiKSUApwEZGEUoCLiCTU/wd7wC6It6MRcgAAAABJRU5ErkJggg==\n",
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
    "plt.scatter(X, y)\n",
    "plt.plot(X, f(X), color='r')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([14.97792648])"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "reg.predict([[0.7]])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 2. Complete the unfinished KNN Model using pure python to solve the previous Line-Regression problem. (8 points)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<评阅点>:\n",
    "> + 是否完成了KNN模型 (4')\n",
    "+ 是否能够预测新的数据 (4')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "from scipy.spatial.distance import cosine"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "def distance(x1, x2):\n",
    "    return cosine(x1, x2)\n",
    "\n",
    "def predict(x, k=5):\n",
    "    # KNN 在预测的时候，需要做大量的计算\n",
    "    \n",
    "    most_similars = sorted(model(X, y), key=lambda xi: distance(xi[0], x))[:k]\n",
    "    \n",
    "    y_hats = [_y for x, _y in most_similars]\n",
    "    \n",
    "    print(most_similars)\n",
    "    \n",
    "    return np.mean(y_hats)\n",
    "\n",
    "def model(X, y):\n",
    "    \"\"\"保存数据即可\"\"\"\n",
    "    \n",
    "    return [(Xi, yi) for Xi, yi in zip(X, y)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.collections.PathCollection at 0x17a7f72a978>"
      ]
     },
     "execution_count": 9,
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
    "plt.scatter(X, y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[(0.6544632894827641, 16.769808079069037), (0.7573356166226171, 18.14829726274307), (0.9252232353096417, 18.397991353149198), (0.5259191968220142, 13.04731723741499), (0.6611261013562946, 16.85908975817435)]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "16.64450073811013"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "predict(0.8)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 3. Re-code the Decision Tree, which could sort the features by salience. (12 points)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<评阅点>\n",
    "> + 是否实现了信息熵 (1' )\n",
    "+ 是否实现了最优先特征点的选择(5')\n",
    "+ 是否实现了持续的特征选则(6')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "1.找到第一个 𝑠𝑎𝑙𝑖𝑒𝑛𝑡 𝑓𝑒𝑎𝑡𝑢𝑟𝑒  \n",
    "2.根据第一个 𝑠𝑎𝑙𝑖𝑒𝑛𝑡 𝑓𝑒𝑎𝑡𝑢𝑟𝑒 切割数据集，得到分数据集  \n",
    "3.对每个分数据集重复上面两个步骤，直到没有 𝑓𝑒𝑎𝑡𝑢𝑟𝑒 可以分割"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from icecream import ic   \n",
    "from collections import Counter\n",
    "import numpy as np\n",
    "\n",
    "# 假设某数据\n",
    "mock_data = {\n",
    "    'gender':['F', 'F', 'F', 'F', 'M', 'M', 'M'],\n",
    "    'income': ['+10', '-10', '+10', '+10', '+10', '+10', '-10'],\n",
    "    'family_number': [1, 1, 2, 2, 1, 1, 2],\n",
    "    #'pet': [1, 1, 1, 0, 0, 0, 1],\n",
    "    'bought': [1, 1, 1, 0, 0, 0, 1],\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset = pd.DataFrame.from_dict(mock_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "def find_the_optimal_spilter(training_data: pd.DataFrame, target: str) -> str:\n",
    "    x_fields = set(training_data.columns.tolist()) - {target}\n",
    "    \n",
    "    spliter = None\n",
    "    min_entropy = float('inf')\n",
    "    \n",
    "    for f in x_fields:\n",
    "        ic(f)\n",
    "        values = set(training_data[f])\n",
    "        ic(values)\n",
    "        for v in values:\n",
    "            sub_spliter_1 = training_data[training_data[f] == v][target].tolist()\n",
    "            ic(sub_spliter_1)\n",
    "            # split by the current feature and one value\n",
    "\n",
    "            entropy_1 = entropy(sub_spliter_1)\n",
    "            ic(entropy_1)\n",
    "\n",
    "            sub_spliter_2 = training_data[training_data[f] != v][target].tolist()\n",
    "            ic(sub_spliter_2)\n",
    "\n",
    "            entropy_2 = entropy(sub_spliter_2)\n",
    "            ic(entropy_2)\n",
    "\n",
    "            entropy_v = entropy_1 + entropy_2\n",
    "            ic(entropy_v)\n",
    "\n",
    "            if entropy_v <= min_entropy:\n",
    "                min_entropy = entropy_v\n",
    "                spliter = (f, v)\n",
    "    \n",
    "    print('spliter is: {}'.format(spliter))\n",
    "    print('the min entropy is: {}'.format(min_entropy))\n",
    "    \n",
    "    return spliter\n",
    "\n",
    "def entropy(elements):\n",
    "    '''群体的混乱程度'''\n",
    "    counter = Counter(elements)\n",
    "    probs = [counter[c] / len(elements) for c in set(elements)]\n",
    "    ic(probs)\n",
    "    return - sum(p * np.log(p) for p in probs)\n",
    "\n",
    "def create_tree(training_data: pd.DataFrame, target):\n",
    "    \n",
    "     # np tolist()将数组或列表转换成列表,这里有没有tolist()好像都一样 \n",
    "    x_fields = set(training_data.columns.tolist()) - {target}   \n",
    "    \n",
    "    if not x_fields:\n",
    "        return list(training_data[target])[0]\n",
    "    \n",
    "    best_featLabel = find_the_optimal_spilter(training_data, target)[0]\n",
    "    my_tree = {best_featLabel:{}}\n",
    "    \n",
    "    feat_values = training_data[best_featLabel]\n",
    "    uniqueVals=set(feat_values)\n",
    "    \n",
    "    for value in uniqueVals:\n",
    "        sub_dataset = split_dataset(training_data, best_featLabel, value)\n",
    "        my_tree[best_featLabel][value]=create_tree(sub_dataset, target)\n",
    "    return my_tree\n",
    "\n",
    "def split_dataset(formal_data,feature_splited_by,value_splited_by):\n",
    "    '''\n",
    "    formal_data：需要被分割的数据集\n",
    "    feature_splited_by：分割数据集所需要的feature\n",
    "    '''\n",
    "    x = formal_data[feature_splited_by] == value_splited_by\n",
    "    sub_dataframe = formal_data[x].drop(feature_splited_by,axis=1)\n",
    "    return sub_dataframe\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "#find_the_optimal_spilter(dataset, target='bought')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "ic| f: 'gender'\n",
      "ic| values: {'F', 'M'}\n",
      "ic| sub_spliter_1: [1, 1, 1, 0]\n",
      "ic| probs: [0.25, 0.75]\n",
      "ic| entropy_1: 0.5623351446188083\n",
      "ic| sub_spliter_2: [0, 0, 1]\n",
      "ic| probs: [0.6666666666666666, 0.3333333333333333]\n",
      "ic| entropy_2: 0.6365141682948128\n",
      "ic| entropy_v: 1.198849312913621\n",
      "ic| sub_spliter_1: [0, 0, 1]\n",
      "ic| probs: [0.6666666666666666, 0.3333333333333333]\n",
      "ic| entropy_1: 0.6365141682948128\n",
      "ic| sub_spliter_2: [1, 1, 1, 0]\n",
      "ic| probs: [0.25, 0.75]\n",
      "ic| entropy_2: 0.5623351446188083\n",
      "ic| entropy_v: 1.198849312913621\n",
      "ic| f: 'income'\n",
      "ic| values: {'+10', '-10'}\n",
      "ic| sub_spliter_1: [1, 1, 0, 0, 0]\n",
      "ic| probs: [0.6, 0.4]\n",
      "ic| entropy_1: 0.6730116670092565\n",
      "ic| sub_spliter_2: [1, 1]\n",
      "ic| probs: [1.0]\n",
      "ic| entropy_2: -0.0\n",
      "ic| entropy_v: 0.6730116670092565\n",
      "ic| sub_spliter_1: [1, 1]\n",
      "ic| probs: [1.0]\n",
      "ic| entropy_1: -0.0\n",
      "ic| sub_spliter_2: [1, 1, 0, 0, 0]\n",
      "ic| probs: [0.6, 0.4]\n",
      "ic| entropy_2: 0.6730116670092565\n",
      "ic| entropy_v: 0.6730116670092565\n",
      "ic| f: 'family_number'\n",
      "ic| values: {1, 2}\n",
      "ic| sub_spliter_1: [1, 1, 0, 0]\n",
      "ic| probs: [0.5, 0.5]\n",
      "ic| entropy_1: 0.6931471805599453\n",
      "ic| sub_spliter_2: [1, 0, 1]\n",
      "ic| probs: [0.3333333333333333, 0.6666666666666666]\n",
      "ic| entropy_2: 0.6365141682948128\n",
      "ic| entropy_v: 1.3296613488547582\n",
      "ic| sub_spliter_1: [1, 0, 1]\n",
      "ic| probs: [0.3333333333333333, 0.6666666666666666]\n",
      "ic| entropy_1: 0.6365141682948128\n",
      "ic| sub_spliter_2: [1, 1, 0, 0]\n",
      "ic| probs: [0.5, 0.5]\n",
      "ic| entropy_2: 0.6931471805599453\n",
      "ic| entropy_v: 1.3296613488547582\n",
      "ic| f: 'gender'\n",
      "ic| values: {'F', 'M'}\n",
      "ic| sub_spliter_1: [1, 1, 0]\n",
      "ic| probs: [0.3333333333333333, 0.6666666666666666]\n",
      "ic| entropy_1: 0.6365141682948128\n",
      "ic| sub_spliter_2: [0, 0]\n",
      "ic| probs: [1.0]\n",
      "ic| entropy_2: -0.0\n",
      "ic| entropy_v: 0.6365141682948128\n",
      "ic| sub_spliter_1: [0"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "spliter is: ('income', '-10')\n",
      "the min entropy is: 0.6730116670092565\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      ", 0]\n",
      "ic| probs: [1.0]\n",
      "ic| entropy_1: -0.0\n",
      "ic| sub_spliter_2: [1, 1, 0]\n",
      "ic| probs: [0.3333333333333333, 0.6666666666666666]\n",
      "ic| entropy_2: 0.6365141682948128\n",
      "ic| entropy_v: 0.6365141682948128\n",
      "ic| f: 'family_number'\n",
      "ic| values: {1, 2}\n",
      "ic| sub_spliter_1: [1, 0, 0]\n",
      "ic| probs: [0.6666666666666666, 0.3333333333333333]\n",
      "ic| entropy_1: 0.6365141682948128\n",
      "ic| sub_spliter_2: [1, 0]\n",
      "ic| probs: [0.5, 0.5]\n",
      "ic| entropy_2: 0.6931471805599453\n",
      "ic| entropy_v: 1.3296613488547582\n",
      "ic| sub_spliter_1: [1, 0]\n",
      "ic| probs: [0.5, 0.5]\n",
      "ic| entropy_1: 0.6931471805599453\n",
      "ic| sub_spliter_2: [1, 0, 0]\n",
      "ic| probs: [0.6666666666666666, 0.3333333333333333]\n",
      "ic| entropy_2: 0.6365141682948128\n",
      "ic| entropy_v: 1.3296613488547582\n",
      "ic| f: 'family_number'\n",
      "ic| values: {1, 2}\n",
      "ic| sub_spliter_1: [1]\n",
      "ic| probs: [1.0]\n",
      "ic| entropy_1: -0.0\n",
      "ic| sub_spliter_2: [1, 0]\n",
      "ic| probs: [0.5, 0.5]\n",
      "ic| entropy_2: 0.6931471805599453\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "spliter is: ('gender', 'M')\n",
      "the min entropy is: 0.6365141682948128\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "ic| entropy_v: 0.6931471805599453\n",
      "ic| sub_spliter_1: [1, 0]\n",
      "ic| probs: [0.5, 0.5]\n",
      "ic| entropy_1: 0.6931471805599453\n",
      "ic| sub_spliter_2: [1]\n",
      "ic| probs: [1.0]\n",
      "ic| entropy_2: -0.0\n",
      "ic| entropy_v: 0.6931471805599453\n",
      "ic| f: 'family_number'\n",
      "ic| values: {1}\n",
      "ic| sub_spliter_1: [0, 0]\n",
      "ic| probs: [1.0]\n",
      "ic| entropy_1: -0.0\n",
      "ic| sub_spliter_2: []\n",
      "ic| probs: []\n",
      "ic| entropy_2: 0\n",
      "ic| entropy_v: 0.0\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "spliter is: ('family_number', 2)\n",
      "the min entropy is: 0.6931471805599453\n",
      "spliter is: ('family_number', 1)\n",
      "the min entropy is: 0.0\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "ic| f: 'gender'\n",
      "ic| values: {'F', 'M'}\n",
      "ic| sub_spliter_1: [1]\n",
      "ic| probs: [1.0]\n",
      "ic| entropy_1: -0.0\n",
      "ic| sub_spliter_2: [1]\n",
      "ic| probs: [1.0]\n",
      "ic| entropy_2: -0.0\n",
      "ic| entropy_v: -0.0\n",
      "ic| sub_spliter_1: [1]\n",
      "ic| probs: [1.0]\n",
      "ic| entropy_1: -0.0\n",
      "ic| sub_spliter_2: [1]\n",
      "ic| probs: [1.0]\n",
      "ic| entropy_2: -0.0\n",
      "ic| entropy_v: -0.0\n",
      "ic| f: 'family_number'\n",
      "ic| values: {1, 2}\n",
      "ic| sub_spliter_1: [1]\n",
      "ic| probs: [1.0]\n",
      "ic| entropy_1: -0.0\n",
      "ic| sub_spliter_2: [1]\n",
      "ic| probs: [1.0]\n",
      "ic| entropy_2: -0.0\n",
      "ic| entropy_v: -0.0\n",
      "ic| sub_spliter_1: [1]\n",
      "ic| probs: [1.0]\n",
      "ic| entropy_1: -0.0\n",
      "ic| sub_spliter_2: [1]\n",
      "ic| probs: [1.0]\n",
      "ic| entropy_2: -0.0\n",
      "ic| entropy_v: -0.0\n",
      "ic| f: 'gender'\n",
      "ic| values: {'F'}\n",
      "ic| sub_spliter_1: [1]\n",
      "ic| probs: [1.0]\n",
      "ic| entropy_1: -0.0\n",
      "ic| sub_spliter_2: []\n",
      "ic| probs: []\n",
      "ic| entropy_2: 0\n",
      "ic| entropy_v: 0.0\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "spliter is: ('family_number', 2)\n",
      "the min entropy is: -0.0\n",
      "spliter is: ('gender', 'F')\n",
      "the min entropy is: 0.0\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "ic| f: 'gender'\n",
      "ic| values: {'M'}\n",
      "ic| sub_spliter_1: [1]\n",
      "ic| probs: [1.0]\n",
      "ic| entropy_1: -0.0\n",
      "ic| sub_spliter_2: []\n",
      "ic| probs: []\n",
      "ic| entropy_2: 0\n",
      "ic| entropy_v: 0.0\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "spliter is: ('gender', 'M')\n",
      "the min entropy is: 0.0\n"
     ]
    }
   ],
   "source": [
    "tree = create_tree(dataset, 'bought')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'income': {'+10': {'gender': {'F': {'family_number': {1: 1, 2: 1}},\n",
       "    'M': {'family_number': {1: 0}}}},\n",
       "  '-10': {'family_number': {1: {'gender': {'F': 1}},\n",
       "    2: {'gender': {'M': 1}}}}}}"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tree"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 网上参考\n",
    "import matplotlib.pyplot as plt\n",
    " \n",
    "#定义判断结点形状,其中boxstyle表示文本框类型,fc指的是注释框颜色的深度\n",
    "decisionNode = dict(boxstyle=\"round4\", color='r', fc='0.9')\n",
    "#定义叶结点形状\n",
    "leafNode = dict(boxstyle=\"circle\", color='m')\n",
    "#定义父节点指向子节点或叶子的箭头形状\n",
    "arrow_args = dict(arrowstyle=\"<-\", color='g')\n",
    " \n",
    "def plot_node(node_txt, center_point, parent_point, node_style):\n",
    "    '''\n",
    "    绘制父子节点，节点间的箭头，并填充箭头中间上的文本\n",
    "    :param node_txt:文本内容\n",
    "    :param center_point:文本中心点\n",
    "    :param parent_point:指向文本中心的点\n",
    "    '''\n",
    "    createPlot.ax1.annotate(node_txt, \n",
    "                            xy=parent_point,\n",
    "                            xycoords='axes fraction',\n",
    "                            xytext=center_point,\n",
    "                            textcoords='axes fraction',\n",
    "                            va=\"center\",\n",
    "                            ha=\"center\",\n",
    "                            bbox=node_style,\n",
    "                            arrowprops=arrow_args)\n",
    " \n",
    "def get_leafs_num(tree_dict):\n",
    "    '''\n",
    "    获取叶节点的个数\n",
    "    :param tree_dict:树的数据字典\n",
    "    :return tree_dict的叶节点总个数\n",
    "    '''\n",
    "    #tree_dict的叶节点总数\n",
    "    leafs_num = 0\n",
    "    \n",
    "    #字典的第一个键，也就是树的第一个节点\n",
    "    root = list(tree_dict.keys())[0]\n",
    "    #这个键所对应的值，即该节点的所有子树。\n",
    "    child_tree_dict =tree_dict[root]\n",
    "    for key in child_tree_dict.keys():\n",
    "        #检测子树是否字典型\n",
    "        if type(child_tree_dict[key]).__name__=='dict':\n",
    "            #子树是字典型，则当前树的叶节点数加上此子树的叶节点数\n",
    "            leafs_num += get_leafs_num(child_tree_dict[key])\n",
    "        else:\n",
    "            #子树不是字典型，则当前树的叶节点数加1\n",
    "            leafs_num += 1\n",
    " \n",
    "    #返回tree_dict的叶节点总数\n",
    "    return leafs_num\n",
    " \n",
    "def get_tree_max_depth(tree_dict):\n",
    "    '''\n",
    "    求树的最深层数\n",
    "    :param tree_dict:树的字典存储\n",
    "    :return tree_dict的最深层数\n",
    "    '''\n",
    "    #tree_dict的最深层数\n",
    "    max_depth = 0\n",
    "    \n",
    "    #树的根节点\n",
    "    root = list(tree_dict.keys())[0]\n",
    "    #当前树的所有子树的字典\n",
    "    child_tree_dict = tree_dict[root]\n",
    "    \n",
    "    for key in child_tree_dict.keys():\n",
    "        #树的当前分支的层数\n",
    "        this_path_depth = 0\n",
    "        #检测子树是否字典型\n",
    "        if type(child_tree_dict[key]).__name__ == 'dict':\n",
    "            #如果子树是字典型，则当前分支的层数需要加上子树的最深层数\n",
    "            this_path_depth = 1 + get_tree_max_depth(child_tree_dict[key])\n",
    "        else:\n",
    "            #如果子树不是字典型，则是叶节点，则当前分支的层数为1\n",
    "            this_path_depth = 1\n",
    "        if this_path_depth > max_depth:\n",
    "            max_depth = this_path_depth\n",
    "    \n",
    "    #返回tree_dict的最深层数\n",
    "    return max_depth\n",
    " \n",
    "def plot_mid_text(center_point, parent_point, txt_str):\n",
    "    '''\n",
    "    计算父节点和子节点的中间位置，并在父子节点间填充文本信息\n",
    "    :param center_point:文本中心点\n",
    "    :param parent_point:指向文本中心点的点\n",
    "    '''\n",
    "    \n",
    "    x_mid = (parent_point[0] - center_point[0])/2.0 + center_point[0]\n",
    "    y_mid = (parent_point[1] - center_point[1])/2.0 + center_point[1]\n",
    "    createPlot.ax1.text(x_mid, y_mid, txt_str)\n",
    "    return\n",
    " \n",
    "def plotTree(tree_dict, parent_point, node_txt):\n",
    "    '''\n",
    "    绘制树\n",
    "    :param tree_dict:树\n",
    "    :param parent_point:父节点位置\n",
    "    :param node_txt:节点内容\n",
    "    '''\n",
    "    \n",
    "    leafs_num = get_leafs_num(tree_dict)\n",
    "    root = list(tree_dict.keys())[0]\n",
    "    #plotTree.totalW表示树的深度\n",
    "    center_point = (plotTree.xOff+(1.0+float(leafs_num))/2.0/plotTree.totalW,plotTree.yOff)\n",
    "    #填充node_txt内容\n",
    "    plot_mid_text(center_point, parent_point, node_txt)\n",
    "    #绘制箭头上的内容\n",
    "    plot_node(root, center_point, parent_point, decisionNode)\n",
    "    #子树\n",
    "    child_tree_dict = tree_dict[root]\n",
    "    plotTree.yOff=plotTree.yOff-1.0/plotTree.totalD\n",
    "    #因从上往下画，所以需要依次递减y的坐标值，plotTree.totalD表示存储树的深度\n",
    "    for key in child_tree_dict.keys():\n",
    "        if type(child_tree_dict[key]).__name__ == 'dict':\n",
    "            plotTree(child_tree_dict[key],center_point,str(key))\n",
    "        else:\n",
    "            plotTree.xOff=plotTree.xOff+1.0/plotTree.totalW\n",
    "            plot_node(child_tree_dict[key],(plotTree.xOff,plotTree.yOff),center_point,leafNode)\n",
    "            plot_mid_text((plotTree.xOff,plotTree.yOff),center_point,str(key))\n",
    "    #h绘制完所有子节点后，增加全局变量Y的偏移\n",
    "    plotTree.yOff=plotTree.yOff+1.0/plotTree.totalD\n",
    " \n",
    "    return\n",
    " \n",
    "def createPlot(tree_dict):\n",
    "    '''\n",
    "    绘制决策树图形\n",
    "    :param tree_dict\n",
    "    :return 无\n",
    "    '''\n",
    "    #设置绘图区域的背景色\n",
    "    fig=plt.figure(1,facecolor='white')\n",
    "    #清空绘图区域\n",
    "    fig.clf()\n",
    "    #定义横纵坐标轴,注意不要设置xticks和yticks的值!!!\n",
    "    axprops = dict(xticks=[], yticks=[])\n",
    "    createPlot.ax1=plt.subplot(111, frameon=False, **axprops)\n",
    "    #由全局变量createPlot.ax1定义一个绘图区，111表示一行一列的第一个，frameon表示边框,**axprops不显示刻度\n",
    "    plotTree.totalW=float(get_leafs_num(tree_dict))\n",
    "    plotTree.totalD=float(get_tree_max_depth(tree_dict))\n",
    "    plotTree.xOff=-0.5/plotTree.totalW;\n",
    "    plotTree.yOff=1.0;\n",
    "    plotTree(tree_dict, (0.5,1.0), '')\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "    createPlot(tree)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 4. Finish the K-Means using 2-D matplotlib (8 points)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<评阅点>\n",
    "> + 是否完成了KMeans模型，基于scikit-learning (3')\n",
    "+ 是否完成了可视化任务（5'）"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
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
    "%matplotlib inline\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import random\n",
    "random_data = np.random.random((30, 2))\n",
    "X = random_data[:, 0]\n",
    "y = random_data[:, 1]\n",
    "plt.scatter(X, y, marker='o')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
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
    "from sklearn.cluster import KMeans\n",
    "y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(random_data)\n",
    "plt.scatter(X, y, c=y_pred)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part-2 Question and Answer 问答"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 1. What's the *model*? why  all the models are wrong, but some are useful? (5 points) "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Ans:"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<评阅点>\n",
    "> + 对模型的理解是否正确,对模型的抽象性是否正确(5')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "答：现实世界的影响因素非常的多，我们不能站在上帝视角统计所有的因素和数据，因此模型是对现实的一种近似，从而它不是正确的。  \n",
    "对于模型，我们关注的并不是现实是否与模型一模一样，而是，我们是否能够从这个模型中发现更多。"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 2. What's the underfitting and overfitting? List the reasons that could make model overfitting or underfitting. (10 points)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Ans:"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<评阅点>\n",
    "> + 对过拟合和欠拟合的理解是否正确 (3')\n",
    "+ 对欠拟合产生的原因是否理解正确(2')\n",
    "+ 对过拟合产生的原因是否理解正确(5')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "答：过拟合欠拟合是导致模型泛化能力不高的两种常见原因，都是模型学习能力与数据复杂度之间失配的结果。“欠拟合”常常在模型学习能力较弱，而数据复杂度较高的情况出现，此时模型由于学习能力不足，无法学习到数据集中的“一般规律”，因而导致泛化能力弱。与之相反，“过拟合”常常在模型学习能力过强的情况中出现，此时的模型学习能力太强，以至于将训练集单个样本自身的特点都能捕捉到，并将其认为是“一般规律”，同样这种情况也会导致模型泛化能力下降。过拟合与欠拟合的区别在于，欠拟合在训练集和测试集上的性能都较差，而过拟合往往能较好地学习训练集数据的性质，而在测试集上的性能较差。在神经网络训练的过程中，欠拟合主要表现为输出结果的高偏差，而过拟合主要表现为输出结果的高方差。  \n",
    "\n",
    "欠拟合出现的原因：模型复杂度过低，不能很好的捕捉特征；特征量过少    \n",
    "过拟合出现的原因：在对模型进行训练时，有可能遇到训练数据不够，即训练数据无法对整个数据的分布进行估计；权值学习迭代次数足够多,拟合了训练数据中的噪声和训练样例中没有代表性的特征."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 3. What's the precision, recall, AUC, F1, F2score. What are they mainly target on? (12')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Ans:"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<评阅点>\n",
    "> + 对precision, recall, AUC, F1, F2 理解是否正确(6‘)\n",
    "+ 对precision, recall, AUC, F1, F2的使用侧重点是否理解正确 (6’)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "答：患病 = 1， 未患病 = 0   \n",
    "Precision(查准率)：我查出来所有患病的有多少个是真的患病了。 查准率过高，会有大量患病者成“漏网之鱼”。   \n",
    "recall（召回率，查全率）：所有真的患病的人，有多少个被我查出来了。  召回率过高，会有大量未患病被误诊。   \n",
    "\n",
    "Precision 和 recall 互为牵制关系，则需要有一个合适的阈值，即 F1 Score  \n",
    "\n",
    "AUG：ROC（receiver operating characteristic curve）由TPR和FPR组成，其中TPR = TP / (TP + FN) = Recall（TRP是召回率），FPR = FP / (FP + TN)。我们以FPR为x轴，TPR为y轴画图所得曲线即为ROC。ROC与FPR所围成的面积就是AUG。 ROC是光滑的，一般就没有太大的过拟合，这时候只需要调整AUC，使得面积越大模型越好。"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 4. Based on our course and yourself mind, what's the machine learning?  (8')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Ans:"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<评阅点> 开放式问题，是否能说出来机器学习这种思维方式和传统的分析式编程的区别（8'）"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "答：机器学习是给予机器知识（数据），自主式的完成学习和认知任务。   \n",
    "传统编程看似有智能的效果，实际是人为把各种情况规则编写到程序中，一旦规则改变或增加就需要不断的修改程序，非真正的智能。"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 5. \"正确定义了机器学习模型的评价标准(evaluation)， 问题基本上就已经解决一半\". 这句话是否正确？你是怎么看待的？ (8‘)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<评阅点> 开放式问题，主要看能理解评价指标对机器学习模型的重要性."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "答：评价标准即是对模型衡量的一个标尺，衡量的标准都不确定，训练出来的模型就无法评判。 甚至如果是错误的评价标准，那训练出看似再好的模型也是徒劳。评价标准是方向，正确的定义了模型的评价标准就是选择了正确的方向，在正确的方向上不断的改进模型才是有用功。    "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part-03 Programming Practice 编程练习"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "1. In our course and previous practice, we complete some importance components of Decision Tree. In this problem, you need to build a **completed** Decision Tree Model. You show finish a `predicate()` function, which accepts three parameters **<gender, income, family_number>**, and outputs the predicated 'bought': 1 or 0.  (20 points)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'income': {'+10': {'gender': {'F': {'family_number': {1: 1, 2: 1}},\n",
       "    'M': {'family_number': {1: 0}}}},\n",
       "  '-10': {'family_number': {1: {'gender': {'F': 1}},\n",
       "    2: {'gender': {'M': 1}}}}}}"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tree"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    " createPlot(tree)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_dict = {\n",
    "    'pet':'0',\n",
    "    'gender':'M',\n",
    "    'income':'+10',\n",
    "    'family_number':'1'\n",
    "}\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 有bug\n",
    "def decision_tree_precision(tree_dict, test_dict):\n",
    "    if type(tree_dict) == int :\n",
    "        return tree_dict\n",
    "\n",
    "    top = list(tree_dict)[0]\n",
    "    child_tree = tree_dict[top]\n",
    "\n",
    "    if len(child_tree.keys()) == 1:\n",
    "        k = list(child_tree.keys())[0]  # keys()返回dict_keys类型，其性质类似集合(set)而不是列表(list)\n",
    "        value = child_tree[k]\n",
    "        return value\n",
    "\n",
    "    for key in child_tree.keys():  # 0\n",
    "\n",
    "        if str(key) == test_dict[top]:\n",
    "            # 进入下一层\n",
    "            child_tree = child_tree[key]\n",
    "            predict = decision_tree_precision(child_tree, test_dict)\n",
    "            return predict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "decision_tree_precision(tree, test_dict)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<评阅点>\n",
    "> + 是否将之前的决策树模型的部分进行合并组装， predicate函数能够顺利运行(8')\n",
    "+ 是够能够输入未曾见过的X变量，例如gender, income, family_number 分别是： <M, -10, 1>, 模型能够预测出结果 (12')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "2. 将上一节课(第二节课)的线性回归问题中的Loss函数改成\"绝对值\"，并且改变其偏导的求值方式，观察其结果的变化。(19 point)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<评阅点>\n",
    "+ 是否将Loss改成了“绝对值”(3')\n",
    "+ 是否完成了偏导的重新定义(5')\n",
    "+ 新的模型Loss是否能够收敛 (11’)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "$$loss = \\frac{1}{n}\\sum\\limits_{i = 1}^{n}|y_i - \\hat y_i|$$    \n",
    "$$loss = \\frac{1}{n}\\sum\\limits_{i = 1}^{n}|y_i - (k  x_i + b)|$$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "当$y_i >\\hat y_i$时： $$|y_i-\\hat y|=y_i-\\hat y=y_i-kx_i-b$$    \n",
    "$$\\frac{\\partial loss}{ \\partial k}=-\\frac{1}{n}x_i$$     \n",
    "$$\\frac{\\partial loss}{ \\partial b}=-\\frac{1}{n}$$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "当$y_i <\\hat y_i$时： $$|y_i-\\hat y|=\\hat y-y_i=kx_i+b-y_i$$     \n",
    "$$\\frac{\\partial loss}{ \\partial k}=\\frac{1}{n}x_i$$    \n",
    "$$\\frac{\\partial loss}{ \\partial b}=\\frac{1}{n}$$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "def price(rm, k, b):\n",
    "    \"\"\"define target function\"\"\"\n",
    "    \n",
    "    return k * rm + b\n",
    "\n",
    "def loss(y,y_hat):\n",
    "    \"\"\"define loss function \"\"\"\n",
    "    \n",
    "    return sum(abs(y_i - y_hat_i) for y_i, y_hat_i in zip(list(y),list(y_hat)))/len(list(y))\n",
    "\n",
    "def partial_derivative_k(x, y, y_hat): \n",
    "    \"\"\"define partial derivative k\"\"\"\n",
    "    \n",
    "    n = len(y)\n",
    "    gradient = 0\n",
    "    for x_i, y_i, y_hat_i in zip(list(x),list(y),list(y_hat)):\n",
    "        if y_i > y_hat_i:\n",
    "            gradient += -x_i\n",
    "        else:\n",
    "            gradient += x_i\n",
    "    return 1/n * gradient\n",
    "\n",
    "def partial_derivative_b(y, y_hat):\n",
    "    \"\"\"define partial derivative b\"\"\"\n",
    "    \n",
    "    n = len(y)\n",
    "    gradient = 0\n",
    "    for y_i, y_hat_i in zip(list(y),list(y_hat)):\n",
    "        if y_i > y_hat_i:\n",
    "            gradient += -1\n",
    "        else:\n",
    "            gradient += 1\n",
    "    return 1/n * gradient"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
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
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 处理数据\n",
    "from sklearn.datasets import load_boston\n",
    "import random\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "dataset = load_boston()\n",
    "x,y=dataset['data'],dataset['target']\n",
    "X_rm = x[:,5]    # 只选择第个5个特征 ，即RM\n",
    "X_rm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.collections.PathCollection at 0x17a03919668>"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXEAAAD4CAYAAAAaT9YAAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy8li6FKAAAgAElEQVR4nO2dbZBc5ZXf/2daF+gha7dkjx3RRhbeuKS1LEtjJkaJqlwWrC17MTABBKZwikq5Qj64EkO5ZhmniCW2SNBm4pj9sJUqyk6WFCwrXuyxsBLLWyBnK1TBZuRBZhWk2tpFCLcIyEaDjdSgnpmTD913dPv2fXn69n3v/69K1Zruvvc+93b3eZ57zv+cI6oKQgghxWQk6wEQQgiJDo04IYQUGBpxQggpMDTihBBSYGjECSGkwKxK82Af/OAHdf369WkekhBCCs/hw4d/papjXq+lasTXr1+Pubm5NA9JCCGFR0Re9XuN7hRCCCkwNOKEEFJgaMQJIaTA0IgTQkiBoREnhJACY6ROEZETAH4LYAnAoqpOiMgaAPsArAdwAsAtqnommWGSOJidb2Dm4HGcWmjisloVUzs3YHK8nvWwIhP3+aR5feI8Vtqfa1zHs/fTWGiiIoIl1ZXHWtVCa2kZZ88vAQBqVQt7rt+EyfH6QMd3b7tj4xgOHTsd63fovqeP4sy5Vs+4k0JMqhh2jPiEqv7K8dx/BPCWqu4VkWkAq1X1nqD9TExMKCWG2TA738C3fvASmq2lleeqVgUP3Li5kIY87vNJ8/rEeay0P9e4jue1nzCsEcGtn7kcTx1uRDq+yTEH/Q5NPXkEraVum2qNCGZ2bRno8xCRw6o64fXaIO6UGwA83Pn/wwAmB9gXSZiZg8d7vrzN1hJmDh7PaESDEff5pHl94jxW2p9rXMfz2k8YrWXFYy+8Fvn4Jscc9DvkNuBAe9xJ/s5MjbgC+KmIHBaROzvPfVhVXweAzuOHvDYUkTtFZE5E5k6fPj34iEkkTi00+3o+78R9PmlenziPlfbnGtfxoo5vycdzYLI/02PG/R0aZJ8mmBrx7ar6aQBfAvB1Efms6QFU9SFVnVDVibExz6xRkgKX1ap9PZ934j6fNK9PnMdK+3ON63hRx1cRibw/02PG/R0aZJ8mGBlxVT3VeXwTwA8BfAbAGyKyFgA6j28mNUgyOFM7N6BqVbqeq1oVTO3ckNGIBiPu80nz+sR5rLQ/17iO57WfMKwRwW1XXR75+CbHHPQ7ZFV6JxlrRBL9nYWqU0TkUgAjqvrbzv+/AOCPAOwHcAeAvZ3HHyU2SjIwdlClLOqUuM8nzesT57HS/lzjOp5zP/2qUyY+uibS8b3GbqtT7DE4feJRzyl36hQR+Rjaq2+gbfT/XFX/vYh8AMDjANYBOAlgl6q+FbQvqlMIIV5kKX8tgnIrSJ0SuhJX1b8HsMXj+V8DuGbw4RFChhm3EW0sNPGtH7wEoP/VsN/+gyaIIMVNXox4EMzYJIRkSpIySXuCaCw0obgwQczON1beU3TlFo04ISRTkjSiJhNE0ZVbNOKEkExJ0oiaTBBFV27RiBNCMiVJI2oyQUyO1/HAjZtRr1UhAOq1aq6CmmGk2p6NEELcJCmTnNq5wVN54p4gJsfrhTHabmjECSGZk5QRLVt+hBc04oSQxMhD+eMir7JNoBEnhCRC0vpv0oaBTUJIIpSt/HFeoREnhMTO7HwDjYIn0RQFGnFCSKzYbhQ/ipJEUxToEyeExEpQBx23vC8Pgc+iQyNOCImVIHeJM4mGgc94oDuFEBIrfu6Seq1qXD2QmEMjTgiJFdM0+jSrB87ON7B977O4YvoAtu99tquKYdGhO4UQEiumWZKX1aqeCpa4A59ld9vQiBNCYsckS9K0rsmgFL3pQxg04oSQTEirrknRmz6EQSNOCDEiCTlgGnVN0nLbZAUDm4SQUEzanEXdb9IBx6I3fQiDRpwQEoqfX3nP/qOR95nUxOCm6E0fwqA7hRASip//eKHZwux8I5JBTDPgWOZytFyJE0JCCfIf37XvxUiukLIHHNOCRpwQEkqY/ziKKyTNLvNlTvahESeEhDI5XsfqUSvwPf2mzKcVcEzL954VNOKEECN2X7epx+i6aSw0jVe8aQUcy16jhYFNQogRzuQcv4YPAqy8ZpLenkbAsey+d67ECSHGTI7X8dz01Xjw1q09q3IBoK7352HFm6bvPQtoxAkhfePlCnEbcJusV7xevnerIjj73mIpAp10pxBSMPLSDcftCtm+99lcpre7a7TURi288+4iFpotAMWvasiVOCEFIs9Kizynt9tuoFf2XovRi1ahtdx935AHt09UaMQJKRB5VloUJb29bIFOulMIKRB5N0BFSG8vW1VDrsQJKRBlV1qkQZ7dPlGgESekQJTNAGVBUdw+phi7U0SkAmAOQENVvywiawDsA7AewAkAt6jqmSQGSQhpk1Y3nLJTBLePKf34xL8B4GUA7+v8PQ3gGVXdKyLTnb/viXl8hBAXRTdAeZFIlgUjd4qIfATAtQC+53j6BgAPd/7/MIDJeIdGCCkbeZZIFhVTn/iDAP4QwLLjuQ+r6usA0Hn8kNeGInKniMyJyNzp06cHGiwhpNjkWSJZVEKNuIh8GcCbqno4ygFU9SFVnVDVibGxsSi7IISUhLxLJIuIiU98O4DrReQPAFwC4H0i8giAN0Rkraq+LiJrAbyZ5EAJIcWnbBrtPBC6ElfVb6nqR1R1PYCvAHhWVb8KYD+AOzpvuwPAjxIbJSGkFMQpkSxzt55+GCRjcy+Ax0XkawBOAtgVz5AIIWUlLomkHSC1/etFL2I1CKLqV0AyfiYmJnRubi614xFCyolfxcR6rYrnpq/OYETJIiKHVXXC6zXWTiFkCCibNpsB0gsw7Z6QklNGbTZryFyARpyQklM2bfbsfAPnzi/2PD+sNWToTiGkoJi6SPyaGhfR9eAOaNrUqhb2XL+p0C6iqNCIE1JA7p19CY8+f3Klr6WfOmN2vuHZwBgopuvB664CAC69eNVQGnCA7hRCCsfsfKPLgNt4uUhmDh73NOACFNL1wIBmLzTihBQMP8MM9BozP+OmKKaemgHNXmjECSkYQatOtzELMm7rC5jpyKYYvdCIk6GlqGnbfobZy0UytXMDrIr47qux0MTd+17EvbMvxTnExChbV544YGCTDCVFTtue2rmhR6EhAG7fts577CFJ2Qrg0edPYuKjawDkv2tQ0ZtixA2NOBlKgrTTeTcQ/dQfmTl4HK3l8NIaCuC+p4/i3dZyISe2YYbuFDKUFF3lMDlex9TODbisVsWphSZmDh73dAf1cz5nzrVKlRQ0LHAlToaSote1NnUH+Z1nPww6sZWtbkve4EqcDCV5VzmEBV1NU+m9zrNfBpnYyli3JW/QiJOhJM8qBxPD57c6biw0u4y+8zyjMOjEVra6LXmE7hQytORV5WASdK2NWjhzruW5vdu1Mjlex9yrb+GR50+GHrtqjWDNpRfH5vooeuyhCNCIE5IzTAxfWC8Xt9F/7IXXjI69uKyx+qyLHnsoAnSnEJIzTFLL3256r8KdOI3+kmEHr9aSxurqyHvsoQzQiBOSM0wMn8lK1vmeivhnbbqJ09WR59hDWaA7hZCcYZLM45W16cRt9G+76nIjnzgQv6sjr7GHskAjTkgOCTN89mv3PX10JcBp1w2vexj9+yc344c/b+DseW+jb+Pl6qDOO9/QiBOSQ0wN5zvvXmhTpgCsEfF977kQA35hL93jmHryCFpL7ecbC01MPXkEAFPx8wKNOCE5wzQbc8/+oz11UVrLij37j2JyvI7Z+Qb27D+KhU4QdERMVC3LmHrigpG+7+mjKwZ85RhLivuePkojnhMY2CQkZ5gmyCz4KFQWmq32CvqJI13vMaiDBaA9EdjH8tOi+z1P0ocrcUISIqovOY4EmaDqhRWRUMkhk3GKA404IQnQb71yp8Ef8TGybtXIap+szdWjVmDRq2VV1EMKY9nHqlUtzxV/rWr5bkvShe4UQhKgn5oh7lopfqvks+8tdtVP2X3dpp6uPVZFcO2n1iJIFW6Xr/XDDo4CwJ7rN8EakZ7X91y/KeAIJE1oxAlJgH5cIl4G34uFZqurENbkeB0zN2/pSqSZuXkLDh077dvMx6rISh1yL0SAmV1bVu4WJsfrmNnlOobjdZI9dKcQkgD91Azpx//sronipSe/e9+LvtvbafU7No7hqcONrsmjalWYTVlAaMQJSYCpnRsw9cSRruCi003hpN/GDWFGP2x/jYUmnjrcwE1X1nHo2OmewKvTP18btfDOu4sr58GWbfmDRpyQpHA7pn0c1WEp9G7eHxJUNNlfs7WEQ8dO47npq7uedwdkvQKnRelFOizQJ05IAswcPO6ZJOMV2LSLRK0eNVN8nD2/GNgZx110yg+vFf2e/UeNJhNKEPMDjTghCRDUecfLAE+O1zH/7S8YGXL3ZODVym1yvI7npq/GK3uv9e3q4/bPz843fBOIwrYl2UEjTkgCBBk5d6s1pxE2zYS0JwmTVm6mNb1N64izHni+oBEnJAGCGhQ79eJuI2yKPUkE6dHtyeHufS/i4lUjWD1qBdb0DnKR1KrB25LsCA1sisglAP4KwMWd9z+pqrtFZA2AfQDWAzgB4BZVPZPcUAkpDraRu8tH7merR0w14k6cK+Egt40zQLnQbKFqVfDdW7f6GmA/VcvqUQvz3/5CX2Mk6WGyEn8PwNWqugXAVgBfFJFtAKYBPKOqHwfwTOdvQogDv446gvYqvB9pIQBcetEFLffsfAMjPvuviPTdZd7P7bL7OmZn5pnQlbiqKoB3On9anX8K4AYAn+s8/zCAnwG4J/YREpIQSTY7sN0kfin0ivYq3KQYlRO7Jvi9sy/h0edPerpgqlbFd3Uf5DIx6ShE8oeowRdIRCoADgP4RwD+VFXvEZEFVa053nNGVVd7bHsngDsBYN26dVe++uqrsQ2ekKi49dBAvBmL43/009Agpd2Jp19qVQtvN1ue21ZE8J1btmDm4HHPVX69Vu3RhpP8IyKHVXXC6zWjZB9VXQKwVURqAH4oIp80PbiqPgTgIQCYmJiI8p0lJHaCAoJRjLg7y9FEZWIHJ72MbdAKPUgGuKSKu/e9iNqoBWtEujJGqSopJ32pU1R1AW23yRcBvCEiawGg8/hm7KMjJCHiqNlt41aYmMoEd2wcw46NY56vbfvY6sBEnSBWxiBUlQwDJuqUMQAtVV0QkSqA3wfwxwD2A7gDwN7O44+SHCghcdJPgSo/7NV3v8FJm6Du8yd+3cTt29b1+L2rVgWXWCNGE0VrSfHbdxcDFSmk+JisxNcCOCQivwDwfwD8par+GG3j/XkR+VsAn+/8TUghME2A8cO5+k6CUwtN3D+5GbdvW7eicKmI4KYr69h93SZfDbqbJdWe5B9SLkzUKb8AMO7x/K8BXJPEoAhJmqhKjEFX36ZcVqtidr6BfX/92opvfEkV+/76NUx8dA0euHFz19jPnV/0XZ2zYFW5MVKnxMXExITOzc2ldjySf5KU+cWNl6LFD6siuPSiVXi72fZN9/MzsyqCmZu3dHWqd1KrWnhxd3fyTdjYBMAre681HwTJFQOrUwhJgn77UGaNaXZl3TUZXTF9oL8DdQx+UDd7N/axvvn4EaP+nKQ8FMKIF2m1RsyJW+aXNGHKFT+deb9NH1rL3iVrndiVCp3Yf3vp3yktLC+5L4BlUqWNFJM4ZX5pELSaDZLwRTGgjYUmRi3/n6ffb8BdS5zSwvKT+5V40VZrxJw4ZH6D4nWXB3gHPL065vitvt37rVojaLaWjcdVEcHFVgXnfLYJ+g149d0k5SX3RrxoqzVijp9RTOvW38snP/XEEUCw0pXHy08f5trz2q9VEc8MSj8f+5IqFkK04PwNEKAARjwPqzWSDFkXXPK6y3MaWRvnqtdkleu53yXFpRdVsNxaxpLqiub70LHTvv5yCVG1KID10wdQq1rYc/0mrr6HlNz7xAdNyiD5JeuAdT8rWZP32k0Y/Izy2fNLXZrvpw43sGPjmG/ijsd84slCs4WpJ44wTjSk5N6IM1BTTvIQsO7nbi7svVEyOJutJfz4yOu4JCCACaArY9MPE0ULKSe5d6cADNSUkTwErL188taIdPnEAbM7vygdeoDgioQ2y6o40UnUuWL6gG/5WvrIh5NCGHFSPvIQsPbzyXs9FzaxBI27Xqvi7HuLxp3k3TjvAoI054wTDSc04iQT8hKw9rvL6/duwO987CYM/aTsO7Eq0nUXMLVzA6aePNJ1pwC07yAYJxpOaMRJJmQtL7RxB1d3bBzDoWOn+w62hp2P16o/qGgV0G5QvPu6btWJ/f/7nj66si3VKcMNjTjJhKzkhe4OPO+8u7giK2wsNLtqfAfVcpmdb/QY0puurOPAL15fee7iVcEBy2s/tRZPHW50GX67ZZu7/oqToBhR1oofkj6sYkhKRZARi+rSqIhgWXVlf3OvvuXb0GEEgDPH0jbKq10TBtBeqTu14u6em3Y2KGA22SXdN5RkR1AVQxpxUgrcK2MbpxEL0nBnhe0z9xtbrWrhvcVlI8Pstw82Ry4+LEVLSk3QCrvZWsKe/Ucx9+pbuTPgwIUmyX7qFi9Fi58UMw+KH5I+uU/2ISSMMI32QrMV2M8yS+wEnn5VOV6G2W8flB6WG67ESWFJq1Vakiyp4orpA6iNWp4FsvyaItvt29zKGneglCUqyg+NOImVtNQRUYOUeUSBHkNtywYB7yYPOzaO9VRKfOT5k6haI1g9amHhXIvqlCGBRpzERprt1qKmuReF9xbbGhc/Kabf+bdrlgu+e+tWGu8hYWiMOPWzydNPPZR+mjF4UfZgXVj527v3vWi0LSk/Q2HEi9aQt6iYqiOiNmNw0m/fyiISNFGFnX/ZJzlygaFQpwStEEl8mKoj9uw/6tmMwV0PJOgz2rFxbICRFoMgVYlXnX3TbUm5GIqVOPWz6WBSD2V2vtFXNT/7M5qdb2DP/qORKwEWjTBViVcNFdNtSbkYipU49bPpYNLAo9+7H1tKN/XEkaEx4BURo1T5yfE65r/9BTx461Y2TRlihiLtnjUl8kNQUwOrIj3NGB64cXPhteBRqNeqPQFeBueHl6FPu8+6IW8eyIsB8AvI2WVXbYNdEVnxiQ+bARdcSMe3A7xzr77VlcjD4HwxSON3NxQr8WEnT3ciYWPJQxJPrWqhtbSMs+cHG8ODt27FNx8/stIc2Y1VEUDRlaXprmRoUxHx3A+LW+WXOH93QSvxofCJDzt5UueE+c3zkMSz0GwNbMCB9rn6GXAAmLl5C2Z2bem6Fn7v9tsPg/P5Ja3f3VC4U4advKlzgpoalMUorR61ALQNs195WPsaOK+FXzlZv5U4g/P5Ja3fHVfiQ0DS6pzZ+Qa2730WV0wfwPa9z2J2vhF5H+k595LDqgh2X9eue+Kl57Zrn3hdM7/333bV5Z7PU0qYX9JSxdGIDwF+hiEOA2D7/RoLTSguBNz6MeTOfRSdighmbt7Stcp2u49uurKOpw43PK+Zn7vp/snNofJNki+S/N05YWBzSEgqSh5HN5mgjjt1j+bFeTX2pkErduAZHuL63Q29xJAE+6EHIcjvZ/oF9tuHACtGzbmvPGD7qO1Hu7Ex0DbSQeectxgFSY6kfndOQo24iFwO4L8D+Ido94B9SFX/RETWANgHYD2AEwBuUdUzyQ2V5JHaqOXZtKA2ahkXHfNbXY+IYP30AYwIsJwzZ/l3btnScx6mhdb8zpdBShIFE5/4IoBvqurvAdgG4Osi8gkA0wCeUdWPA3im8zcZImbnG3jn3UXP1xaaLWN5lV8xJ1uNkTcDXqtangb8m48fMTrntHylZDgIXYmr6usAXu/8/7ci8jKAOoAbAHyu87aHAfwMwD2JjJLkkpmDx7sSVZz4hVq8XAbujNoRHzldXth02e90/W2vwE213MwgJnHSl09cRNYDGAfwAoAPdww8VPV1EfmQzzZ3ArgTANatWzfIWEnOiOLDdboM/BpD3BXQ8CAPPPd3b+He2Zdw/+RmAOEJSl5ukjR8pWQ4MJYYisg/APAUgLtU9Tem26nqQ6o6oaoTY2PlrwE9TPTrw3W6DLykiVNPHmk3hygAj73w2sr/g9QydJOQpDEy4iJioW3AH1XVH3SefkNE1nZeXwvgzWSGSPJKWGMCJybp9a0l9XXP5A3bdTI734D4vMe0pCwhg2CiThEA3wfwsqr+Z8dL+wHcAWBv5/FHiYyQ5BanQQ7Tbrv1z0WX041IsL5d4K1gISRuTHzi2wH8cwAviYjtrPy3aBvvx0XkawBOAtiVzBBJnnH6dj/x7/4nzrWWe95j1xFxkuekHSM02I1SjPsJUgZC3Smq+r9VVVT1U6q6tfPvf6jqr1X1GlX9eOfxrTQGTPLLf7jxU+3yqg6cdUSc5LFHZmXEzzGCHpdJ71TVS7/lBwiJAjM2c0xeGjmY0o907tCx02kPL5BRa8TzLmIQbI14nj8zUnxoxHOKafZf0mPodxIxlc7lzSceZsCjukfydp79ULRFxLDCKoY5JetGDnFUJwza94j4uy7KRFFT6ZP8/Em80IjnlKyLJCU1iYRlN+aRqNONn0Y8jvrrSZP1IoKYQyOeU9IqKO9HUpPIIO3XbGMaEH803kc/3L5tnbEe3sav3ndRVrhZLyKIOTTiOSXrIklJTSJRjUCtauH2betQr1UHKoilnX2ZUq9VVxoyVPpwAZ07710YrCgr3KwXEcQcGvGcEtZQOGmSmkQGMQJ2N5y0cJ7v5Hgd37lli/GK/My5Fu7e9yLWu1wmRVnhZr2IIOZQnZJjsiqSZKsSmq2lnqYHg45naueGLtWNKQvN3prlUQnaV71W9VVj2P83LdBl3zA4lUVBtdPt9mx5gJUWiwON+BBhIhlzSxuXVFdWYHH8gPtJ1R+EKLpvr/ZofpUW+8V2mfhNYkuqqUtIw2ClxWJAd8qQYBpQS8NnOzlex3PTV+PE3muNt7l4VfBXdfWoBek81qoWmiEG3MRV4HfNqla0n82pheaKm8zLv55H3zjJPzTiQ4KpcY7bZxsmpzMNMr636G+U67Uqdl+3Ce+vtlvFLTRbgck5taqFB27c3HXsSzwMs981u6RPpYqNHQ+YHK9j2bCBBCFh0IgPCabGOU5VQtjq/97Zlwb2dVetCnZsHMO3fmC2L2tEsOf6di0X58Rw5lyr587E75otePQUNRmnc6VP9QeJCxrxIcHPONgBNZs4VQl79h/1Xf3Pzjfw6PMn+94n0K7T7VTsHDp2OjRQar9/Zle7PKzJnUmQoa0bGFvbYeJWFs3ON3D2vV4JItUfJAoMbA4JpgG1uFQJs/MN35XxqYUmZg4ej1yPZFkVrzj86XeHqEW8ApZ+QVXn81M7N2DqiSM9jSpOde4sBME1VdTj2O7Asc2IdE8iWaqSqEYpFjTiQ4L9Y/zm40d6Ut7d1fbiUCUEBegu68j4ovJ+lx89qDa57W7ZvvfZLuNU8WnG3BNw9MjvUcdjmCF3n6dfxqo9T2RR6AzIR8E1Eg26U0qOM7A4c/C4cUf2QQna39TODQP5ft9+t9UVKPVrE7d61MJNV9ZXkoRsv/xd+170vQ5LqivupZmDx9FaCr5fsFfbfu4V93maXOcsVCpFySQlvdCIlxivwKJf4nhcATV70vAzfatHLUyO1/vqz+lGFV2BUgA92a0P3roVu6/bhMdeeK3vxCI7wGk6sTUWmp7X1svHbXqd01apFCWTlPRCd0qJ8Vpd+RnXODrt+Pl7bapWZaXLj32Lft/TR3HGpfYIc1E4sVeLz01f3XXbP0i1RHuf/baQc7pXKiKePm7TjNXLatVUfdR+50q1TP7hStyQIpQPddOPAYqj005QhUK/2i+/afaqNGxjaIrXanGQaolA+9pFuVuwx25PHl5JVc7EpVFrpKelnVM2mVa1Q9ZKKS404gYUpXyok9n5xsCGsF/C/OBOid3W+34a6Ju2fc3AhWCjXxVB92pxdj6+Qlm2m6Yf3GfklFW69ewKwa3/+PKeQmdesskkfdRZF1wj0RFNsTj/xMSEzs3NpXa8uNi+91lPo+AlXcsLfmP2I45zCTqmvf8wl0vQePy2XT1qrbhp9uw/GluxLOcYvI7dj9tH0FbVeI3NLi5mu05qo1aPi8m5n1f6KFdAyoGIHFbVCa/X6BM3oIhBn6CxVa1KlzHq57Y5yE+7Y+MYHvFJ4LHHY+LmsCqCs+8t4orpA13HsI/jNtRnzrUw9cQRQBCoJunH6DrHDPQW7rIliqb7DDLM9p2dfV383gfQR016oTvFgCKmSPuNzb5NjnLbHOZWCvKr2+MJm/ikYxXt+ifuY0yO13Hpxb1rj9ayhsoB+23r476GTlWN7QYyMeBVq4KgG147CGqyH/qoiRsacQOKGPQJGrNdRfCVvdf2qDqCCNMSh/nEgfCJT4CeDMlmawn3PX105e8od0D1WjXQkHqNo7HQ7Ali9xswtSfJtwNcPKYKGvqoiRc04gYUMeiTxJjD3Ep+BrpWtbokdkGKD7/Wa2fOtVaMaZQ7oB0bx/pqr+Zu6BDWmccLAVYmSb8xrx61jAKn9Vo11983kh30iRtSxAL5cY85TEvspYGuWpWVqoFeHYP80t+9sEsDROkOdOjYadx21eW+PnsbLx+3syxBP9pxp+H2uzZ2QDZMX5/nuz6SLVyJE2P8VtHnzi+utBbzW/07/enAhY5B/STj2Ktg+zj9cGqhifsnN+Or29Z1SRa3/+6arvH6jcY+9tTODUaudbfhDbo27tfsxhZFuesj2UKJoSFFrPCWxJhn5xueMr6qVQk0Nv1KHr1wyw6jyCiDrsHsfMOzQJj72OunDwQep1a1sOf6Tbn/fpDiECQx5ErcgKIm+yQxZj91iB189Mtq7ceXbFUE1khvFqPbpeB1Z2CNSE8GpE3QNQhK03cfO8y3HtSFiJC4oRE3oIgV3pIcs59BPnOu5TtphAUjnY0eZm7egpldW0KDsl4uipldWzBz8xbfYKHfNfBTnVREeo4d5gLK+3eDlAsGNg0oU7JPHGM2De45A4JBiUBAb6MHwKyOtV/wdgR9Ki0AAAj4SURBVHK8jiumD3j6uL2ugd91WVbt2X/d4Pzz/N0g5YIrcQPKlOwTx5j7KQxlG7OwAlv9jsukIFk/12D0Iu/z8XqvyfnXRq3CFUwjxYRG3ICyJfsMipcbw69rvUmmZr/jMvX3m16De2dfwtnzHq6UEfEcl/P8gd5EUKsieOfdxULFUEhxoTvFgLj6TkYhqsIk7TF/ectaPHW44VuTxc8F4+VzDsJPQeJuMQeYX4PHXnjN81jLy72uFOe+nVUZncc4+95ij3rHa3yDUkTFFIkfSgxzjFflvDApn71dkj9uv3HddGUdh46d9jxu1HMJO66TqBX+giSDJyLsz88XH2cFwjiuJykOA1UxFJH/CuDLAN5U1U92nlsDYB+A9QBOALhFVc/ENWDSJkhhEqR1Trrhrd+4Dh077VvONo47g7C6JVH9/UFZo3YSUz+k0SUnyneDlBMTn/ifAfii67lpAM+o6scBPNP5m8RMFIWJqbRwkE5FUZUvUQtvmexfgMj+/tuuutz3tShSwTRiKEVUTJFkCF2Jq+pfich619M3APhc5/8PA/gZgHtiHBdBtBWdyY970NV6Vv0Yg6SNiuh3GvdPbg6tg+7E7a7asXGsx430wI2bE3VpsScmsYmqTvmwqr4OAJ3HD/m9UUTuFJE5EZk7fXrwPo7DRJQVnYmsbtBEoKzUOkF1S/ptoWa6vVfrN7cy5pHnT/YoUQAMdNcRRhEVUyQZEpcYqupDqjqhqhNjY4N3VB8mopSTNflxD3ornlVp3snxOm7ftq7HkMdhvEyNokk98TQyNotYHpkkQ1SJ4RsislZVXxeRtQDejHNQZWJQpUi/5WRNAohx3IpnVZr3/snNmPjoGt/zS1qSaTrRpeGbLmJ5ZBI/UY34fgB3ANjbefxRbCMqEWkoRbwI+3H71bZO+lY8ioH128Zru36vt9e+w5pFm5YcoG+apIWJxPAxtIOYHxSRXwLYjbbxflxEvgbgJIBdSQ6yqORVBhZ3IpCJcY4yoZls4zw2ENzQYdDxAN4ToJs4JkQm8hBTTNQpt/m8dE3MYykdeZaBuVeztuSwX6NhagyjTGhh24Ql/9h4Xe+oE6zXBOilThnE4GZ1B0eKCdPuE6QoMrBBjIapMYwyoYVtY9q02Ot6DzLBJu2LzusdHMknLICVIEWRgQ0iOTQ1hlGqKoZtY3pH43W981yZMs93cCR/0IgnSB5lYF6ZmoMYDVNjGGVCC9vGxOCuHrU8r3deJ9jZ+QZGfDoH5WGCIfmD7pSEyZMMzM9tUhu1cOZcq+f9XkbDK1sxqHqhTZRgatg2YUFGZzf5fvcddt72e+MMQPbTIo4QG1YxzJg0VQh+jYVrVQvvLS6HVsQzqV74/qoFEWDhXCsVVYXz+iV17KDz9prAot5t+X0+FRF855YtuVkMkPQZqIohSY60VQh+7pG3my1899atoZNJWPXCLFQVadzp+J33Yy+8ZlTX3JR+WsQRYkMjniFpqxCC1DImxjCKWqQMqgq/8/YrXxs1AFkUNRPJFwxsZkjaKoRBg3lR1SJJnM8gpXT7xe+8KzEHIPMabCX5hkY8Q9KWuQ2qlomqFon7fEx7bMaF33nfdtXlsRrdPKqZSP6hOyVDsqhhMogPOYpaJInzSdttE3TeQcW4oh6LRpv0A9UpGVO2GhlpnI9JD8uyXVcy3FCdkmPKtvJK43zCAoCsPUKGCfrESeEI880P2rmIkCLBlThJjDjrhzsJ882z9ggZJmjESSIkVT/cJshtQ701GSboTiGJEMWlEZcbhHprMkxwJU4SIYn64abE3bmIkDxDI04SIYpLI043SNlUP4T4QXcKSYQk6ocTQnrhSpwkQhL1wwkhvTBjkxBCck5QxibdKYQQUmBoxAkhpMDQiBNCSIGhESeEkAJDI04IIQUmVXWKiJwG8GpqB4zGBwH8KutBpADPs3wMy7kO43l+VFXHvN6UqhEvAiIy5yflKRM8z/IxLOfK8+yG7hRCCCkwNOKEEFJgaMR7eSjrAaQEz7N8DMu58jwd0CdOCCEFhitxQggpMDTihBBSYGjEHYhIRUTmReTHWY8lSUTkhIi8JCIvikhpy0qKSE1EnhSRYyLysoj8k6zHFDcisqHzOdr/fiMid2U9riQQkbtF5KiI/I2IPCYil2Q9piQQkW90zvGoyWfJeuLdfAPAywDel/VAUmCHqpY9YeJPAPxEVW8WkYsAjGY9oLhR1eMAtgLtRQiABoAfZjqoBBCROoB/A+ATqtoUkccBfAXAn2U6sJgRkU8C+JcAPgPgPICfiMgBVf1bv224Eu8gIh8BcC2A72U9FjI4IvI+AJ8F8H0AUNXzqrqQ7agS5xoAf6eqec+KjsoqAFURWYX2hHwq4/Ekwe8BeF5Vz6nqIoD/BeCfBW1AI36BBwH8IYDlrAeSAgrgpyJyWETuzHowCfExAKcB/LeOi+x7InJp1oNKmK8AeCzrQSSBqjYA/CcAJwG8DuBtVf1ptqNKhL8B8FkR+YCIjAL4AwCXB21AIw5ARL4M4E1VPZz1WFJiu6p+GsCXAHxdRD6b9YASYBWATwP4L6o6DuAsgOlsh5QcHXfR9QCeyHosSSAiqwHcAOAKAJcBuFREvprtqOJHVV8G8McA/hLATwAcAbAYtA2NeJvtAK4XkRMA/gLA1SLySLZDSg5VPdV5fBNt/+lnsh1RIvwSwC9V9YXO30+ibdTLypcA/FxV38h6IAnx+wBeUdXTqtoC8AMA/zTjMSWCqn5fVT+tqp8F8BYAX384QCMOAFDVb6nqR1R1Pdq3pM+qaulmeQAQkUtF5Hfs/wP4Atq3cKVCVf8fgNdEZEPnqWsA/N8Mh5Q0t6GkrpQOJwFsE5FRERG0P8+XMx5TIojIhzqP6wDciJDPleqU4ePDAH7Y/h1gFYA/V9WfZDukxPjXAB7tuBr+HsC/yHg8idDxnX4ewL/KeixJoaoviMiTAH6OtnthHuVNv39KRD4AoAXg66p6JujNTLsnhJACQ3cKIYQUGBpxQggpMDTihBBSYGjECSGkwNCIE0JIgaERJ4SQAkMjTgghBeb/A0T4tUWrpYfdAAAAAElFTkSuQmCC\n",
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
    "plt.scatter(X_rm,y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Iteration 0, the loss is 375.7439515621081, parameters k is -54.886942022775465 and b is -8.266781985079035\n",
      "Iteration 1, the loss is 371.6942886238391, parameters k is -54.25847858404028 and b is -8.166781985079036\n",
      "Iteration 2, the loss is 367.6446256855707, parameters k is -53.6300151453051 and b is -8.066781985079036\n",
      "Iteration 3, the loss is 363.5949627473022, parameters k is -53.00155170656992 and b is -7.966781985079036\n",
      "Iteration 4, the loss is 359.54529980903385, parameters k is -52.37308826783474 and b is -7.866781985079037\n",
      "Iteration 5, the loss is 355.4956368707656, parameters k is -51.74462482909956 and b is -7.766781985079037\n",
      "Iteration 6, the loss is 351.44597393249757, parameters k is -51.116161390364375 and b is -7.666781985079037\n",
      "Iteration 7, the loss is 347.39631099422854, parameters k is -50.487697951629194 and b is -7.566781985079038\n",
      "Iteration 8, the loss is 343.3466480559598, parameters k is -49.85923451289401 and b is -7.466781985079038\n",
      "Iteration 9, the loss is 339.2969851176913, parameters k is -49.23077107415883 and b is -7.366781985079038\n",
      "Iteration 10, the loss is 335.2473221794234, parameters k is -48.60230763542365 and b is -7.266781985079039\n",
      "Iteration 11, the loss is 331.1976592411549, parameters k is -47.97384419668847 and b is -7.166781985079039\n",
      "Iteration 12, the loss is 327.1479963028862, parameters k is -47.345380757953286 and b is -7.066781985079039\n",
      "Iteration 13, the loss is 323.09833336461736, parameters k is -46.716917319218105 and b is -6.96678198507904\n",
      "Iteration 14, the loss is 319.0486704263491, parameters k is -46.08845388048292 and b is -6.86678198507904\n",
      "Iteration 15, the loss is 314.9990074880807, parameters k is -45.45999044174774 and b is -6.7667819850790405\n",
      "Iteration 16, the loss is 310.94934454981245, parameters k is -44.83152700301256 and b is -6.666781985079041\n",
      "Iteration 17, the loss is 306.8996816115438, parameters k is -44.20306356427738 and b is -6.566781985079041\n",
      "Iteration 18, the loss is 302.8500186732756, parameters k is -43.5746001255422 and b is -6.466781985079042\n",
      "Iteration 19, the loss is 298.80035573500703, parameters k is -42.946136686807016 and b is -6.366781985079042\n",
      "Iteration 20, the loss is 294.7506927967385, parameters k is -42.317673248071834 and b is -6.266781985079042\n",
      "Iteration 21, the loss is 290.70102985847, parameters k is -41.68920980933665 and b is -6.166781985079043\n",
      "Iteration 22, the loss is 286.6513669202015, parameters k is -41.06074637060147 and b is -6.066781985079043\n",
      "Iteration 23, the loss is 282.60170398193304, parameters k is -40.43228293186629 and b is -5.966781985079043\n",
      "Iteration 24, the loss is 278.55204104366436, parameters k is -39.80381949313111 and b is -5.866781985079044\n",
      "Iteration 25, the loss is 274.50237810539625, parameters k is -39.17535605439593 and b is -5.766781985079044\n",
      "Iteration 26, the loss is 270.45271516712756, parameters k is -38.546892615660745 and b is -5.666781985079044\n",
      "Iteration 27, the loss is 266.4030522288591, parameters k is -37.918429176925564 and b is -5.566781985079045\n",
      "Iteration 28, the loss is 262.3533892905902, parameters k is -37.28996573819038 and b is -5.466781985079045\n",
      "Iteration 29, the loss is 258.30372635232186, parameters k is -36.6615022994552 and b is -5.3667819850790455\n",
      "Iteration 30, the loss is 254.25406341405363, parameters k is -36.03303886072002 and b is -5.266781985079046\n",
      "Iteration 31, the loss is 250.20440047578495, parameters k is -35.40457542198484 and b is -5.166781985079046\n",
      "Iteration 32, the loss is 246.15473753751687, parameters k is -34.776111983249656 and b is -5.0667819850790465\n",
      "Iteration 33, the loss is 242.10507459924838, parameters k is -34.147648544514475 and b is -4.966781985079047\n",
      "Iteration 34, the loss is 238.0554116609797, parameters k is -33.51918510577929 and b is -4.866781985079047\n",
      "Iteration 35, the loss is 234.0057487227114, parameters k is -32.89072166704411 and b is -4.766781985079048\n",
      "Iteration 36, the loss is 229.9560857844428, parameters k is -32.26225822830893 and b is -4.666781985079048\n",
      "Iteration 37, the loss is 225.90642284617437, parameters k is -31.633794789573752 and b is -4.566781985079048\n",
      "Iteration 38, the loss is 221.85675990790614, parameters k is -31.005331350838574 and b is -4.466781985079049\n",
      "Iteration 39, the loss is 217.80709696963763, parameters k is -30.376867912103396 and b is -4.366781985079049\n",
      "Iteration 40, the loss is 213.75743403136923, parameters k is -29.74840447336822 and b is -4.266781985079049\n",
      "Iteration 41, the loss is 209.70777109310058, parameters k is -29.11994103463304 and b is -4.16678198507905\n",
      "Iteration 42, the loss is 205.65810815483226, parameters k is -28.491477595897862 and b is -4.06678198507905\n",
      "Iteration 43, the loss is 201.6084452165639, parameters k is -27.863014157162684 and b is -3.96678198507905\n",
      "Iteration 44, the loss is 197.5587822782952, parameters k is -27.234550718427506 and b is -3.86678198507905\n",
      "Iteration 45, the loss is 193.509119340027, parameters k is -26.60608727969233 and b is -3.76678198507905\n",
      "Iteration 46, the loss is 189.45945640175836, parameters k is -25.97762384095715 and b is -3.6667819850790497\n",
      "Iteration 47, the loss is 185.4097934634902, parameters k is -25.349160402221973 and b is -3.5667819850790496\n",
      "Iteration 48, the loss is 181.36013052522168, parameters k is -24.720696963486795 and b is -3.4667819850790496\n",
      "Iteration 49, the loss is 177.3104675869529, parameters k is -24.092233524751617 and b is -3.3667819850790495\n",
      "Iteration 50, the loss is 173.26080464868454, parameters k is -23.46377008601644 and b is -3.2667819850790494\n",
      "Iteration 51, the loss is 169.21114171041637, parameters k is -22.83530664728126 and b is -3.1667819850790493\n",
      "Iteration 52, the loss is 165.16147877214777, parameters k is -22.206843208546083 and b is -3.066781985079049\n",
      "Iteration 53, the loss is 161.11181583387926, parameters k is -21.578379769810905 and b is -2.966781985079049\n",
      "Iteration 54, the loss is 157.0621528956108, parameters k is -20.949916331075727 and b is -2.866781985079049\n",
      "Iteration 55, the loss is 153.01248995734232, parameters k is -20.32145289234055 and b is -2.766781985079049\n",
      "Iteration 56, the loss is 148.96282701907398, parameters k is -19.69298945360537 and b is -2.666781985079049\n",
      "Iteration 57, the loss is 144.9131640808056, parameters k is -19.064526014870193 and b is -2.5667819850790488\n",
      "Iteration 58, the loss is 140.86350114253716, parameters k is -18.436062576135015 and b is -2.4667819850790487\n",
      "Iteration 59, the loss is 136.81383820426854, parameters k is -17.807599137399837 and b is -2.3667819850790486\n",
      "Iteration 60, the loss is 132.76417526600017, parameters k is -17.17913569866466 and b is -2.2667819850790485\n",
      "Iteration 61, the loss is 128.7145123277317, parameters k is -16.55067225992948 and b is -2.1667819850790484\n",
      "Iteration 62, the loss is 124.66484938946331, parameters k is -15.922208821194303 and b is -2.0667819850790483\n",
      "Iteration 63, the loss is 120.61518645119484, parameters k is -15.293745382459125 and b is -1.9667819850790482\n",
      "Iteration 64, the loss is 116.56552351292648, parameters k is -14.665281943723947 and b is -1.8667819850790481\n",
      "Iteration 65, the loss is 112.51586057465806, parameters k is -14.03681850498877 and b is -1.766781985079048\n",
      "Iteration 66, the loss is 108.46619763638957, parameters k is -13.408355066253591 and b is -1.666781985079048\n",
      "Iteration 67, the loss is 104.41653469812101, parameters k is -12.779891627518413 and b is -1.5667819850790479\n",
      "Iteration 68, the loss is 100.36687175985259, parameters k is -12.151428188783235 and b is -1.4667819850790478\n",
      "Iteration 69, the loss is 96.3172088215842, parameters k is -11.522964750048057 and b is -1.3667819850790477\n",
      "Iteration 70, the loss is 92.26754588331569, parameters k is -10.89450131131288 and b is -1.2667819850790476\n",
      "Iteration 71, the loss is 88.21788294504726, parameters k is -10.266037872577702 and b is -1.1667819850790475\n",
      "Iteration 72, the loss is 84.16822000677871, parameters k is -9.637574433842524 and b is -1.0667819850790474\n",
      "Iteration 73, the loss is 80.11855706851038, parameters k is -9.009110995107346 and b is -0.9667819850790474\n",
      "Iteration 74, the loss is 76.06889413024183, parameters k is -8.380647556372168 and b is -0.8667819850790475\n",
      "Iteration 75, the loss is 72.01923119197345, parameters k is -7.752184117636989 and b is -0.7667819850790475\n",
      "Iteration 76, the loss is 67.96956825370495, parameters k is -7.12372067890181 and b is -0.6667819850790475\n",
      "Iteration 77, the loss is 63.91990531543656, parameters k is -6.495257240166631 and b is -0.5667819850790475\n",
      "Iteration 78, the loss is 59.87024237716811, parameters k is -5.866793801431452 and b is -0.46678198507904756\n",
      "Iteration 79, the loss is 55.820579438899664, parameters k is -5.238330362696273 and b is -0.3667819850790476\n",
      "Iteration 80, the loss is 51.770916500631195, parameters k is -4.6098669239610945 and b is -0.2667819850790476\n",
      "Iteration 81, the loss is 47.7212535623627, parameters k is -3.9814034852259157 and b is -0.16678198507904762\n",
      "Iteration 82, the loss is 43.671590624094335, parameters k is -3.352940046490737 and b is -0.06678198507904763\n",
      "Iteration 83, the loss is 39.62192768582583, parameters k is -2.724476607755558 and b is 0.03321801492095236\n",
      "Iteration 84, the loss is 35.57226474755736, parameters k is -2.096013169020379 and b is 0.13321801492095237\n",
      "Iteration 85, the loss is 31.522601809288904, parameters k is -1.4675497302852003 and b is 0.23321801492095234\n",
      "Iteration 86, the loss is 27.472938871020485, parameters k is -0.8390862915500216 and b is 0.3332180149209523\n",
      "Iteration 87, the loss is 23.42327593275199, parameters k is -0.21062285281484283 and b is 0.4332180149209523\n",
      "Iteration 88, the loss is 19.37361299448354, parameters k is 0.4178405859203359 and b is 0.5332180149209523\n",
      "Iteration 89, the loss is 15.344250536409348, parameters k is 1.0463040246555146 and b is 0.6332180149209523\n",
      "Iteration 90, the loss is 11.638190290213599, parameters k is 1.6579556056831835 and b is 0.73045121650198\n",
      "Iteration 91, the loss is 8.6620249283474, parameters k is 2.223239597778046 and b is 0.8197792797430867\n",
      "Iteration 92, the loss is 6.88323253429293, parameters k is 2.6691093606238963 and b is 0.8893444971343911\n",
      "Iteration 93, the loss is 5.881052484267274, parameters k is 3.0079498744578883 and b is 0.9411231532608734\n",
      "Iteration 94, the loss is 5.51731100851083, parameters k is 3.2318390048926706 and b is 0.9735342204545493\n",
      "Iteration 95, the loss is 5.466180880148568, parameters k is 3.3276937479756747 and b is 0.9846014141304387\n",
      "Iteration 96, the loss is 5.464420217537342, parameters k is 3.3485998744578884 and b is 0.9834156433794505\n",
      "Iteration 97, the loss is 5.464191907602417, parameters k is 3.348021811216782 and b is 0.978672560375498\n",
      "Iteration 98, the loss is 5.463971994467652, parameters k is 3.3474437479756753 and b is 0.9739294773715454\n",
      "Iteration 99, the loss is 5.463767088358239, parameters k is 3.349123787501367 and b is 0.969581651284589\n",
      "Iteration 100, the loss is 5.463558223913511, parameters k is 3.3485457242602603 and b is 0.9648385682806364\n",
      "Iteration 101, the loss is 5.4633422691140705, parameters k is 3.350225763785952 and b is 0.9604907421936799\n",
      "Iteration 102, the loss is 5.463144453359379, parameters k is 3.3496477005448453 and b is 0.9557476591897274\n",
      "Iteration 103, the loss is 5.4629271921144715, parameters k is 3.351327740070537 and b is 0.9513998331027709\n",
      "Iteration 104, the loss is 5.462720940560666, parameters k is 3.3530077795962283 and b is 0.9470520070158144\n",
      "Iteration 105, the loss is 5.462513421560329, parameters k is 3.3524297163551218 and b is 0.9423089240118618\n",
      "Iteration 106, the loss is 5.462296160315431, parameters k is 3.3541097558808133 and b is 0.9379610979249053\n",
      "Iteration 107, the loss is 5.462099612007265, parameters k is 3.355789795406505 and b is 0.9336132718379488\n",
      "Iteration 108, the loss is 5.461882389761281, parameters k is 3.3552117321653983 and b is 0.9288701888339963\n",
      "Iteration 109, the loss is 5.461674792763093, parameters k is 3.3568917716910898 and b is 0.9245223627470398\n",
      "Iteration 110, the loss is 5.4614686192071344, parameters k is 3.3563137084499832 and b is 0.9197792797430873\n",
      "Iteration 111, the loss is 5.461251357962236, parameters k is 3.3579937479756747 and b is 0.9154314536561308\n",
      "Iteration 112, the loss is 5.461053464209689, parameters k is 3.3596737875013662 and b is 0.9110836275691743\n",
      "Iteration 113, the loss is 5.460837587408104, parameters k is 3.3590957242602597 and b is 0.9063405445652217\n",
      "Iteration 114, the loss is 5.4606286449655155, parameters k is 3.360775763785951 and b is 0.9019927184782652\n",
      "Iteration 115, the loss is 5.460423816853954, parameters k is 3.3601977005448447 and b is 0.8972496354743127\n",
      "Iteration 116, the loss is 5.460206555609056, parameters k is 3.361877740070536 and b is 0.8929018093873562\n",
      "Iteration 117, the loss is 5.460007316412112, parameters k is 3.3635577795962277 and b is 0.8885539833003997\n",
      "Iteration 118, the loss is 5.4597927850549155, parameters k is 3.362979716355121 and b is 0.8838109002964472\n",
      "Iteration 119, the loss is 5.459582497167946, parameters k is 3.3646597558808127 and b is 0.8794630742094907\n",
      "Iteration 120, the loss is 5.459379014500781, parameters k is 3.364081692639706 and b is 0.8747199912055381\n",
      "Iteration 121, the loss is 5.459161753255871, parameters k is 3.3657617321653976 and b is 0.8703721651185816\n",
      "Iteration 122, the loss is 5.458961168614535, parameters k is 3.367441771691089 and b is 0.8660243390316251\n",
      "Iteration 123, the loss is 5.4587479827017305, parameters k is 3.3668637084499826 and b is 0.8612812560276726\n",
      "Iteration 124, the loss is 5.458536349370372, parameters k is 3.368543747975674 and b is 0.8569334299407161\n",
      "Iteration 125, the loss is 5.458334212147591, parameters k is 3.3679656847345676 and b is 0.8521903469367635\n",
      "Iteration 126, the loss is 5.458116950902685, parameters k is 3.369645724260259 and b is 0.847842520849807\n",
      "Iteration 127, the loss is 5.457915020816963, parameters k is 3.3713257637859506 and b is 0.8434946947628505\n",
      "Iteration 128, the loss is 5.457703180348543, parameters k is 3.370747700544844 and b is 0.838751611758898\n",
      "Iteration 129, the loss is 5.4574902015728055, parameters k is 3.3724277400705356 and b is 0.8344037856719415\n",
      "Iteration 130, the loss is 5.457289409794405, parameters k is 3.371849676829429 and b is 0.829660702667989\n",
      "Iteration 131, the loss is 5.457072148549496, parameters k is 3.3735297163551206 and b is 0.8253128765810325\n",
      "Iteration 132, the loss is 5.456868873019386, parameters k is 3.375209755880812 and b is 0.820965050494076\n",
      "Iteration 133, the loss is 5.45665837799536, parameters k is 3.3746316926397055 and b is 0.8162219674901234\n",
      "Iteration 134, the loss is 5.456444053775212, parameters k is 3.376311732165397 and b is 0.8118741414031669\n",
      "Iteration 135, the loss is 5.4562446074412225, parameters k is 3.3757336689242905 and b is 0.8071310583992144\n",
      "Iteration 136, the loss is 5.456027346196319, parameters k is 3.377413708449982 and b is 0.8027832323122579\n",
      "Iteration 137, the loss is 5.455822725221812, parameters k is 3.3790937479756735 and b is 0.7984354062253014\n",
      "Iteration 138, the loss is 5.4556135756421735, parameters k is 3.378515684734567 and b is 0.7936923232213489\n",
      "Iteration 139, the loss is 5.45539790597764, parameters k is 3.3801957242602585 and b is 0.7893444971343924\n",
      "Iteration 140, the loss is 5.455199805088034, parameters k is 3.379617661019152 and b is 0.7846014141304398\n",
      "Iteration 141, the loss is 5.454982543843133, parameters k is 3.3812977005448435 and b is 0.7802535880434833\n",
      "Iteration 142, the loss is 5.454776577424233, parameters k is 3.382977740070535 and b is 0.7759057619565268\n",
      "Iteration 143, the loss is 5.45456877328899, parameters k is 3.3823996768294284 and b is 0.7711626789525743\n",
      "Iteration 144, the loss is 5.454351758180067, parameters k is 3.38407971635512 and b is 0.7668148528656178\n",
      "Iteration 145, the loss is 5.454155002734856, parameters k is 3.3835016531140134 and b is 0.7620717698616652\n",
      "Iteration 146, the loss is 5.4539377414899395, parameters k is 3.385181692639705 and b is 0.7577239437747088\n",
      "Iteration 147, the loss is 5.453730429626659, parameters k is 3.3868617321653964 and b is 0.7533761176877523\n",
      "Iteration 148, the loss is 5.4535239709358025, parameters k is 3.38628366892429 and b is 0.7486330346837997\n",
      "Iteration 149, the loss is 5.453306709690902, parameters k is 3.3879637084499814 and b is 0.7442852085968432\n",
      "Iteration 150, the loss is 5.453109101073254, parameters k is 3.389643747975673 and b is 0.7399373825098867\n",
      "Iteration 151, the loss is 5.452892939136765, parameters k is 3.3890656847345664 and b is 0.7351942995059342\n",
      "Iteration 152, the loss is 5.452684281829088, parameters k is 3.390745724260258 and b is 0.7308464734189777\n",
      "Iteration 153, the loss is 5.452479168582626, parameters k is 3.3901676610191513 and b is 0.7261033904150251\n",
      "Iteration 154, the loss is 5.452261907337722, parameters k is 3.391847700544843 and b is 0.7217555643280686\n",
      "Iteration 155, the loss is 5.452062953275676, parameters k is 3.3935277400705344 and b is 0.7174077382411121\n",
      "Iteration 156, the loss is 5.4518481367835765, parameters k is 3.392949676829428 and b is 0.7126646552371596\n",
      "Iteration 157, the loss is 5.451638134031506, parameters k is 3.3946297163551193 and b is 0.7083168291502031\n",
      "Iteration 158, the loss is 5.451434366229428, parameters k is 3.394051653114013 and b is 0.7035737461462506\n",
      "Iteration 159, the loss is 5.45121710498453, parameters k is 3.3957316926397043 and b is 0.6992259200592941\n",
      "Iteration 160, the loss is 5.4510168054781065, parameters k is 3.397411732165396 and b is 0.6948780939723376\n",
      "Iteration 161, the loss is 5.450803334430392, parameters k is 3.3968336689242893 and b is 0.690135010968385\n",
      "Iteration 162, the loss is 5.450591986233938, parameters k is 3.398513708449981 and b is 0.6857871848814285\n",
      "Iteration 163, the loss is 5.450389563876256, parameters k is 3.3979356452088743 and b is 0.681044101877476\n",
      "Iteration 164, the loss is 5.4501723026313496, parameters k is 3.3996156847345658 and b is 0.6766962757905195\n",
      "Iteration 165, the loss is 5.449970657680527, parameters k is 3.4012957242602573 and b is 0.672348449703563\n",
      "Iteration 166, the loss is 5.4497585320772055, parameters k is 3.4007176610191507 and b is 0.6676053666996105\n",
      "Iteration 167, the loss is 5.44954583843636, parameters k is 3.4023977005448423 and b is 0.663257540612654\n",
      "Iteration 168, the loss is 5.449344761523067, parameters k is 3.4018196373037357 and b is 0.6585144576087014\n",
      "Iteration 169, the loss is 5.449127500278166, parameters k is 3.4034996768294272 and b is 0.6541666315217449\n",
      "Iteration 170, the loss is 5.448924509882952, parameters k is 3.4051797163551187 and b is 0.6498188054347884\n",
      "Iteration 171, the loss is 5.448713729724024, parameters k is 3.404601653114012 and b is 0.6450757224308359\n",
      "Iteration 172, the loss is 5.448499690638787, parameters k is 3.4062816926397037 and b is 0.6407278963438794\n",
      "Iteration 173, the loss is 5.448299959169886, parameters k is 3.405703629398597 and b is 0.6359848133399268\n",
      "Iteration 174, the loss is 5.448082697924977, parameters k is 3.4073836689242887 and b is 0.6316369872529703\n",
      "Iteration 175, the loss is 5.447878362085379, parameters k is 3.40906370844998 and b is 0.6272891611660139\n",
      "Iteration 176, the loss is 5.447668927370839, parameters k is 3.4084856452088736 and b is 0.6225460781620613\n",
      "Iteration 177, the loss is 5.4474535428412105, parameters k is 3.410165684734565 and b is 0.6181982520751048\n",
      "Iteration 178, the loss is 5.4472551568167, parameters k is 3.4095876214934586 and b is 0.6134551690711523\n",
      "Iteration 179, the loss is 5.447037895571791, parameters k is 3.41126766101915 and b is 0.6091073429841958\n",
      "Iteration 180, the loss is 5.446832214287801, parameters k is 3.4129477005448416 and b is 0.6047595168972393\n",
      "Iteration 181, the loss is 5.446624125017657, parameters k is 3.412369637303735 and b is 0.6000164338932867\n",
      "Iteration 182, the loss is 5.446407395043636, parameters k is 3.4140496768294266 and b is 0.5956686078063302\n",
      "Iteration 183, the loss is 5.446210354463515, parameters k is 3.41347161358832 and b is 0.5909255248023777\n",
      "Iteration 184, the loss is 5.445993093218615, parameters k is 3.4151516531140116 and b is 0.5865776987154212\n",
      "Iteration 185, the loss is 5.445786356984176, parameters k is 3.416831692639703 and b is 0.5822298726284647\n",
      "Iteration 186, the loss is 5.445602790553359, parameters k is 3.413833866552747 and b is 0.5770915327075161\n",
      "Iteration 187, the loss is 5.445385529308454, parameters k is 3.4155139060784383 and b is 0.5727437066205596\n",
      "Iteration 188, the loss is 5.445168268063554, parameters k is 3.41719394560413 and b is 0.5683958805336031\n",
      "Iteration 189, the loss is 5.444951006818648, parameters k is 3.4188739851298213 and b is 0.5640480544466466\n",
      "Iteration 190, the loss is 5.44474029631888, parameters k is 3.420554024655513 and b is 0.5597002283596901\n",
      "Iteration 191, the loss is 5.444560704153402, parameters k is 3.4175561985685565 and b is 0.5545618884387415\n",
      "Iteration 192, the loss is 5.444343442908496, parameters k is 3.419236238094248 and b is 0.550214062351785\n",
      "Iteration 193, the loss is 5.444126181663599, parameters k is 3.4209162776199395 and b is 0.5458662362648286\n",
      "Iteration 194, the loss is 5.443908920418694, parameters k is 3.422596317145631 and b is 0.5415184101778721\n",
      "Iteration 195, the loss is 5.44369423565358, parameters k is 3.4242763566713226 and b is 0.5371705840909156\n",
      "Iteration 196, the loss is 5.4435186177534405, parameters k is 3.4212785305843663 and b is 0.532032244169967\n",
      "Iteration 197, the loss is 5.443301356508537, parameters k is 3.4229585701100578 and b is 0.5276844180830105\n",
      "Iteration 198, the loss is 5.4430840952636315, parameters k is 3.4246386096357493 and b is 0.523336591996054\n",
      "Iteration 199, the loss is 5.442866834018735, parameters k is 3.426318649161441 and b is 0.5189887659090975\n"
     ]
    }
   ],
   "source": [
    "#initialized parameters\n",
    "\n",
    "k = random.random() * 200 - 100  # -100 100\n",
    "b = random.random() * 200 - 100  # -100 100\n",
    "\n",
    "learning_rate = 1e-1\n",
    "iteration_num = 200 # 迭代次数\n",
    "losses = []\n",
    "\n",
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
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x17a03983518>]"
      ]
     },
     "execution_count": 30,
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
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.collections.PathCollection at 0x17a039f1a58>"
      ]
     },
     "execution_count": 31,
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
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": []
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
 "nbformat_minor": 2
}
