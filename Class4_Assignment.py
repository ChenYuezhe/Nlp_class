{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Assignment 4"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1. 复习上课内容以及复现课程代码"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "在本部分，你需要复习上课内容和课程代码后，自己复现课程代码。"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. 回答一下理论题目"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 1. What does a neuron compute?"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "答：神经元计算一个线性函数，再计算一个激活函数（sigmoid，relu等）"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "####  2. Why we use non-linear activation funcitons in neural networks?"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "答：如果使用线性函数，每一层输出都是上层输入的线性函数，无论神经网络有多少层，输出都是输入的线性组合。"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 3. What is the 'Logistic Loss' ?"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "$$L = - (ylog(\\hat{y}) + (1 - y)log(1 - \\hat{y}))$$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 4. Assume that you are building a binary classifier for detecting if an image containing cats, which activation functions would you recommen using for the output layer ?\n",
    "\n",
    "A. ReLU    \n",
    "B. Leaky ReLU    \n",
    "C. sigmoid    \n",
    "D. tanh  "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "选C。sigmoid一般只在二分类输出层使用；tanh一般比sigmoid效果号；大部分情况使用RelU，leaky relu也可以。"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 5. Why we don't use zero initialization for all parameters ?"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "答：会导致神经网络对称性问题。w，b全部为0会使得每个神经元得计算都是相同的。"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 6. Can you implement the softmax function using python ? "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "答：sigmoid是softmax的特殊情况。"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3.实践题"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### In this practical part, you will build a simple digits recognizer to check if the digit in the image is larger than 5. This assignmnet will guide you step by step to finish your first small project in this course ."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 1 - Packages  \n",
    "sklearn is a famous package for machine learning.   \n",
    "matplotlib is a common package for vasualization."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 346,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn import datasets\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.model_selection import train_test_split\n",
    "import random"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 2 - Overvie of the dataset  \n",
    "    - a training set has m_train images labeled as 0 if the digit < 5 or 1 if the digit >= 5\n",
    "    - a test set contains m_test images labels as if the digit < 5 or 1 if the digit >= 5\n",
    "    - eah image if of shape (num_px, num_px ). Thus, each image is square(height=num_px and  width = num_px)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 347,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Loading the data \n",
    "digits = datasets.load_digits()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 348,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAV0AAADSCAYAAADpGRMOAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy8li6FKAAARkElEQVR4nO3df2xO9/vH8atFZzYtpU03RRGTiK2KyGbzazqWWFcM67IYm6xmmV/LpCzSIV2QycJkxIbWMI0t2obJrE1UYtkWZISJZUFbdNpRSikVPd8/lvT7XfY51/l+7rqvc6rPx1/TK+c+1333Pi/HfV97vyMcx3EEAGAi0u8GAKA1IXQBwBChCwCGCF0AMEToAoAhQhcADLUNx4N+9913an3lypWuteHDh7vWsrKyXGsxMTHejQVYRkaGa+369euutfnz57vWxo4d26yeguDnn392rWVmZrrW+vfv71rLz89vVk/htmHDBrW+atUq11r37t1da3v37nWttfTrR7tGPvjgA9faF198EY52VGEJ3Vu3bqn1ixcvutauXLniWmtsbAy5p6C7dOmSa+3atWuutfr6+nC0Exi3b992rVVUVLjWYmNjw9GOCS1ARETKy8tda5GR7v94fZCvH+25VVdXG3bijY8XAMAQoQsAhghdADBE6AKAobB8kaZNGYiInDt3zrV29epV15r25ciuXbvUc06ZMkWt+61Tp06utYMHD7rWDhw44FpLT09vVk8Wjh07ptZHjx7tWtO+cS8rKwu1JROLFi1yrXm9lzdu3OhamzVrlmvt6NGjrrXU1FT1nEGXl5fnWhs4cKBdI/8P3OkCgCFCFwAMEboAYIjQBQBDhC4AGCJ0AcBQyCNj2viJNhImInLmzBnXWu/evV1rL7zwQkj9iPg/MuY1GlVaWhrS4wZtHOa/VVhYqNaTk5NdaxMmTHCtLVu2LOSeLGiL9XiNXA4ePNi11qtXL9daSx4L09YfEdFHxrRFoZozWpiUlBTScdzpAoAhQhcADBG6AGCI0AUAQ4QuABgidAHAEKELAIZCntPVlmAcNGiQeqw2i6vR5hODYM2aNa61pUuXqsfW1taGdM5Ro0aFdFxQaDOUIvospHZs0Je11K6Bs2fPqsdqc/DaLK52zXbu3Fk9p9+0OVwRfd52xowZrjXtPaQttyrifU274U4XAAwRugBgiNAFAEOELgAYInQBwBChCwCGwjIypi3B2BxBH3nRxk+0sRWR0Pv3WvIuCLQetTE7Ee+lH914jRgFmddIZU1NjWtNGxnTaiUlJeo5La6voqIi19qCBQvUY6dPnx7SOdeuXetay83NDekxvXCnCwCGCF0AMEToAoAhQhcADBG6AGCI0AUAQyGPjGkjJF4782q0sbAjR4641qZOnRryOVsybZfhoOwUrK3GpI3seNHGybxWiGrJtGtPG/2aNWuWa23VqlXqOVeuXOndWDPFxMSEVBMR2bp1q2vNayduN9pu083BnS4AGCJ0AcAQoQsAhghdADBE6AKAIUIXAAyFPDKmrYSkjXaJiHzzzTch1TRZWVkhHYfw01ZYKy0tVY89fvy4a00b6dE2pnzzzTfVc/q9qeWiRYvUeqibTxYXF7vWgjByqW2y6rWanjYWpj2utjpZuMYOudMFAEOELgAYInQBwBChCwCGCF0AMEToAoAhQhcADIVlTtdrmThtpnbIkCGuteYsGek3r5k/bTZU2yVVm3P12oHYirbEpNeye1pdWzJSe82SkpLUc/o9p+u1825mZmZIj6vN4m7cuDGkxwwK7fqqra11rflxjXCnCwCGCF0AMEToAoAhQhcADBG6AGAo5OkFzaOPPqrWu3Xr5lqLi4u73+20CPHx8a61nj17uta6du0ajnYCIyoqyrUW6msWGxvbrJ7CLTo6Wq1r14+mS5cuIR3XErRt6x5l2nuhffv24WhHFeE4jmN+VgBopfh4AQAMEboAYIjQBQBDhC4AGCJ0AcAQoQsAhghdADBE6AKAIUIXAAwRugBgKDChW1paKmlpaTJu3DiZO3eu1NXV+d1SIDiOI1lZWbJ582a/WwmEoqIiefnllyU9PV0yMjLkxIkTfrcUCNu3b5fx48fLSy+9JLNnz5YrV6743VJglJSUSEpKit9tNAlE6NbU1MjixYtl3bp1sn//funevbusXr3a77Z8d+bMGZk+fbrs37/f71YC4ezZs/LJJ5/Ipk2bpKioSGbPni1z5szxuy3fnTx5UrZs2SL5+fmyd+9eSUpKkrVr1/rdViCUlZV5bh9mLRChe+jQIXnyySeb9q567bXXZM+ePdLa1+LZsWOHTJkyRV588UW/WwmEqKgoycnJaVpdbMCAAXL58mVpaGjwuTN/DRgwQPbv3y8dO3aUO3fuSFVVleeefK1BfX29LFy4UBYtWuR3K/8QiNC9dOmSJCQkNP05ISFB6urq5ObNmz525b/s7GxJS0vzu43ASExMlFGjRonI3x+7rFixQp5//nl1+cfWol27dlJSUiIjRoyQw4cPy6RJk/xuyXfZ2dny6quvSr9+/fxu5R8CEbqNjY0SERHxr59HRgaiPQTMrVu3ZN68eVJRUSE5OTl+txMYqamp8ssvv8icOXNk5syZ0tjY6HdLvtmxY4e0bdtWJk+e7Hcr/xKIVHvsscekurq66c9VVVUSExMjHTp08LErBFFlZaVkZGRImzZt5KuvvvJc8Ls1KC8vlyNHjjT9+ZVXXpHKykp16/EHXUFBgZw4cULS09MlMzNTbt++Lenp6VJVVeV3a+HZOeK/9dxzz8mqVaukrKxMkpKSJD8/X8aMGeN3WwiYuro6mTZtmkycOFHee+89v9sJjL/++kvef/99KSwslNjYWNmzZ4/07dtXOnfu7Hdrvvn222+b/vvChQuSlpYmRUVFPnb0vwIRul26dJEVK1bI3Llz5e7du9KjR4/AfeMI/+3YsUMqKyuluLhYiouLm36el5fXqgNmyJAh8s4778gbb7whbdq0kfj4ePn888/9bgsu2K4HAAwF4jNdAGgtCF0AMEToAoAhQhcADBG6AGCI0AUAQ4QuABgidAHAEKELAIYIXQAwROgCgCFCFwAMEboAYIjQBQBDhC4AGCJ0AcAQoQsAhghdADBE6AKAIUIXAAwRugBgiNAFAEOELgAYInQBwBChCwCGCF0AMEToAoAhQhcADBG6AGCI0AUAQ4QuABgidAHAEKELAIYIXQAwROgCgCFCFwAMEboAYIjQBQBDhC4AGCJ0AcAQoQsAhghdADDUNhwPmpGRodYTExNda6tXr77f7bQI2mt2/fp119q+ffvC0Y6ZLVu2qHXtuf/www+utVOnTrnWoqOj1XMeOnToXz+LiIiQjh07qsfdL8uXL1fr2vOePHmya+2tt95yrXm9Jn7LzMxU69r7JD8//3630yxhCd1Lly6p9fbt24fjtC2a9ppdu3bNsBNb2sUioj/3yspK11p5eblrLSYmRj1nY2Pjv34WGWn3j8Kamhq1fuHCBdea9nr9p+fVUlRXV6v1lnSN8PECABgidAHAEKELAIYIXQAwFOE4jnO/HzQpKUmta19yaHr27OlaKysrC+kxrRQVFan1CRMmuNY++ugj19rSpUtDbSkQ1qxZE/KxAwcODOlxvb50KS0tDbWl+2LUqFFqPdT3unZd+v2cRfTn1atXr7CcMzk52bV27NixsJyTO10AMEToAoAhQhcADBG6AGCI0AUAQ4QuABgKy9oLnTp1Uuuh/n/x2iiN1xiQV0/hpo19edHGyVq6+fPnh3ysNi6njR8FYTxKo43CieijX3l5ea417Rrwek28xtjuh+asnzBy5EjXWtBG5bjTBQBDhC4AGCJ0AcAQoQsAhghdADBE6AKAIUIXAAyFZU7Xa2nH48ePu9Zqa2tda9r8ot9zuF68ZhC1Jea85jaDTpuFbM6cZKjLQhYWFqr1GTNmhPS494vX+VNSUlxr2nyydo14XbMWmtOD9jvV5tz92FuNO10AMEToAoAhQhcADBG6AGCI0AUAQ4QuABgKy8iY10iONiak7cC5YMGCUFtq1hKC94PXaIo2LqONRmnjMEEYAxLR+/DacTXUkTLtPWixTGFzNGeM6eDBg661c+fOudaC8F7RRtq0kUoRkc6dO7vW5s2b51rT3n9euy6H+ppxpwsAhghdADBE6AKAIUIXAAwRugBgiNAFAENhGRnzEo6RHa/xDr95jZdooz7aCJE2Rvfrr7+q57RavUx77l7jhRERESEdG/SxMG1UafTo0eqx2s7S2nWgjRd6/R78HinzGi3U6qG+z73GTL1eMzfc6QKAIUIXAAwRugBgiNAFAEOELgAYInQBwFBYRsaKiorUekxMjGtt6dKlIZ1TG4cJAq/NBrXRL21cRxsR8hppCcKGl15jOdp7ZeTIkfe7HTPa71R7ziL6a6a9H7QNLfPy8tRzhnpdWtHey9rrpT3vUEfCvHCnCwCGCF0AMEToAoAhQhcADBG6AGCI0AUAQ4QuABgKy5zugQMH1PratWtDetzp06e71oK+lJ/XnK42X6nNEmrPO+izyyLeu/1u3brVtabtHht0Wu9e72Vt51ttxjc9Pd215vdu2V68+tOWdtSWRtXef+GaY+dOFwAMEboAYIjQBQBDhC4AGCJ0AcBQWKYXYmNj1XrPnj1DetyuXbuGdFwQREbqf79pr1mPHj1ca/Hx8a61qKgo78Z89vjjj6v1hx9+2KiT4NB+pyL69RMdHR3S43q9P/3mlSkJCQmutQ4dOrjW/LhGIhzHcczPCgCtVLD/egOABwyhCwCGCF0AMEToAoAhQhcADBG6AGCI0AUAQ4QuABgidAHAEKELAIbCsvZCKFauXCnff/9908r3vXr1kjVr1vjclb9+//13ycnJkRs3bkhkZKQsX75cBgwY4HdbviosLJTc3NymP9+4cUOqqqrk4MGDLXptjuYqLi6Wzz77TCIjIyUmJkZycnLUNTtag23btsn27dulffv20qdPH8nOzg7GbiNOQEydOtU5evSo320Exq1bt5xnn33WKS0tdRzHcYqLi51x48b53FWwNDQ0OFOnTnV27tzpdyu+qq+vd5KTk52ysjLHcRwnNzfXefvtt33uyl8//fSTM3z4cOfPP/90HMdxCgoKnDlz5vjc1d8Ccafb0NAgp06dkk2bNsn58+clKSlJFi9e7LkC1YPsxx9/lO7du8vIkSNFRGTMmDGSmJjoc1fB8uWXX0psbKxkZGT43Yqv7t27J47jyI0bN0RE5ObNm/LQQw/53JW/fvvtNxk2bFjT6mNjx46VJUuWSENDg++r7wUidKuqquTpp5+W+fPnS9++fWXz5s3y7rvvSkFBgURERPjdni/OnTsncXFx8uGHH8rp06clOjpaFi5c6HdbgVFTUyO5ubmye/duv1vx3SOPPCLLli2TjIwM6dSpkzQ2NsrOnTv9bstXycnJsm3bNrl48aJ069ZNdu/eLXfv3pVr1655Lp0Zdn7fav8njY2NTkpKilNRUeF3K75Zv36989RTTznHjh1zHOfvjxeGDRvm3Llzx+fOgmHDhg1OVlaW320EwunTp53U1FSnvLzccRzH2bp1q5OWluY0Njb63Jm/du3a5UyYMMGZOHGis337dmfo0KFOTU2N3205gZheOH36tBQWFv7jZ47jSLt27XzqyH/x8fHSp08fSU5OFhGR1NRUuXfvnpw/f97nzoJh3759MmnSJL/bCIRDhw7JoEGDmr44e/311+WPP/6Qq1ev+tyZf+rq6mTo0KFSUFAgu3fvltTUVBGRQHyRFojQjYyMlI8//rgpUL7++mvp16+fuhr8g27EiBFy4cIFOXnypIiIHD58WCIiIvhcV0Rqa2uloqJCUlJS/G4lEPr37y+HDx+Wy5cvi4hISUmJJCYmeu628CCrrq6WadOmSV1dnYiIbNiwQcaPHx+IjysD8ZnuE088IUuWLJHZs2fLvXv3JCEhQT799FO/2/JVXFycfP7557Js2TKpr6+XqKgoWbduXav/gkREpLy8XOLi4lr1v4T+r2eeeUZmzpwp06ZNk3bt2klMTIysX7/e77Z81bt3b8nMzJQpU6ZIY2OjDB48WLKzs/1uS0TYrgcATAXi4wUAaC0IXQAwROgCgCFCFwAMEboAYIjQBQBDhC4AGPofp/oNjoBPOjcAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 10 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Vilizating the data\n",
    "for i in range(1,11):\n",
    "    plt.subplot(2,5,i)\n",
    "    plt.imshow(digits.data[i-1].reshape([8,8]),cmap=plt.cm.gray_r)\n",
    "    plt.text(3,10,str(digits.target[i-1]))\n",
    "    plt.xticks([])\n",
    "    plt.yticks([])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 349,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split the data into training set and test set \n",
    "X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 350,
   "metadata": {},
   "outputs": [],
   "source": [
    "# reformulate the label. \n",
    "# If the digit is smaller than 5, the label is 0.\n",
    "# If the digit is larger than 5, the label is 1.\n",
    "\n",
    "y_train[y_train < 5 ] = 0\n",
    "y_train[y_train >= 5] = 1\n",
    "y_test[y_test < 5] = 0\n",
    "y_test[y_test >= 5] = 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 351,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(1347, 64)\n",
      "(450, 64)\n",
      "(1347,)\n",
      "(450,)\n",
      "450\n"
     ]
    }
   ],
   "source": [
    "print(X_train.shape)\n",
    "print(X_test.shape)\n",
    "print(y_train.shape)\n",
    "print(y_test.shape)\n",
    "print(len(y_test))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 3- Architecture of the neural network"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![title](./networks.png)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 352,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "'pwd' 不是内部或外部命令，也不是可运行的程序\n",
      "或批处理文件。\n"
     ]
    }
   ],
   "source": [
    "!pwd"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Mathematical expression of the algorithm:"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "For one example $x^{(i)}$:   \n",
    " $$ z^{(i)} = w^T * x^{(i)} +b $$   \n",
    " $$ y^{(i)} = a^{(i)} = sigmoid(z^{(i)})$$   \n",
    " $$L(a^{(i)},y^{(i)}) = -y^{(i)} log(a^{(i)})-(1-y^{(i)})log(1-a^{(i)})$$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The total cost over all training examples:\n",
    "$$ J = \\frac{1}{m}\\sum_{i=1}^{m}L(a^{(i)},y^{(i)}) $$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 4 - Building the algorithm"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### 4.1- Activation function    \n",
    "###### Exercise:\n",
    "Finish the sigmoid funciton "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 353,
   "metadata": {},
   "outputs": [],
   "source": [
    "def sigmoid(z):\n",
    "    '''\n",
    "    Compute the sigmoid of z\n",
    "    Arguments: z -- a scalar or numpy array of any size.\n",
    "    \n",
    "    Return:\n",
    "    s -- sigmoid(z)\n",
    "    '''\n",
    "    s = 1./(1 + np.exp(-1 * z))\n",
    "    \n",
    "    return s"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 354,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "sigmoid([0,2]) = [0.5        0.88079708]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "# Test your code \n",
    "# The result should be [0.5 0.88079708]\n",
    "print(\"sigmoid([0,2]) = \" + str(sigmoid(np.array([0,2]))))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### 4.1-Initializaing parameters\n",
    "###### Exercise:\n",
    "Finishe the initialize_parameters function below"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 355,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Random innitialize the parameters\n",
    "\n",
    "def initialize_parameters(dim):\n",
    "    '''\n",
    "    Argument: dim -- size of the w vector\n",
    "    \n",
    "    Returns:\n",
    "    w -- initialized vector of shape (dim,1)\n",
    "    b -- initializaed scalar\n",
    "    '''\n",
    "    \n",
    "    w = np.random.randn(dim, 1)\n",
    "    b = random.random()\n",
    "    \n",
    "    assert(w.shape == (dim,1))\n",
    "    assert(isinstance(b,float) or isinstance(b,int))\n",
    "    \n",
    "    return w,b"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 4.3-Forward and backward propagation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "###### Some mathematical expressions\n",
    "Forward Propagation:   \n",
    ". X    \n",
    ". A = $\\sigma(w^T*X+b) = (a^{(1)},a^{(2)},...,a^{(m)}$   \n",
    ". J = $-\\frac{1}{m} \\sum_{i=1}^{m}y^{(i)}log(a^{(i)}+(1-y^{(i)})log(1-a^{(i)})$       "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Some derivative: \n",
    "$$\\frac{\\partial{J}}{\\partial{w}} = \\frac{1}{m}X*(A-Y)^T$$   \n",
    "$$\\frac{\\partial{J}}{\\partial{b}} = \\frac{1}{m}\\sum_{i=1}^m(a^{(i)}-y^{(i)}) $$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "###### Exercise:\n",
    "Finish the function below:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 356,
   "metadata": {},
   "outputs": [],
   "source": [
    "def propagate(w,b,X,Y):\n",
    "    '''\n",
    "    Implement the cost function and its gradient for the propagation\n",
    "    \n",
    "    Arguments:\n",
    "    w - weights\n",
    "    b - bias\n",
    "    X - data\n",
    "    Y - ground truth\n",
    "    '''\n",
    "    m = X.shape[1]\n",
    "    A = sigmoid(np.dot(w.T, X.T) + b)\n",
    "    \n",
    "    cost = -1/m * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))\n",
    "    \n",
    "    dw = (1/m * np.dot(A - Y, X)).T\n",
    "    db = 1/m * np.sum(A-Y, axis=0, keepdims=False)\n",
    "    \n",
    "    assert (dw.shape == w.shape)\n",
    "    assert (db.dtype == float)\n",
    "    cost = np.squeeze(cost)\n",
    "    assert (cost.shape == ())\n",
    "    \n",
    "    grads = {'dw': dw,\n",
    "             'db': db}\n",
    "    return grads, cost"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### 4.4 -Optimization"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "###### Exercise:\n",
    "Minimizing the cost function using gradient descent.   \n",
    "$$\\theta = \\theta - \\alpha*d\\theta$$ where $\\alpha$ is the learning rate."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 357,
   "metadata": {},
   "outputs": [],
   "source": [
    "def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):\n",
    "    '''\n",
    "    This function optimize w and b by running a gradient descen algorithm\n",
    "    \n",
    "    Arguments:\n",
    "    w - weights\n",
    "    b - bias\n",
    "    X - data\n",
    "    Y - ground truth\n",
    "    num_iterations -- number of iterations of the optimization loop\n",
    "    learning_rate -- learning rate of the gradient descent update rule\n",
    "    print_cost -- True to print the loss every 100 steps\n",
    "    \n",
    "    Returns:\n",
    "    params - dictionary containing the weights w and bias b\n",
    "    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function\n",
    "    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.\n",
    "    \n",
    "    '''\n",
    "    \n",
    "    costs = []\n",
    "    \n",
    "    for i in range(num_iterations):\n",
    "        \n",
    "        grads, cost = propagate(w,b,X,Y)\n",
    "        \n",
    "        dw = grads['dw']\n",
    "        db = grads['db']\n",
    "        \n",
    "        \n",
    "        w +=  -1 * learning_rate * dw\n",
    "        b +=  -1 * learning_rate * db\n",
    "        \n",
    "        if i % 100 == 0:\n",
    "            costs.append(cost)\n",
    "        if print_cost and i % 100 == 0:\n",
    "            print (\"Cost after iteration %i: %f\" %(i, cost))\n",
    "    \n",
    "    params = {\"w\":w,\n",
    "              \"b\":b}\n",
    "    \n",
    "    grads = {\"dw\":dw,\n",
    "             \"db\":db}\n",
    "    \n",
    "    return params, grads, costs"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "###### Exercise\n",
    "The previous function will output the learned w and b. We are able to use w and b to predict the labels for a dataset X. Implement the predict() function.    \n",
    "Two steps to finish this task:   \n",
    "1. Calculate $\\hat{Y} = A = \\sigma(w^T*X+b)$   \n",
    "2. Convert the entries of a into 0 (if activation <= 0.5) or 1 (if activation > 0.5), stores the predictions in a vector Y_prediction. If you wish, you can use an if/else statement in a for loop (though there is also a way to vectorize this)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 358,
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict(w, b, X):\n",
    "    '''\n",
    "    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)\n",
    "    \n",
    "    Arguments:\n",
    "    w -- weights\n",
    "    b -- bias \n",
    "    X -- data \n",
    "    \n",
    "    Returns:\n",
    "    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X\n",
    "    '''\n",
    "    m = X.shape[0]\n",
    "    Y_prediction = np.zeros((1, m))\n",
    "    w = w.reshape(X.shape[1], 1)\n",
    "\n",
    "    A = sigmoid(np.dot(w.T, X.T) + b)\n",
    "\n",
    "    for i in range(A.shape[1]):\n",
    "        if A[0, i] >= 0.5:\n",
    "            Y_prediction[0, i] = 1\n",
    "\n",
    "        elif A[0, i] < 0.5:\n",
    "            Y_prediction[0, i] = 0\n",
    "\n",
    "    assert (Y_prediction.shape == (1, m))\n",
    "\n",
    "    return Y_prediction"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### 5- Merge all functions into a model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Congratulations !! You have finished all the necessary components for constructing a model. Now, Let's take the challenge to merge all the implemented function into one model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 359,
   "metadata": {},
   "outputs": [],
   "source": [
    "def model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate,print_cost):\n",
    "    \"\"\"\n",
    "    Build the logistic regression model by calling all the functions you have implemented.\n",
    "    Arguments:\n",
    "    X_train - training set\n",
    "    Y_train - training label\n",
    "    X_test - test set\n",
    "    Y_test - test label\n",
    "    num_iteration - hyperparameter representing the number of iterations to optimize the parameters\n",
    "    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()\n",
    "    print_cost -- Set to true to print the cost every 100 iterations\n",
    "    \n",
    "    Returns:\n",
    "    d - dictionary should contain following information w,b,training_accuracy, test_accuracy,cost\n",
    "    eg: d = {\"w\":w,\n",
    "             \"b\":b,\n",
    "             \"training_accuracy\": traing_accuracy,\n",
    "             \"test_accuracy\":test_accuracy,\n",
    "             \"cost\":cost}\n",
    "    \"\"\"\n",
    "    w1, b1 = initialize_parameters(X_train.shape[1])\n",
    "    w2, b2 = initialize_parameters(X_test.shape[1])\n",
    "    params1, grads1, costs1 = optimize(w1, b1, X_train, Y_train, num_iterations, learning_rate, print_cost)\n",
    "    params2, grads2, costs2 = optimize(w1, b1, X_test, Y_test, num_iterations, learning_rate, print_cost)\n",
    "\n",
    "    w1 = params1[\"w\"]\n",
    "    b1 = params1[\"b\"]\n",
    "    \n",
    "    Y_prediction_train = predict(w1, b1, X_train)\n",
    "    Y_prediction_test = predict(w2, b2, X_test)\n",
    "   \n",
    "\n",
    "    print(\"train accuracy: {} %\".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))\n",
    "    print(\"test accuracy: {} %\".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))\n",
    "    train_accuracy = 100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100\n",
    "    test_accuracy = 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100\n",
    "    \n",
    "    d = {\"costs1\": costs1,\n",
    "         \"costs2\": costs2,\n",
    "         \"Y_prediction_test\": Y_prediction_test,\n",
    "         \"Y_prediction_train\": Y_prediction_train,\n",
    "         \"w\": w1,\n",
    "         \"b\": b1,\n",
    "         \"learning_rate\": learning_rate,\n",
    "         \"num_iterations\": num_iterations,\n",
    "        \"train_accuracy\":train_accuracy,\n",
    "        \"test_accuracy\":test_accuracy}\n",
    "\n",
    "\n",
    "\n",
    "    return d"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 360,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Cost after iteration 0: nan\n",
      "Cost after iteration 100: nan"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "f:\\app\\python3.7\\install\\lib\\site-packages\\ipykernel_launcher.py:14: RuntimeWarning: divide by zero encountered in log\n",
      "  \n",
      "f:\\app\\python3.7\\install\\lib\\site-packages\\ipykernel_launcher.py:14: RuntimeWarning: invalid value encountered in multiply\n",
      "  \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Cost after iteration 200: nan\n",
      "Cost after iteration 300: nan\n",
      "Cost after iteration 400: nan\n",
      "Cost after iteration 500: 12.441340\n",
      "Cost after iteration 600: 10.880995\n",
      "Cost after iteration 700: 9.712934\n",
      "Cost after iteration 800: 8.794902\n",
      "Cost after iteration 900: 15.098275\n",
      "Cost after iteration 1000: 8.708537\n",
      "Cost after iteration 1100: 11.222968\n",
      "Cost after iteration 1200: 7.463667\n",
      "Cost after iteration 1300: 26.728535\n",
      "Cost after iteration 1400: 7.200988\n",
      "Cost after iteration 1500: 16.066212\n",
      "Cost after iteration 1600: 7.112504\n",
      "Cost after iteration 1700: 12.536454\n",
      "Cost after iteration 1800: 7.022312\n",
      "Cost after iteration 1900: 10.984357\n",
      "Cost after iteration 2000: 6.969202\n",
      "Cost after iteration 2100: 9.463432\n",
      "Cost after iteration 2200: 6.958475\n",
      "Cost after iteration 2300: 8.186186\n",
      "Cost after iteration 2400: 7.000409\n",
      "Cost after iteration 2500: 7.297720\n",
      "Cost after iteration 2600: 7.152729\n",
      "Cost after iteration 2700: 6.796564\n",
      "Cost after iteration 2800: 7.736924\n",
      "Cost after iteration 2900: 6.572632\n",
      "Cost after iteration 3000: 10.925665\n",
      "Cost after iteration 3100: 6.501229\n",
      "Cost after iteration 3200: 23.169133\n",
      "Cost after iteration 3300: 6.505529\n",
      "Cost after iteration 3400: 31.080415\n",
      "Cost after iteration 3500: 6.557014\n",
      "Cost after iteration 3600: 13.342622\n",
      "Cost after iteration 3700: 6.662330\n",
      "Cost after iteration 3800: 7.275047\n",
      "Cost after iteration 3900: 6.940677\n",
      "Cost after iteration 4000: 6.434820\n",
      "Cost after iteration 4100: 10.064559\n",
      "Cost after iteration 4200: 6.380735\n",
      "Cost after iteration 4300: 31.860517\n",
      "Cost after iteration 4400: 6.453875\n",
      "Cost after iteration 4500: 14.341708\n",
      "Cost after iteration 4600: 6.601289\n",
      "Cost after iteration 4700: 6.776635\n",
      "Cost after iteration 4800: 7.123831\n",
      "Cost after iteration 4900: 6.314672\n",
      "Cost after iteration 5000: 19.654208\n",
      "Cost after iteration 5100: 6.366753\n",
      "Cost after iteration 5200: 22.269650\n",
      "Cost after iteration 5300: 6.512762\n",
      "Cost after iteration 5400: 6.976610\n",
      "Cost after iteration 5500: 6.951846\n",
      "Cost after iteration 5600: 6.272569\n",
      "Cost after iteration 5700: 20.492951\n",
      "Cost after iteration 5800: 6.339562\n",
      "Cost after iteration 5900: 17.497514\n",
      "Cost after iteration 6000: 6.520112\n",
      "Cost after iteration 6100: 6.513107\n",
      "Cost after iteration 6200: 7.529546\n",
      "Cost after iteration 6300: 6.247523\n",
      "Cost after iteration 6400: 33.650888\n",
      "Cost after iteration 6500: 6.375431\n",
      "Cost after iteration 6600: 8.265304\n",
      "Cost after iteration 6700: 6.694288\n",
      "Cost after iteration 6800: 6.220650\n",
      "Cost after iteration 6900: 19.026606\n",
      "Cost after iteration 7000: 6.294382\n",
      "Cost after iteration 7100: 13.037717\n",
      "Cost after iteration 7200: 6.536920\n",
      "Cost after iteration 7300: 6.242475\n",
      "Cost after iteration 7400: 12.052040\n",
      "Cost after iteration 7500: 6.249665\n",
      "Cost after iteration 7600: 17.377635\n",
      "Cost after iteration 7700: 6.481356\n",
      "Cost after iteration 7800: 6.245304\n",
      "Cost after iteration 7900: 11.148119\n",
      "Cost after iteration 8000: 6.230903\n",
      "Cost after iteration 8100: 16.130771\n",
      "Cost after iteration 8200: 6.483216\n",
      "Cost after iteration 8300: 6.191829\n",
      "Cost after iteration 8400: 14.504316\n",
      "Cost after iteration 8500: 6.236342\n",
      "Cost after iteration 8600: 10.888495\n",
      "Cost after iteration 8700: 6.553910\n",
      "Cost after iteration 8800: 6.139479\n",
      "Cost after iteration 8900: 26.227483\n",
      "Cost after iteration 9000: 6.271139\n",
      "Cost after iteration 9100: 7.296060\n",
      "Cost after iteration 9200: 6.843363\n",
      "Cost after iteration 9300: 6.134764\n",
      "Cost after iteration 9400: 33.452397\n",
      "Cost after iteration 9500: 6.349292\n",
      "Cost after iteration 9600: 6.240776\n",
      "Cost after iteration 9700: 10.171734\n",
      "Cost after iteration 9800: 6.179466\n",
      "Cost after iteration 9900: 11.132474\n",
      "Cost after iteration 10000: 6.529943\n",
      "Cost after iteration 10100: 6.093454\n",
      "Cost after iteration 10200: 33.543345\n",
      "Cost after iteration 10300: 6.277007\n",
      "Cost after iteration 10400: 6.366697\n",
      "Cost after iteration 10500: 8.332495\n",
      "Cost after iteration 10600: 6.141253\n",
      "Cost after iteration 10700: 12.365945\n",
      "Cost after iteration 10800: 6.490725\n",
      "Cost after iteration 10900: 6.067800\n",
      "Cost after iteration 11000: 34.826603\n",
      "Cost after iteration 11100: 6.268918\n",
      "Cost after iteration 11200: 6.216584\n",
      "Cost after iteration 11300: 10.160074\n",
      "Cost after iteration 11400: 6.141697\n",
      "Cost after iteration 11500: 8.725271\n",
      "Cost after iteration 11600: 6.617202\n",
      "Cost after iteration 11700: 6.058458\n",
      "Cost after iteration 11800: 29.823229\n",
      "Cost after iteration 11900: 6.325075\n",
      "Cost after iteration 12000: 6.047150\n",
      "Cost after iteration 12100: 23.392828\n",
      "Cost after iteration 12200: 6.186713\n",
      "Cost after iteration 12300: 6.426820\n",
      "Cost after iteration 12400: 7.959032\n",
      "Cost after iteration 12500: 6.090564\n",
      "Cost after iteration 12600: 9.918724\n",
      "Cost after iteration 12700: 6.532123\n",
      "Cost after iteration 12800: 6.024001\n",
      "Cost after iteration 12900: 29.430294\n",
      "Cost after iteration 13000: 6.305186\n",
      "Cost after iteration 13100: 6.004441\n",
      "Cost after iteration 13200: 28.575440\n",
      "Cost after iteration 13300: 6.182998\n",
      "Cost after iteration 13400: 6.158499\n",
      "Cost after iteration 13500: 10.622180\n",
      "Cost after iteration 13600: 6.094453\n",
      "Cost after iteration 13700: 7.220984\n",
      "Cost after iteration 13800: 6.877788\n",
      "Cost after iteration 13900: 6.026701\n",
      "Cost after iteration 14000: 13.103244\n",
      "Cost after iteration 14100: 6.420119\n",
      "Cost after iteration 14200: 5.978341\n",
      "Cost after iteration 14300: 31.585441\n",
      "Cost after iteration 14400: 6.264186\n",
      "Cost after iteration 14500: 5.960986\n",
      "Cost after iteration 14600: 31.478560\n",
      "Cost after iteration 14700: 6.165019\n",
      "Cost after iteration 14800: 6.026139\n",
      "Cost after iteration 14900: 15.261765\n",
      "Cost after iteration 0: 2.834290\n",
      "Cost after iteration 100: 2.183847\n",
      "Cost after iteration 200: 1.883492\n",
      "Cost after iteration 300: 1.708309\n",
      "Cost after iteration 400: 1.608601\n",
      "Cost after iteration 500: 1.549926\n",
      "Cost after iteration 600: 1.512406\n",
      "Cost after iteration 700: 1.486039\n",
      "Cost after iteration 800: 1.466038\n",
      "Cost after iteration 900: 1.450062\n",
      "Cost after iteration 1000: 1.436873\n",
      "Cost after iteration 1100: 1.425751\n",
      "Cost after iteration 1200: 1.416233\n",
      "Cost after iteration 1300: 1.407996\n",
      "Cost after iteration 1400: 1.400801\n",
      "Cost after iteration 1500: 1.394467\n",
      "Cost after iteration 1600: 1.388849\n",
      "Cost after iteration 1700: 1.383833\n",
      "Cost after iteration 1800: 1.379325\n",
      "Cost after iteration 1900: 1.375247\n",
      "Cost after iteration 2000: 1.371537\n",
      "Cost after iteration 2100: 1.368142\n",
      "Cost after iteration 2200: 1.365018\n",
      "Cost after iteration 2300: 1.362128\n",
      "Cost after iteration 2400: 1.359442\n",
      "Cost after iteration 2500: 1.356933\n",
      "Cost after iteration 2600: 1.354579\n",
      "Cost after iteration 2700: 1.352362\n",
      "Cost after iteration 2800: 1.350265\n",
      "Cost after iteration 2900: 1.348275\n",
      "Cost after iteration 3000: 1.346380\n",
      "Cost after iteration 3100: 1.344570\n",
      "Cost after iteration 3200: 1.342836\n",
      "Cost after iteration 3300: 1.341170\n",
      "Cost after iteration 3400: 1.339567\n",
      "Cost after iteration 3500: 1.338019\n",
      "Cost after iteration 3600: 1.336523\n",
      "Cost after iteration 3700: 1.335072\n",
      "Cost after iteration 3800: 1.333664\n",
      "Cost after iteration 3900: 1.332295\n",
      "Cost after iteration 4000: 1.330961\n",
      "Cost after iteration 4100: 1.329661\n",
      "Cost after iteration 4200: 1.328390\n",
      "Cost after iteration 4300: 1.327148\n",
      "Cost after iteration 4400: 1.325932\n",
      "Cost after iteration 4500: 1.324739\n",
      "Cost after iteration 4600: 1.323570\n",
      "Cost after iteration 4700: 1.322421\n",
      "Cost after iteration 4800: 1.321292\n",
      "Cost after iteration 4900: 1.320181\n",
      "Cost after iteration 5000: 1.319087\n",
      "Cost after iteration 5100: 1.318010\n",
      "Cost after iteration 5200: 1.316947\n",
      "Cost after iteration 5300: 1.315899\n",
      "Cost after iteration 5400: 1.314865\n",
      "Cost after iteration 5500: 1.313843\n",
      "Cost after iteration 5600: 1.312834\n",
      "Cost after iteration 5700: 1.311836\n",
      "Cost after iteration 5800: 1.310848\n",
      "Cost after iteration 5900: 1.309871\n",
      "Cost after iteration 6000: 1.308904\n",
      "Cost after iteration 6100: 1.307947\n",
      "Cost after iteration 6200: 1.306998\n",
      "Cost after iteration 6300: 1.306058\n",
      "Cost after iteration 6400: 1.305126\n",
      "Cost after iteration 6500: 1.304201\n",
      "Cost after iteration 6600: 1.303284\n",
      "Cost after iteration 6700: 1.302375\n",
      "Cost after iteration 6800: 1.301472\n",
      "Cost after iteration 6900: 1.300575\n",
      "Cost after iteration 7000: 1.299686\n",
      "Cost after iteration 7100: 1.298802\n",
      "Cost after iteration 7200: 1.297924\n",
      "Cost after iteration 7300: 1.297051\n",
      "Cost after iteration 7400: 1.296184\n",
      "Cost after iteration 7500: 1.295323\n",
      "Cost after iteration 7600: 1.294466\n",
      "Cost after iteration 7700: 1.293614\n",
      "Cost after iteration 7800: 1.292767\n",
      "Cost after iteration 7900: 1.291925\n",
      "Cost after iteration 8000: 1.291087\n",
      "Cost after iteration 8100: 1.290253\n",
      "Cost after iteration 8200: 1.289423\n",
      "Cost after iteration 8300: 1.288598\n",
      "Cost after iteration 8400: 1.287776\n",
      "Cost after iteration 8500: 1.286958\n",
      "Cost after iteration 8600: 1.286144\n",
      "Cost after iteration 8700: 1.285333\n",
      "Cost after iteration 8800: 1.284526\n",
      "Cost after iteration 8900: 1.283722\n",
      "Cost after iteration 9000: 1.282922\n",
      "Cost after iteration 9100: 1.282125\n",
      "Cost after iteration 9200: 1.281330\n",
      "Cost after iteration 9300: 1.280539\n",
      "Cost after iteration 9400: 1.279751\n",
      "Cost after iteration 9500: 1.278966\n",
      "Cost after iteration 9600: 1.278183\n",
      "Cost after iteration 9700: 1.277404\n",
      "Cost after iteration 9800: 1.276627\n",
      "Cost after iteration 9900: 1.275852\n",
      "Cost after iteration 10000: 1.275080\n",
      "Cost after iteration 10100: 1.274311\n",
      "Cost after iteration 10200: 1.273544\n",
      "Cost after iteration 10300: 1.272780\n",
      "Cost after iteration 10400: 1.272017\n",
      "Cost after iteration 10500: 1.271258\n",
      "Cost after iteration 10600: 1.270500\n",
      "Cost after iteration 10700: 1.269744\n",
      "Cost after iteration 10800: 1.268991\n",
      "Cost after iteration 10900: 1.268240\n",
      "Cost after iteration 11000: 1.267491\n",
      "Cost after iteration 11100: 1.266744\n",
      "Cost after iteration 11200: 1.265999\n",
      "Cost after iteration 11300: 1.265255\n",
      "Cost after iteration 11400: 1.264514\n",
      "Cost after iteration 11500: 1.263775\n",
      "Cost after iteration 11600: 1.263037\n",
      "Cost after iteration 11700: 1.262302\n",
      "Cost after iteration 11800: 1.261568\n",
      "Cost after iteration 11900: 1.260835\n",
      "Cost after iteration 12000: 1.260105\n",
      "Cost after iteration 12100: 1.259376\n",
      "Cost after iteration 12200: 1.258649\n",
      "Cost after iteration 12300: 1.257924\n",
      "Cost after iteration 12400: 1.257200\n",
      "Cost after iteration 12500: 1.256477\n",
      "Cost after iteration 12600: 1.255757\n",
      "Cost after iteration 12700: 1.255038\n",
      "Cost after iteration 12800: 1.254320\n",
      "Cost after iteration 12900: 1.253604\n",
      "Cost after iteration 13000: 1.252889\n",
      "Cost after iteration 13100: 1.252176\n",
      "Cost after iteration 13200: 1.251464\n",
      "Cost after iteration 13300: 1.250753\n",
      "Cost after iteration 13400: 1.250044\n",
      "Cost after iteration 13500: 1.249336\n",
      "Cost after iteration 13600: 1.248630\n",
      "Cost after iteration 13700: 1.247925\n",
      "Cost after iteration 13800: 1.247221\n",
      "Cost after iteration 13900: 1.246519\n",
      "Cost after iteration 14000: 1.245818\n",
      "Cost after iteration 14100: 1.245118\n",
      "Cost after iteration 14200: 1.244419\n",
      "Cost after iteration 14300: 1.243722\n",
      "Cost after iteration 14400: 1.243026\n",
      "Cost after iteration 14500: 1.242331\n",
      "Cost after iteration 14600: 1.241637\n",
      "Cost after iteration 14700: 1.240944\n",
      "Cost after iteration 14800: 1.240253\n",
      "Cost after iteration 14900: 1.239562\n",
      "train accuracy: 87.15664439495174 %\n",
      "test accuracy: 47.77777777777777 %\n"
     ]
    }
   ],
   "source": [
    "d = model(X_train, y_train, X_test, y_test, 15000, 0.001 ,True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 361,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[nan, nan, nan, nan, nan, 12.44134020984273, 10.880994916060606, 9.712933941038477, 8.794901960180542, 15.098275066481845, 8.708537429244313, 11.222968438481661, 7.463667260443196, 26.728535144848067, 7.200987525342982, 16.066212186240172, 7.112504090021973, 12.536454186155915, 7.022312080021829, 10.98435707319138, 6.969202341163888, 9.463432164237952, 6.958475152468764, 8.186186346323586, 7.000408716315087, 7.297719501865411, 7.152728977076856, 6.796563659266718, 7.736924110573133, 6.57263235424013, 10.925664505862546, 6.501228954912451, 23.169132618561633, 6.505529225492204, 31.080415147063093, 6.557013696867881, 13.342622031501211, 6.662330343268008, 7.275046864615836, 6.940676753430335, 6.434820058978788, 10.06455862047472, 6.380734712904123, 31.86051695941275, 6.453874865757645, 14.341707740840516, 6.601288574883521, 6.776634507781983, 7.123831187708397, 6.314672055239832, 19.65420815280529, 6.3667525985905264, 22.269649513696088, 6.512761607635751, 6.976610239932855, 6.951845697544776, 6.272569365050757, 20.49295129320908, 6.339561799736304, 17.497513722315322, 6.520112460338254, 6.513107200967864, 7.529545908498971, 6.247522717368772, 33.65088793668931, 6.3754313629897865, 8.265304231754092, 6.6942876832459906, 6.220650090613153, 19.026606357948108, 6.294381581569784, 13.037716559959893, 6.536919753701767, 6.242475423142768, 12.052039997823856, 6.249664694554992, 17.37763469720924, 6.4813562661889845, 6.245303873479768, 11.148118607706609, 6.230903371256213, 16.130770513464682, 6.4832156545080935, 6.191828573251636, 14.504315814994346, 6.236341993605986, 10.88849485356176, 6.55391046140947, 6.139479045932662, 26.22748333933221, 6.271138919253472, 7.29606044801443, 6.843362604976914, 6.134763981350753, 33.45239714752249, 6.349291715904241, 6.24077624943357, 10.171734013709703, 6.179465742435936, 11.132473867200982, 6.529942794631777, 6.093454368170205, 33.54334467517184, 6.277007346290022, 6.366697394648696, 8.332495090624555, 6.141253170659203, 12.365945202202838, 6.490725290881068, 6.067800185245101, 34.826603184437246, 6.2689178031767865, 6.216584497407705, 10.16007444940485, 6.141696813449444, 8.725271104830416, 6.617202390464462, 6.058457719009176, 29.823229231066495, 6.325075053838914, 6.047150399889317, 23.392827509793637, 6.18671315767452, 6.426819710940936, 7.959032428442766, 6.090564347244181, 9.91872384566176, 6.532122930636899, 6.024000689858875, 29.4302939399159, 6.30518588857257, 6.004440591299545, 28.575440230434474, 6.182998281292665, 6.158499128194519, 10.622180430780869, 6.094452748804904, 7.2209842519168905, 6.87778761903864, 6.026701006955559, 13.103243907188027, 6.42011892596812, 5.978341141379464, 31.585441042324423, 6.264185853988406, 5.9609859491894275, 31.47856029128529, 6.165019006249723, 6.026138739875342, 15.261764758636614]\n",
      "[2.834290091638736, 2.1838469001652974, 1.883492251807089, 1.7083086960901137, 1.608601354502832, 1.549926435502362, 1.512405976263405, 1.4860387440593879, 1.466038216354388, 1.4500616439307794, 1.4368729067443526, 1.4257511750531806, 1.4162329379018952, 1.4079955264245887, 1.400801012505422, 1.3944669003342764, 1.3888494411635173, 1.3838333361044841, 1.3793249252512487, 1.3752474237413141, 1.3715374419125077, 1.36814235857201, 1.365018289074168, 1.3621284849126316, 1.3594420565238718, 1.3569329443217149, 1.3545790840511494, 1.352361726422574, 1.3502648804881807, 1.3482748569757097, 1.3463798927616493, 1.3445698414271514, 1.342835917756371, 1.3411704863392335, 1.3395668862815877, 1.338019285510379, 1.3365225593652341, 1.3350721891467472, 1.3336641770888913, 1.332294974872374, 1.3309614233246965, 1.3296607013832573, 1.3283902827482683, 1.3271478989372354, 1.325931507684538, 1.324739265818127, 1.323569505898717, 1.3224207160316823, 1.3212915223636141, 1.3201806738585202, 1.319087029016515, 1.3180095442534343, 1.3169472637054427, 1.315899310260256, 1.3148648776476057, 1.3138432234472108, 1.312833662893861, 1.3118355633769296, 1.3108483395465118, 1.3098714489508123, 1.3089043881398916, 1.3079466891797473, 1.3069979165281977, 1.3060576642304351, 1.3051255533975574, 1.3042012299360595, 1.3032843625002704, 1.3023746406431789, 1.301471773144069, 1.3005754864939783, 1.299685523522221, 1.2988016421491955, 1.2979236142523716, 1.2970512246338668, 1.2961842700793116, 1.2953225584988557, 1.2944659081421708, 1.2936141468801965, 1.2927671115471566, 1.2919246473370596, 1.2910866072495177, 1.29025285158025, 1.289423247452125, 1.2885976683830145, 1.2877759938871214, 1.286958109106766, 1.2861439044719294, 1.285333275385121, 1.2845261219293538, 1.2837223485972626, 1.2829218640395532, 1.2821245808311705, 1.2813304152537102, 1.2805392870927412, 1.2797511194488331, 1.2789658385611888, 1.2781833736428814, 1.2774036567267957, 1.2766266225214324, 1.275852208275838, 1.2750803536529571, 1.2743110006107887, 1.2735440932907642, 1.2727795779128268, 1.2720174026767241, 1.27125751766908, 1.2704998747758296, 1.2697444275996566, 1.268991131382079, 1.2682399429298754, 1.2674908205455595, 1.2667437239616328, 1.2659986142783692, 1.2652554539049081, 1.264514206503434, 1.263774836936261, 1.2630373112156286, 1.2623015964560516, 1.2615676608290591, 1.2608354735201879, 1.2601050046880884, 1.2593762254256218, 1.258649107722833, 1.2579236244316871, 1.2571997492324778, 1.2564774566017998, 1.255756721782012, 1.2550375207520985, 1.254319830199857, 1.253603627495341, 1.2528888906654876, 1.2521755983698712, 1.2514637298775169, 1.2507532650447308, 1.2500441842938814, 1.2493364685930948, 1.2486300994368058, 1.2479250588271378, 1.2472213292560532, 1.2465188936882519, 1.245817735544768, 1.2451178386872428, 1.2444191874028339, 1.2437217663897315, 1.2430255607432588, 1.2423305559425208, 1.2416367378375832, 1.2409440926371533, 1.240252606896742, 1.2395622675072824]\n"
     ]
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "print(d[\"costs1\"])\n",
    "print(d[\"costs2\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 362,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x1b80aacff60>]"
      ]
     },
     "execution_count": 362,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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WLAzfE12IqZgD/vC97//fTzyyCO455574Lourr32WlxzzTW48cYb8dd//de47LLL8OEPfxgPPfQQjh07th193lUQXlkBxqyILL/tRBHV/yYZhdZa2eHs0EsNog5QTLAT8K+XtT39GeqRv+Md78BXvvIV2LaNlZUVeJ6HjY0NHD16FFdccQVs28bJkydx//33b0d/dx2K2ICXY9Kq2BXhTU4yaAH3qxCvvqJWQkiirEiBz2xapKJWarUabr31Vpw4cQLXXHMNFhcXMT8/L/7/0KFDuHDhQmmd3M0oclnrTRjnXEStkUlGEWqRvElF3PPM249LDUnxm52o3z6UWuH4+Mc/jt///d/HRz7yESwsLMAwDPF/lFLlfRocODAT+mx+fnakNnYK29nP6ekmAGB2tjnycfXvT03X/TYbY3Wt4/rScdmDYBjG2PR3O/th22xdnufe8+dyz55Wpr57XrAcyNoGAPz3x14FIRT/89uvjOznJEDua6/vAmD3SD8Hft/27Z/BnP/MlY2hhvyll17CYDDAVVddhVarhePHj+P++++HZQXkz9LSEg4dOjTSgVdW2oqnNT8/i6Wl8a/Kt9393Nhgqb5ra92RjhvVz82NnmhzXK510vVcXmFJUK5HxqK/233v+30HALC21sl87x2X5R+sXNzC0nRt5D64kiG/eLGDpanR2wCAe//ppyCU4ldety+yn+MOva9d35D3+m7oHHr+fVta2kS/U5whN00j0gEGUlArZ86cwU033YTBYIDBYIBvfvOb+MAHPoCXX34Zp0+fhud5uPfee/HOd76zsA5XCFCsakV9HXfs9iBbIYqTnEWz5GufV4t+KZVaSHoud0I2O9QjP3bsGJ588km8733vg2VZOH78OE6cOIH9+/fj1KlT6Pf7OHbsGN7znvdsR393HQoNdk6YnG+3y94K0YDnbEO+9nm16GXdx0eePo9/eHgBf/x7vzYyxZsVSWqynXjOUnHkp06dwqlTp5TPrrnmGtxzzz2ldKpCgCLkYxyTlmAzaf0tGsWk6Kuvo/ch+DvfBs60NBXHuZUOzq104BEK29omQ+6fStQ57cS4rTI7xxxF1rmYNA83oFZ2th87hWKKZuW754FmJR/F48VUCiwCO5FvkEa1sp0L38qQjzmCQVpAWxMm59v1CUEFbHWXd0VXGEdOyruPwYRXSvORoAnH3Iny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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(range(len(d[\"costs1\"])), d[\"costs1\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 363,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x1b80ab3b470>]"
      ]
     },
     "execution_count": 363,
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
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(range(len(d[\"costs2\"])), d[\"costs2\"])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4.选做题"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Congratulations on building your first logistic regression model. It is your time to analyze it further."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### 4.1 Observe the effect of learning rate on the leraning process.   \n",
    "Hits: plot the learning curve with different learning rate"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 364,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "f:\\app\\python3.7\\install\\lib\\site-packages\\ipykernel_launcher.py:14: RuntimeWarning: divide by zero encountered in log\n",
      "  \n",
      "f:\\app\\python3.7\\install\\lib\\site-packages\\ipykernel_launcher.py:14: RuntimeWarning: invalid value encountered in multiply\n",
      "  \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Cost after iteration 0: nan\n",
      "Cost after iteration 100: nan\n",
      "Cost after iteration 200: nan\n",
      "Cost after iteration 300: nan\n",
      "Cost after iteration 400: nan\n",
      "Cost after iteration 500: nan\n",
      "Cost after iteration 600: nan\n",
      "Cost after iteration 700: nan\n",
      "Cost after iteration 0: nan\n",
      "Cost after iteration 100: 3.551128\n",
      "Cost after iteration 200: 3.143515\n",
      "Cost after iteration 300: 2.868094\n",
      "Cost after iteration 400: 2.663828\n",
      "Cost after iteration 500: 2.507570\n",
      "Cost after iteration 600: 2.386441\n",
      "Cost after iteration 700: 2.290857\n",
      "train accuracy: 86.26577579806978 %\n",
      "test accuracy: 52.88888888888889 %\n",
      "[86.26577579806978]\n",
      "[52.88888888888889]\n",
      "[0.001]\n",
      "Cost after iteration 0: nan\n",
      "Cost after iteration 100: nan\n",
      "Cost after iteration 200: nan\n",
      "Cost after iteration 300: nan\n",
      "Cost after iteration 400: nan\n",
      "Cost after iteration 500: nan\n",
      "Cost after iteration 600: nan\n",
      "Cost after iteration 700: nan\n",
      "Cost after iteration 0: nan\n",
      "Cost after iteration 100: nan\n",
      "Cost after iteration 200: nan\n",
      "Cost after iteration 300: nan\n",
      "Cost after iteration 400: nan\n",
      "Cost after iteration 500: nan\n",
      "Cost after iteration 600: nan\n",
      "Cost after iteration 700: 2.454766\n",
      "train accuracy: 86.56273199703044 %\n",
      "test accuracy: 52.22222222222222 %\n",
      "[86.26577579806978, 86.56273199703044]\n",
      "[52.88888888888889, 52.22222222222222]\n",
      "[0.001, 0.002]\n",
      "Cost after iteration 0: nan\n",
      "Cost after iteration 100: nan\n",
      "Cost after iteration 200: nan\n",
      "Cost after iteration 300: nan\n",
      "Cost after iteration 400: nan\n",
      "Cost after iteration 500: nan\n",
      "Cost after iteration 600: nan\n",
      "Cost after iteration 700: nan\n",
      "Cost after iteration 0: nan\n",
      "Cost after iteration 100: nan\n",
      "Cost after iteration 200: nan\n",
      "Cost after iteration 300: nan\n",
      "Cost after iteration 400: nan\n",
      "Cost after iteration 500: nan\n",
      "Cost after iteration 600: nan\n",
      "Cost after iteration 700: nan\n",
      "train accuracy: 86.6369710467706 %\n",
      "test accuracy: 48.22222222222222 %\n",
      "[86.26577579806978, 86.56273199703044, 86.6369710467706]\n",
      "[52.88888888888889, 52.22222222222222, 48.22222222222222]\n",
      "[0.001, 0.002, 0.003]\n",
      "Cost after iteration 0: nan\n",
      "Cost after iteration 100: nan\n",
      "Cost after iteration 200: nan\n",
      "Cost after iteration 300: nan\n",
      "Cost after iteration 400: nan\n",
      "Cost after iteration 500: nan\n",
      "Cost after iteration 600: nan\n",
      "Cost after iteration 700: nan\n",
      "Cost after iteration 0: nan\n",
      "Cost after iteration 100: nan\n",
      "Cost after iteration 200: nan\n",
      "Cost after iteration 300: nan\n",
      "Cost after iteration 400: nan\n",
      "Cost after iteration 500: nan\n",
      "Cost after iteration 600: nan\n",
      "Cost after iteration 700: nan\n",
      "train accuracy: 86.6369710467706 %\n",
      "test accuracy: 50.0 %\n",
      "[86.26577579806978, 86.56273199703044, 86.6369710467706, 86.6369710467706]\n",
      "[52.88888888888889, 52.22222222222222, 48.22222222222222, 50.0]\n",
      "[0.001, 0.002, 0.003, 0.004]\n",
      "Cost after iteration 0: nan\n",
      "Cost after iteration 100: nan\n",
      "Cost after iteration 200: nan\n",
      "Cost after iteration 300: nan\n",
      "Cost after iteration 400: nan\n",
      "Cost after iteration 500: nan\n",
      "Cost after iteration 600: nan\n",
      "Cost after iteration 700: nan\n",
      "Cost after iteration 0: nan\n",
      "Cost after iteration 100: nan\n",
      "Cost after iteration 200: nan\n",
      "Cost after iteration 300: nan\n",
      "Cost after iteration 400: nan\n",
      "Cost after iteration 500: nan\n",
      "Cost after iteration 600: nan\n",
      "Cost after iteration 700: nan\n",
      "train accuracy: 86.41425389755011 %\n",
      "test accuracy: 56.0 %\n",
      "[86.26577579806978, 86.56273199703044, 86.6369710467706, 86.6369710467706, 86.41425389755011]\n",
      "[52.88888888888889, 52.22222222222222, 48.22222222222222, 50.0, 56.0]\n",
      "[0.001, 0.002, 0.003, 0.004, 0.005]\n",
      "Cost after iteration 0: nan\n",
      "Cost after iteration 100: nan\n",
      "Cost after iteration 200: nan\n",
      "Cost after iteration 300: nan\n",
      "Cost after iteration 400: nan\n",
      "Cost after iteration 500: nan\n",
      "Cost after iteration 600: nan\n",
      "Cost after iteration 700: nan\n",
      "Cost after iteration 0: nan\n",
      "Cost after iteration 100: nan\n",
      "Cost after iteration 200: nan\n",
      "Cost after iteration 300: nan\n",
      "Cost after iteration 400: nan\n",
      "Cost after iteration 500: nan\n",
      "Cost after iteration 600: nan\n",
      "Cost after iteration 700: nan\n",
      "train accuracy: 85.89458054936897 %\n",
      "test accuracy: 56.44444444444445 %\n",
      "[86.26577579806978, 86.56273199703044, 86.6369710467706, 86.6369710467706, 86.41425389755011, 85.89458054936897]\n",
      "[52.88888888888889, 52.22222222222222, 48.22222222222222, 50.0, 56.0, 56.44444444444445]\n",
      "[0.001, 0.002, 0.003, 0.004, 0.005, 0.006]\n",
      "Cost after iteration 0: nan\n",
      "Cost after iteration 100: nan\n",
      "Cost after iteration 200: nan\n",
      "Cost after iteration 300: nan\n",
      "Cost after iteration 400: nan\n",
      "Cost after iteration 500: nan\n",
      "Cost after iteration 600: nan\n",
      "Cost after iteration 700: nan\n",
      "Cost after iteration 0: nan\n",
      "Cost after iteration 100: nan\n",
      "Cost after iteration 200: nan\n",
      "Cost after iteration 300: nan\n",
      "Cost after iteration 400: nan\n",
      "Cost after iteration 500: nan\n",
      "Cost after iteration 600: nan\n",
      "Cost after iteration 700: nan\n",
      "train accuracy: 86.8596881959911 %\n",
      "test accuracy: 41.11111111111111 %\n",
      "[86.26577579806978, 86.56273199703044, 86.6369710467706, 86.6369710467706, 86.41425389755011, 85.89458054936897, 86.8596881959911]\n",
      "[52.88888888888889, 52.22222222222222, 48.22222222222222, 50.0, 56.0, 56.44444444444445, 41.11111111111111]\n",
      "[0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007]\n",
      "Cost after iteration 0: nan\n",
      "Cost after iteration 100: nan\n",
      "Cost after iteration 200: nan\n",
      "Cost after iteration 300: nan\n",
      "Cost after iteration 400: nan\n",
      "Cost after iteration 500: nan\n",
      "Cost after iteration 600: nan\n",
      "Cost after iteration 700: nan\n",
      "Cost after iteration 0: nan\n",
      "Cost after iteration 100: nan\n",
      "Cost after iteration 200: nan\n",
      "Cost after iteration 300: nan\n",
      "Cost after iteration 400: nan\n",
      "Cost after iteration 500: nan\n",
      "Cost after iteration 600: nan\n",
      "Cost after iteration 700: nan\n",
      "train accuracy: 86.78544914625093 %\n",
      "test accuracy: 57.111111111111114 %\n",
      "[86.26577579806978, 86.56273199703044, 86.6369710467706, 86.6369710467706, 86.41425389755011, 85.89458054936897, 86.8596881959911, 86.78544914625093]\n",
      "[52.88888888888889, 52.22222222222222, 48.22222222222222, 50.0, 56.0, 56.44444444444445, 41.11111111111111, 57.111111111111114]\n",
      "[0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008]\n",
      "Cost after iteration 0: nan\n",
      "Cost after iteration 100: nan\n",
      "Cost after iteration 200: nan\n",
      "Cost after iteration 300: nan\n",
      "Cost after iteration 400: nan\n",
      "Cost after iteration 500: nan\n",
      "Cost after iteration 600: nan\n",
      "Cost after iteration 700: nan\n",
      "Cost after iteration 0: nan\n",
      "Cost after iteration 100: nan\n",
      "Cost after iteration 200: nan\n",
      "Cost after iteration 300: nan\n",
      "Cost after iteration 400: nan\n",
      "Cost after iteration 500: nan\n",
      "Cost after iteration 600: nan\n",
      "Cost after iteration 700: nan\n",
      "train accuracy: 86.8596881959911 %\n",
      "test accuracy: 52.44444444444444 %\n",
      "[86.26577579806978, 86.56273199703044, 86.6369710467706, 86.6369710467706, 86.41425389755011, 85.89458054936897, 86.8596881959911, 86.78544914625093, 86.8596881959911]\n",
      "[52.88888888888889, 52.22222222222222, 48.22222222222222, 50.0, 56.0, 56.44444444444445, 41.11111111111111, 57.111111111111114, 52.44444444444444]\n",
      "[0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009000000000000001]\n"
     ]
    }
   ],
   "source": [
    "learning_rate = []\n",
    "test_accuracy = []\n",
    "train_accuracy = []\n",
    "for i in np.arange(0.001, 0.01, 0.001):\n",
    "    \n",
    "    d = model(X_train, y_train, X_test, y_test, 800, i ,True)\n",
    "    \n",
    "    train_accuracy.append(d[\"train_accuracy\"])\n",
    "    test_accuracy.append(d[\"test_accuracy\"])\n",
    "    learning_rate.append(d[\"learning_rate\"])\n",
    "    \n",
    "    print(train_accuracy)\n",
    "    print(test_accuracy)\n",
    "    print(learning_rate)\n",
    "  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 365,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "No handles with labels found to put in legend.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.legend.Legend at 0x1b80ac200f0>"
      ]
     },
     "execution_count": 365,
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
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(learning_rate, test_accuracy, color='r')\n",
    "plt.plot(learning_rate, train_accuracy, color='g')\n",
    "plt.xlabel('learning rate')\n",
    "plt.ylabel('accuracy')\n",
    "plt.legend()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### 4.2 Observe the effect of iteration_num on the test accuracy."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 366,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "f:\\app\\python3.7\\install\\lib\\site-packages\\ipykernel_launcher.py:14: RuntimeWarning: divide by zero encountered in log\n",
      "  \n",
      "f:\\app\\python3.7\\install\\lib\\site-packages\\ipykernel_launcher.py:14: RuntimeWarning: invalid value encountered in multiply\n",
      "  \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Cost after iteration 0: nan\n",
      "Cost after iteration 100: nan\n",
      "Cost after iteration 200: nan\n",
      "Cost after iteration 300: nan\n",
      "Cost after iteration 400: nan\n",
      "Cost after iteration 0: nan\n",
      "Cost after iteration 100: nan\n",
      "Cost after iteration 200: nan\n",
      "Cost after iteration 300: nan\n",
      "Cost after iteration 400: nan\n",
      "train accuracy: 86.6369710467706 %\n",
      "test accuracy: 50.666666666666664 %\n",
      "[86.6369710467706]\n",
      "[50.666666666666664]\n",
      "Cost after iteration 0: nan\n",
      "Cost after iteration 100: nan\n",
      "Cost after iteration 200: nan\n",
      "Cost after iteration 300: nan\n",
      "Cost after iteration 400: 13.185143\n",
      "Cost after iteration 500: 11.718869\n",
      "Cost after iteration 0: 4.398175\n",
      "Cost after iteration 100: 3.592661\n",
      "Cost after iteration 200: 3.163059\n",
      "Cost after iteration 300: 2.870243\n",
      "Cost after iteration 400: 2.667061\n",
      "Cost after iteration 500: 2.521065\n",
      "train accuracy: 85.37490720118782 %\n",
      "test accuracy: 57.55555555555556 %\n",
      "[86.6369710467706, 85.37490720118782]\n",
      "[50.666666666666664, 57.55555555555556]\n",
      "Cost after iteration 0: 661.205338\n",
      "Cost after iteration 100: nan\n",
      "Cost after iteration 200: nan\n",
      "Cost after iteration 300: nan\n",
      "Cost after iteration 400: nan\n",
      "Cost after iteration 500: nan\n",
      "Cost after iteration 600: nan\n",
      "Cost after iteration 0: 5.798619\n",
      "Cost after iteration 100: 4.687034\n",
      "Cost after iteration 200: 4.065009\n",
      "Cost after iteration 300: 3.635486\n",
      "Cost after iteration 400: 3.347728\n",
      "Cost after iteration 500: 3.139623\n",
      "Cost after iteration 600: 2.974707\n",
      "train accuracy: 86.19153674832963 %\n",
      "test accuracy: 39.55555555555556 %\n",
      "[86.6369710467706, 85.37490720118782, 86.19153674832963]\n",
      "[50.666666666666664, 57.55555555555556, 39.55555555555556]\n",
      "Cost after iteration 0: nan\n",
      "Cost after iteration 100: nan\n",
      "Cost after iteration 200: nan\n",
      "Cost after iteration 300: nan\n",
      "Cost after iteration 400: nan\n",
      "Cost after iteration 500: 9.763574\n",
      "Cost after iteration 600: 8.663475\n",
      "Cost after iteration 700: nan\n",
      "Cost after iteration 0: 3.462372\n",
      "Cost after iteration 100: 2.737874\n",
      "Cost after iteration 200: 2.453705\n",
      "Cost after iteration 300: 2.275384\n",
      "Cost after iteration 400: 2.152336\n",
      "Cost after iteration 500: 2.060911\n",
      "Cost after iteration 600: 1.988814\n",
      "Cost after iteration 700: 1.929303\n",
      "train accuracy: 86.93392724573125 %\n",
      "test accuracy: 65.11111111111111 %\n",
      "[86.6369710467706, 85.37490720118782, 86.19153674832963, 86.93392724573125]\n",
      "[50.666666666666664, 57.55555555555556, 39.55555555555556, 65.11111111111111]\n",
      "Cost after iteration 0: 995.666421\n",
      "Cost after iteration 100: nan\n",
      "Cost after iteration 200: nan\n",
      "Cost after iteration 300: nan\n",
      "Cost after iteration 400: nan\n",
      "Cost after iteration 500: nan\n",
      "Cost after iteration 600: nan\n",
      "Cost after iteration 700: nan\n",
      "Cost after iteration 800: 8.487611\n",
      "Cost after iteration 0: 3.753715\n",
      "Cost after iteration 100: 2.895098\n",
      "Cost after iteration 200: 2.521428\n",
      "Cost after iteration 300: 2.279668\n",
      "Cost after iteration 400: 2.123670\n",
      "Cost after iteration 500: 2.018748\n",
      "Cost after iteration 600: 1.942967\n",
      "Cost after iteration 700: 1.884575\n",
      "Cost after iteration 800: 1.837587\n",
      "train accuracy: 87.37936154417224 %\n",
      "test accuracy: 50.0 %\n",
      "[86.6369710467706, 85.37490720118782, 86.19153674832963, 86.93392724573125, 87.37936154417224]\n",
      "[50.666666666666664, 57.55555555555556, 39.55555555555556, 65.11111111111111, 50.0]\n"
     ]
    }
   ],
   "source": [
    "test_accuracy = []\n",
    "train_accuracy = []\n",
    "num_iterations = []\n",
    "for i in range(500, 1000, 100):\n",
    "    \n",
    "    d = model(X_train, y_train, X_test, y_test, i, 0.001 ,True)\n",
    "    \n",
    "    train_accuracy.append(d[\"train_accuracy\"])\n",
    "    test_accuracy.append(d[\"test_accuracy\"])\n",
    "    num_iterations.append(d[\"num_iterations\"])\n",
    "    \n",
    "    print(train_accuracy)\n",
    "    print(test_accuracy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 369,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "No handles with labels found to put in legend.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.legend.Legend at 0x1b80ad38160>"
      ]
     },
     "execution_count": 369,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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pAAIIYSXkgIghBBeSgqAEEJ4qQZ3K4gLGTJkCKdOncJoLP+VXnjhBYqLi5k1axZ2u50+ffowadIkAH7++WeeeuopiouLuemmm3j++ecrl6uLXKtXr2bfvn34+voCMG7cOHr27FmnubZt28arr75KSUkJXbp04emnnyYpKUnz9qoq1/Tp0zVvrzVr1rBq1arK9ykpKdxzzz306NFD0za7UK6SkhLN22z9+vUsX74cgG7duvHkk0/Wi7+xqnLVh7+x5cuX88knn2A2m7nzzjsZM2aM+u2lNAJut1vp2rWr4nA4KqeVlJQo8fHxyokTJxSHw6EMHz5c+fLLLxVFUZS77rpL+e677xRFUZTp06cr77//fp3lUhRFufvuu5WMjIzz5q+rXCdOnFC6du2qpKenK2VlZcqgQYOUL7/8UvP2ulAurdvrXIcPH1Z69uyppKWlad5mVeXKycnRvM1sNpty8803Kzk5OYrD4VAGDhyobN26VfP2qirXzp07NW+v0xkKCwsVp9OpPProo8r69etVb69GMQT0+++/AzB8+HD69evHqlWrOHDgAC1atCA2Nhaj0Ujfvn3ZvHkzqamplJaW0r59e6D8kZWbN2+us1wlJSWkpaUxY8YM+vbty6JFi3C73XWa6/PPP+fOO+8kIiICk8lEYmIivr6+mrdXVbnatGmjeXud67nnnmPSpEkkJydr3mZV5fL19dW8zVwuF263m5KSEpxOJ06nk4CAAM3bq6pcFotF8/b66aef6Nq1KwEBARgMBm699VbWrFmjens1igJQUFBAXFwcS5Ys4Z133mH16tWkpaVhtVor5wkPDycjI4PMzMyzplutVjIyMuos1/r16+nUqRMvv/wyH330EXv37uXjjz+u01zHjx/H5XIxevRo7rnnHj744IPztq9Fe1WVy263a95eZ0pKSqK0tJQ+ffrUizarKld2drbmbRYQEMCECRPo06cP8fHxREdH14v2qiqX1WrVvL2uvfZaduzYQV5eHna7nW3btvHtt9+q3l6N4juADh060KFDh8r3AwcOZNGiRdx4442V0xRFQafT4Xa70el0502vq1y///47S5YsqZw2ZMgQ1q1bx+WXX15nuVwuF3v37mXlypX4+fkxZswYfHx8qtx+XbZXVblatGiheXudafXq1Tz88MMAF2ybumyzqnLFxsZq3ma//PILn3zyCV988QWBgYE8/vjjHDt2TPP2qirXf//7X83bKy4ujgEDBjBkyBCaNGlCXFwcO3bsUL29GsURwN69e9m1a1fle0VRiI6OJisrq3JaVlYW4eHhREREnDU9Ozub8PDwOsuVmprKli1bzppmNBrrNFdYWBhxcXE0bdoUHx8fevToQVJSkubtVVWutWvXat5ep5WVlbFnzx5uv/12gPMyaNFmVeU6dOiQ5m22Y8cO4uLiCA0NxWw2M2DAAHbv3q15e1WV66uvvtK8vYqKiujVqxefffYZK1euxGw207FjR9Xbq1EUgMLCQubOnYvdbqeoqIi1a9cyefJkjh49WjmssHHjRrp160Z0dDQWi4V9+/YB5WcEdOvWrc5yDR06lJdffpn8/HwcDgcffvghPXv2rNNc3bt3Z8eOHRQUFOByufjqq6/o3bu35u1VVa4ePXpo3l6nHTp0iJYtW+Ln5wdAu3btNG+zqnIpiqJ5m7Vp04akpCRsNhuKorBt27Z60V5V5QoMDNS8vVJSUnjsscdwOp0UFhby8ccfM3HiRNXbq1EMAXXv3p39+/fTv39/3G43gwcPpkOHDsyePZuEhATsdjvx8fH07t0bgHnz5vH0009TVFTEtddey0MPPVRnuTp27MioUaMYNGgQTqeTXr16cffdd9dprnbt2jFy5EgGDx6Mw+GgS5cuDBo0iFatWmnaXlXlGjJkCEajUdP2Oi05OZmIiIjK9xaLRfO/sapytWnTRvO/sa5du/LTTz8xYMAATCYTbdu2JSEhgS5dumjaXlXlmj9/Ph9//LGm7dWmTRt69epFv379cLlcDBs2jBtvvFH1vy95IpgQQnipRjEEJIQQwnNSAIQQwktJARBCCC8lBUAIIbyUFAAhhPBSUgBEg/bDDz8wfvx4AA4cOMDMmTNrdf1r1qzh/fffB+Bf//pX5V0khWgMGsV1AMJ7tW3blkWLFgHw66+/1vq9Wvbt20fr1q0BGDRoUK2uWwitSQEQDdru3bt58cUXWbFiBYsWLaKwsJDp06cza9Ystm3bxrJly3A4HPj4+PDkk0/SoUMHFi9ezPfff09mZiZXXXUV06ZNY+bMmeTk5JCVlUV0dDQLFizg22+/Zdu2bezcuRMfHx9OnTpFbm4uM2fO5MiRI7zwwgvk5eWh0+kYPnw4/fv3Z/fu3SQmJhIbG8uRI0dwOp08//zzZ92X6nTuC803bdo0WrduzYgRIwDOen/77bdz99138/XXX5Ofn8/IkSP59ttv+fHHHzEajSxbtoxmzZpp8U8hGiAZAhKNQmRkJOPHj+emm25i1qxZHDt2jMTERJYvX866det48cUXSUhIwGazAZCamsratWuZN28e//73v2nfvj0ffvghW7duxcfHh/Xr19OzZ09uv/12hg0bxoMPPli5LafTyZgxYxgyZAifffYZK1as4J///CffffcdUD4UNXz4cNatW8eAAQNITEysMnN15zuX3W7no48+YsKECcycOZOhQ4eyYcMGIiMjWbt27SW2pPAmcgQgGqWdO3eSmZnJsGHDKqfpdDpOnDgBQPv27SufoDR06FD27t3L22+/zbFjxzhy5Ajt2rW74LqPHTuG3W6nV69eADRr1oxevXrx1VdfccsttxAVFcXVV18NwDXXXHPBTrm6853r9HZjY2MJCwujTZs2ADRv3pz8/PxqrUMIkAIgGim3201cXBwLFiyonJaenk54eDiff/555Y3TAF555RUOHDjAvffeyy233ILT6eRid0hxuVzn3X5XURScTicAPj4+ldN1Ot0F13Wh+c5dxuFwnLWc2WyufG0ymS6YU4g/I0NAotEwGAyVnXBcXBw7d+7kt99+A2D79u3069eP0tLS85bbsWMHQ4cOpX///oSGhpKUlITL5Tpvnae1atUKo9HIf//7XwAyMjLYsmULnTt3rpXfIyQkhIMHD1au+5tvvqmV9QpxLjkCEI1G+/btWbJkCePGjePVV1/lhRdeYPLkyZX3d1+2bBn+/v7nLTd27Fjmzp3LwoULMZlM3HDDDZVDRd26dWP27NlnzW8ymVi6dCkvvfQSixcvxuVyMXbsWDp16sTu3bsv+fcYMmQIjz/+OHfccQcxMTF06tTpktcpRFXkbqBCCOGlZAhICCG8lBQAIYTwUlIAhBDCS0kBEEIILyUFQAghvJQUACGE8FJSAIQQwktJARBCCC/1/5RZvtSekGvaAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(iteration_num, test_accuracy, color='r')\n",
    "plt.plot(iteration_num, train_accuracy, color='g')\n",
    "plt.xlabel('iteration num')\n",
    "plt.ylabel('accuracy')\n",
    "plt.legend()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Challenge ! ! !\n",
    "\n",
    "The original data have images labeled 0,1,2,3,4,5,6,7,8,9. In our logistic model, we only detect if the digit in the image is larger or smaller than 5. Now, Let's go for a more challenging problem. Try to use softmax function to build a model to recognize which digit (0,1,2,3,4,5,6,7,8,9) is in the image."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Congratulations ! You have completed assigment 4. "
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
 "nbformat_minor": 2
}
