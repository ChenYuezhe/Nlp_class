{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Build Graph"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "coordination_source = \"\"\"\n",
    "{name:'兰州', geoCoord:[103.73, 36.03]},\n",
    "{name:'嘉峪关', geoCoord:[98.17, 39.47]},\n",
    "{name:'西宁', geoCoord:[101.74, 36.56]},\n",
    "{name:'成都', geoCoord:[104.06, 30.67]},\n",
    "{name:'石家庄', geoCoord:[114.48, 38.03]},\n",
    "{name:'拉萨', geoCoord:[102.73, 25.04]},\n",
    "{name:'贵阳', geoCoord:[106.71, 26.57]},\n",
    "{name:'武汉', geoCoord:[114.31, 30.52]},\n",
    "{name:'郑州', geoCoord:[113.65, 34.76]},\n",
    "{name:'济南', geoCoord:[117, 36.65]},\n",
    "{name:'南京', geoCoord:[118.78, 32.04]},\n",
    "{name:'合肥', geoCoord:[117.27, 31.86]},\n",
    "{name:'杭州', geoCoord:[120.19, 30.26]},\n",
    "{name:'南昌', geoCoord:[115.89, 28.68]},\n",
    "{name:'福州', geoCoord:[119.3, 26.08]},\n",
    "{name:'广州', geoCoord:[113.23, 23.16]},\n",
    "{name:'长沙', geoCoord:[113, 28.21]},\n",
    "//{name:'海口', geoCoord:[110.35, 20.02]},\n",
    "{name:'沈阳', geoCoord:[123.38, 41.8]},\n",
    "{name:'长春', geoCoord:[125.35, 43.88]},\n",
    "{name:'哈尔滨', geoCoord:[126.63, 45.75]},\n",
    "{name:'太原', geoCoord:[112.53, 37.87]},\n",
    "{name:'西安', geoCoord:[108.95, 34.27]},\n",
    "//{name:'台湾', geoCoord:[121.30, 25.03]},\n",
    "{name:'北京', geoCoord:[116.46, 39.92]},\n",
    "{name:'上海', geoCoord:[121.48, 31.22]},\n",
    "{name:'重庆', geoCoord:[106.54, 29.59]},\n",
    "{name:'天津', geoCoord:[117.2, 39.13]},\n",
    "{name:'呼和浩特', geoCoord:[111.65, 40.82]},\n",
    "{name:'南宁', geoCoord:[108.33, 22.84]},\n",
    "//{name:'西藏', geoCoord:[91.11, 29.97]},\n",
    "{name:'银川', geoCoord:[106.27, 38.47]},\n",
    "{name:'乌鲁木齐', geoCoord:[87.68, 43.77]},\n",
    "{name:'香港', geoCoord:[114.17, 22.28]},\n",
    "{name:'澳门', geoCoord:[113.54, 22.19]}\n",
    "\"\"\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Get data from source using regular expression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [],
   "source": [
    "import re"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### ---------------------------------------------------------------------------------------------------------------------------------------------------------------"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### regular expression\n",
    "[a-z]    \n",
    "[A-Z]    \n",
    "[^a]: negation   \n",
    "colou?r:  ? zero or one of its previous character  \n",
    "* : zero or more of its previous character  \n",
    "+: one or more  \n",
    ".:match any single character  \n",
    "^:start of the line  \n",
    "$:end of the line  \n",
    "| [cat|dog] : cat or dog  \n",
    "(da): make the string da like a character  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [],
   "source": [
    "l = \"color or colour\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['color', 'colour']"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pattern = re.compile(\"colou?r\")\n",
    "pattern.findall(l)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[]"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "re.findall(\"{A-Z}\",l)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### ---------------------------------------------------------------------------------------------------------------------------------------------------------------"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_city_info(city_coordination):\n",
    "    city_location = {}\n",
    "    for line in city_coordination.split(\"\\n\"):\n",
    "        if line.startswith(\"//\"): continue\n",
    "        if line.strip() == \"\":continue\n",
    "            \n",
    "        city = re.findall(\"name:'(\\w+)'\",line)[0]\n",
    "        x_y = re.findall(\"Coord:\\[(\\d+.\\d+),\\s(\\d+.\\d+)\\]\",line)[0]\n",
    "        x_y = tuple(map(float,x_y))\n",
    "        city_location[city] = x_y\n",
    "    return city_location"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    ">[字符串Srting的基本内置函数与用法](https://www.cnblogs.com/gongdada/p/10928620.html)  \n",
    ">[re模块findall等用法](https://www.cnblogs.com/syw20170419/p/9749809.html)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [],
   "source": [
    "city_info = get_city_info(coordination_source)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'金安桥': (116.16, 39.92), '四道桥': (116.13, 39.91)}"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "city_info"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Compute distance between cities"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [],
   "source": [
    "import math\n",
    "\n",
    "def geo_distance(origin, destination):\n",
    "    \"\"\"\n",
    "    Calculate the Haversine distance.\n",
    "\n",
    "    Parameters\n",
    "    ----------\n",
    "    origin : tuple of float\n",
    "        (lat, long)\n",
    "    destination : tuple of float\n",
    "        (lat, long)\n",
    "\n",
    "    Returns\n",
    "    -------\n",
    "    distance_in_km : float\n",
    "\n",
    "    Examples\n",
    "    --------\n",
    "    >>> origin = (48.1372, 11.5756)  # Munich\n",
    "    >>> destination = (52.5186, 13.4083)  # Berlin\n",
    "    >>> round(distance(origin, destination), 1)\n",
    "    504.2\n",
    "    \"\"\"\n",
    "    lat1, lon1 = origin\n",
    "    lat2, lon2 = destination\n",
    "    radius = 6371  # km\n",
    "\n",
    "    dlat = math.radians(lat2 - lat1)\n",
    "    dlon = math.radians(lon2 - lon1)\n",
    "    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +\n",
    "         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *\n",
    "         math.sin(dlon / 2) * math.sin(dlon / 2))\n",
    "    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))\n",
    "    d = radius * c\n",
    "\n",
    "    return d"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_city_distance(city1, city2):\n",
    "    return geo_distance(city_info[city1], city_info[city2])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [
    {
     "ename": "KeyError",
     "evalue": "'杭州'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-63-855a5a231356>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mget_city_distance\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"杭州\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m\"上海\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32m<ipython-input-62-26f9bd6439a4>\u001b[0m in \u001b[0;36mget_city_distance\u001b[1;34m(city1, city2)\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[1;32mdef\u001b[0m \u001b[0mget_city_distance\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mcity1\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcity2\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 2\u001b[1;33m     \u001b[1;32mreturn\u001b[0m \u001b[0mgeo_distance\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mcity_info\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mcity1\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcity_info\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mcity2\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mKeyError\u001b[0m: '杭州'"
     ]
    }
   ],
   "source": [
    "get_city_distance(\"杭州\", \"上海\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Draw the gragh"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [],
   "source": [
    "import networkx as nx   # networkx 用来画图\n",
    "import matplotlib as plt\n",
    "# %matplotlib inline\n",
    "\n",
    "plt.rcParams['font.sans-serif'] = ['SimHei']\n",
    "plt.rcParams['axes.unicode_minus'] = False"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "dict_keys(['金安桥', '四道桥'])"
      ]
     },
     "execution_count": 65,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "city_info.keys()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [],
   "source": [
    "city_graph = nx.Graph()   # 画一个图  ()内为空就是列表，（）内若为字典就是连线图\n",
    "\n",
    "city_graph.add_nodes_from(list(city_info.keys()))    # 图上加点"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAb4AAAEuCAYAAADx63eqAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy8li6FKAAAOf0lEQVR4nO3cf8zWdb3H8dfl/QtB6ObQDpWgdwZH5s6gkxyoNjthpcTCu9GBseh0GibUsB/CynSVzomsVawYlVtbR3Q21GHWUMLjzKVEnN1LTUuTpplKalQo983Fj/u+r/MHi43VqUwP7Ob9ePx139f1ua7r87n+ee5zXZ/v1Wi1Wq0AQBEnHe8JAMCxJHwAlCJ8AJQifACUInwAlCJ8AJQifACUInwAlCJ8AJQifACUInwAlCJ8AJQifACUInwAlCJ8AJQifACUInwAlCJ8AJQifACUInwAlCJ8AJQifACUInwAlCJ8AJQifACUInwAlCJ8AJQifACUInwAlCJ8AJQifACUInwAlCJ8AJQifACUInwAlCJ8AJQifACUInwAlCJ8AJQifACUInwAlCJ8AJQifACUInwAlCJ8AJQifACUInwAlCJ8AJQifACUInwAlCJ8AJQifACUInwAlCJ8AJQifACUInwAlCJ8AJQifACUInwAlCJ8AJQifACUInwAlCJ8AJQifACUInwAlCJ8AJQifACUInwAlCJ8AJQifACUInwAlCJ8AJQifACUInwAlCJ8AJQifACUInwAlCJ8AJQifACUInwAlCJ8AJQifAAj2O7du4/Zax04cOCYvdb/J+EDGKEOHTqUOXPmZPXq1X/T+I997GPZtWvXXxzT29ubBx54IEnS39+fhx56KDfccEMuvPDCTJw4MY8//vjLnvfxJnwAI9SKFSsyf/783H333bn11lv/6vgzzzwza9eu/YtjRo0alTFjxqTVauWiiy7KmjVr8vGPfzxvetObsnHjxowdO/aVmv5x0368JwDASzM8PJxLLrkkw8PDueaaa/Liiy/mggsuyE9/+tNcfvnl6ezsTJJce+21+fznP5/TTjvtqMfPnDkzSfKb3/wmK1asyOWXX55Wq5VDhw6l0WhkeHg4w8PD2bBhQzo6OjJjxoysWLEiyYnxcacdH8AI8qtf/Srnn39+nn766Vx11VV59tlns2/fvmzYsCE//vGP84Y3vCFXXHFFdu3alc7OzixatCh9fX25+eabM2HylLz7s/+VNdffnr6+vnz0ox9NW1vbked961vfmk2bNmXJkiXZvHlzzjnnnEyfPj2/+93vMm7cuLztbW/L4sWLj/M78PI1Wq1W63hPAoC/7sEHH8y8efOyevXqbNq0KUNDQ9m5c2fGjBmTSZMm5cCBA1m7dm3WrVuXVatW5TWveU0OHDiQiRMnZsF/LMu25xs5+U0X5OSOtqxb/C/5pzH7kySnn356kuTpp5/O5MmTs2TJkmzYsCFtbW1ZtmxZent7c9ddd2X+/Pk599xzj+db8IoQPoARpNls5uSTTz7y/2c/+9lMmTIlH/rQh44aNzQ0lLPPPjtjxozJ8PBw+n5yf06acHoabYe/4eruGM6U101Is9nMN77xjcyaNSvr16/PpZdemgULFuS8887L4OBgNmzYkK1bt+bFF1/M/Pnzc+WVV2bu3LnHcsmvOOEDGGHOPvvstFqtnHTSSdm1a1e6uroyYcKEDA0Npdls5tFHHz1q/Cc/+ck0O8fnh6Nmp3loKENP9OXVv7or9//43iNjBgYGMnv27JxxxhlZs2ZNbrrppmzfvj2vetWr8r73vS8dHR2544478uijj2b16tWZM2fOsV72K8Z3fAAjzPe///2sWrUqfX19Wbp0aa644or09fXlAx/4QDZv3nzU2F//+te5/vrrM/yHp7P4H5/Lv581Lp0P3JJvfOWLR43bunVrFi9enNGjR6erqysrV67M5s2bMzAwkClTpiRJpk6dmm3bto3o6CXCBzDiDA4OZtWqVXnqqaeO3DYwMJAvfvGL6erqOmrsaaedlt/+9rdZunRpfvbD27P2P/8tJw396cnMBQsW5LLLLjvyf3d3dzo7O/PII49k+vTpSZJGo5FGo/H/tKpjx+UMACPMa1/72ixfvjzXXXfdkdu+9a1v5YMf/GAmT5581Ni9e/fmvvvuyy233JKHH3443/3ud7N///5cfPHF6ezszNVXX33kwEpbW1tarVb++A3YjTfemJ6enj+J6UgnfAAjyPLly3PfffclSTo6OrJnz550dHSkv78/48ePz5YtW7Jv37489thjufTSS3PdddflXe96VxYtWpRvfvObRy5fWLhwYa6//vp84QtfyDnnnJOOjo4kh6/TO3jwYF544YWsW7cu69evT3L4V2JOhGv4EodbAE5YBw8eTEdHx0v6eHLPnj055ZRT0t5+4u6LhA+AUhxuAaAU4QOgFOEDoBThA6AU4QOgFOEDoBThA6AU4QOgFOEDoBThA6AU4QOgFOEDoBThA6AU4QOgFOEDoBThA6AU4QOgFOEDoBThA6AU4QOgFOEDoBThA6AU4QOgFOEDoBThA6AU4QOgFOEDoBThA6AU4QOgFOEDoBThA6AU4QOgFOEDoBThA6AU4QOgFOEDoBThA6AU4QOgFOEDoBThA6AU4QOgFOEDoBThA6AU4QOgFOEDoBThA6AU4QOgFOEDoBThA6AU4QOgFOEDoBThA6AU4QOgFOEDoBThA6AU4QOgFOEDoBThA6AU4QOgFOEDoBThA6AU4QOgFOEDoBThA6AU4QOgFOEDoBThA6AU4QOgFOEDoBThA6AU4QOgFOEDoBThA6AU4QOgFOEDoBThA6AU4QOgFOEDoBThA6AU4QOgFOEDoBThA6AU4QOgFOEDoBThA6AU4QOgFOEDoBThA6AU4QOgFOEDoBThA6AU4QOgFOEDoBThA6AU4QOgFOEDoBThA6AU4QOgFOEDoBThA6AU4QOgFOEDoBThA6AU4QOgFOEDoBThA6AU4QOgFOEDoBThA6AU4QOgFOF7GQ4cOHC8pwDAS1QqfM1mM3v37k2SDA0N5eKLL06z2UySfOUrX8nWrVv/5DG9vb154IEHkiT9/f156KGHcsMNN+TCCy/MxIkT8/jjjx+7BQDwsrUf7wm8XDNnzszg4GDa29vz/PPPZ+zYsWk2m2lvb8+4cePSbDYzbdq0bNq0KStXrswZZ5yRlStXptVqZdKkSfnBD36Q888/P1//+tezbdu2DA0NJUna2tqSJKNGjcqYMWPSarVy0UUXpdFoZMuWLbn66quzcOHCjB079nguH4CXaMSHr9ls5vbbb09PT0/e//735yMf+UjuvvvuTJo0KR/+8Idzzz335Etf+lKSZN68efna176WadOmZfXq1Wlra8vNN9+cRYsWZcaMGXnve9+bwcHBfPrTn86CBQty6NChNBqNDA8PZ3h4OBs2bEhHR0dmzJiRFStWJPFxJ8BIM+LD19HRceTvPXv2ZPz48X8ypr398DLnzp2b8847L11dXZk6dWoefvjh3H///Wlra8sll1ySJHnLW96SU089NU888UQWLlyYBx98ML/85S/zuc99Ltdcc0327duX3//+9xk3blze+MY3ZsKECfnOd75zbBYLwMs24sP3R61WKz//+c8zderUP3v/zp0788QTT6S7uzuzZs3K9773vfzoRz/K1H+dk9e9pTc7fvFM/ufOW9PV1ZVTTz01r3/963Pbbbdl8uTJmTZtWt7znvekt7c3y5YtS29vb+66667Mnz8/55577jFeKQAvxwkTvo0bN+bNb35zRo0alSQZHBw86v5nnnkmd9xxR7Zv354dO3akvb09P9rRlzt/sjPDreS/G8mofb89sjtMkttuuy2jR49Oo9HIt7/97QwODuaxxx7LO9/5zsyaNSvz58/PlVdemblz5x7TtQLw9zshwnfvvffmsssuy7Zt25Ik06dPz7Jly/LlL385y5cvT5K8/e1vT09PT/r6+pIkjUYjp/7z7Dz72rceeZ5xv9icRqORJBkYGMi1116bd7zjHfnMZz6Tm266Kdu3b8+rX/3q3Hrrreno6MhZZ52Vq666Kl1dXZkzZ84xXjUAf48T4nKGnp6ebNy4MaeffnqSZMGCBdm9e3d27tyZmTNn/tnHDA0NZdRgf/KHp3Jw95PJH55K52Azw8PDSZKtW7dm8eLFGT16dLq6urJy5cps3rw5AwMDmTJlSpJk6tSp2bZtm+gBjCAjfsfXarUyefLk9PT0/MUxyeETmAMDA2m1Wunv78+Tv3goY//h2fTvH8wpo9rz7HO7cujQoSSH49nb25slS5YkSbq7u9NqtfLII49k+vTpefLJJ9NoNI7sEAEYGUZ8+A4dOpR58+als7Pzz97f39+fM888M3v37s3SpUvTbDbz7ne/O6tWrcqnPvWpjBkz5sjYwcHBPPfcc9mzZ0+6u7vT1taWVqt1JJw33nhjenp60tXVdUzWBsArb8SHb3BwMHfeeef/ueO75557smbNmmzZsiWzZ8/O2rVrs2PHjnz1q1/NJz7xifT392f//v1pNps5cOBA2tvb87Of/Szd3d1JDu8SDx48mBdeeCHr1q3L+vXrkxwOrmv4AEaeRuuP25kRavfu3Rk/fvyRX1p5pe3ZsyennHLKUac9ARi5Rnz4AOClOCFOdQLA30r4AChF+AAoRfgAKEX4AChF+AAoRfgAKEX4AChF+AAoRfgAKEX4AChF+AAoRfgAKEX4AChF+AAoRfgAKEX4AChF+AAoRfgAKEX4AChF+AAoRfgAKEX4AChF+AAoRfgAKEX4AChF+AAoRfgAKEX4AChF+AAoRfgAKEX4AChF+AAoRfgAKEX4AChF+AAoRfgAKEX4AChF+AAoRfgAKEX4AChF+AAoRfgAKEX4AChF+AAoRfgAKEX4AChF+AAoRfgAKEX4AChF+AAoRfgAKEX4AChF+AAoRfgAKEX4AChF+AAoRfgAKEX4AChF+AAoRfgAKEX4AChF+AAoRfgAKEX4AChF+AAoRfgAKEX4AChF+AAoRfgAKEX4AChF+AAoRfgAKEX4AChF+AAoRfgAKEX4AChF+AAoRfgAKEX4AChF+AAoRfgAKOV/AVBkfFvDtLcvAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "nx.draw(city_graph, city_info, with_labels=True, node_size=10)  # nx.draw(图， 图的内容（这里是列表，下面那个是字典）， with_labels=是否显示标签， node_size=点的大小)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Build connection between. Let's assume that two cities are connected if their distance is less than 700 km."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "threshold = 700   # defined the threshold"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "from collections import defaultdict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "def build_connection(city_info):\n",
    "    cities_connection = defaultdict(list)   # 建了一个字典\n",
    "    cities = list(city_info.keys())\n",
    "    for c1 in cities:\n",
    "        for c2 in cities:\n",
    "            if c1 == c2: continue\n",
    "            \n",
    "            if get_city_distance(c1, c2) < threshold:\n",
    "                cities_connection[c1].append(c2)\n",
    "    return cities_connection\n",
    "cities_connection = build_connection(city_info)"
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
       "defaultdict(list,\n",
       "            {'兰州': ['嘉峪关', '西宁', '成都', '拉萨', '贵阳', '西安', '重庆', '南宁', '银川'],\n",
       "             '嘉峪关': ['兰州', '西宁', '成都', '拉萨'],\n",
       "             '西宁': ['兰州', '嘉峪关', '成都', '拉萨', '贵阳', '重庆', '银川'],\n",
       "             '成都': ['兰州', '嘉峪关', '西宁', '拉萨', '贵阳', '西安', '重庆', '南宁', '银川'],\n",
       "             '石家庄': ['武汉',\n",
       "              '郑州',\n",
       "              '济南',\n",
       "              '南京',\n",
       "              '合肥',\n",
       "              '南昌',\n",
       "              '广州',\n",
       "              '长沙',\n",
       "              '太原',\n",
       "              '西安',\n",
       "              '北京',\n",
       "              '天津',\n",
       "              '呼和浩特'],\n",
       "             '拉萨': ['兰州', '嘉峪关', '西宁', '成都', '贵阳', '重庆', '南宁', '银川'],\n",
       "             '贵阳': ['兰州', '西宁', '成都', '拉萨', '西安', '重庆', '南宁', '银川'],\n",
       "             '武汉': ['石家庄',\n",
       "              '郑州',\n",
       "              '济南',\n",
       "              '南京',\n",
       "              '合肥',\n",
       "              '杭州',\n",
       "              '南昌',\n",
       "              '福州',\n",
       "              '广州',\n",
       "              '长沙',\n",
       "              '太原',\n",
       "              '西安',\n",
       "              '北京',\n",
       "              '天津',\n",
       "              '呼和浩特',\n",
       "              '香港',\n",
       "              '澳门'],\n",
       "             '郑州': ['石家庄',\n",
       "              '武汉',\n",
       "              '济南',\n",
       "              '南京',\n",
       "              '合肥',\n",
       "              '南昌',\n",
       "              '广州',\n",
       "              '长沙',\n",
       "              '太原',\n",
       "              '西安',\n",
       "              '北京',\n",
       "              '天津',\n",
       "              '呼和浩特',\n",
       "              '香港',\n",
       "              '澳门'],\n",
       "             '济南': ['石家庄',\n",
       "              '武汉',\n",
       "              '郑州',\n",
       "              '南京',\n",
       "              '合肥',\n",
       "              '杭州',\n",
       "              '南昌',\n",
       "              '福州',\n",
       "              '长沙',\n",
       "              '太原',\n",
       "              '北京',\n",
       "              '上海',\n",
       "              '天津',\n",
       "              '呼和浩特'],\n",
       "             '南京': ['石家庄',\n",
       "              '武汉',\n",
       "              '郑州',\n",
       "              '济南',\n",
       "              '合肥',\n",
       "              '杭州',\n",
       "              '南昌',\n",
       "              '福州',\n",
       "              '长沙',\n",
       "              '北京',\n",
       "              '上海',\n",
       "              '天津'],\n",
       "             '合肥': ['石家庄',\n",
       "              '武汉',\n",
       "              '郑州',\n",
       "              '济南',\n",
       "              '南京',\n",
       "              '杭州',\n",
       "              '南昌',\n",
       "              '福州',\n",
       "              '广州',\n",
       "              '长沙',\n",
       "              '太原',\n",
       "              '北京',\n",
       "              '上海',\n",
       "              '天津',\n",
       "              '香港',\n",
       "              '澳门'],\n",
       "             '杭州': ['武汉', '济南', '南京', '合肥', '南昌', '福州', '北京', '上海', '天津'],\n",
       "             '南昌': ['石家庄',\n",
       "              '武汉',\n",
       "              '郑州',\n",
       "              '济南',\n",
       "              '南京',\n",
       "              '合肥',\n",
       "              '杭州',\n",
       "              '福州',\n",
       "              '广州',\n",
       "              '长沙',\n",
       "              '太原',\n",
       "              '北京',\n",
       "              '上海',\n",
       "              '天津',\n",
       "              '香港',\n",
       "              '澳门'],\n",
       "             '福州': ['武汉',\n",
       "              '济南',\n",
       "              '南京',\n",
       "              '合肥',\n",
       "              '杭州',\n",
       "              '南昌',\n",
       "              '广州',\n",
       "              '上海',\n",
       "              '香港',\n",
       "              '澳门'],\n",
       "             '广州': ['石家庄',\n",
       "              '武汉',\n",
       "              '郑州',\n",
       "              '合肥',\n",
       "              '南昌',\n",
       "              '福州',\n",
       "              '长沙',\n",
       "              '太原',\n",
       "              '西安',\n",
       "              '南宁',\n",
       "              '香港',\n",
       "              '澳门'],\n",
       "             '长沙': ['石家庄',\n",
       "              '武汉',\n",
       "              '郑州',\n",
       "              '济南',\n",
       "              '南京',\n",
       "              '合肥',\n",
       "              '南昌',\n",
       "              '广州',\n",
       "              '太原',\n",
       "              '西安',\n",
       "              '北京',\n",
       "              '天津',\n",
       "              '呼和浩特',\n",
       "              '南宁',\n",
       "              '香港',\n",
       "              '澳门'],\n",
       "             '沈阳': ['长春', '哈尔滨', '上海'],\n",
       "             '长春': ['沈阳', '哈尔滨'],\n",
       "             '哈尔滨': ['沈阳', '长春'],\n",
       "             '太原': ['石家庄',\n",
       "              '武汉',\n",
       "              '郑州',\n",
       "              '济南',\n",
       "              '合肥',\n",
       "              '南昌',\n",
       "              '广州',\n",
       "              '长沙',\n",
       "              '西安',\n",
       "              '北京',\n",
       "              '天津',\n",
       "              '呼和浩特',\n",
       "              '银川',\n",
       "              '澳门'],\n",
       "             '西安': ['兰州',\n",
       "              '成都',\n",
       "              '石家庄',\n",
       "              '贵阳',\n",
       "              '武汉',\n",
       "              '郑州',\n",
       "              '广州',\n",
       "              '长沙',\n",
       "              '太原',\n",
       "              '重庆',\n",
       "              '呼和浩特',\n",
       "              '南宁',\n",
       "              '银川'],\n",
       "             '北京': ['石家庄',\n",
       "              '武汉',\n",
       "              '郑州',\n",
       "              '济南',\n",
       "              '南京',\n",
       "              '合肥',\n",
       "              '杭州',\n",
       "              '南昌',\n",
       "              '长沙',\n",
       "              '太原',\n",
       "              '天津',\n",
       "              '呼和浩特'],\n",
       "             '上海': ['济南', '南京', '合肥', '杭州', '南昌', '福州', '沈阳', '天津'],\n",
       "             '重庆': ['兰州', '西宁', '成都', '拉萨', '贵阳', '西安', '呼和浩特', '南宁', '银川'],\n",
       "             '天津': ['石家庄',\n",
       "              '武汉',\n",
       "              '郑州',\n",
       "              '济南',\n",
       "              '南京',\n",
       "              '合肥',\n",
       "              '杭州',\n",
       "              '南昌',\n",
       "              '长沙',\n",
       "              '太原',\n",
       "              '北京',\n",
       "              '上海',\n",
       "              '呼和浩特'],\n",
       "             '呼和浩特': ['石家庄',\n",
       "              '武汉',\n",
       "              '郑州',\n",
       "              '济南',\n",
       "              '长沙',\n",
       "              '太原',\n",
       "              '西安',\n",
       "              '北京',\n",
       "              '重庆',\n",
       "              '天津',\n",
       "              '银川'],\n",
       "             '南宁': ['兰州',\n",
       "              '成都',\n",
       "              '拉萨',\n",
       "              '贵阳',\n",
       "              '广州',\n",
       "              '长沙',\n",
       "              '西安',\n",
       "              '重庆',\n",
       "              '银川',\n",
       "              '香港',\n",
       "              '澳门'],\n",
       "             '银川': ['兰州',\n",
       "              '西宁',\n",
       "              '成都',\n",
       "              '拉萨',\n",
       "              '贵阳',\n",
       "              '太原',\n",
       "              '西安',\n",
       "              '重庆',\n",
       "              '呼和浩特',\n",
       "              '南宁'],\n",
       "             '香港': ['武汉', '郑州', '合肥', '南昌', '福州', '广州', '长沙', '南宁', '澳门'],\n",
       "             '澳门': ['武汉',\n",
       "              '郑州',\n",
       "              '合肥',\n",
       "              '南昌',\n",
       "              '福州',\n",
       "              '广州',\n",
       "              '长沙',\n",
       "              '太原',\n",
       "              '南宁',\n",
       "              '香港']})"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cities_connection"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Draw connection graph"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "cities_connection_graph = nx.Graph(cities_connection)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
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
    "nx.draw(cities_connection_graph,city_info,with_labels=True,node_size=10)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### BFS 1 version|"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['济南', '南京', '合肥', '杭州', '南昌', '福州', '沈阳', '天津']\n"
     ]
    }
   ],
   "source": [
    "pathes = [['上海']]   #  附加，记得删了\n",
    "path = pathes.pop(0)\n",
    "froniter = path[-1]\n",
    "successsors = cities_connection[froniter]\n",
    "\n",
    "print(successsors)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "def search_1(graph,start,destination):\n",
    "    pathes = [[start]]  # list 用来存储待搜索路径\n",
    "    visited = set() # set用来存储已搜索的节点\n",
    "    \n",
    "    while pathes:\n",
    "        path = pathes.pop(0)  #提取第一条路径\n",
    "        froniter = path[-1]   #提取即将要探索的节点\n",
    "        \n",
    "        if froniter in visited: continue  #检查如果该点已经探索过 则不用再探索\n",
    "            \n",
    "        successsors = graph[froniter]\n",
    "        \n",
    "        for city in successsors:      #遍历子节点\n",
    "            if city in path: continue  # check loop #检查会不会形成环\n",
    "            \n",
    "            new_path = path+[city]\n",
    "            \n",
    "            pathes.append(new_path)  #bfs     #将新路径加到list里面\n",
    "            #pathes = [new_path] + pathes #dfs\n",
    "            #print(pathes)\n",
    "            if city == destination:  #检查目的地是不是已经搜索到了\n",
    "                return new_path\n",
    "        visited.add(froniter)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['上海', '合肥', '香港']"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "search_1(cities_connection,\"上海\",\"香港\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Optimal search using variation of BFS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [],
   "source": [
    "def search_2(graph,start,destination,search_strategy):\n",
    "    pathes = [[start]]\n",
    "    visited = set()# ！\n",
    "    while pathes:\n",
    "        path = pathes.pop(0)  # path 列表\n",
    "        froniter = path[-1]   # froniter 字符\n",
    "        if froniter in visited : continue# ！\n",
    "            \n",
    "        if froniter == destination:# ！\n",
    "            return path# ！\n",
    "        \n",
    "        successsors = graph[froniter]\n",
    "        \n",
    "        for city in successsors:\n",
    "            if city in path: continue  # check loop\n",
    "            \n",
    "            new_path = path+[city]\n",
    "            \n",
    "            pathes.append(new_path)  #bfs\n",
    "            \n",
    "        pathes = search_strategy(pathes)\n",
    "       \n",
    "        visited.add(froniter) # ！\n",
    "       # if pathes and (destination == pathes[0][-1]):\n",
    "       #     return pathes[0]  \n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_distance_of_path(pathes):\n",
    "    distance = 0\n",
    "    \n",
    "    for i,_ in enumerate(pathes[:-1]):\n",
    "        \n",
    "        distance += get_city_distance(pathes[i],pathes[i+1])\n",
    "        \n",
    "    return distance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [],
   "source": [
    "def sort_by_distance(pathes):\n",
    "  \n",
    "    return sorted(pathes,key=get_distance_of_path)    \n",
    "    #sorted（可迭代的对象，key=用来比较的对象，reverse=Ture降序 False升序（默认） 返回值是重新排列后的列表）"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "978.120319439158"
      ]
     },
     "execution_count": 69,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "get_distance_of_path([\"上海\",\"福州\",\"香港\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1045.1004188056104"
      ]
     },
     "execution_count": 70,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "get_distance_of_path([\"上海\",\"合肥\",\"香港\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['上海', '福州', '香港']"
      ]
     },
     "execution_count": 66,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "search_2(cities_connection,\"上海\",\"香港\",search_strategy=sort_by_distance)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['上海', '合肥', '香港']"
      ]
     },
     "execution_count": 68,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "search_2(cities_connection,\"上海\",\"香港\",search_strategy=lambda x:x)"
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
