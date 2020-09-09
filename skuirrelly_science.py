import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as BS
import warnings
from sys import argv
import seaborn as sns
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from math import sqrt
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

# making sure we can see the whole table
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 70000)


