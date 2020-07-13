import pandas as pd
import numpy as np
import warnings
from sys import argv

warnings.filterwarnings('ignore')

# making sure we can see the whole table
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 25)
pd.set_option('display.max_rows', 5000)

pff_projections = 'data/PFFprojections2020Week1.csv'

#create dataframe from pff projectsions
df_pff = pd.read_csv(pff_projections)

#test to make sure dataframe is working
print(df_pff.head())

#testing to see type of data
print(type(df_pff['fantasyPoints'][3]))


#        #trying to import Drafkings Data
dk_salaries = 'data/DKSalaries.csv'
df_dk = pd.read_csv(dk_salaries, skiprows=7) #because of format from DK, need to skip the first 7 rows
# cleaning up DKSalary download
df_dk = df_dk.drop(df_dk.columns[[0, 1, 2, 3, 4, 5, 6]], axis=1)
df_dk = df_dk.drop(['Name + ID', 'ID', 'Game Info'], 1)
print(df_dk.head())

#           time for the backtracking recursion hopefully
MAX_SALARY = 50000

