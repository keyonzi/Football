# unfinished code to take showdown generator and turn into a format easy to upload to draftkings


import pandas as pd
import numpy as np
import warnings
import copy
from collections import Counter

warnings.filterwarnings('ignore')

# making sure we can see the whole table
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 25)
pd.set_option('display.max_rows', 5000)

#           load data sources
linups_csv = 'data/Lineups/Week{w}/{m}_{d}_20_{teamA}vs{teamB}/{style}.csv'
dk_salaries = 'data/Lineups/Week{w}/{m}_{d}_20_{teamA}vs{teamB}/DKSalaries.csv'

week = input('What week number: ')
month = input('What month is it: ')
day = input('What day is the game: ')
teamA = input('Who is team A? ').upper()
teamB = input('Who is team B? ').upper()
style = input('Cash or GPP? ').upper()

# loading the lineups
file_path = linups_csv.format(w=week, m=month, d=day, teamA=teamA, teamB=teamB, style=style)
df_lineups = pd.read_csv(file_path)
df_lineups = df_lineups.drop(df_lineups.columns[[0]], axis=1)
df_lineups = df_lineups.drop(['Cash', 'GPP', 'Total Proj', 'Total Salary'], 1)
df_lineups = df_lineups.drop_duplicates()

# loading the salaries so can pull the ID numbers
salary_path = dk_salaries.format(w=week, m=month, d=day, teamA=teamA, teamB=teamB)
df_salaries = pd.read_csv(salary_path, skiprows=7)

df_salaries = df_salaries.drop(df_salaries.columns[[0, 1, 2, 3, 4, 5, 6]], axis=1)
df_salaries = df_salaries.drop(['Name + ID', 'Position', 'TeamAbbrev', 'AvgPointsPerGame',
                                'Salary', 'Game Info'], 1)
print(df_salaries)


print(df_lineups)
print(df_salaries)

df_temp = pd.DataFrame()

dict_salaries = df_salaries.to_dict('series')

for k, v in dict_salaries['Name'].items():
    # print('k: ' + str(k) + ' v: ' + str(v))
    if v == 'Patrick Mahomes':
        print(dict_salaries['ID'][k])

# print(dict_salaries['Name'])