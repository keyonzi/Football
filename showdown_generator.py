import pandas as pd
import numpy as np
import warnings
from player import Player

from sys import argv

warnings.filterwarnings('ignore')

# making sure we can see the whole table
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 25)
pd.set_option('display.max_rows', 5000)


#           load data sources
pff_projections = 'data/PFFprojections2020Week1.csv'
df_pff = pd.read_csv(pff_projections)
print(df_pff.head())

dk_salaries = 'data/DKSalaries.csv'
df_dk = pd.read_csv(dk_salaries, skiprows=7) #because of format from DK, need to skip the first 7 rows
# cleaning up DKSalary download
df_dk = df_dk.drop(df_dk.columns[[0, 1, 2, 3, 4, 5, 6]], axis=1)
df_dk = df_dk.drop(['Name + ID', 'ID', 'Game Info'], 1)
print(df_dk.head())



#print(type(df_pff['Roster Position'][3]))




#       add projection to salaries
df_pff_proj = df_pff[['playerName', 'fantasyPoints']]   #creating a new temp table to merge
df_pff_proj['Name'] = df_pff_proj['playerName']
df_dk = pd.merge(df_dk,df_pff_proj,on='Name', how='inner')
df_dk = df_dk.drop(['playerName'], axis=1)
print(df_dk.head())

# add captain multiplier to projections in table
df_dk["fantasyPoints"] = np.where(df_dk["Roster Position"] == 'CPT', round(df_dk['fantasyPoints'] * 1.5, 2), df_dk['fantasyPoints'])
print(df_dk.head())


#       attempt to make player objects that can be used later to create lineups
play = Player()
print(df_dk.iloc[:1]['fantasyPoints'])
dak = Player(df_dk.iloc[:1]['Name'])
print("This is his name: " + dak.name)

players_list = []
for index, row in df_dk.iterrows():
    #print(row['fantasyPoints'], row['Name'])
    row['Name'] = Player(row['Name'], row['TeamAbbrev'])
    print(row['Name'].tm)


#           time for the backtracking recursion hopefully
MAX_SALARY = 50000

