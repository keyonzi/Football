import pandas as pd
import numpy as np
import warnings
from player import Player
import copy

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
df_dk['Roster'] = df_dk['Roster Position']
df_dk = df_dk.drop(['Roster Position'], axis=1)
print(df_dk.head())


#       attempt to make player objects that can be used later to create lineups
play = Player()
print(df_dk.iloc[:1]['fantasyPoints'])
dak = Player(df_dk.iloc[:1]['Name'])
print("This is his name: " + dak.name)
print("Is he used? " + str(dak.used))
dak.flip_used()
print("Is he used? " + str(dak.used))

players_list = []
cpt_list = []
flx_list = []
for index, row in df_dk.iterrows():
    # using "row['Name'] because it is creating the object and its called that person's name, then gets added to the
    # player_list above
    row['Name'] = Player(row['Name'], row['TeamAbbrev'], row['fantasyPoints'], row['Position'], row['Roster'],
                         row['Salary'], row['AvgPointsPerGame'])
    players_list.append(row['Name'])
    if row['Name'].ros == 'CPT':
        cpt_list.append(row['Name'])
    else:
        flx_list.append(row['Name'])


# a test to see that the object is set correctly
for i in cpt_list:
    print(i.__str__())
    print(i.ros)

for i in flx_list:
    print(i.__str__())
    print(i.ros)

for i in players_list:
    print(i.__str__())

#TODO: for testing loop in recursion shortening captatin list
cpt_list = cpt_list[:2]
flx_list = flx_list[:6]


#           time for the backtracking recursion hopefully
MAX_SALARY = 50000
max_total = 0
min_total = 0
lineup_count = 0
ROSTER_SIZE = 6

# example of how to add rows line by line
#df_final = pd.DataFrame(columns=['Test', 'Scooby', 'Doo'])
#df_final = df_final.append({'Test': i.name}, ignore_index=True)
# df_final.loc[1] = i.name you can also do it this way via loops
#print(df_final.head())

Roster = {'cpt': {'name': 'Bob', 'pro': 22.3, 'salary': 20},
     'flx1': {'name': 'Kim', 'pro': 22.3, 'salary': 10},
     'flx2': {'name': 'Sam', 'pro': 22.3, 'salary': 8},
    'flx3': {'name': 'Kim', 'pro': 10.3, 'salary': 30},
    'flx4': {'name': 'Kim', 'pro': 15, 'salary': 22},
    'flx5': {'name': 'Kim', 'pro': 22.3, 'salary': 5},
          }

# roster2 = {'cpt': ['bob', 'test'], 'pro': [22.4, 35.4], 'salary': [20, 15],
#         'flx1': ['scoob', 'daffy'], 'pro': [18.4, 12.4], 'salary': [10, 15],
#         'flx2': ['baby', 'zoe'], 'pro': [18.4, 12.4], 'salary': [10, 15],
#            }

# create the dataframe that will hold all the lineups
df_final = pd.DataFrame(columns=['CPT', 'Proj', 'Salary', 'flx1', 'Proj1', 'Salary1', 'flx2', 'Proj2', 'Salary2',
                                 'flx3', 'Proj3', 'Salary3', 'flx4', 'Proj4', 'Salary4', 'flx5', 'Proj5', 'Salary5',
                                 'Total Proj', 'Total Salary'])


# in order to use backtracking, we need a valid check. so this is attempt to make a check function

def is_valid_roster(df_temp):
    result = False
    df = df_temp
    if df.iat[0, 19] > 0:
        result = True
    return result

df_temp = pd.DataFrame(columns=['CPT', 'Proj', 'Salary', 'flx1', 'Proj1', 'Salary1', 'flx2', 'Proj2', 'Salary2',
                                 'flx3', 'Proj3', 'Salary3', 'flx4', 'Proj4', 'Salary4', 'flx5', 'Proj5', 'Salary5',
                                 'Total Proj', 'Total Salary'])

#df_temp = df_temp.astype({'flx1': 'string', 'flx2': 'string', 'flx3': 'string', 'flx4': 'string', 'flx5': 'string'}).dtypes
df_temp['flx1']= df_temp['flx1'].astype(str)

def create_roster(captains, flex, df_temp, slot, df_final):
    cpt = copy.deepcopy(captains)
    flex = copy.deepcopy(flex)
    df = df_temp
    df_f = df_final
    i = slot

    # i feel like i shouldn't need this, cause im doing it to the temp object being passed in
    df['flx1'] = df['flx1'].astype(str)
    df['flx2'] = df['flx2'].astype(str)
    df['flx3'] = df['flx3'].astype(str)
    df['flx4'] = df['flx4'].astype(str)
    df['flx5'] = df['flx5'].astype(str)



    # df = df.append({'CPT': cpt[5].name, 'Proj': cpt[5].proj, 'Salary': cpt[5].sal,
    #                 'flx1': flex[3].name, 'Proj1': flex[3].proj, 'Salary1': flex[3].sal,
    #                 'Total Salary': 0}, ignore_index=True)

    #   working on figuring out the stopping point fo the backtracking
    # if df.iat[0, 19] > 0:
    #     return


    #while cpt or flex is not empty:
    print(i)
    if i > 5:
        print('last check_____________')
        print(df)
        df_f = df_f.append(df)
        print(df_f)
        return df
    else:
        print("This is I LOOP:" + str(i))
        if cpt and flex:
            print('TRUE')
        while cpt and flex:
            print("This is the cpt COUNT " + str(len(cpt)))
            print("This is the flx COUNT " + str(len(flex)))
            if i == 0:
                captain_player = cpt.pop(0)
                print("this is your captain:")
                print(captain_player)
                #df = df.append({'CPT': captain_player.name, 'Proj': captain_player.proj, 'Salary': captain_player.sal}, ignore_index=True)
                df.at[0, 'CPT'] = captain_player.name
                df.at[0, 'Proj'] = captain_player.proj
                df.at[0, 'Salary'] = captain_player.sal
                df = create_roster(cpt, flex, df, i + 1, df_f)
                print("Captain Loop:")

                #i = i - 1
                #df = df.iloc[0:, 3:]
                print(df)

                #cpt = cpt.append(captain_player)

            else:
                flex_player = flex.pop(0)
                print('The is your FLEX PLAYER')
                print(flex_player)
                #df = df.append({'flx1': flex_player.name, 'Proj1': flex_player.proj, 'Salary1': flex_player.sal}, ignore_index=True)
                df.at[0, 'flx{}'.format(i)] = flex_player.name
                df.at[0, 'Proj{}'.format(i)] = flex_player.proj
                df.at[0, 'Salary{}'.format(i)] = flex_player.sal
                df = create_roster(cpt, flex, df, i + 1, df_f)

                #df.at[0, 'flx{}'.format(i)] = np.nan
                #df.at[0, 'Proj{}'.format(i)] = np.nan
                #df.at[0, 'Salary{}'.format(i)] = np.nan

                #i = i - 1
                print(i)

                # df = df.iloc[0:, 0:(i+2)]
                # print("flex loop")
                # print(df)
                #flex = flex.append(flex_player)


    # for i in range(len(cpt)):
    #     for j in range(len(flex)):
    #         df = df.append({'CPT': cpt[i].name, 'Proj': cpt[i].proj, 'Salary': cpt[i].sal,
    #                     'flx1': flex[j].name, 'Proj1': flex[j].proj, 'Salary1': flex[j].sal,
    #                     'Total Salary': j}, ignore_index=True)
    # if is_valid_roster(df):
    #     return

    #print(df)

    return df


slot_number = 0
df_test = pd.DataFrame()
df_test = df_test.append(create_roster(cpt_list, flx_list, df_temp, slot_number, df_final))
print('final product')
print(df_test)
#print(df_final.iloc[0,3:])
#df_final = df_final.iloc[0:, 3:] #this keeps the last part of the table

# df_final = df_final.append({'CPT': cpt_list[5].name, 'Proj': cpt_list[5].proj, 'Salary': cpt_list[5].sal,
#                 'flx1': flx_list[3].name, 'Proj1': flx_list[3].proj, 'Salary1': flx_list[3].sal,
#                 'Total Salary': 0}, ignore_index=True)

#df_final = df_final.iloc[0:1, (1+2):]


# print("flex loop")
# print(df_final)
#
# print(df_final.xs(['flx1','Proj1'], axis=1))
#df_final.at[0, 'flx1'] = np.nan
#print(type(df_final['Proj'][0]))
#print(df_final)

#create_roster(cpt_list, flx_list, df_temp, slot_number)

#print(df_final)



#test = pd.DataFrame.from_dict(roster2)
#print(test)

#def valid_linup(pos, lineup):



