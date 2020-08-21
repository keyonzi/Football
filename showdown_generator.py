import pandas as pd
import numpy as np
import warnings
from player import Player
import copy
from datetime import datetime
from collections import Counter

from sys import argv

# timing the script
begin_time = datetime.now()

warnings.filterwarnings('ignore')

# making sure we can see the whole table
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 25)
pd.set_option('display.max_rows', 5000)


# fixing the names in DST so PFF matches with DK salaries
def fix_defense(player):
    if player.find('DST') != -1:
        player = player[:-3]
        print(player)
    return player


#           load data sources
pff_projections = 'data/PFFprojections2020Week1.csv'
df_pff = pd.read_csv(pff_projections)
print(df_pff.head())

# applying defesnse fix function
df_pff['playerName'] = df_pff['playerName'].apply(fix_defense)
print(df_pff.head())


dk_salaries = 'data/DKSalaries.csv'
df_dk = pd.read_csv(dk_salaries, skiprows=7) # because of format from DK, need to skip the first 7 rows
# cleaning up DKSalary download
df_dk = df_dk.drop(df_dk.columns[[0, 1, 2, 3, 4, 5, 6]], axis=1)
df_dk = df_dk.drop(['Name + ID', 'ID', 'Game Info'], 1)
print(df_dk.head())

#       add projection to salaries
df_pff_proj = df_pff[['playerName', 'fantasyPoints']]   # creating a new temp table to merge
df_pff_proj['Name'] = df_pff_proj['playerName']
df_dk = pd.merge(df_dk, df_pff_proj, on='Name', how='left')
df_dk = df_dk.drop(['playerName'], axis=1)
print(df_dk.head())

# making a list of the teams for use in checking lineups, based on DK download
print("MAKING LIST")
team_list = df_dk.TeamAbbrev.to_list()
set_team_list = set(team_list)
team_list = list(set_team_list)
# for i in range(len(team_list)):
#     print(team_list[i])
team_count = Counter(team_list)
# reset count to 0
team_count[team_list[0]] = 0
team_count[team_list[1]] = 0

# add captain multiplier to projections in table
df_dk["fantasyPoints"] = np.where(df_dk["Roster Position"] == 'CPT', round(df_dk['fantasyPoints'] * 1.5, 2), df_dk['fantasyPoints'])
df_dk['Roster'] = df_dk['Roster Position']
df_dk = df_dk.drop(['Roster Position'], axis=1)
print(df_dk.head())

# adding average fantasy points, for 0 projections
df_dk['fantasyPoints'] = df_dk['fantasyPoints'].fillna(df_dk['AvgPointsPerGame'])
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
    # print(i.ros)

for i in flx_list:
    print(i.__str__())
    # print(i.ros)

for i in players_list:
    print(i.__str__())

#for testing loop in recursion shortening captain list
# cpt_list = cpt_list[:12]
# flx_list = flx_list[:12]


#           time for the backtracking recursion hopefully

MAX_SALARY = 50000.0
max_total = 0
min_total = 0
lineup_count = 0
ROSTER_SIZE = 6

# create the dataframe that will hold all the lineups
df_final = pd.DataFrame(columns=['CPT', 'Proj', 'Salary', 'flx1', 'Proj1', 'Salary1', 'flx2', 'Proj2', 'Salary2',
                                 'flx3', 'Proj3', 'Salary3', 'flx4', 'Proj4', 'Salary4', 'flx5', 'Proj5', 'Salary5',
                                 'Total Proj', 'Total Salary'])


# in order to use backtracking, we need a valid check. so this is attempt to make a check function

def is_valid_roster(df_temp):
    df = df_temp

    # setting salary column
    df_salary = df[{'Salary', 'Salary1', 'Salary2', 'Salary3', 'Salary5'}]
    df_salary['Total Salary'] = df_salary.sum(axis=1)
    df.at[0, 'Total Salary'] = df_salary.at[0, 'Total Salary']

    # calculating the total projection
    df_proj = df[{'Proj', 'Proj1', 'Proj2', 'Proj3', 'Proj4', 'Proj5'}]
    df_proj['Total Proj'] = df_proj.sum(axis=1)
    df.at[0, 'Total Proj'] = df_proj.at[0, 'Total Proj']

    return df


df_temp = pd.DataFrame(columns=['CPT', 'Proj', 'Salary', 'flx1', 'Proj1', 'Salary1', 'flx2', 'Proj2', 'Salary2',
                                 'flx3', 'Proj3', 'Salary3', 'flx4', 'Proj4', 'Salary4', 'flx5', 'Proj5', 'Salary5',
                                 'Total Proj', 'Total Salary'])


def create_roster(captains, flex, df_temp, slot, df_final, team_counter):
    cpt = copy.deepcopy(captains)
    flex = copy.deepcopy(flex)
    df = df_temp
    df_f = df_final
    i = slot
    team_ct = team_counter

    # i feel like i shouldn't need this, cause im doing it to the temp object being passed in
    df['flx1'] = df['flx1'].astype(str)
    df['flx2'] = df['flx2'].astype(str)
    df['flx3'] = df['flx3'].astype(str)
    df['flx4'] = df['flx4'].astype(str)
    df['flx5'] = df['flx5'].astype(str)

    if i > 5:
        df = is_valid_roster(df)

        if team_ct[team_list[0]] == 6 or team_ct[team_list[1]] == 6:
            return df_f

        if df.at[0, 'Total Salary'] < MAX_SALARY:
            df_f = df_f.append(df, ignore_index=True)

        return df_f
    else:
        for j in range(len(flex)):
            if i == 0 and len(cpt) != 0:
                # setting the captain
                captain_player = cpt.pop(0)
                df.at[0, 'CPT'] = captain_player.name
                df.at[0, 'Proj'] = captain_player.proj * captain_player.mult
                df.at[0, 'Salary'] = captain_player.sal
                # counting the team for the player
                team_ct[captain_player.tm] += 1
                df_f = create_roster(cpt, flex, df, i + 1, df_f, team_ct)
                # backtracking the count
                team_ct[captain_player.tm] -= 1
            elif i > 0:
                # setting the flex players, trying all till none left
                flex_player = flex.pop(0)
                captain_player = df.iat[0, 0]
                # if the flex is same as captain, skip
                if captain_player == flex_player.name:
                    continue
                # counting the team for the player
                team_ct[flex_player.tm] += 1
                # setting the cells as the flex player
                df.at[0, 'flx{}'.format(i)] = flex_player.name
                df.at[0, 'Proj{}'.format(i)] = flex_player.proj * flex_player.mult
                df.at[0, 'Salary{}'.format(i)] = flex_player.sal
                df_f = create_roster(cpt, flex, df, i + 1, df_f, team_ct)

                # rolling back the change (backtracking
                df.at[0, 'flx{}'.format(i)] = np.nan
                df.at[0, 'Proj{}'.format(i)] = np.nan
                df.at[0, 'Salary{}'.format(i)] = np.nan
                team_ct[flex_player.tm] -= 1

    index = df_f.index
    print("Number of Rows: " + str(len(index)))
    return df_f

def modify_players(captains, flex):
    cpt_list = captains
    flx_list = flex

    for i in range(len(flx_list)):
        print(str(i) + ': ', end="")
        print(flx_list[i].name)

    choice = ' '
    while choice != '':
        choice = input('Selection: ')
        try:
            flex_p = flx_list[int(choice)]
            cpt_p = cpt_list[int(choice)]
        except IndexError:
            print('Selection not valid. Please try again.')
            continue
        except ValueError:
            choice = ''
            continue

        # updating your selected choice
        print("Player Selected: " + flex_p.name)
        print("Player Current Multiplier: " + str(flex_p.mult))
        mult = float(input("Enter multiplier: "))
        flex_p.set_multiplier(mult)
        cpt_p.set_multiplier(mult)

        flx_list = [flex_p if x == flex_p else x for x in flx_list]
        cpt_list = [cpt_list if x == cpt_list else x for x in cpt_list]

        print('Here is your player pool, select the player you wish to update:', end="\n\n")
        for i in range(len(flx_list)):
            print(str(i) + ': ', end="")
            print(flx_list[i].name)

        print()
    return cpt_list, flx_list

def edit_avail_players(captains, flex):
    cpt_list = captains
    flx_list = flex

    # print('Here is your player pool. 0 - to remove players, 1 - change modifiers, 2-exit and run generator:', end="\n\n")
    print("")
    choice = 0
    while choice != 3:
        choice = int(input('Choose your option: 0- to remove players, 1- change modifiers, 2- exit and run generator: '))
        if choice == 0:
            cpt_list, flx_list = remove_players(cpt_list, flx_list)
        elif choice == 1:
            cpt_list, flx_list = modify_players(cpt_list, flx_list)
        else:
            break

    return cpt_list, flx_list

def remove_players(captains, flex):
    cpt_list = captains
    flx_list = flex

    for i in range(len(flx_list)):
        print(str(i) + ': ', end="")
        print(flx_list[i].name)

    choice = ' '
    while choice != '':
        choice = input('Selection: ')
        try:
            del flx_list[int(choice)]
            del cpt_list[int(choice)]
        except ValueError:
            print('Main menu')
            print('_' * 50)
            continue
        except IndexError:
            print('Selection not valid. Please try again.')

        # deleting your selected choice

        print('Here is your player pool, select the player you wish to remove:', end="\n\n")
        for i in range(len(flx_list)):
            print(str(i) + ': ', end="")
            print(flx_list[i].name)

        print()
    return cpt_list, flx_list


# building player pool
cpt_list, flx_list = edit_avail_players(cpt_list, flx_list)

# a test to see that the object is set correctly
for i in cpt_list:
    print(i.__str__())
    # print(i.ros)

for i in flx_list:
    print(i.__str__())
    # print(i.ros)

for i in players_list:
    print(i.__str__())


slot_number = 0
df_test = pd.DataFrame()
df_final = create_roster(cpt_list, flx_list, df_temp, slot_number, df_final, team_count)
print('final product')
print(df_final)
print('-----------------')

# sort by projections
df_final = df_final.sort_values(by=['Total Proj'], ascending=False)
print(df_final)

print('-----------------')
# taking the top 20
df_final = df_final.head(n=20)
print(df_final)


# calculating how long it takes to run
print(datetime.now() - begin_time)