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
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')

# making sure we can see the whole table
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 70000)


# adding week numbers to each week's csv, then making a consolidated df with all the week numbers
# YEAR_BASE = 'data/weekly/2/'
WEEKLY_BASE_URL = 'data/weekly/{}/week{}.csv'
var_df = pd.DataFrame()

for year in range(2007, 2020):
    for week in range(1, 18):
        week_df = pd.read_csv(WEEKLY_BASE_URL.format(year, week))
        # let's also create a week column to keep track of the weeks
        week_df['Week'] = week
        week_df['Year'] = year
        week_df['Player'] = week_df['Player'].map(lambda x: x.replace('.', '').replace("\'", ""))
        var_df = pd.concat([var_df, week_df], ignore_index=True)

var_df = var_df[var_df['Pos'].isin(['QB', 'TE', 'K', 'RB', 'WR'])]
var_df = var_df.replace({'Tm': {'STL': 'LAR', 'SDG': 'LAC'}}) # compensate for acronyms changing over time

print(var_df.head(50))


# loading historical projections
PROJECTIONS_FILE = 'data/Projections/{year}/FD_{year}_week_{week}.csv'
proj_df = pd.DataFrame()
for year in range(2015, 2020):
    for week in range(1, 17):
        proj_df_temp = pd.read_csv(PROJECTIONS_FILE.format(year=year, week=week))
        proj_df_temp['Week'] = week
        proj_df_temp['Year'] = year
        # proj_df_temp['Name'] = proj_df_temp['Name'].map(lambda x: x.lstrip('Jr\.').rstrip('Sr\.')
        #                                                .rstrip(' II').rstrip(' III'))
        proj_df_temp['Name'] = proj_df_temp['Name'].map(lambda x: x.replace('Jr.', '').replace('Sr.', '')
                                                        .replace(' II', '').replace(' III', '').replace('.', '')
                                                        .replace("\'", "").replace('Will Fuller V', 'Will Fuller'))

        proj_df = pd.concat([proj_df, proj_df_temp], ignore_index=True)

print('PROJECTIONS1')
print(proj_df.head(50))
proj_df = proj_df[proj_df['Position'].isin(['QB', 'TE', 'K', 'RB', 'WR', 'DST'])]
proj_df = proj_df.rename(columns={'Team': 'Tm', 'FantasyPointsDraftKings': 'DKProj'})
proj_df = proj_df[['Tm', 'Position', 'Name', 'Week', 'DKProj', 'Year', 'Opponent']]
proj_df = proj_df.rename(columns={'Name': 'Player'})
proj_df = proj_df.replace({'Tm': {'GB': 'GNB', 'NE': 'NWE', 'KC': 'KAN', 'SF': 'SFO', 'TB': 'TAM', 'NO': 'NOR',
                                  'LV': 'OAK'}}) # compensate for acronyms changing over time
proj_df = proj_df.replace({'Opponent': {'GB': 'GNB', 'NE': 'NWE', 'KC': 'KAN', 'SF': 'SFO', 'TB': 'TAM', 'NO': 'NOR',
                                  'LV': 'OAK'}}) # compensate for acronyms changing over time

# odds_df = odds_df.replace({'Tm': {'STL': 'LAR', 'SDG': 'LAC'}}) # compensate for acronyms changing over time

print('PROJECTIONS TABLE')
print(proj_df.head(200))


# loading historical DVOA data from 2007 - 2019
DVOA_FILE = 'data/DVOA_HISTORICAL_DATA.csv'
dvoa_df = pd.read_csv(DVOA_FILE)

print('velma')
print(dvoa_df.head(50))

# loading all of the odds from 2007 - 2020
NFL_ODDS_BASE = 'data/NFLOdds/nflodds{}.csv'
odds_df = pd.DataFrame()
for year in range(2007, 2020):
    odds_temp_df = pd.read_csv(NFL_ODDS_BASE.format(year), skip_blank_lines=True)
    odds_temp_df['Year'] = year
    odds_df = odds_df.append(odds_temp_df, ignore_index=True)

# get rid of all the emptry rows at bottom
odds_df = odds_df.dropna()


# dates are currently floats. fixing them so they look like normal dates as strings
def get_week_num(row):
    date = row['Date']
    year = row['Year']
    date_string = ""
    if date[0] == '9':
        date_string = date[0] + "/" + date[1:3] + "/" + str(year)
        return date_string
    elif date[0:2] == '10' or date[0:2] == '11' or date[0:2] == '12':
        if len(date) == 4:
            date_string = date[0:2] + "/" + date[2:4] + "/" + str(year)
            return date_string
        elif len(date) == 3:
            date_string = date[0] + "/" + date[1:3] + "/" + str(year + 1)
            return date_string
    elif date[0] == '1' or date[0] == '2':
        date_string = date[0] + "/" + date[1:3] + "/" + str(year + 1)
        return date_string


# converting values in date column to string so can run a function on them
odds_df['Date'] = odds_df['Date'].apply(lambda x: str(x))   # converting to string (easier to index)
odds_df['Date'] = odds_df['Date'].apply(lambda x: x.split('.')) # removing decimal
odds_df['Date'] = odds_df['Date'].apply(lambda x: (x[0]))   # getting first number (dont think i need anymore)
odds_df['Date'] = odds_df.apply(lambda x: get_week_num(x), axis=1)  # applying my conversion function to each row

# couldn't figure out how to change where pandas starts week, so instead converting strings in date/time, subtracting
# one from it, then running the week method and adding a column
odds_df['Date'] = pd.to_datetime(odds_df['Date']) + pd.DateOffset(-1)
odds_df['Week'] = odds_df['Date'].dt.week - 35      # making sure the week numbers match nfl
odds_df['Week'] = odds_df['Week'].apply(lambda x: (x + 52) if x < 0 else x)     # using all the data (past week 17)


# update team name to match weekly scoring data
odds_team_list = {
    'GreenBay': 'GNB',
    'NewEngland': 'NWE',
    'Pittsburgh': 'PIT',
    'Dallas': 'DAL',
    'NYGiants': 'NYG',
    'LAChargers': 'LAC',
    'Seattle': 'SEA',
    'Cincinnati': 'CIN',
    'Arizona': 'ARI',
    'Detroit': 'DET',
    'Carolina': 'CAR',
    'LARams': 'LAR',
    'Cleveland': 'CLE',
    'Tennessee': 'TEN',
    'Indianapolis': 'IND',
    'KansasCity': 'KAN',
    'Jacksonville': 'JAX',
    'Chicago': 'CHI',
    'Atlanta': 'ATL',
    'Minnesota': 'MIN',
    'Philadelphia': 'PHI',
    'Washington': 'WAS',
    'NYJets': 'NYJ',
    'Baltimore': 'BAL',
    'Miami': 'MIA',
    'SanFrancisco': 'SFO',
    'TampaBay': 'TAM',
    'Buffalo': 'BUF',
    'NewOrleans': 'NOR',
    'Oakland': 'OAK',
    'Denver': 'DEN',
    'Houston': 'HOU',
    'HoustonTexans': 'HOU',
    'St.Louis': 'STL',
    'SanDiego': 'SDG',
    'LosAngeles': 'LAR'
    }

team_num_list = {
    'GNB': 1,
    'NWE': 2,
    'PIT': 3,
    'DAL': 4,
    'NYG': 5,
    'LAC': 6,
    'SEA': 7,
    'CIN': 8,
    'ARI': 9,
    'DET': 10,
    'CAR': 11,
    'LAR': 12,
    'CLE': 13,
    'TEN': 14,
    'IND': 15,
    'KAN': 16,
    'JAX': 17,
    'CHI': 18,
    'ATL': 19,
    'MIN': 20,
    'PHI': 21,
    'WAS': 22,
    'NYJ': 23,
    'BAL': 24,
    'MIA': 25,
    'SFO': 26,
    'TAM': 27,
    'BUF': 28,
    'NOR': 29,
    'OAK': 30,
    'DEN': 31,
    'HOU': 32
    }

position_list = {
    'Wide Receiver':'WR',
    'Running Back': 'RB',
    'Tight End': 'TE',
    'Quarterback': 'QB',
    'Kicker': 'K',
    'Defense': 'DEF'

}

# try to loop through to change the team names then concatenate at end
odds_df['Tm'] = odds_df['Team'].map(odds_team_list)
odds_df = odds_df.replace({'Tm': {'STL': 'LAR', 'SDG': 'LAC'}}) # compensate for acronyms changing over time
# clean up the 'pk' values
odds_df = odds_df.replace('pk', '0')

# print('dafny')
# print(odds_df)

# trying to make columns for spread and over under
# start working code. get rid of columns don't really need
odds_df_slim = odds_df.drop(['Date', 'Team', '1st', '2nd', '3rd', '4th', 'Final', 'Open', '2H'], axis=1, inplace=False)
odds_df_slim['Spread'] = '' # create empty new column
odds_df_slim['O/U'] = ''    # create empty new column
odds_dict = odds_df_slim.to_dict()  # create a dictionary based on the dataframe you already have

# loop through dictionary to update the values you need in the empty columns (now in dictionary form)
for k, v in odds_dict['ML'].items():        # check moneyline to figure out location of spread
    if (k % 2) == 0 and v > 0:      # if k is even, then you have the visiting team,and can take it from there.
        odds_dict['O/U'][k] = odds_dict['Close'][k]
        odds_dict['Spread'][k] = odds_dict['Close'][k+1]
        odds_dict['O/U'][k+1] = odds_dict['Close'][k]
        odds_dict['Spread'][k+1] = '-' + odds_dict['Close'][k + 1]
    elif (k % 2) == 0 and v < 0:
        if odds_dict['ML'][k+1] < 0:
            if float(odds_dict['Close'][k]) > 2:
                odds_dict['Spread'][k] = odds_dict['Close'][k+1]
                odds_dict['O/U'][k] = odds_dict['Close'][k]
                odds_dict['O/U'][k + 1] = odds_dict['Close'][k]
                odds_dict['Spread'][k + 1] = odds_dict['Close'][k+1]
            else:
                odds_dict['Spread'][k] = odds_dict['Close'][k]
                odds_dict['O/U'][k] = odds_dict['Close'][k + 1]
                odds_dict['O/U'][k + 1] = odds_dict['Close'][k + 1]
                odds_dict['Spread'][k + 1] = odds_dict['Close'][k]
        else:
            odds_dict['O/U'][k] = odds_dict['Close'][k+1]
            odds_dict['Spread'][k] = odds_dict['Close'][k]
            odds_dict['O/U'][k + 1] = odds_dict['Close'][k+1]
            odds_dict['Spread'][k + 1] = '-' + odds_dict['Close'][k]

odds_df_final = pd.DataFrame.from_dict(odds_dict)

# merge the two tables, the one for sports betting, and the weekly
print('Scooby Doo')
print(var_df.head(50))
print('then...')
print(odds_df_final.head(50))


# added this here so I can quickly spot check and make sure all the players match up with betting lines since 2007


filename = ('var_df').upper() + '.csv'
var_df.to_csv('data/{}'.format(filename))

filename = ('odds_df_final').upper() + '.csv'
odds_df_final.to_csv('data/{}'.format(filename))

filename = ('dvoa_df').upper() + '.csv'
dvoa_df.to_csv('data/{}'.format(filename))

filename = ('odds_df').upper() + '.csv'
odds_df.to_csv('data/{}'.format(filename))

filename = ('proj_df').upper() + '.csv'
proj_df.to_csv('data/{}'.format(filename))

result = pd.merge(var_df, odds_df_final, how='inner', on=['Week', 'Tm', 'Year'])
result = pd.merge(result, dvoa_df, how='inner', on=['Week', 'Tm', 'Year'])
result = pd.merge(result, proj_df, how='inner', on=['Week', 'Tm', 'Year', 'Player'])
result = result.drop(['Unnamed: 0'], 1)

filename = ('merge_result_check_DVOA').upper() + '.csv'
result.to_csv('data/{}'.format(filename))

print('STEFFI DOO!')

# convert the series into non string, so math works on it
result['Spread'] = pd.to_numeric(result['Spread'], errors='coerce')
result['O/U'] = pd.to_numeric(result['O/U'], errors='coerce')

# remove non starting positions, drop standard and half cause they suck
result = result[result['Pos'].isin(['QB', 'TE', 'K', 'RB', 'WR'])]
result = result.drop(['StandardFantasyPoints', 'HalfPPRFantasyPoints', 'PassingYds', 'PassingTD', 'Int', 'PassingAtt',
                      'Cmp', 'RushingAtt', 'RushingYds', 'Rec', 'Tgt', 'ReceivingTD', 'FL', 'RushingTD',
                      'ReceivingYds'], axis=1)
print('merge result')
print(result.head(50))

result['pro_diff'] = result['PPRFantasyPoints'] - result['DKProj']
result['pro_diff%'] = result['pro_diff'] / result['DKProj']

# create columns for projected score, and % of fantasy points out of projected score
def project_score(row):
    if row['Spread'] < 0:
        return (row['O/U']/2) - (row['Spread']/2)
    elif row['Spread'] > 0:
        return (row['O/U'] / 2) + (row['Spread'] / 2)
    else:
        return row['O/U'] / 2


def get_next_score(row):
    week = row['Week']
    week = week + 1
    player = row['Player']

    if week == 17:
        return row['PPRFantasyPoints']

    answer = result.loc[(result.Player == player) & (result.Week == (week))]

    if answer.empty:
        return 0
    answer = answer.iloc[0].at['PPRFantasyPoints']

    return answer


def get_next_percent(row):
    week = row['Week']
    week = week + 1
    player = row['Player']

    if week == 17:
        return row['% of Score']

    answer = result.loc[(result.Player == player) & (result.Week == week)]

    if answer.empty:
        return 0
    answer = answer.iloc[0].at['% of Score']

    return answer

def get_opp_dvoa(row):
    opp = row['Opponent']
    week = row['Week']
    year = row['Year']
    answer = result.loc[(result.Tm == opp) & (result.Week == week) & (result.Year == year)]
    answer = answer.iloc[0].at['DEFENSEDVOA']

    return answer


# apply function to have 'next week' score in table as well, also add % of team score
result['PPRFantasyPoints_next'] = result.apply(get_next_score, axis=1)
result['Score'] = result.apply(project_score, axis=1)

# changing to % of projection, since technically won't have this number later, so lets see if works
result['% of Score'] = result['DKProj']/result['Score'] * 100
# result['% of Score'] = result['PPRFantasyPoints']/result['Score'] * 100
result['% of Score_next'] = result.apply(get_next_percent, axis=1)

result = result.replace({'Opponent': {'STL': 'LAR', 'SDG': 'LAC'}}) # compensate for acronyms changing over time
result['Opp DVOA'] = result.apply(get_opp_dvoa, axis=1)

# just playing around with looking at correlation for specific players
# lamar = result[result['Player'] == 'Julio Jones']
# print(lamar)
# print('For the position ' + lamar.iat[0, 0] + ' the p value for ML is: ', end='')
# print(pearsonr(lamar['PPRFantasyPoints'], lamar['ML']), end=' for O/U it is: ')
# print(pearsonr(lamar['PPRFantasyPoints'], lamar['O/U']))
# print(pearsonr(lamar['ML'], lamar['% of Score']))
# print(pearsonr(lamar['O/U'], lamar['% of Score']))
# print(pearsonr(lamar['Spread'], lamar['% of Score']))


# looking at beginning p values before digging deep
result = result.replace([np.nan, -np.nan], 0)  # remove inf values as a result of dividing by 0


for k, v in position_list.items():
    if v == 'K':
        break
    pos_result = result[result['Pos'] == v]
    print('For the position ' + str(k) + ' the p value for Total DVOA is: ', end='')
    print(pearsonr(pos_result['pro_diff'], pos_result['ML']), end=' for O/U it is: ')
    print(pearsonr(pos_result['pro_diff'], pos_result['O/U']))


print()
print()

#                           going to play around with 'learn', and 'train'

# replacing string for numbers (don't know why)
pos_map = {
    'RB': 1,
    'WR': 2,
    'TE': 3,
    'QB': 4,
     'K': 5,
    # 'DEF': 6
}

result['Pos_Num'] = result['Pos'].replace(pos_map)
result['Tm_Num'] = result['Tm'].replace(team_num_list)
print('test opp dvoa')
print(result.head(50))


# want to try running model for each team, even though doesn't look good overall (I think)
# removing for now, but can be used later, remember change 'team_result' to result, can make better later
# for k, v in odds_team_list.items():
#     team_result = result[result['Tm'] == v]

#               build a model based off betting info and fantasy_points next (barely know what i'm doing)

# TODO: temp switch to test only a couple teams. They would be the teams in the showdown. Prompt, or Automatically determined later
# manual testing hack for running for two teams
# team1_result = result.loc[(result['Tm'] == 'HOU')]
# team2_result = result.loc[(result['Tm'] == 'KAN')]
# team_result_temp = team1_result.append(team2_result, ignore_index=True)

# print(team_result_temp)

# result = result.loc[(result['Pos'] == 'RB')]

# going to try and run the numbers per player...I know its nuts, but going to try
result['Player_hash'] = pd.factorize(result['Player'])[0]

filename = ('result').upper() + '.csv'
result.to_csv('data/{}'.format(filename))

x = result[['ML',


                        'DEFENSEDVOA',
                        'OFFENSEDVOA',
                        'Spread',
                        'Score',
                        'O/U',
                        'Tm_Num',
                        'DKProj',
                        'Opp DVOA',
                        'Pos_Num',
                        # 'Player_hash',
                        # 'PPRFantasyPoints',
                        # '% of Score',
                        # 'Week',
                        ]].values



# y = result[['pro_diff%']].values
# y = result[['pro_diff']].values

y = result[['pro_diff']].values
# y = result[['PPRFantasyPoints_next']].values
# y = result[['% of Score_next']].values  # need to figure out exactly what I'm doing here.

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=4, test_size=0.2)
lr = LinearRegression()
scores = cross_val_score(lr, x, y, cv=9)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

x_corr = result[['ML',

                            'DEFENSEDVOA',
                            'OFFENSEDVOA',
                            'Spread',
                            'Score',
                            'O/U',
                            'Tm_Num',
                            'DKProj',
                            'Opp DVOA',
                            'Pos_Num',
                            # 'Player_hash',
                            # 'PPRFantasyPoints',
                            # '% of Score',
                            # 'Week',
                            ]].corr()
vif = pd.DataFrame(np.linalg.inv(x_corr.values), index= x_corr.index, columns=x_corr.columns)
vif_mask = np.zeros_like(vif, dtype=np.bool); vif_mask[np.triu_indices_from(vif_mask)] = True

print(vif.mask(vif_mask))

# score the model
model = lr.fit(x_train, y_train)
y_pred = model.predict(x_test)
results = pd.DataFrame({'Predicted': y_pred.flatten(), 'Actual': y_test.flatten()})

print('Adj. R^2 for our model for team ' + str(k) + ' is: ', model.score(x_test, y_test))
print('RMSE:', sqrt(mean_squared_error(y_test, y_pred)))

plt.scatter(y_test, y_pred)
plt.show()

# Going to try and see how it performs projecting score in week 2 of year
#  based on week 1 projections, and previous years
#   load data sources
PROJECTIONS_FILE_2020 = 'data/Projections/2020/FD_2020_week_1_trial.csv'
y_pro_df = pd.read_csv(PROJECTIONS_FILE_2020)
y_pro_df['Tm_Num'] = y_pro_df['Tm'].replace(team_num_list)

# creating new table with player hash and merging them for the predictive run

print('before')
print(y_pro_df.head())

temp_df = result[['Player', 'Player_hash']]
temp_df = temp_df.drop_duplicates()

# accidently removed rookies or any new players
# y_pro_df = pd.merge(y_pro_df, temp_df, how='left', on=['Player'])

print('post merge')
print(y_pro_df.head())


y_pro_df['Predicted2020_week1'] = model.predict(y_pro_df[[  #should the df be 'result'?
                              'ML',
                              'DEFENSEDVOA',
                              'OFFENSEDVOA',
                              'Spread',
                              'Score',
                              'O/U',
                              'Tm_Num',
                              'DKProj',
                              'Opp DVOA',
                              'Pos_Num',
                              # 'Player_hash',
                              # 'PPRFantasyPoints',
                              # '% of Score',
                              # 'Week',
                              ]].values)
print('this is where we are')


y_pro_df['pred%'] = y_pro_df['DKProj'] + y_pro_df['Predicted2020_week1']
y_pro_df['mult'] = y_pro_df['pred%'] / y_pro_df['DKProj']
print(y_pro_df)

# odds_df['Week'] = odds_df['Week'].apply(lambda x: (x + 52) if x < 0 else x)

filename = ('DFS_multipliers').upper() + '.csv'
y_pro_df.to_csv('data/{}'.format(filename))


#                   saving files for the future... maybe
# checking if there is an argument to the command for '--save' so will save table to a csv for upload

filename = ('Historical Weekly_Performance_Betting_Data').upper() + '.csv'
result.to_csv('data/{}'.format(filename))

try:
    if argv[1] == '--save':
        filename = ('Historical_Weekly_Performance_Betting_Data').upper() + '.csv'
        result.to_csv('data/{}'.format(filename))
except IndexError:
    print(result.head())


# # trying graphs...
# sns.set_style('whitegrid')
#
# # create a canvas with matplotlib
# fig, ax = plt.subplots()
# fig.set_size_inches(15, 10)
#
# # basic regression scatter plot with trendline (FAIL)
# plot = sns.regplot(
#     x=result['Spread'],
#     y=result['PPRFantasyPoints'],
#     scatter=True,)
#
# #plt.show()
#
# # try ploting again with positions colored
#
# #Make sure there is an adequete sample size
#
# fig, ax = plt.subplots()
# fig.set_size_inches(15, 10)
#
#
# #example of regplot
# plot = sns.regplot(
#     x=result['Spread'],
#     y=result['PPRFantasyPoints'],
#     scatter=True)

# qb_result = result[result['PassingYds'] > 0]
# rb_result = result[result['RushingAtt'] > 5]
# car_result = result[result['Tm'] == 'CAR']

# fig, ax = plt.subplots()
# fig.set_size_inches(15, 10)
# sns.scatterplot(x='Spread',
#                 y='PPRFantasyPoints',
#                 data=rb_result,
#                 hue='Tm',
#                 size='O/U',
#                 )
#
# plt.show()
# #for testing single graph
# sns.lmplot(data=car_result, x='Spread', y='PPRFantasyPoints', hue='Pos', height=10, fit_reg=True)


# trying heat map stuff

def make_heatmap(df):
    fig, ax = plt.subplots(); fig.set_size_inches(7, 5);

    mask = np.zeros_like(df.corr(), dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    heatmap = sns.heatmap(df.corr(), mask=mask, cmap=sns.diverging_palette(0, 230))
    return heatmap


# goes through every team and plots points against spread (use later)
def linear_regress_test(odds, xaxis='Spread', yaxis='PPRFantasyPoints'):
    # xaxis = type
    for k, v in odds.items():
        team_result = result[result['Tm'] == v]
        sns.lmplot(data=team_result, x=xaxis, y=yaxis, hue='Pos', height=10, fit_reg=True)
        plt.title(v)    # adds title of team
        plt.show()


def create_team_heatmaps(odds, positions):
    for k, v in odds.items():
        team_result = result[result['Tm'] == v]
        for pos, abr in positions.items():
            team_pos_result = team_result[team_result['Pos'] == abr]
            make_heatmap(team_pos_result)
            plt.title(v + ' ' + abr)  # adds title of team and position for heatmap
            plt.show()


# runs the heatmaps, save time by commenting out
# create_team_heatmaps(odds_team_list, position_list)

# runs linear regression graphs for every team, TIME SAVE. 'O/U','Spread', 'ML', and column are options
x_axis = 'O/U'
# can try running linear vs " % of Score " as well, leave blank for 'PPRFantasyPoints'
y_axis = 'PPRFantasyPoints'
linear_regress_test(odds_team_list, x_axis, y_axis)


# try sci-learn stuff

