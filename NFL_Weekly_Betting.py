import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as BS
import warnings
from sys import argv
import seaborn as sns
from matplotlib import pyplot as plt


warnings.filterwarnings('ignore')

# making sure we can see the whole table
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 25)
pd.set_option('display.max_rows', 5000)


# adding week numbers to each week's csv, then making a consolidated df with all the week numbers
WEEKLY_BASE_URL = 'data/weekly/2019/week{}.csv'

var_df = pd.DataFrame()


for week in range(1, 18):
    week_df = pd.read_csv(WEEKLY_BASE_URL.format(week))
    #let's also create a week column to keep track of the weeks
    week_df['Week'] = week
    var_df = pd.concat([var_df, week_df], ignore_index=True)
    #var_df = var_df.append(week_df, ignore_index=True) #another way of appending data

print(var_df)


# trying to get week numbers in the historical nfl odds df
NFL_ODDS_2019_CSV = 'data/nflodds2019.csv'

odds_df = pd.DataFrame()
odds_df = pd.read_csv(NFL_ODDS_2019_CSV)

# dates are currently floats. fixing them so they look like normal dates as strings
def get_week_num(date):
    date_string = ""
    if date[0] == '9':
        date_string = date[0] + "/" + date[1:3] + "/2019"
        return (date_string)
    elif date[0:2] == '10' or date[0:2] == '11' or date[0:2] == '12' :
        date_string = date[0:2] + "/" + date[2:4] + "/2019"
        return (date_string)
    elif date[0] == '1' or date[0] == '2':
        date_string = date[0] + "/" + date[1:3] + "/2020"
        return (date_string)


#converting values in date column to string so can run a function on them
odds_df['Date'] = odds_df['Date'].apply(lambda x: str(x))   #converting to string (easier to index)
odds_df['Date'] = odds_df['Date'].apply(lambda x: x.split('.')) #removing decimal
odds_df['Date'] = odds_df['Date'].apply(lambda x: (x[0]))   #getting first number (dont think i need anymore)
odds_df['Date'] = odds_df['Date'].apply(lambda x: get_week_num(x))  #applying my converion function to each row

# couldn't figure out how to change where pandas starts week, so instead converting strings in date/time, subtracting
# one from it, then running the week method and adding a column
odds_df['Date'] = pd.to_datetime(odds_df['Date']) + pd.DateOffset(-1)
odds_df['Week'] = odds_df['Date'].dt.week - 35  # making sure the week numbers match nfl

# trying to graph a linear regression model



# update team name to match weekly scoring data
odds_team_list = {
    'GreenBay':'GNB',
    'NewEngland': 'NWE',
    'Pittsburgh': 'PIT',
    'Dallas': 'DAL',
    'NYGiants': 'NYG',
    'LAChargers':'LAC',
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
    }

position_list = {
    'Wide Receiver':'WR',
    'Running Back': 'RB',
    'Tight End': 'TE',
    'Quarterback': 'QB',
    'Kicker': 'K'

}


# try to loop through to change the team names then concatenate at end

odds_df['Tm'] = odds_df['Team'].map(odds_team_list)
print(odds_df)

# trying to make columns for spread and over under

#using indexing you may be able to figure this out
print(odds_df.iloc[0:4,1])


# start working code. get rid of columns don't really need
odds_df_slim = odds_df.drop(['Date', 'Team', '1st', '2nd', '3rd', '4th', 'Final', 'Open', '2H'], axis=1, inplace=False)
odds_df_slim['Spread'] = '' # create empty new column
odds_df_slim['O/U'] = ''    # create empty new column
odds_dict = odds_df_slim.to_dict()  # create a dictionary based on the dataframe you already have

# loop through dictionary to update the values you need in the empty columns (now in dictionary form)
for k, v in odds_dict['ML'].items():        #check moneyline to figure out location of spread
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
print(odds_df_final)

# merge the two tables, the one for sports betting, and the weekly
result = pd.merge(var_df, odds_df_final, how='outer', on=['Week', 'Tm'])

# convert the series into non string, so math works on it
result['Spread'] = pd.to_numeric(result['Spread'], errors='coerce')
result['O/U'] = pd.to_numeric(result['O/U'], errors='coerce')

# remove non starting positions
result = result[result['Pos'].isin(['QB', 'TE', 'K', 'RB', 'WR'])]


# checking if there is an argument to the command for '--save' so will save table to a csv for upload

filename = ('2019_Weekly_Performance_Betting_Data').upper() + '.csv'
result.to_csv('data/{}'.format(filename))

try:
    if argv[1] == '--save':
        filename = ('2019_Weekly_Performance_Betting_Data').upper() + '.csv'
        result.to_csv('data/{}'.format(filename))
except IndexError:
    print(result.head())


#trying graphs...



sns.set_style('whitegrid')

#create a canvas with matplotlib
fig, ax = plt.subplots()
fig.set_size_inches(15, 10)


#print(type(result['O/U'][3]))

#basic regression scatter plot with trendline (FAIL)
plot = sns.regplot(
    x=result['Spread'],
    y=result['PPRFantasyPoints'],
    scatter=True,)

#plt.show()

# try ploting again with positions colored

#Make sure there is an adequete sample size

fig, ax = plt.subplots()
fig.set_size_inches(15, 10)


#example of regplot
plot = sns.regplot(
    x=result['Spread'],
    y=result['PPRFantasyPoints'],
    scatter=True)



qb_result = result[result['PassingYds'] > 0]
rb_result = result[result['RushingAtt'] > 5]
car_result = result[result['Tm'] == 'CAR']

fig, ax = plt.subplots()
fig.set_size_inches(15, 10)
sns.scatterplot(x='Spread',
                y='PPRFantasyPoints',
                data=rb_result,
                hue='Tm',
                size='O/U',
                )

plt.show()
#for testing single graph
sns.lmplot(data=car_result, x='Spread', y='PPRFantasyPoints', hue='Pos', height=10, fit_reg=True)


# trying heat map stuff

def make_heatmap(df):
    fig, ax = plt.subplots(); fig.set_size_inches(7, 5);

    mask = np.zeros_like(df.corr(), dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    heatmap = sns.heatmap(df.corr(), mask=mask, cmap=sns.diverging_palette(0, 230))
    return heatmap

#make_heatmap(result)


#goes through every team and plots points against spread (use later)
for k, v in odds_team_list.items():
    team_result = result[result['Tm'] == v]
    sns.lmplot(data=team_result, x='Spread', y='PPRFantasyPoints', hue='Pos', height=10, fit_reg=True)
    #make_heatmap(team_result)
    plt.title(v)    # adds title of team
    plt.show()

#loop for getting team and position based heatmap
for k, v in odds_team_list.items():
    team_result = result[result['Tm'] == v]
    for pos, abr in position_list.items():
        team_pos_result = team_result[team_result['Pos'] == abr]
        make_heatmap(team_pos_result)
        plt.title(v)    # adds title of team
        plt.show()

# try sci-learn stuff

plt.show()

plt.show()