
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as BS
import warnings

warnings.filterwarnings('ignore')

# making sure we can see the whole table
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 25)

# getting the projections from api and saving in a data frame
res = requests.get('https://www.fantasyfootballdatapros.com/api/projections')

if res.ok:
    data = res.json() #converting response from API to json response (list)

    #not sure I need df_values, but in video
    df_values = {
        'player_name': [],
        'pos': [],
        'team': [],
        'projection': []
    }

    df = pd.DataFrame(data)

    #separate dataframes for each position
    qb_df = df[df['pos'] == 'QB']
    rb_df = df[df['pos'] == 'RB']
    wr_df = df[df['pos'] == 'WR']
    te_df = df[df['pos'] == 'TE']

    #saving all projectsions to file (doubt need)
    df.to_csv('data/projections.csv')

print(df.head())

# ________getting adp for later use__________

scoring_input = input("Are you looking for type: [ppr, half, 2qb, standard] ")
if scoring_input == 'ppr':
    adp_url = "https://fantasyfootballcalculator.com/adp/ppr"
elif scoring_input == 'half':
    adp_url = 'https://fantasyfootballcalculator.com/adp/half-ppr'
elif scoring_input == '2qb':
    adp_url = 'https://fantasyfootballcalculator.com/adp/2qb'
else: adp_url = 'https://fantasyfootballcalculator.com/adp'

def get_adp(url):
    res = requests.get(url)  # grabs full HTML of the URL above based on letter
    soup = BS(res.content, 'html.parser')  # turns full HTML into a beutiful soup object
    table = soup.find('table')  # finds the section of the full player list
    table = str(table)
    adp_df = pd.read_html(table)[0]  # reads the string and turns into a list, and we grab the first one
    return adp_df

# gives you a whole df with adp from calc website
adp_df = get_adp(adp_url)

print(adp_df)

# attempt to remove kickers and defense from list
adp_df = adp_df[adp_df.Pos != 'DEF']
adp_df = adp_df[adp_df.Pos != 'PK']

print(adp_df)

replacement_players = []

def get_replacements(adp_df):
    positions = ['RB', 'WR', 'TE', 'QB']
    rounds = input("How many non kicker/defense starting spots do you have? ")
    size = input('What is the league size: ')
    # have to disable this feature until i can pull better projections
    #draft_pool_limit = int(rounds) * int(size)
    draft_pool_limit = 155
    print("We will limit the draft pool to " + str(draft_pool_limit) + " for our VOA calculations.")
    #draft_pool_limit = input('What do you want the draft pool limit to be? ')
    adp_df = adp_df[:int(draft_pool_limit)] #cuttind down list to top x amount of players
    print(adp_df)
    for pos in positions:
        pos_df = adp_df[adp_df['Pos'] == pos]
        replacement_play = pos_df.iloc[-1,:]    #grabbing the last guy of that position
        replacement_player = replacement_play['Name'] , replacement_play['Pos']   #grabbing the name of the player
        replacement_players.append(replacement_player)   #adding it to replacement player list


get_replacements(adp_df)
#rprint(replacement_players)

replacement_values = []

def find_VOA(projections):
    print(replacement_players)
    for player_name, position in replacement_players:
        if position == 'RB':
            replacement_player = rb_df[rb_df['player_name'].apply(lambda x: x.strip().lower()) == player_name.strip().lower()]
            replacement_value = replacement_player['projection'].max()
            replacement_values.append((position, replacement_value))
        if position == 'WR':
            replacement_player = wr_df[wr_df['player_name'].apply(lambda x: x.strip().lower()) == player_name.strip().lower()]
            replacement_value = replacement_player['projection'].max()
            replacement_values.append((position, replacement_value))
        if position == 'TE':
            replacement_player = te_df[te_df['player_name'].apply(lambda x: x.strip().lower()) == player_name.strip().lower()]
            replacement_value = replacement_player['projection'].max()
            replacement_values.append((position, replacement_value))
        if position == 'QB':
            replacement_player = qb_df[qb_df['player_name'].apply(lambda x: x.strip().lower()) == player_name.strip().lower()]
            replacement_value = replacement_player['projection'].max()
            replacement_values.append((position, replacement_value))



find_VOA(df)
print(replacement_values)

def add_VOA():
    for position, replacement_value in replacement_values:
        if position == 'RB':
            rb_df['VOR'] = rb_df.loc[:,'projection'] - replacement_value
        elif position == 'WR':
            wr_df['VOR'] = wr_df['projection'] - replacement_value
        elif position == 'TE':
            te_df['VOR'] = te_df['projection'] - replacement_value
        elif position == 'QB':
            qb_df['VOR'] = qb_df['projection'] - replacement_value

# run function to get VOA and add to individual tables
add_VOA()
df = pd.concat([wr_df, qb_df, rb_df, te_df]).sort_values(by=['VOR'], ascending = False).reset_index()
df.index = df.index + 1
print(df)
