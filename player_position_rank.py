# Used this file to build position rankings in the main file as a test, so could just re-run this part of it
# Hopefully this will be useful down the road for classification problems

import pandas as pd
import numpy as np
import warnings


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

for year in range(2007, 2008):
    for week in range(1, 2):
        week_df = pd.read_csv(WEEKLY_BASE_URL.format(year, week))
        # let's also create a week column to keep track of the weeks
        week_df['Week'] = week
        week_df['Year'] = year
        week_df['Player'] = week_df['Player'].map(lambda x: x.replace('.', '').replace("\'", ""))

        df_qb = week_df[week_df['Pos'] == 'QB']
        df_te = week_df[week_df['Pos'] == 'TE']
        df_k = week_df[week_df['Pos'] == 'K']
        df_rb = week_df[week_df['Pos'] == 'RB']
        df_wr = week_df[week_df['Pos'] == 'WR']

        # getting the position rankings
        df_qb = df_qb.sort_values(by=['PPRFantasyPoints'], ascending=False)
        df_qb = df_qb.reset_index(drop=True)
        df_qb['PosRank'] = df_qb.index + 1
        df_qb['Week'] = week
        df_qb['Year'] = year
        df_qb['Player'] = df_qb['Player'].map(lambda x: x.replace('.', '').replace("\'", ""))
        var_df = pd.concat([var_df, df_qb], ignore_index=True)
        # var_df = var_df[var_df['Pos'].isin(['QB', 'TE', 'K', 'RB', 'WR'])]

        # getting the position rankings
        df_te = df_te.sort_values(by=['PPRFantasyPoints'], ascending=False)
        df_te = df_te.reset_index(drop=True)
        df_te['PosRank'] = df_te.index + 1
        df_te['Week'] = week
        df_te['Year'] = year
        df_te['Player'] = df_te['Player'].map(lambda x: x.replace('.', '').replace("\'", ""))
        var_df = pd.concat([var_df, df_te], ignore_index=True)
        # var_df = var_df[var_df['Pos'].isin(['QB', 'TE', 'K', 'RB', 'WR'])]

        # getting the position rankings
        df_k = df_k.sort_values(by=['PPRFantasyPoints'], ascending=False)
        df_k = df_k.reset_index(drop=True)
        df_k['PosRank'] = df_k.index + 1
        df_k['Week'] = week
        df_k['Year'] = year
        df_k['Player'] = df_k['Player'].map(lambda x: x.replace('.', '').replace("\'", ""))
        var_df = pd.concat([var_df, df_k], ignore_index=True)
        # var_df = var_df[var_df['Pos'].isin(['QB', 'TE', 'K', 'RB', 'WR'])]

        # getting the position rankings
        df_rb = df_rb.sort_values(by=['PPRFantasyPoints'], ascending=False)
        df_rb = df_rb.reset_index(drop=True)
        df_rb['PosRank'] = df_rb.index + 1
        df_rb['Week'] = week
        df_rb['Year'] = year
        df_rb['Player'] = df_rb['Player'].map(lambda x: x.replace('.', '').replace("\'", ""))
        var_df = pd.concat([var_df, df_rb], ignore_index=True)
        # var_df = var_df[var_df['Pos'].isin(['QB', 'TE', 'K', 'RB', 'WR'])]

        # getting the position rankings
        df_wr = df_wr.sort_values(by=['PPRFantasyPoints'], ascending=False)
        df_wr = df_wr.reset_index(drop=True)
        df_wr['PosRank'] = df_wr.index + 1
        df_wr['Week'] = week
        df_wr['Year'] = year
        df_wr['Player'] = df_wr['Player'].map(lambda x: x.replace('.', '').replace("\'", ""))
        var_df = pd.concat([var_df, df_wr], ignore_index=True)
        # var_df = var_df[var_df['Pos'].isin(['QB', 'TE', 'K', 'RB', 'WR'])]

var_df = var_df[var_df['Pos'].isin(['QB', 'TE', 'K', 'RB', 'WR'])]
var_df = var_df.replace({'Tm': {'STL': 'LAR', 'SDG': 'LAC'}}) # compensate for acronyms changing over time

print(var_df)



