import requests
from bs4 import BeautifulSoup as BS
import pandas as pd
import numpy as np
import re
import collections
from sys import argv

desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 25)
pd.set_option('display.max_rows', 5000)


dvoa_url = 'https://www.footballoutsiders.com/dvoa-ratings/{year}/week-{num}-dvoa-ratings'

# create a separate object for each year, and each week of the stats table for DVOA
section_list = []
run = True
dvoa_dict = collections.defaultdict(dict)
for year in range(2007, 2020):
    for week in range(1, 17):
        print("This is the year: " + str(year) + ' and week ' + str(week))
        if year == 2015 and week == 16:   # no tables this week for some reason (silly fix)
            run = not run
        if run:
            access_url = dvoa_url.format(year=year, num=week)
            res = requests.get(access_url)  # grabs full HTML of the URL above based on letter
            soup = BS(res.content, 'html.parser')  # turns full HTML into a BS object
            searchtext = re.compile(r'\s+please\s+use\s+the\s+following\s+format\s+', re.IGNORECASE)
            foundtext = soup.find('p', text=searchtext)
            # over the years, they change the page structure, this is to compensate
            if not foundtext:
                searchtext = re.compile(r'reason\s+unrelated\s+to\s+DVOA', re.IGNORECASE)
                foundtext = soup.find('strong', text=searchtext)
            if not foundtext:
                searchtext = re.compile(r'reason\s+unrelated\s+to\s+DVOA', re.IGNORECASE)
                foundtext = soup.find('b', text=searchtext)
            section = foundtext.findNext('table')  # finds the dvoa table
            # for some reason this one week, the above code doesn't work because table in a <center> tag
            if not section:
                foundtext = soup.find('center')
                section = foundtext.findNext('table')  # finds the dvoa table
            # print(section)
            section_list.append(section)
            dvoa_dict[year][week] = section
        run = True


dvoa_df = pd.DataFrame()

print('scooby')
# print(dvoa_dict[2018][1])

# adding year and week numbers, so can be merged easily with other data
for year, v in dvoa_dict.items():
    for week, w in dvoa_dict[year].items():
        table = str(w)
        df = pd.read_html(table, header=0)
        df[0]['Year'] = year
        df[0]['Week'] = week
        # read_html creates it as a list in the 0 index. so can't do dataframe functions on a list. this just converts
        df = df[0]
        # pre 2018 they had a 'header' in the middle of the table, which would break everything. this fixes it
        # if the header isn't there, it will move on

        # df = df.dropna(subset=['Unnamed: 0'], axis='rows')
        try:
            df = df.dropna(subset=['Unnamed: 0'], axis='rows')
        except KeyError:
            pass
        # renaming columns because in week one they don't have the 'defense' adjusted numbers yet. But for our purpose
        # it should be fine
        df = df.rename(columns={"TOTALVOA": "TOTALDVOA", "OFFENSEVOA": "OFFENSEDVOA", 'DEFENSEVOA': 'DEFENSEDVOA',
                                'WEIGHTEDVOA': 'WEIGHTEDDVOA', 'S.T.VOA': 'S.T.DVOA', 'WEIGHTED  DVOA': 'WEIGHTEDDVOA',
                                'TOTAL  DVOA': 'TOTALDVOA', 'OFFENSE DVOA': 'OFFENSEDVOA', 'DEFENSE DVOA': 'DEFENSEDVOA',
                                'S.T. DVOA': 'S.T.DVOA', 'TOTAL  DAVE': 'DAVE', 'OFF. DVOA': 'OFFENSEDVOA',
                                'DEF. DVOA': 'DEFENSEDVOA', 'WEI.  DVOA': 'WEIGHTEDDVOA'})
        # filtering out all of the DVOA rankings
        df = df.filter(regex='(VOA|Week|Year|TEAM|DAVE)')
        df = df.rename(columns={'DAVE': 'WEIGHTEDDVOA'})
        df = df.rename(columns={'WEI.DVOA':'WEIGHTEDDVOA'})
        print(df)
        dvoa_df = dvoa_df.append(df, ignore_index=False)

# removing these columns. Weighted is only used for certain years and only certain times of the year, so hard to make
# data match up. Same with non-adjust. Dave was dropped completely in 2008. So not useful
# put in try block, because when testing with smaller data sets, these columns may not exist
# dvoa_df = dvoa_df.drop(['WEIGHTEDDVOA', 'NON-ADJTOT VOA', 'TOTALDAVE', 'TOTAL DAVE'], 1)

try:
    dvoa_df = dvoa_df.drop(['TOTALDAVE'], 1)
    dvoa_df = dvoa_df.drop(['TOTAL DAVE'], 1)
    dvoa_df = dvoa_df.drop(['WEIGHTEDDVOA'], 1)
    dvoa_df = dvoa_df.drop(['NON-ADJTOT VOA'], 1)
except KeyError:
    pass

# changing team names so stay the same over the years, since teams move, or their abbreviations change.
dvoa_df = dvoa_df.replace({'TEAM': {'STL': 'LAR', 'LARM': 'LAR', 'SDC': 'LAC', 'LACH': 'LAC', 'GB': 'GNB', 'JAC': 'JAX',
                          'JAG': 'JAX', 'KC': 'KAN', 'NE': 'NWE', 'NO': 'NOR', 'SF': 'SFO', 'TB': 'TAM', 'SD': 'LAC'}})


# saving file, without parameter. Should remove later
filename = 'DVOA_Historical_Data'.upper() + '.csv'
dvoa_df.to_csv('data/{}'.format(filename))

try:
    if argv[1] == '--save':
        filename = ('DVOA_Historical_Data').upper() + '.csv'
        dvoa_df.to_csv('data/{}'.format(filename))
except IndexError:
    print(dvoa_df.head())
