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

pro_url = 'https://fantasydata.com/nfl/fantasy-football-weekly-projections?' \
          '&season={year}&seasontype=1&scope=2&scoringsystem=4&startweek={week}&endweek={week}'



pro_dict = collections.defaultdict(dict)
for year in range(2015, 2020):
    for week in range(1, 17):
        # print("This is the year: " + str(year) + ' and week ' + str(week))
        print('FD_' + str(year) + '_week_' + str(week))
        access_url = pro_url.format(year=year, week=week)
        print(access_url)
        # res = requests.get(access_url)  # grabs full HTML of the URL above based on letter
        # soup = BS(res.content, 'html.parser')  # turns full HTML into a BS object
        # searchtext = re.compile(r'Fantasy\s+Football\s+Projections', re.IGNORECASE)
        # foundtext = soup.find('h1', text=searchtext)
        # section = foundtext.findNext('table')  # finds the projection table
        # stats = soup.find('table', id='stats_grid')
        # # print(stats)
        # pro_dict[year][week] = stats


# for year, v in pro_dict.items():
#     for week, w in pro_dict[year].items():
#         table = str(w)
#         df = pd.read_html(table, header=0)
#         df[0]['Year'] = year
#         df[0]['Week'] = week
#         # read_html creates it as a list in the 0 index. so can't do dataframe functions on a list. this just converts
#         df = df[0]
#         # pre 2018 they had a 'header' in the middle of the table, which would break everything. this fixes it
#         # if the header isn't there, it will move on
#         print(df)