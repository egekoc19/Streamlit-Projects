import streamlit as st
import pandas as pd
import numpy as np
import datetime
import base64
import matplotlib.pyplot as plt


st.title("Looking at How NBA Shots are Changing")
st.write("In this app we are going to take a look at how 3 point shots changed from 1980 ")
st.write(" Here is the data source 👉 [How 3p shooting is changing NBA by David Lancharro](https://www.kaggle.com/datasets/drgilermo/nba-players-stats)")

url1= 'https://raw.githubusercontent.com/egekoc19/Streamlit-Projects/main/Seasons_Stats.csv'
st.write("Our table looks like this: ")
fullstats = pd.read_csv(url1,index_col=[0])
st.write(fullstats.head(100))
st.code("""
fullstats = pd.read_csv("Seasons_Stats.csv",index_col=[0])
""")
st.markdown('##')


st.write('TOT means “Two Other Teams” meaning that player played for two (or more) teams that season. So we are going to ignore those. We will also fillna() function to get rid of NaN values')
st.write("Also, since 3 point shot was made official in 1980, we are only going to look at the data from that year. We will also change the year column to date format")
fullstats = fullstats[fullstats.Tm != 'TOT']
stats = fullstats.loc[fullstats['Year'] > 1979]
stats= stats.fillna(0)
stats['Year'] = pd.to_datetime(stats['Year'], format='%Y').dt.year
st.write(stats.head(100))
st.code("""
fullstats = fullstats[fullstats.Tm != 'TOT']
stats= stats.fillna(0)
stats = fullstats.loc[fullstats['Year'] > 1979]
stats['Year'] = pd.to_datetime(stats['Year'], format='%Y').dt.year
""")
st.markdown('##')

st.write("In order to see yearly changes, we use groupby function and select some stats that we will use")
comparison_list = ['Year','Pos', 'Player', 'G','FG','FGA', '3P', '3PA', '3P%', '2P','2PA','2P%','eFG%',]
comparison_stats_by_years = stats[comparison_list]
comparison_stats_by_years_sum = comparison_stats_by_years.groupby('Year',as_index = False).sum()
st.write(comparison_stats_by_years_sum)
st.code("""
comparison_list = ['Year','Pos', 'Player', 'G','FG','FGA', '3P', '3PA', '3P%', '2P','2PA','2P%','eFG%',]
comparison_stats = stats[comparison_list]
comparison_stats_by_years_sum = comparison_stats.groupby('Year',as_index = False).sum()
""")
st.markdown('##')

graph1_year = comparison_stats_by_years_sum['Year']
graph1_2pa = comparison_stats_by_years_sum['2PA']
graph1_3pa = comparison_stats_by_years_sum['3PA']
graph1_fga = comparison_stats_by_years_sum['FGA']

graph1 = plt.figure(num = 1, figsize=(8, 5))
plt.plot(graph1_year, graph1_2pa, label = '2 Point Attempted', color='red')
plt.plot(graph1_year, graph1_3pa, label = '3 Point Attempted', color = 'blue')
plt.plot(graph1_year, graph1_fga, label = 'Total Attempted', color='purple',
        )
plt.legend()
st.pyplot(graph1)

st.code("""
graph_year = comparison_stats_by_years_sum['Year']
graph1_2pa = comparison_stats_by_years_sum['2PA']
graph1_3pa = comparison_stats_by_years_sum['3PA']
graph1_fga = comparison_stats_by_years_sum['FGA']

graph1 = plt.figure(num = 3, figsize=(8, 5))
plt.plot(graph1_year, graph1_2pa, label = '2 Point Attempted', color='red')
plt.plot(graph1_year, graph1_3pa, label = '3 Point Attempted', color = 'blue')
plt.plot(graph1_year, graph1_fga, label = 'Total Attempted', color='purple',
        )
plt.legend()
st.pyplot(graph1)
""")

st.write("We can see that the amount of 3 point shots are increasing "
"(In 1995 and 2011 the seasons were not played completely. That is the reason of the sudden drops)")


st.markdown('##')

st.write("Now we are going to look at general averages data")
st.write(" Here is the data source 👉 [NBA League Averages - Totals](https://www.basketball-reference.com/leagues/NBA_stats_totals.html)")

url2 = 'https://raw.githubusercontent.com/egekoc19/Streamlit-Projects/main/general_stats.csv'
general_averages = pd.read_csv(url2,index_col=[0])
st.dataframe(general_averages)
st.write("We can look at how 3 point shot has been improving over seasons now.")

graph2_seasons = np.arange(1980,2022)
graph2_3pp = general_averages['3P%'].iloc[::-1]
graph2_fgg = general_averages['FG%'].iloc[::-1]

graph2 = plt.figure(num = 2, figsize=(10, 5))
plt.plot(graph2_seasons, graph2_3pp, label = '3P%', color = 'blue')
plt.plot(graph2_seasons, graph2_fgg, label = 'FG%(Field Goal)', color='purple',
        )
plt.legend()
st.pyplot(graph2)
st.write("Here we can see that the percentage of 3 point shot has been increasing steadily leaguewide.")
st.code("""
graph2_seasons = np.arange(1980,2022)
graph2_3pp = general_averages['3P%'].iloc[::-1]
graph2_fgg = general_averages['FG%'].iloc[::-1]

graph2 = plt.figure(num = 2, figsize=(10, 5))
plt.plot(graph2_seasons, graph2_3pp, label = '3P%', color = 'blue')
plt.plot(graph2_seasons, graph2_fgg, label = 'FG%(Field Goal)', color='purple',
        )
plt.legend()
st.pyplot(graph2)
""")
st.markdown("###")

st.write("We can look at the current season and see the current 3 point percentages of the teams")
st.write(" Here is the data source 👉 [Teams Shooting Dashboard General - NBA.com]https://www.nba.com/stats/teams/shots-general)")
url3 = 'https://raw.githubusercontent.com/egekoc19/Streamlit-Projects/main/21-22.csv'
current_season = pd.read_csv(url3, index_col=[0])
st.dataframe(current_season)
team_list = current_season['Team']
threeP = current_season['3P%']

graph3 = plt.figure(num = 3, figsize=(8, 5))
plt.scatter(team_list, threeP, label = '3P%', color='purple')
plt.legend()
plt.xticks(rotation=75, ha='right')
plt.grid()
st.pyplot(graph3)
st.code("""
current_season = pd.read_csv("21-22.csv", index_col=[0])
st.dataframe(current_season)
team_list = current_season['Team']
threeP = current_season['3P%']

graph3 = plt.figure(num = 3, figsize=(8, 5))
plt.scatter(team_list, threeP, label = '3P%', color='purple')
plt.legend()
plt.xticks(rotation=75, ha='right')
plt.grid()
st.pyplot(graph3)
""")



