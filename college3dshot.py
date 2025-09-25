import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go  # Import Plotly graph objects separately
from datetime import datetime, timedelta
from random import randint
from courtCoordinates import CourtCoordinates
import numpy as np
import re
from bs4 import BeautifulSoup
import json

st.set_page_config(layout='wide',page_title='CBB Shot Analysis',page_icon='🏀')
def display_player_image(teamid, player_id, width2, caption2):
    # Try the image with -1 first
    image_url = f"https://storage.googleapis.com/cbb-image-files/player-headshots/{teamid}-{player_id}.png"
    
    # Check if the image with -1 is available
    response = requests.head(image_url)
    
    if response.status_code == 200:
        # If image is available, display it
        st.markdown(
            f'<div style="display: flex; flex-direction: column; align-items: center;">'
            f'<img src="{image_url}" style="width: {width2}px;">'
            f'<p style="text-align: center;">{caption2}</p>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        # If -1 image is not available, try the image with -2
        image_url = f"https://www.sports-reference.com/req/202302071/cbb/images/players/{player_id}-2.jpg"
        response = requests.head(image_url)
        
        if response.status_code == 200:
            # If image with -2 is available, display it
            st.markdown(
                f'<div style="display: flex; flex-direction: column; align-items: center;">'
                f'<img src="{image_url}" style="width: {width2}px;">'
                f'<p style="text-align: center;">{caption2}</p>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            # If neither -1 nor -2 image is available, use a fallback image
            image_url = "https://cdn.nba.com/headshots/nba/latest/1040x760/fallback.png"
            st.markdown(
                f'<div style="display: flex; flex-direction: column; align-items: center;">'
                f'<img src="{image_url}" style="width: {width2}px;">'
                f'<p style="text-align: center;">{"Image Unavailable"}</p>'
                f'</div>',
                unsafe_allow_html=True
            )
def display_player_image2(player_id, width2, caption2):
    # Try the image with -1 first
    image_url = f"https://storage.googleapis.com/cbb-image-files/team-logos/{player_id}.png"

    
    # Check if the image with -1 is available
    response = requests.head(image_url)
    
    if response.status_code == 200:
        # If image is available, display it
        st.markdown(
            f'<div style="display: flex; flex-direction: column; align-items: center;">'
            f'<img src="{image_url}" style="width: {width2}px;">'
            f'<p style="text-align: center;">{caption2}</p>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        # If -1 image is not available, try the image with -2
        image_url = f"https://cdn.ssref.net/req/202411081/tlogo/ncaa/{player_id}.png"

        response = requests.head(image_url)
        
        if response.status_code == 200:
            # If image with -2 is available, display it
            st.markdown(
                f'<div style="display: flex; flex-direction: column; align-items: center;">'
                f'<img src="{image_url}" style="width: {width2}px;">'
                f'<p style="text-align: center;">{caption2}</p>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            # If neither -1 nor -2 image is available, use a fallback image
            image_url = "https://cdn.nba.com/headshots/nba/latest/1040x760/fallback.png"
            st.markdown(
                f'<div style="display: flex; flex-direction: column; align-items: center;">'
                f'<img src="{image_url}" style="width: {width2}px;">'
                f'<p style="text-align: center;">{"Image Unavailable"}</p>'
                f'</div>',
                unsafe_allow_html=True
            )
# Define the API endpoint
st.markdown("""
    <h1 style="text-align: center; font-size: 100px;">CBB Shot Analysis</h1>
""", unsafe_allow_html=True)
st.sidebar.markdown("""
    <h1 style="text-align: center; font-size: 25px;">CBB Shot Analysis</h1>
""", unsafe_allow_html=True)
teamurl = st.secrets["stats"]["teamurl"]
response = requests.get(teamurl)
data = response.json()
teamdf = pd.DataFrame(data)
teamdf['team_display'] = teamdf['teamMarket'] + ' ' + teamdf['teamName']

# Get unique teams (now with combined 'teamMarket - teamName' string)
uniqueteams = teamdf['team_display'].unique()
uniqueteams = sorted(uniqueteams)
# Streamlit selectbox to select a team
filterby = st.selectbox('View by',['Team','Player'])
selected = False
if filterby == 'Team':
    genderselect = st.selectbox('',['Men','Women'])
    if genderselect == 'Men':
        genderselect = 'MALE'
    else:
        genderselect = 'FEMALE'
    teamselect = st.selectbox('Select a team', [''] + uniqueteams)
    if teamselect:
        teamparts = teamselect.split(' - ')
        teamname = teamparts[-1]
        teammarket = teamparts[0]
        teamdf2 = teamdf
        teamdf = teamdf[(teamdf['team_display'] == teamselect) & (teamdf['gender'] == str(genderselect))]
        teamid = teamdf['teamId'].iloc[0]
        division = teamdf['divisionId'].iloc[0]
        teamname = teamdf['team_display'].iloc[0]
        teammarket = teamdf['teamMarket'].iloc[0]
        color1 = teamdf['hexColor1'].iloc[0]
        color2 = teamdf['hexColor2'].iloc[0]
        # st.write(teamid)
    
        seasonurl = st.secrets["stats"]["seasonurl"]
        response = requests.get(seasonurl)
        data = response.json()
        seasonsdf = pd.DataFrame(data)
        seasonsdf = seasonsdf[seasonsdf['startYear'] >= 2018]
        seasonsdf['SelSeason'] = seasonsdf['competitionName'] 
        seasonsdf = seasonsdf[seasonsdf['gender'] == genderselect]
    
        uniqueseasons = seasonsdf['SelSeason'].unique()
        def extract_year(season):
            match = re.search(r'\d{4}', season)
            return int(match.group()) if match else 0
        
        # Sort seasons by year
        uniqueseasons = sorted(uniqueseasons, key=extract_year)
        selectseason = st.selectbox('Select a season',[''] + uniqueseasons)
        if selectseason:
            seasonparts = selectseason.split(' ')
            season = seasonparts[0]
            seasonsdf = seasonsdf[seasonsdf['competitionName'] == selectseason]
            competitionid = seasonsdf['competitionId'].iloc[0]
            # st.write(competitionid)
        
            url = st.secrets["stats"]["pbp"]
        
            # Send the GET request to the API
            response = requests.get(url)
        
            # Check if the request was successful
        
            # Parse the response as JSON
            data = response.json()
        
            # Convert JSON data to a pandas DataFrame
            shotdf = pd.DataFrame(data)
        
        
            # Display the DataFrame using Streamlit
            # print(df)
            # df.to_csv('cbbshotdata.csv')
        
        
            playersurl = st.secrets["stats"]["playersurl"] + f'?competitionId={competitionid}&divisionId={division}&teamId={teamid}&pass=false'
            response = requests.get(playersurl)
            data = response.json()
            playersdf = pd.DataFrame(data)
            playersdf2 = playersdf
            playersdf = playersdf[playersdf['teamId'] == teamid]
            playersdf['TeamName'] = playersdf['teamMarket'] + ' ' + playersdf['teamName']
            # st.write(playersdf)
            shotdf = shotdf.merge(playersdf[['playerId', 'fullName']], on='playerId', how='left')
            shotdf = shotdf.merge(playersdf[['teamId', 'TeamName']], on='teamId', how='left')
            teamdf2['TeamName'] = teamdf2['teamMarket'] + ' ' + teamdf2['teamName']
            shotdf = shotdf.merge(teamdf2[['teamId', 'TeamName']], left_on='awayId', right_on='teamId', how='left')
            player_id_to_name = playersdf2.set_index('playerId')['fullName'].to_dict()
            shotdf['assisterName'] = shotdf['assisterId'].map(player_id_to_name)
        
            # st.write(shotdf)
        
            # Now, check if the merged TeamName (from awayId) matches the existing homeId's teamName
            # If they match, re-merge using homeId instead of awayId
        
            # Create a temporary column to store the TeamName from the first merge
            shotdf['TeamName_temp'] = shotdf['TeamName_y']
        
            # Perform the second merge based on homeId for those where awayId teamName matches homeId
            shotdf.loc[shotdf['TeamName_temp'] == shotdf['TeamName_x'], 'TeamName_y'] = shotdf.merge(teamdf2[['teamId', 'TeamName']], left_on='homeId', right_on='teamId', how='left')['TeamName']
        
            # Drop temporary columns
            shotdf.drop(columns=['TeamName_temp', 'teamId_y'], inplace=True)
            shotdf.rename(columns={'TeamName_x': 'TeamName', 'TeamName_y': 'OpponentName'}, inplace=True)
            def get_player_names(lineup_str):
                # Split the homeLineupId string by dashes to get individual playerIds
                player_ids = map(int, lineup_str.split('-'))
                
                # Map playerIds to their fullNames using playerdf2
                player_names = playersdf2[playersdf2['playerId'].isin(player_ids)]['fullName'].values
                
                # Return the player names as a string, joined by commas
                return ', '.join(player_names)
        
            # Apply the function to the homeLineupId column to create the homelineup column
            # shotdf['homeLineup'] = shotdf['homeLineupId'].apply(get_player_names)
            # shotdf['awayLineup'] = shotdf['awayLineupId'].apply(get_player_names)
            uniqueplayers = shotdf['fullName'].unique()
            # playerselect = st.multiselect('Select players',uniqueplayers)
            # shotdf = shotdf[shotdf['fullName'].isin(playerselect)]
            # st.write(shotdf)
            # unqiuedates = shotdf['gameDate'].unique()
            shotdf.drop_duplicates(subset=['clock','gameDate','period'],inplace=True)
            shotdf['color1'] = color1
            shotdf['color2'] = color2
            selected = True
        
            # st.write(unqiuedates)

else:
    # st.warning('This list only contains draft prospects')
    genderselect = st.selectbox('',['Men','Women'])
    if genderselect == 'Men':
        genderselect = 'MALE'
    else:
        genderselect = 'FEMALE'
    seasonurl = st.secrets["stats"]["seasonurl"]
    response = requests.get(seasonurl)
    data = response.json()
    seasonsdf = pd.DataFrame(data)
    seasonsdf = seasonsdf[seasonsdf['startYear'] >= 2020]
    seasonsdf['SelSeason'] = seasonsdf['competitionName'] 
    seasonsdf = seasonsdf[seasonsdf['gender'] == genderselect]

    uniqueseasons = seasonsdf['SelSeason'].unique()
    def extract_year(season):
        match = re.search(r'\d{4}', season)
        return int(match.group()) if match else 0
    
    # Sort seasons by year
    uniqueseasons = sorted(uniqueseasons, key=extract_year)
    selectseason = st.selectbox('Select a season', [''] + uniqueseasons)
    selectdiv = st.selectbox('Select a division',[1,2,3])
    if selectseason:
        seasonparts = selectseason.split(' ')
        season = seasonparts[0]
        seasonsdf = seasonsdf[seasonsdf['competitionName'] == selectseason]
        competitionid = seasonsdf['competitionId'].iloc[0]
        # division = st.selectbox('Select a division',[1,2,3])
        playersurl = st.secrets["stats"]["playersurl"] + f'?competitionId={competitionid}&divisionId={selectdiv}&pass=false'
        response = requests.get(playersurl)
        data = response.json()
        playersdf = pd.DataFrame(data)
        playersdf['PlayerInfo'] = playersdf['fullName'] + ' - ' + playersdf['teamMarket'] + ' ' + playersdf['teamName']
        playersdf['TeamName'] = playersdf['teamMarket'] + ' ' + playersdf['teamName']
        uniqueplayers = playersdf['PlayerInfo'].unique()
        uniqueplayers = sorted(uniqueplayers)
        playerselect = st.selectbox('Select a player', [''] + uniqueplayers)
        if playerselect:
            playerparts = playerselect.split(' - ')
            playername = playerparts[0]
            playersdf = playersdf[(playersdf['fullName'] == playername) & (playersdf['TeamName'] == playerparts[-1])]
            playerid = playersdf['playerId'].iloc[0]

            playerstatsurl = st.secrets["stats"]["playerstatsurl"] + f'?competitionId={competitionid}&playerId={playerid}&scope=season&fields[]=mins&fields[]=fullName&pass=false'  
            response = requests.get(playerstatsurl)
            data = response.json()
            playerstats = pd.DataFrame(data)

            result = playerstats.groupby(['fullName', 'classYr', 'position'])[['mins','ptsScored','reb','ast','stl','tov','fga','fgm','fga3','fgm3','fta','ftm','pf']].mean().reset_index()

            result['fgPct'] = result['fgm'] / result['fga']
            result['fg3Pct'] = result['fgm3'] / result['fga3']
            result['ftPct'] = result['ftm'] / result['fta']

            result = result.round({'mins': 1, 'ptsScored': 1, 'reb': 1, 'ast': 1, 'stl': 1, 'tov': 1, 'fga': 1, 'fgm': 1, 'fga3': 1, 'fgm3': 1, 'fta': 1, 'ftm': 1, 'pf': 1, 'fgPct': 3, 'fg3Pct': 3, 'ftPct': 3})
            result = result.rename(columns={
                'fullName':'Name',
                'classYr':'Class',
                'position':'Position',
                'mins': 'MPG',
                'ptsScored': 'PPG',
                'reb': 'RPG',
                'ast': 'APG',
                'stl': 'SPG',
                'tov': 'TPG',
                'fga': 'FGA',
                'fgm': 'FGM',
                'fga3': '3PA',
                'fgm3': '3PM',
                'fta': 'FTA',
                'ftm': 'FTM',
                'pf': 'PF',
                'fgPct': 'FG%',
                'fg3Pct': '3P%',
                'ftPct': 'FT%'
            })

            teamid = playersdf['teamId'].iloc[0]
            playerteamurl = st.secrets["stats"]["playerteamurl"] + f'?teamid={teamid}&gender={genderselect}&competitionId={competitionid}&playerid={playerid}'
            response = requests.get(playerteamurl)
            data2 = response.json()
            playersdf2 = pd.DataFrame(data2)

            url = st.secrets["stats"]["pbp"] + f'?competitionId={competitionid}&playerId={playerid}'
        
            # Send the GET request to the API
            response = requests.get(url)
        
            # Check if the request was successful
        
            # Parse the response as JSON
            data = response.json()
        
            # Convert JSON data to a pandas DataFrame
            shotdf = pd.DataFrame(data)
            shotdf = shotdf.merge(playersdf[['playerId', 'fullName']], on='playerId', how='left')
            shotdf = shotdf.merge(playersdf[['teamId', 'TeamName']], on='teamId', how='left')
            teamdf['TeamName'] = teamdf['teamMarket'] + ' ' + teamdf['teamName']
            shotdf = shotdf.merge(teamdf[['teamId', 'TeamName']], left_on='awayId', right_on='teamId', how='left')
            player_id_to_name = playersdf2.set_index('playerId')['fullName'].to_dict()
            shotdf['assisterName'] = shotdf['assisterId'].map(player_id_to_name)
            teamid = shotdf['teamId_x'].iloc[0]
            teamdf = teamdf[teamdf['teamId'] == teamid]
            color1 = teamdf['hexColor1'].iloc[0]
            color2 = teamdf['hexColor2'].iloc[0]
            shotdf['color1'] = color1
            shotdf['color2'] = color2
            # st.write(shotdf)
        
            # st.write(shotdf)
        
            # Now, check if the merged TeamName (from awayId) matches the existing homeId's teamName
            # If they match, re-merge using homeId instead of awayId
        
            # Create a temporary column to store the TeamName from the first merge
            shotdf['TeamName_temp'] = shotdf['TeamName_y']
        
            # Perform the second merge based on homeId for those where awayId teamName matches homeId
            shotdf.loc[shotdf['TeamName_temp'] == shotdf['TeamName_x'], 'TeamName_y'] = shotdf.merge(teamdf[['teamId', 'TeamName']], left_on='homeId', right_on='teamId', how='left')['TeamName']
        
            # Drop temporary columns
            shotdf.drop(columns=['TeamName_temp'], inplace=True)
            shotdf.rename(columns={'TeamName_x': 'TeamName', 'TeamName_y': 'OpponentName'}, inplace=True)
            selected = True
            teamid = shotdf['possTeamId'].iloc[0]
            # unqiuedates = shotdf['gameDate'].unique()
            # unique_dates = sorted(shotdf['gameDate'].unique())
        
            # st.write(shotdf[['gameDate']])
            # st.write(unique_dates)
            # st.write(shotdf)
if selected:
    st.sidebar.header("Filter Options")
    teamcolors = st.sidebar.checkbox('Team Colors')
    make = st.sidebar.checkbox('Make Shot Path',value=True)
    miss = st.sidebar.checkbox('Miss Shot Path')
    
    # Filter by Success (Made or Missed shots)
    success_filter = st.sidebar.radio(
        "Shot Success", 
        options=["All", "Made", "Missed"], 
        index=0
    )
    
    # Filter by Shot Distance (Slider)
    shot_dist_min, shot_dist_max = st.sidebar.slider(
        "Shot Distance (ft)", 
        min_value=int(shotdf['shotDist'].min()), 
        max_value=int(shotdf['shotDist'].max()), 
        value=(int(shotdf['shotDist'].min()), int(shotdf['shotDist'].max()))
    )
    
    # Filter by Shot Type (Action Type and Sub Type)
    action_type_filter = st.sidebar.selectbox(
        "Shot Action Type", 
        options=["All"] + shotdf['actionType'].unique().tolist()
    )
    
    sub_type_filter = st.sidebar.multiselect(
        "Shot Sub Type", 
        options= shotdf['subType'].unique().tolist()
    )
    
    # Filter by Period Number
    period_filter = st.sidebar.selectbox(
        "Half", 
        options=["All"] + sorted(shotdf['periodNumber'].unique().tolist())
    )
    shotdf['clock'] = pd.to_datetime(shotdf['clock'], format='%H:%M:%S', errors='coerce').dt.time
    default_time = datetime.strptime("00:00:00", "%H:%M:%S").time()
    shotdf['clock'] = shotdf['clock'].fillna(default_time)
    
    # Streamlit app layout
    # Get the minimum and maximum times from the data
    min_time = min(shotdf['clock'])
    max_time = max(shotdf['clock'])
    
    # Streamlit sliders for selecting start and end time
    start_time, end_time = st.sidebar.slider(
        "Time", 
        min_value=min_time, 
        max_value=max_time, 
        value=(min_time, max_time), 
        format="HH:mm:ss"
    )
    
    shotclockmin, shotclockmax = st.sidebar.slider(
        "Shot Clock (seconds)", 
        min_value=int(shotdf['shotClock'].min()), 
        max_value=int(shotdf['shotClock'].max()), 
        value=(int(shotdf['shotClock'].min()), int(shotdf['shotClock'].max()))
    )
    # Filter by Game Result (Home Win or Away Win)
    game_result_filter = st.sidebar.radio(
        "Game Result", 
        options=["All", "Home Win", "Away Win"], 
        index=0
    )
    
    # Filter by Player Name
    if filterby == 'Team':
        player_name_filter = st.sidebar.multiselect(
            "Player Name", 
            options=shotdf['fullName'].unique().tolist()
        )
    
    
    # Filter by Opponent Name
    opponent_name_filter = st.sidebar.multiselect(
        "Opponent Name", 
        options= shotdf['OpponentName'].unique().tolist()
    )
    
    assisted_by_filter = st.sidebar.multiselect(
        "Assisted By", 
        options= shotdf['assisterName'].unique().tolist()
    )
    
    # Filter by Date Range (if gameDate is in datetime format)
    game_date_min, game_date_max = st.sidebar.date_input(
        "Game Date Range", 
        min_value=pd.to_datetime(shotdf['gameDate']).min().date(),
        max_value=pd.to_datetime(shotdf['gameDate']).max().date(),
        value=(pd.to_datetime(shotdf['gameDate']).min().date(), pd.to_datetime(shotdf['gameDate']).max().date())
    )
    
    # Applying filters to the data
    
    # Success filter
    if success_filter == "Made":
        shotdf = shotdf[shotdf['success'] == 1]
    elif success_filter == "Missed":
        shotdf = shotdf[shotdf['success'] == 0]
    
    # Shot Distance filter
    shotdf = shotdf[
        (shotdf['shotDist'] >= shot_dist_min) & 
        (shotdf['shotDist'] <= shot_dist_max)
    ]
    shotdf = shotdf[
        (shotdf['shotClock'] >= shotclockmin) & 
        (shotdf['shotClock'] <= shotclockmax)
    ]
    
    shotdf = shotdf[
        (shotdf['clock'] >= start_time) & 
        (shotdf['clock'] <= end_time)
    ]
    
    # Action Type filter
    if action_type_filter != "All":
        shotdf = shotdf[shotdf['actionType'] == action_type_filter]
    
    # Sub Type filter
    if sub_type_filter:
        shotdf = shotdf[shotdf['subType'].isin(sub_type_filter)]
    
    # Period filter
    if period_filter != "All":
        shotdf = shotdf[shotdf['periodNumber'] == period_filter]
    
    # Game Result filter (Home Win or Away Win)
    if game_result_filter == "Home Win":
        shotdf = shotdf[shotdf['didHomeWin'] == True]
    elif game_result_filter == "Away Win":
        shotdf = shotdf[shotdf['didHomeWin'] == False]
    
    # Player Name filter
    if filterby == 'Team':
        if player_name_filter:
            shotdf = shotdf[shotdf['fullName'].isin(player_name_filter)]
    
    # Opponent Name filter
    if opponent_name_filter:
        shotdf = shotdf[shotdf['OpponentName'].isin(opponent_name_filter)]
    
    if assisted_by_filter:
        shotdf = shotdf[shotdf['assisterName'].isin(assisted_by_filter)]
    # Game Date filter
    shotdf = shotdf[
        (pd.to_datetime(shotdf['gameDate']).dt.date >= game_date_min) &
        (pd.to_datetime(shotdf['gameDate']).dt.date <= game_date_max)
    ]
    court = CourtCoordinates(season)
    court_lines_df = court.get_coordinates()
    # st.write(court_lines_df)
    fig = px.line_3d(
            data_frame=court_lines_df,
            x='x',
            y='y',
            z='z',
            line_group='line_group_id',
            color='line_group_id',
            color_discrete_map={
                'court': 'black',
                'hoop': '#e47041',
                'net': '#D3D3D3',
                'backboard': 'gray',
                'backboard2': 'gray',
                'free_throw_line': 'black',
                'hoop2':'#D3D3D3',
                'free_throw_line2': 'black',
                'free_throw_line3': 'black',
                'free_throw_line4': 'black',
                'free_throw_line5': 'black',
            }
        )
    fig.update_traces(hovertemplate=None, hoverinfo='skip', showlegend=False)
    fig.update_traces(line=dict(width=6))
    court_perimeter_bounds = np.array([[-250, 0, -0.2], [250, 0, -0.2], [250, 450, -0.2], [-250, 450, -0.2], [-250, 0, -0.2]])
    
    # Extract x, y, and z values for the mesh
    court_x = court_perimeter_bounds[:, 0]
    court_y = court_perimeter_bounds[:, 1]
    court_z = court_perimeter_bounds[:, 2]
    
    # Add a square mesh to represent the court floor at z=0
    fig.add_trace(go.Mesh3d(
        x=court_x,
        y=court_y,
        z=court_z,
        color='#d2a679',
        # opacity=0.5,
        name='Court Floor',
        hoverinfo='none',
        showscale=False
    ))
    fig.update_layout(    
        margin=dict(l=20, r=20, t=20, b=20),
        scene_aspectmode="data",
        height=600,
        scene_camera=dict(
            eye=dict(x=1.3, y=0, z=0.7)
        ),
        scene=dict(
            xaxis=dict(title='', showticklabels=False, showgrid=False),
            yaxis=dict(title='', showticklabels=False, showgrid=False),
            zaxis=dict(title='',  showticklabels=False, showgrid=False, showbackground=False, backgroundcolor='#d2a679'),
        ),
        showlegend=False,
        legend=dict(
            yanchor='top',
            y=0.05,
            x=0.2,
            xanchor='left',
            orientation='h',
            font=dict(size=15, color='gray'),
            bgcolor='rgba(0, 0, 0, 0)',
            title='',
            itemsizing='constant'
        )
    )
    hover_data = shotdf.apply(lambda row: f"""
        <b>Player:</b> {row['fullName']}<br>
        <b>Date:</b> {row['gameDate']}<br>
        <b>Vs:</b> {row['OpponentName']}<br>
        <b>Half:</b> {row['period'][-1]}<br>
        <b>Time:</b> {row['clock']}<br>
        <b>Result:</b> {'Made' if row['success'] else 'Missed'}<br>
        <b>Shot Distance:</b> {row['shotDist']} ft<br>
        <b>Shot Type:</b> {row['actionType']} ({row['subType']})<br>
        <b>Shot Clock:</b> {row['shotClock']}<br>
        <b>Assisted by:</b> {row['assisterName']}<br>
    """, axis=1)
    shotdf['x'] = (-shotdf['x']*10)+250
    shotdf['y'] = (450-shotdf['y']*10)+10
    if teamcolors:
        shotdf['color'] = color2
    else:
        shotdf['color'] = np.where(shotdf['success'] == True, 'green', 'red')
    shotdf['symbol'] = np.where(shotdf['success'] == True, 'circle', 'cross')
    shotdf['size'] = np.where(shotdf['success'] == True, 10, 8)
    if st.sidebar.checkbox('Animated'):
        newdf = shotdf.copy()
        newdf = newdf[newdf['success'] == True]
        newdf = newdf[newdf['shotDist'] > 3]
        # if len(newdf) > 150: 
        #     st.error(f'Too many shots. Only showing first 150 shots.')
        #     newdf = newdf.head(150)
        # else:
        #     newdf = newdf
        if len(newdf) >= 100:
                default = 10
        elif len(newdf) >= 75:
            default = 8
        elif len(newdf) >= 50:
            default = 5
        elif len(newdf) >= 20:
            default = 3
        else:
            default = 1
        cl1, cl2 = st.columns(2)
        with cl1:
            shotgroup = st.number_input("Number of Shots Together", min_value=1, max_value=10, step=1, value=default)
        with cl2:
            speed = st.selectbox('Speed',['Fast','Medium','Slow'])
        if speed == 'Fast':
            delay = 0.00000000000000000000001
        elif speed == 'Medium':
            delay = 150
        elif speed == 'Slow':
            delay = 200
        court_perimeter_bounds = np.array([[-250, 0, 0], [250, 0, 0], [250, 450, 0], [-250, 450, 0], [-250, 0, 0]])
        
        # Extract x, y, and z values for the mesh
        court_x = court_perimeter_bounds[:, 0]
        court_y = court_perimeter_bounds[:, 1]
        court_z = court_perimeter_bounds[:, 2]
        
        # Add a square mesh to represent the court floor at z=0
        fig.add_trace(go.Mesh3d(
            x=court_x,
            y=court_y,
            z=court_z-1,
            color='#d2a679',
            opacity=1,
            name='Court Floor',
            hoverinfo='none',
            showscale=False
        ))
        hover_data = newdf.apply(lambda row: f"""
            <b>Player:</b> {row['fullName']}<br>
            <b>Date:</b> {row['gameDate']}<br>
            <b>Vs:</b> {row['OpponentName']}<br>
            <b>Half:</b> {row['period'][-1]}<br>
            <b>Time:</b> {row['clock']}<br>
            <b>Result:</b> {'Made' if row['success'] else 'Missed'}<br>
            <b>Shot Distance:</b> {row['shotDist']} ft<br>
            <b>Shot Type:</b> {row['actionType']} ({row['subType']})<br>
            <b>Shot Clock:</b> {row['shotClock']}<br>
            <b>Assisted by:</b> {row['assisterName']}<br>
        """, axis=1)
       
        court_perimeter_lines = court_lines_df[court_lines_df['line_id'] == 'outside_perimeter']
        three_point_lines = court_lines_df[court_lines_df['line_id'] == 'three_point_line']
        backboard = court_lines_df[court_lines_df['line_id'] == 'backboard']
        backboard2 = court_lines_df[court_lines_df['line_id'] == 'backboard2']
        freethrow = court_lines_df[court_lines_df['line_id'] == 'free_throw_line']
        freethrow2 = court_lines_df[court_lines_df['line_id'] == 'free_throw_line2']
        freethrow3 = court_lines_df[court_lines_df['line_id'] == 'free_throw_line3']
        freethrow4 = court_lines_df[court_lines_df['line_id'] == 'free_throw_line4']
        freethrow5 = court_lines_df[court_lines_df['line_id'] == 'free_throw_line5']
        hoop = court_lines_df[court_lines_df['line_id'] == 'hoop']
        hoop2 = court_lines_df[court_lines_df['line_id'] == 'hoop2']
        
        
        
        
        
        
        
        # Add court lines to the plot (3D scatter)
        fig.add_trace(go.Scatter3d(
            x=court_perimeter_lines['x'],
            y=court_perimeter_lines['y'],
            z=np.zeros(len(court_perimeter_lines)),  # Place court lines on the floor
            mode='lines',
            line=dict(color='black', width=6),
            name="Court Perimeter",
            hoverinfo='none'
        ))
        fig.add_trace(go.Scatter3d(
            x=hoop['x'],
            y=hoop['y'],
            z=hoop['z'],  # Place 3-point line on the floor
            mode='lines',
            line=dict(color='#e47041', width=6),
            name="Hoop",
            hoverinfo='none'
        ))
        fig.add_trace(go.Scatter3d(
            x=hoop2['x'],
            y=hoop2['y'],
            z=hoop2['z'],  # Place 3-point line on the floor
            mode='lines',
            line=dict(color='#D3D3D3', width=6),
            name="Backboard",
            hoverinfo='none'
        ))
        # Add the 3-point line to the plot
        fig.add_trace(go.Scatter3d(
            x=backboard['x'],
            y=backboard['y'],
            z=backboard['z'],  # Place 3-point line on the floor
            mode='lines',
            line=dict(color='grey', width=6),
            name="Backboard",
            hoverinfo='none'
        ))
        fig.add_trace(go.Scatter3d(
            x=backboard2['x'],
            y=backboard2['y'],
            z=backboard2['z'],  # Place 3-point line on the floor
            mode='lines',
            line=dict(color='grey', width=6),
            name="Backboard",
            hoverinfo='none'
        ))
        fig.add_trace(go.Scatter3d(
            x=three_point_lines['x'],
            y=three_point_lines['y'],
            z=np.zeros(len(three_point_lines)),  # Place 3-point line on the floor
            mode='lines',
            line=dict(color='black', width=6),
            name="3-Point Line",
            hoverinfo='none'
        ))
        fig.add_trace(go.Scatter3d(
            x=freethrow['x'],
            y=freethrow['y'],
            z=np.zeros(len(freethrow)),  # Place court lines on the floor
            mode='lines',
            line=dict(color='black', width=6),
            name="Court Perimeter",
            hoverinfo='none'
        ))
        fig.add_trace(go.Scatter3d(
            x=freethrow2['x'],
            y=freethrow2['y'],
            z=np.zeros(len(freethrow2)),  # Place court lines on the floor
            mode='lines',
            line=dict(color='black', width=6),
            name="Court Perimeter",
            hoverinfo='none'
        ))
        fig.add_trace(go.Scatter3d(
            x=freethrow3['x'],
            y=freethrow3['y'],
            z=np.zeros(len(freethrow3)),  # Place court lines on the floor
            mode='lines',
            line=dict(color='black', width=6),
            name="Court Perimeter",
            hoverinfo='none'
        ))
        fig.add_trace(go.Scatter3d(
            x=freethrow4['x'],
            y=freethrow4['y'],
            z=np.zeros(len(freethrow4)),  # Place court lines on the floor
            mode='lines',
            line=dict(color='black', width=6),
            name="Court Perimeter",
            hoverinfo='none'
        ))
        fig.add_trace(go.Scatter3d(
            x=freethrow5['x'],
            y=freethrow5['y'],
            z=np.zeros(len(freethrow5)),  # Place court lines on the floor
            mode='lines',
            line=dict(color='black', width=6),
            name="Court Perimeter",
            hoverinfo='none'
        ))
        x_values = []
        y_values = []
        z_values = []
        # dfmiss = df[df['SHOT_MADE_FLAG'] == 0]
        # df = df[df['SHOT_MADE_FLAG'] == 1]
        

        for index, row in newdf.iterrows():
            
            
        
            x_values.append(row['x'])
            # Append the value from column 'x' to the list
            y_values.append(row['y'])
            z_values.append(0)
        
        
        
        x_values2 = []
        y_values2 = []
        z_values2 = []
        import math
        for index, row in newdf.iterrows():
            # Append the value from column 'x' to the list
        
        
            x_values2.append(court.hoop_loc_x)
        
            y_values2.append(court.hoop_loc_y)
            z_values2.append(100)
        
        def calculate_distance(x1, y1, x2, y2):
            return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Function to generate arc points
        def generate_arc_points(p1, p2, apex, num_points=100):
            t = np.linspace(0, 1, num_points)
            x = (1 - t)**2 * p1[0] + 2 * (1 - t) * t * apex[0] + t**2 * p2[0]
            y = (1 - t)**2 * p1[1] + 2 * (1 - t) * t * apex[1] + t**2 * p2[1]
            z = (1 - t)**2 * p1[2] + 2 * (1 - t) * t * apex[2] + t**2 * p2[2]
            return x, y, z
        
        
       
        
        frames = []
        num_points = 200  # Increase this for more resolution
        segment_size = 20  # Number of points per visible segment
        
        # Function to process shots in batches
        def process_shots_in_batches(shotdf, batch_size=3):
            for batch_start in range(0, len(shotdf), batch_size):
                batch_end = min(batch_start + batch_size, len(shotdf))
                yield shotdf[batch_start:batch_end]
        
        # Generate frames for each batch
        for batch in process_shots_in_batches(newdf, batch_size=shotgroup):
            for t in np.linspace(0, 1, 8):  # Adjust for smoothness
                frame_data = []
                
                for _, row in batch.iterrows():
                    x1, y1 = int(row['x']), int(row['y'])
                    x2, y2 = court.hoop_loc_x, court.hoop_loc_y
                    p2 = np.array([x1, y1, 0])
                    p1 = np.array([x2, y2, 100])
        
                    # Arc height based on shot distance
                    h = (150 if row['shotDist'] <= 15 else
                         200 if row['shotDist'] <= 25 else
                         250 if row['shotDist'] <= 30 else
                         300 if row['shotDist'] <= 50 else
                         325)
                    apex = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2), h])
                    x, y, z = generate_arc_points(p2, p1, apex, num_points)
        
                    # Calculate the start and end of the moving segment
                    total_points = len(x)
                    start_index = int(t * (total_points - segment_size))
                    end_index = start_index + segment_size
        
                    # Ensure indices are within bounds
                    start_index = max(0, start_index)
                    end_index = min(total_points, end_index)
        
                    segment_x = x[start_index:end_index]
                    segment_y = y[start_index:end_index]
                    segment_z = z[start_index:end_index]
        
                    frame_data.append(go.Scatter3d(
                        x=segment_x, y=segment_y, z=segment_z,
                        mode='lines', line=dict(width=6, color=row['color']),
                        hoverinfo='text', hovertext=row.get('hover_text', '')
                    ))

                frames.append(go.Frame(data=frame_data))
        
        
        # Add an initial empty trace for layout
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[]))
        
        # Empty frame at the end for clearing the court
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[]))
        empty_frame_data = []
        for i in range(0,10):
            empty_frame_data.append(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='lines', line=dict(width=6, color='rgba(255, 0, 0, 0)')
        ))
        
        frames.append(go.Frame(data=empty_frame_data))
        
        
        # Add frames to the figure
        fig.frames = frames
        
        
        
        # Layout with animation controls
        fig.update_layout(
            updatemenus=[
                dict(type="buttons",
                     showactive=False,
                     buttons=[
                         dict(label="Play",
                              method="animate",
                              args=[None, {"frame": {"duration": delay, "redraw": True}, "fromcurrent": True}]),
                         dict(label="Pause",
                              method="animate",
                              args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
                     ])
            ],
            # # scene=dict(
            # #     # xaxis=dict(range=[-25, 25], title="X"),
            # #     # yaxis=dict(range=[-50, 50], title="Y"),
            #     zaxis=dict(range=[0, 175], title="Z"),
            # #     # aspectratio=dict(x=1, y=1, z=0.5),
            # # ),
        )

    else:
        fig.add_trace(go.Scatter3d(
            x=shotdf['x'],
            y=shotdf['y'],
            z=[0] * len(shotdf),  # Set z = 0 for all points
            mode='markers',
            marker=dict(size=8, color=shotdf['color'], opacity=0.6,symbol=shotdf['symbol']),
            hoverinfo='text',
            hovertext=hover_data
        ))
        
        dfmiss = shotdf[shotdf['success'] == False]
        df = shotdf[shotdf['success'] == True]
        if miss:
            x_values = []
            y_values = []
            z_values = []
            for index, row in dfmiss.iterrows():
                
                
            
                x_values.append(row['x'])
                # Append the value from column 'x' to the list
                y_values.append(row['y'])
                z_values.append(0)
            
            
            
            x_values2 = []
            y_values2 = []
            z_values2 = []
            for index, row in dfmiss.iterrows():
                # Append the value from column 'x' to the list
            
            
                x_values2.append(court.hoop_loc_x)
            
                y_values2.append(court.hoop_loc_y)
                z_values2.append(100)
            
            import numpy as np
            import plotly.graph_objects as go
            import streamlit as st
            import math
            def calculate_distance(x1, y1, x2, y2):
                """Calculate the distance between two points (x1, y1) and (x2, y2)."""
                return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            def generate_arc_points(p1, p2, apex, num_points=100):
                """Generate points on a quadratic Bezier curve (arc) between p1 and p2 with an apex."""
                t = np.linspace(0, 1, num_points)
                x = (1 - t)**2 * p1[0] + 2 * (1 - t) * t * apex[0] + t**2 * p2[0]
                y = (1 - t)**2 * p1[1] + 2 * (1 - t) * t * apex[1] + t**2 * p2[1]
                z = (1 - t)**2 * p1[2] + 2 * (1 - t) * t * apex[2] + t**2 * p2[2]
                return x, y, z
            
            # Example lists of x and y coordinates
            x_coords = x_values
            y_coords = y_values
            z_value = 0  # Fixed z value
            x_coords2 = x_values2
            y_coords2 = y_values2
            z_value2 = 100
            # st.write(shotdf)
            for i in range(len(dfmiss)):
                
            
                # if df['SHOT_MADE_FLAG'].iloc[i] == 1:
                #     s = 'circle-open'
                #     s2 = 'circle'
                #     size = 9
                #     color = 'green'
                # else:
                #     s = 'cross'
                #     s2 = 'cross'
                #     size = 10
                #     color = 'red'
                # date_str = df['GAME_DATE'].iloc[i]
                # game_date = datetime.strptime(date_str, "%Y%m%d")
                # formatted_date = game_date.strftime("%m/%d/%Y")
                # if int(df['SECONDS_REMAINING'].iloc[i]) < 10:
                #     df['SECONDS_REMAINING'].iloc[i] = '0' + str(df['SECONDS_REMAINING'].iloc[i])
                # hovertemplate= f"Date: {formatted_date}<br>Game: {df['HTM'].iloc[i]} vs {df['VTM'].iloc[i]}<br>Result: {df['EVENT_TYPE'].iloc[i]}<br>Shot Type: {df['ACTION_TYPE'].iloc[i]}<br>Distance: {df['SHOT_DISTANCE'].iloc[i]} ft {df['SHOT_TYPE'].iloc[i]}<br>Quarter: {df['PERIOD'].iloc[i]}<br>Time: {df['MINUTES_REMAINING'].iloc[i]}:{df['SECONDS_REMAINING'].iloc[i]}"
            
                if dfmiss['shotDist'].iloc[i] > 3:
                    x1 = x_coords[i]
                    y1 = y_coords[i]
                    x2 = x_coords2[i]
                    y2 = y_coords2[i]
                    # Define the start and end points
                    p2 = np.array([x1, y1, z_value])
                    p1 = np.array([x2, y2, z_value2])
                    
                    # Apex will be above the line connecting p1 and p2
                    distance = calculate_distance(x1, y1, x2, y2)
                    if dfmiss['shotDist'].iloc[i] > 50:
                        h = randint(255,305)
                    elif dfmiss['shotDist'].iloc[i] > 30:
                        h = randint(230,280)
                    elif dfmiss['shotDist'].iloc[i] > 25:
                        h = randint(180,230)
                    elif dfmiss['shotDist'].iloc[i] > 15:
                        h = randint(180,230)
                    else:
                        h = randint(130,160)
                    if teamcolors:
                        color = dfmiss['color1'].iloc[0]
                    else:
                        if dfmiss['success'].iloc[i] == False:
                            color = 'red'
                        else:
                            color = 'green'
                    apex = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2), h])  # Adjust apex height as needed
                    
                    # Generate arc points
                    row = dfmiss.iloc[i]
                    
                    # Create the hover data string for the current row
                    hover_label = f"""
                    <b>Player:</b> {row['fullName']}<br>
                    <b>Date:</b> {row['gameDate']}<br>
                    <b>Vs:</b> {row['OpponentName']}<br>
                    <b>Half:</b> {row['period'][-1]}<br>
                    <b>Time:</b> {row['clock']}<br>
                    <b>Result:</b> {'Made' if row['success'] else 'Missed'}<br>
                    <b>Shot Distance:</b> {row['shotDist']} ft<br>
                    <b>Shot Type:</b> {row['actionType']} ({row['subType']})<br>
                    <b>Shot Clock:</b> {row['shotClock']}<br>
                    <b>Assisted by:</b> {row['assisterName']}<br>
            
                    """
                    x, y, z = generate_arc_points(p1, p2, apex)
                    fig.add_trace(go.Scatter3d(
                                x=x, y=y, z=z,
                                mode='lines',
                                line=dict(width=8,color = color),
                                opacity =0.5,
                                # name=f'Arc {i + 1}',
                                hoverinfo='text',
                                hovertext=hover_label
                            ))
        if make:
            x_values = []
            y_values = []
            z_values = []
            for index, row in df.iterrows():
                
                
            
                x_values.append(row['x'])
                # Append the value from column 'x' to the list
                y_values.append(row['y'])
                z_values.append(0)
            
            
            
            x_values2 = []
            y_values2 = []
            z_values2 = []
            for index, row in df.iterrows():
                # Append the value from column 'x' to the list
            
            
                x_values2.append(court.hoop_loc_x)
            
                y_values2.append(court.hoop_loc_y)
                z_values2.append(100)
            
            import numpy as np
            import plotly.graph_objects as go
            import streamlit as st
            import math
            def calculate_distance(x1, y1, x2, y2):
                """Calculate the distance between two points (x1, y1) and (x2, y2)."""
                return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            def generate_arc_points(p1, p2, apex, num_points=100):
                """Generate points on a quadratic Bezier curve (arc) between p1 and p2 with an apex."""
                t = np.linspace(0, 1, num_points)
                x = (1 - t)**2 * p1[0] + 2 * (1 - t) * t * apex[0] + t**2 * p2[0]
                y = (1 - t)**2 * p1[1] + 2 * (1 - t) * t * apex[1] + t**2 * p2[1]
                z = (1 - t)**2 * p1[2] + 2 * (1 - t) * t * apex[2] + t**2 * p2[2]
                return x, y, z
            
            # Example lists of x and y coordinates
            x_coords = x_values
            y_coords = y_values
            z_value = 0  # Fixed z value
            x_coords2 = x_values2
            y_coords2 = y_values2
            z_value2 = 100
            # st.write(shotdf)
            for i in range(len(df)):
                
            
                # if df['SHOT_MADE_FLAG'].iloc[i] == 1:
                #     s = 'circle-open'
                #     s2 = 'circle'
                #     size = 9
                #     color = 'green'
                # else:
                #     s = 'cross'
                #     s2 = 'cross'
                #     size = 10
                #     color = 'red'
                # date_str = df['GAME_DATE'].iloc[i]
                # game_date = datetime.strptime(date_str, "%Y%m%d")
                # formatted_date = game_date.strftime("%m/%d/%Y")
                # if int(df['SECONDS_REMAINING'].iloc[i]) < 10:
                #     df['SECONDS_REMAINING'].iloc[i] = '0' + str(df['SECONDS_REMAINING'].iloc[i])
                # hovertemplate= f"Date: {formatted_date}<br>Game: {df['HTM'].iloc[i]} vs {df['VTM'].iloc[i]}<br>Result: {df['EVENT_TYPE'].iloc[i]}<br>Shot Type: {df['ACTION_TYPE'].iloc[i]}<br>Distance: {df['SHOT_DISTANCE'].iloc[i]} ft {df['SHOT_TYPE'].iloc[i]}<br>Quarter: {df['PERIOD'].iloc[i]}<br>Time: {df['MINUTES_REMAINING'].iloc[i]}:{df['SECONDS_REMAINING'].iloc[i]}"
            
                if df['shotDist'].iloc[i] > 3:
                    x1 = x_coords[i]
                    y1 = y_coords[i]
                    x2 = x_coords2[i]
                    y2 = y_coords2[i]
                    # Define the start and end points
                    p2 = np.array([x1, y1, z_value])
                    p1 = np.array([x2, y2, z_value2])
                    
                    # Apex will be above the line connecting p1 and p2
                    distance = calculate_distance(x1, y1, x2, y2)
                    if df['shotDist'].iloc[i] > 50:
                        h = randint(255,305)
                    elif df['shotDist'].iloc[i] > 30:
                        h = randint(230,280)
                    elif df['shotDist'].iloc[i] > 25:
                        h = randint(180,230)
                    elif df['shotDist'].iloc[i] > 15:
                        h = randint(180,230)
                    else:
                        h = randint(130,160)
                    if teamcolors:
                        color = df['color1'].iloc[0]
                    else:
                        if df['success'].iloc[i] == False:
                            color = 'red'
                        else:
                            color = 'green'
                    apex = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2), h])  # Adjust apex height as needed
                    
                    # Generate arc points
                    row = df.iloc[i]
                    
                    # Create the hover data string for the current row
                    hover_label = f"""
                    <b>Player:</b> {row['fullName']}<br>
                    <b>Date:</b> {row['gameDate']}<br>
                    <b>Vs:</b> {row['OpponentName']}<br>
                    <b>Half:</b> {row['period'][-1]}<br>
                    <b>Time:</b> {row['clock']}<br>
                    <b>Result:</b> {'Made' if row['success'] else 'Missed'}<br>
                    <b>Shot Distance:</b> {row['shotDist']} ft<br>
                    <b>Shot Type:</b> {row['actionType']} ({row['subType']})<br>
                    <b>Shot Clock:</b> {row['shotClock']}<br>
                    <b>Assisted by:</b> {row['assisterName']}<br>
            
                    """
                    x, y, z = generate_arc_points(p1, p2, apex)
                    fig.add_trace(go.Scatter3d(
                                x=x, y=y, z=z,
                                mode='lines',
                                line=dict(width=8,color = color),
                                opacity =0.5,
                                # name=f'Arc {i + 1}',
                                hoverinfo='text',
                                hovertext=hover_label
                            ))
    st.warning('Some games are not included in the data so some shots may be missing.')
    if filterby == 'Player':
        if genderselect == 'MALE':
            # playernameimg = playername.lower().replace(" ", "-")
            # url = f'https://www.nbadraft.net/players/{playernameimg}/'
            # response = requests.get(url)
            # soup = BeautifulSoup(response.text, 'html.parser')
            # player_image_tag = soup.find('div', class_='player-image').find('img')  # Find the <img> inside the div
            # player_image_url = player_image_tag['data-lazy-src'] if player_image_tag and 'data-lazy-src' in player_image_tag.attrs else player_image_tag['src'] if player_image_tag else 'N/A'
            # # st.markdown(f"""
            # #             <div style="text-align: center;">
            # #                 <img src="{player_image_url}" alt="{playername}" width="200"/>
            # #             </div>
            # #         """, unsafe_allow_html=True)
            # st.markdown(
            #     f'<div style="display: flex; flex-direction: column; align-items: center;">'
            #     f'<img src="{player_image_url}" style="width: 200px;">'
            #     f'<p style="text-align: center;">{playername}</p>'
            #     f'</div>',
            #     unsafe_allow_html=True
            # )
            display_player_image(teamid,playerid,200,f'{playername}')
            playernameimg = playername.lower().replace(" ", "-")
        else:
            playernameimg = playername.lower().replace(" ", "-")
            display_player_image(teamid,playerid,200,f'{playername}')

    
        # display_player_image(playernameimg,200,f'{playername}')
    else:
        teamnameimg = teammarket.lower().replace(" ","-")
        display_player_image2(teamid,300,'')
    fgperc = st.sidebar.checkbox('Hot Zones')
    # xbinnum = st.sidebar.number_input("Number of x bins:", min_value=10, max_value=50, value=20)
    # ybinnum = st.sidebar.number_input("Number of y bins:", min_value=10, max_value=50, value=10)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig)
    # st.write(len(shotdf))
    
    if fgperc:
    
        x_bins = np.linspace(-270, 270, 20)  # 30 bins along X axis (basketball court length)
        y_bins = np.linspace(-10, 450, 10)   # 20 bins along Y axis (basketball court width)
        
        # Create 2D histograms: one for shot attempts (total shots) and one for made shots
        shot_attempts, x_edges, y_edges = np.histogram2d(shotdf['x'], shotdf['y'], bins=[x_bins, y_bins])
        made_shots, _, _ = np.histogram2d(shotdf['x'][shotdf['success'] == True], shotdf['y'][shotdf['success'] == True], bins=[x_bins, y_bins])
        
        # Calculate the Field Goal Percentage (FG%) for each bin
        fg_percentage = np.divide(made_shots, shot_attempts, where=shot_attempts != 0) * 100  # Avoid division by zero
        
        # Normalize FG% for color mapping (to make sure it stays between 0 and 100)
        fg_percentage_normalized = np.clip(fg_percentage, 0, 100)  # Clamp FG% between 0 and 100
        
        # Calculate the center of each bin for plotting (bin centers)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        
        # Create a meshgrid of X and Y centers for 3D plotting
        X, Y = np.meshgrid(x_centers, y_centers)
        
        # Create hovertext to show FG% for each region
        hovertext = np.array([f'FG%: {fg}%' for fg in fg_percentage.flatten()]).reshape(fg_percentage.shape)
    
        
        # Create the 3D surface plot
        z_max = 100  # Replace with the desired limit
        Z = shot_attempts.T
        Z2 = Z *5
        Z2 = np.minimum(Z2, z_max)
        fig = go.Figure(data=go.Surface(
            z=Z2,  # Shot density (number of shots) as the Z-axis
            x=X,  # X values (bin centers)
            y=Y,  # Y values (bin centers)
            
            # Surface color based on Field Goal Percentage (FG%)
            surfacecolor=fg_percentage.T,  # Use FG% as the surface color
            
            colorscale='hot',  # Color scale based on FG% (you can change this to any scale)
            cmin=0,  # Minimum FG% for color scale
            cmax=100,  # Maximum FG% for color scale
            colorbar=dict(title='Field Goal %'),  # Color bar label
            showscale=False,  # Show the color scale/legend
            hoverinfo='none',  # Show text on hover
            # hovertext=hovertext  # Attach the hovertext showing FG%
        ))
        # df = shotdf
    
        # # Data transformation
        # df['hexX'] = (-df['hexX'] * 2) +50
        # df['hexY'] = 95 - (df['hexY'] * 2)
    
        # # Group by hexX and hexY to calculate shooting percentage
        # bin_stats = (
        #     df.groupby(['hexX', 'hexY'])
        #     .agg(total_shots=('success', 'size'), made_shots=('success', 'sum'))
        #     .reset_index()
        # )
        # bin_stats['shooting_percentage'] = (bin_stats['made_shots'] / bin_stats['total_shots']) * 100
        
        # # Expand each bin for the 3D stacking effect
        # stacked_hexX = []
        # stacked_hexY = []
        # stacked_z = []
        # stacked_colors = []
        # stacked_sizes = []
        # hover_texts = []
        # mult = 5
        # min_size = 10
        # max_size = 25
        
        # for _, row in bin_stats.iterrows():
        #     size = min_size + ((max_size - min_size) * (row['total_shots'] / bin_stats['total_shots'].max()))
        #     for z in range(int(row['total_shots'])):
        #         stacked_hexX.append(row['hexX'] * mult)
        #         stacked_hexY.append(row['hexY'] * mult)
        #         stacked_z.append(z)
        #         stacked_colors.append(row['shooting_percentage'])
        #         stacked_sizes.append(size)
        #         hover_texts.append(
        #             f"Shooting %: {row['shooting_percentage']:.1f}%<br>"
        #             f"{row['made_shots']}/{row['total_shots']}"
        #         )
        
        # # Normalize z values for height
        # max_height = 95
        # normalized_z = [(z / max(stacked_z)) * max_height for z in stacked_z]
        # custom_colorscale = [
        #     [0.0, 'blue'],      # Dark blue at cmin (0)
        #     [0.3, 'lightblue'],     # Light blue at 30%
        #     [0.5, 'white'],         # White at 50% (neutral)
        #     [0.7, 'lightcoral'],    # Light red at 70%
        #     [1.0, 'red']        # Dark red at cmax (100)
        # ]
        # fig = go.Figure(data=[
        #     go.Scatter3d(
        #         x=stacked_hexX,
        #         y=stacked_hexY,
        #         z=len(normalized_z)*[0],
        #         mode='markers',
        #         marker=dict(
        #             size=stacked_sizes,
        #             color=stacked_colors,
        #             colorscale=custom_colorscale,
        #             cmin=0,  # Min shooting percentage
        #             cmax=100,  # Max shooting percentage
        #             opacity=1,
        #             symbol='diamond',
        #             line=dict(width=0,color='black'),
        #             # colorbar=dict(title="Shooting %")
        #         ),
        #         hoverinfo='text',
        #         hovertext=hover_texts,
        #     )
        # ])
        
    else:
        x_bins = np.linspace(-270, 270, 20)  # 30 bins along X axis (basketball court length)
        y_bins = np.linspace(-10, 450, 10)   # 20 bins along Y axis (basketball court width)
    
        
        # Create 2D histogram to get shot density
        shot_density, x_edges, y_edges = np.histogram2d(shotdf['x'], shotdf['y'], bins=[x_bins, y_bins])
        
        # Calculate the center of each bin for plotting
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        
        # Create a meshgrid of X and Y centers for 3D plotting
        X, Y = np.meshgrid(x_centers, y_centers)
        Z = shot_density.T  # Transpose to match the correct orientation for plotting
        Z2 = Z*5
        z_max = 100  # Replace with the desired limit
    
        # Apply the limit to Z values
        Z2 = np.minimum(Z2, z_max)
        # Plot 3D shot density
        hovertext = np.array([f'Shots: {z}' for z in Z.flatten()]).reshape(Z.shape)
        fig = go.Figure(data=go.Surface(
            z=Z2,
            x=X,
            y=Y,
            colorscale='hot',  # You can choose different color scales
            colorbar=dict(title='Shot Density'),
            showscale=False  # Hide the color bar/legend
            ,hoverinfo='text',
            hovertext=hovertext
        ))
    court_perimeter_lines = court_lines_df[court_lines_df['line_id'] == 'outside_perimeter']
    three_point_lines = court_lines_df[court_lines_df['line_id'] == 'three_point_line']
    backboard = court_lines_df[court_lines_df['line_id'] == 'backboard']
    backboard2 = court_lines_df[court_lines_df['line_id'] == 'backboard2']
    freethrow = court_lines_df[court_lines_df['line_id'] == 'free_throw_line']
    freethrow2 = court_lines_df[court_lines_df['line_id'] == 'free_throw_line2']
    freethrow3 = court_lines_df[court_lines_df['line_id'] == 'free_throw_line3']
    freethrow4 = court_lines_df[court_lines_df['line_id'] == 'free_throw_line4']
    freethrow5 = court_lines_df[court_lines_df['line_id'] == 'free_throw_line5']
    hoop = court_lines_df[court_lines_df['line_id'] == 'hoop']
    hoop2 = court_lines_df[court_lines_df['line_id'] == 'hoop2']
    
    
    
    
    
    
    
    # Add court lines to the plot (3D scatter)
    fig.add_trace(go.Scatter3d(
        x=court_perimeter_lines['x'],
        y=court_perimeter_lines['y'],
        z=np.zeros(len(court_perimeter_lines)),  # Place court lines on the floor
        mode='lines',
        line=dict(color='white', width=6),
        name="Court Perimeter",
        hoverinfo='none'
    ))
    fig.add_trace(go.Scatter3d(
        x=hoop['x'],
        y=hoop['y'],
        z=hoop['z'],  # Place 3-point line on the floor
        mode='lines',
        line=dict(color='#e47041', width=6),
        name="Hoop",
        hoverinfo='none'
    ))
    fig.add_trace(go.Scatter3d(
        x=hoop2['x'],
        y=hoop2['y'],
        z=hoop2['z'],  # Place 3-point line on the floor
        mode='lines',
        line=dict(color='#D3D3D3', width=6),
        name="Backboard",
        hoverinfo='none'
    ))
    # Add the 3-point line to the plot
    fig.add_trace(go.Scatter3d(
        x=backboard['x'],
        y=backboard['y'],
        z=backboard['z'],  # Place 3-point line on the floor
        mode='lines',
        line=dict(color='grey', width=6),
        name="Backboard",
        hoverinfo='none'
    ))
    fig.add_trace(go.Scatter3d(
        x=backboard2['x'],
        y=backboard2['y'],
        z=backboard2['z'],  # Place 3-point line on the floor
        mode='lines',
        line=dict(color='grey', width=6),
        name="Backboard",
        hoverinfo='none'
    ))
    fig.add_trace(go.Scatter3d(
        x=three_point_lines['x'],
        y=three_point_lines['y'],
        z=np.zeros(len(three_point_lines)),  # Place 3-point line on the floor
        mode='lines',
        line=dict(color='white', width=6),
        name="3-Point Line",
        hoverinfo='none'
    ))
    fig.add_trace(go.Scatter3d(
        x=freethrow['x'],
        y=freethrow['y'],
        z=np.zeros(len(freethrow)),  # Place court lines on the floor
        mode='lines',
        line=dict(color='white', width=6),
        name="Court Perimeter",
        hoverinfo='none'
    ))
    fig.add_trace(go.Scatter3d(
        x=freethrow2['x'],
        y=freethrow2['y'],
        z=np.zeros(len(freethrow2)),  # Place court lines on the floor
        mode='lines',
        line=dict(color='white', width=6),
        name="Court Perimeter",
        hoverinfo='none'
    ))
    fig.add_trace(go.Scatter3d(
        x=freethrow3['x'],
        y=freethrow3['y'],
        z=np.zeros(len(freethrow3)),  # Place court lines on the floor
        mode='lines',
        line=dict(color='white', width=6),
        name="Court Perimeter",
        hoverinfo='none'
    ))
    fig.add_trace(go.Scatter3d(
        x=freethrow4['x'],
        y=freethrow4['y'],
        z=np.zeros(len(freethrow4)),  # Place court lines on the floor
        mode='lines',
        line=dict(color='white', width=6),
        name="Court Perimeter",
        hoverinfo='none'
    ))
    fig.add_trace(go.Scatter3d(
        x=freethrow5['x'],
        y=freethrow5['y'],
        z=np.zeros(len(freethrow5)),  # Place court lines on the floor
        mode='lines',
        line=dict(color='white', width=6),
        name="Court Perimeter",
        hoverinfo='none'
    ))
    court_perimeter_bounds = np.array([[-250, 0, -0.2], [250, 0, -0.2], [250, 450, -0.2], [-250, 450, -0.2], [-250, 0, -0.2]])
    
    # Extract x, y, and z values for the mesh
    court_x = court_perimeter_bounds[:, 0]
    court_y = court_perimeter_bounds[:, 1]
    court_z = court_perimeter_bounds[:, 2]
    
    # Add a square mesh to represent the court floor at z=0
    fig.add_trace(go.Mesh3d(
        x=court_x,
        y=court_y,
        z=court_z,
        color='black',
        # opacity=0.5,
        name='Court Floor',
        hoverinfo='none',
        showscale=False
    ))
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        scene_aspectmode="data",
        height=600,
        scene_camera=dict(
            eye=dict(x=1.3, y=0, z=0.7)
        ),
        title="",
        scene=dict(
             xaxis=dict(title='', showticklabels=False, showgrid=False),
                yaxis=dict(title='', showticklabels=False, showgrid=False),
                zaxis=dict(title='',  showticklabels=False, showgrid=False,showbackground=False,backgroundcolor='black'),
       
    ),
     showlegend=False
    )
    
    # Show the plot in Streamlit
    with col2:
        st.plotly_chart(fig,use_container_width=True)
    if filterby == 'Player':
        if genderselect == 'MALE':
            pname = playernameimg
            import requests
            from bs4 import BeautifulSoup
            
            try:
                url = f'https://www.nbadraft.net/players/{pname}/'
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')

                player_name = soup.find('h1', class_='player-name').text.strip() if soup.find('h1', class_='player-name') else 'N/A'
                player_height = soup.find('span', class_='player-height').text.strip() if soup.find('span', class_='player-height') else 'N/A'
                player_weight = soup.find('span', class_='player-weight').text.strip() if soup.find('span', class_='player-weight') else 'N/A'
                player_position = soup.find('span', class_='player-position').text.strip() if soup.find('span', class_='player-position') else 'N/A'
                player_team = soup.find('span', class_='team-title').text.strip() if soup.find('span', class_='team-title') else 'N/A'

                attributes = {}
                for row in soup.find_all('div', class_='div-table-row'):
                    attribute_name = row.find('div', class_='div-table-cell attribute-name')
                    attribute_value = row.find('div', class_='div-table-cell attribute-value')
                    
                    if attribute_name and attribute_value:
                        attributes[attribute_name.text.strip()] = attribute_value.text.strip()

                mock_draft = soup.find('div', class_='mock-year')
                big_board = soup.find('div', class_='big-board')
                overall_rank = soup.find('div', class_='overall')

                mock_draft_value = mock_draft.find('span', class_='value').text.strip() if mock_draft and mock_draft.find('span', class_='value') else 'N/A'
                big_board_value = big_board.find('span', class_='value').text.strip() if big_board and big_board.find('span', class_='value') else 'N/A'
                overall_rank_value = overall_rank.find('span', class_='value').text.strip() if overall_rank and overall_rank.find('span', class_='value') else 'N/A'

                player_image_tag = soup.find('div', class_='player-image').find('img')
                player_image_url = player_image_tag['data-lazy-src'] if player_image_tag and 'data-lazy-src' in player_image_tag.attrs else player_image_tag['src'] if player_image_tag else 'N/A'

                # Display player info inside a collapsible section
                with st.expander(f"Player Info"):
                    st.write(result)
                    st.markdown(f"**Height**: {player_height}")
                    st.markdown(f"**Weight**: {player_weight}")
                    st.markdown(f"**Position**: {player_position}")
                    st.markdown(f"**Team**: {player_team}")

                    for i, (attribute, value) in enumerate(attributes.items()):
                        if i >= 3:
                            break
                        st.markdown(f"• **{attribute}**: {value}")


                # # Displaying rankings inside a collapsible section
                # with st.expander("Rankings"):
                #     st.markdown(f"**Mock Draft**: {mock_draft_value}")
                #     st.markdown(f"**Big Board**: {big_board_value}")
                #     st.markdown(f"**Overall Score**: {overall_rank_value}")

            except Exception as e:
                st.error(f"Failed to retrieve data: {e}")
        else:
            with st.expander(f"Player Info"):
                st.write(result)
        with st.expander('AI Player Analysis'):
            aistatsurl = st.secrets["aistats"]["url"] + f'?playerid={playerid}&gender={genderselect}'
            response = requests.get(aistatsurl)
            data = response.json()
            aistatsdf = pd.DataFrame(data)

            API_URL = st.secrets["api"]["url"]
            API_KEY = st.secrets["api"]["key"]
            HEADERS = {
                'Authorization': f'Bearer {API_KEY}',
                'Content-Type': 'application/json'
            }
        
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
        
            def get_scouting_report(player_summary):
                payload = {
                    "model": "DeepSeek-R1-Distill-Llama-70B",
                    "messages": [
                        # {"role": "system", "content": "You are an expert basketball scout. Given a player's statistics and position, write a long, comprehensive and professional scouting report and breakdown of the player. The report must include: 1. Strengths – supported by specific stats. 2. Weaknesses – with clear statistical evidence. 3. Detailed analysis which should be at least 25-30 sentences – explain the player's playing style and tendencies. 4. Improvement recommendations – specific, actionable advice tied to statistical patterns. 5. Justification for recommendations – explain why you made the recommendation by referencing the player's stats and percentages. 6. Player projection/ceiling – if you think the ceiling is an NBA player, compare the player's skillset to 1–2 similar NBA players who are similar in height, position, and playstyle/skillset, otherwise just give a projection of the player. Tailor the analysis based on the player's position (e.g., point guard, center). Be detailed, analytical, and thorough. **Do not include any internal reasoning, planning, or 'thinking aloud'. Only output the final scouting report directly.**"},
                        # {"role": "system", "content": "You are an expert basketball scout. Given a player's statistics and position, write a long, comprehensive and professional scouting report and breakdown of the player. The report must include: 1. Strengths – supported by specific stats. 2. Weaknesses – with clear statistical evidence. 3. Detailed analysis which should be at least 25-30 sentences – explain the player's playing style and tendencies. 4. Improvement recommendations – specific, actionable advice tied to statistical patterns. 5. Justification for recommendations – explain why you made the recommendation by referencing the player's stats and percentages. 6. Player projection/ceiling – give a projection of what the player could develop into based on skillset, playing style and measurements. Tailor the analysis based on the player's position (e.g., point guard, center). Be detailed, analytical, and thorough. **Do not include any internal reasoning, planning, or 'thinking aloud'. Only output the final scouting report directly.**"},
                        {
          "role": "system",
          "content": (
            "You are an expert basketball scout. Given a player's statistics and position, "
            "write a long, comprehensive, and professional scouting report. Use the following Markdown format for readability. "
            "Each section should be clear, in-depth, and supported by specific statistics. Do NOT include internal thoughts, reasoning, or XML tags like <think>. Output only the final report.\n\n"
        
            "Format and writing instructions:\n\n"
        
            "## 🏀 Scouting Report: [Player Name]\n\n"
            
            "---\n\n"
        
            "### **1. Strengths**\n"
            "- List 3 to 5 key strengths in bullet format.\n"
            "- Each strength must be supported by **at least one stat** (e.g., shooting %, assists, rebound rate).\n"
            "- Use clear basketball language (e.g., 'excellent PnR facilitator', 'elite catch-and-shoot threat').\n\n"
        
            "---\n\n"
        
            "### **2. Weaknesses**\n"
            "- List 2 to 4 specific weaknesses with statistical backing.\n"
            "- Be objective but constructive (e.g., 'turnover rate of 3.8 per game is high for a lead guard').\n"
            "- Highlight areas like defense, shot selection, efficiency, or decision-making if relevant.\n\n"
        
            "---\n\n"
        
            "### **3. Playing Style & Tendencies**\n"
            "- Write 4 to 6 detailed paragraphs (at least 25–30 full sentences total).\n"
            "- Describe the player’s on-court behavior, tendencies, positioning, pace, decision-making, scoring style, etc.\n"
            "- Mention any standout habits or traits (e.g., 'frequently pushes tempo after defensive rebounds', or 'reluctant to shoot above-the-break threes').\n"
            "- Tailor the analysis to the player’s position (e.g., describe playmaking for guards, rim protection for centers).\n\n"
        
            "---\n\n"
        
            "### **4. Recommendations for Improvement**\n"
            "- Provide 2 to 3 actionable, realistic suggestions.\n"
            "- Each recommendation must clearly relate to a weakness (e.g., 'reduce isolation possessions in favor of drive-and-kick opportunities').\n"
            "- Keep the advice technical and development-focused.\n\n"
        
            "---\n\n"
        
            "### **5. Justification for Recommendations**\n"
            "- Explain WHY each recommendation was made by citing specific stats, traits, or inefficiencies.\n"
            "- Use comparative benchmarks when relevant (e.g., 'his 59% FT% ranks in the bottom quartile for guards').\n\n"
        
            "---\n\n"
        
            "### **6. Player Projection & Ceiling**\n"
            "- Offer a realistic development path, all players you look at are either NBA or WNBA prospects so decide how they will project in the next level.\n"
            "- Optionally, compare the player’s skillset to **1–2 NBA/WNBA players** who share similar size, position, and playing style.\n"
            "- The comparison should be skill-based and NOT just based on raw stats.\n"
            "- Keep tone professional, realistic, and analytically grounded.\n\n"
        
            "---"
          )
        },
        
        
                        {"role": "user", "content": player_summary}
                    ],
                    "max_tokens": 10000,  # Adjust max_tokens as needed
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "top_k": 50,
                    "stream": True  # Enable streaming
                }
        
                response = requests.post(API_URL, headers=HEADERS, json=payload, stream=True)
        
                if response.status_code == 200:
                    output_placeholder = st.empty()
                    full_report = ""
        
                    for line in response.iter_lines():
                        if line:
                            try:
                                data = line.decode('utf-8')
                                if data.startswith("data:"): 
                                    response_data = data.split("data:")[1].strip()
                                    
                                    if not response_data:
                                        continue
                                    
                                    chunk = json.loads(response_data)
                                    content = chunk["choices"][0]["delta"].get("content", "")
                                    
                                    full_report += content
        
                                    full_report = re.sub(r"<think>.*?</think>", "", full_report, flags=re.DOTALL).strip()
        
                                    output_placeholder.code(full_report, language="markdown")
        
                            except Exception as e:
                                st.error(f"Error processing stream: {e}")
                                break
                    return full_report
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
                    return None
            columns_to_remove = [
                '_id', 'competitionId', 'gender', 'teamId', 'playerId', 'scope', 'tournamentId', 
                'isDraftProspect', 'hasImage', 'conferenceId', 'divisionId', 'isLatest', 'priorTeamId', 
                'priorConferenceId', 'isTransfer', 'inPortal', 'updated', 'isQualArray'
            ]
            aistatsdf = aistatsdf.drop(columns=columns_to_remove)
            row_values_with_columns = [f"{col}: {value}" for col, value in aistatsdf.items()]
            player_summary = "\n".join(row_values_with_columns)
            if st.button('Generate Analysis'):
                with st.spinner('Analyzing player...'):
                    scouting_report = get_scouting_report(player_summary)
                    st.session_state['scouting_report'] = scouting_report
                    if scouting_report:
                        st.success("Analysis complete!")
                    else:
                        st.error("Failed to analyze player.")
    else:
        with st.expander('AI Team Scouting Report'):
            teamurl = st.secrets["aistats"]["teamurl"] + f'?teamid={teamid}&gender={genderselect}&competitionId={competitionid}'
            response = requests.get(teamurl)
            data = response.json()
            teamranks = pd.DataFrame(data)
            teamranks['fullName'] = teamranks['teamMarket'] + ' ' + teamranks['teamName']
            columns_to_remove = [
                '_id', 'competitionId', 'gender', 'teamId', 'scope', 'tournamentId'
            ]
        
            teamranksfiltered = teamranks.drop(columns=columns_to_remove)
            
            row_values_with_columns = [f"{col}: {value}" for col, value in teamranksfiltered.items()]
        
            team_summary = "\n".join(row_values_with_columns)
        
            url = st.secrets["stats"]["teamplayerurl"] + f'?teamid={teamid}&gender={genderselect}&competitionId={competitionid}'
            response = requests.get(url)
            data = response.json()
            oteamplayerstats = pd.DataFrame(data)
            oteamplayerstats = oteamplayerstats.drop(columns=['_id','competitionId','gender','teamId','playerId','tournamentId','scope','hasImage','isDraftProspect','conferenceId','priorTeamId','isLatest','isTransfer','priorConferenceId','inPortal','divisionId','updated','jerseyNum','isQualArray'])
            oteamplayerstats = oteamplayerstats[oteamplayerstats['gp'] >= oteamplayerstats['gp'].mean()]
            oteamplayerstats = oteamplayerstats[oteamplayerstats['usagePct'] >= oteamplayerstats['usagePct'].mean()]
        
            player_stats_summary = ""
            for _, row in oteamplayerstats.iterrows():
                player_lines = [f"{col}: {row[col]}" for col in oteamplayerstats.columns]
                player_stats_summary += "\n" + "\n".join(player_lines) + "\n---"
            team_summary += player_stats_summary
        
        
            API_URL = st.secrets["api"]["url"]
            API_KEY = st.secrets["api"]["key"]
            HEADERS = {
                'Authorization': f'Bearer {API_KEY}',
                'Content-Type': 'application/json'
            }
        
            def get_team_scouting_report(player_summary):
                # Construct payload
                payload = {
                    "model": "DeepSeek-R1-Distill-Llama-70B",
                    "messages": [
                        # {"role": "system", "content": "You are a basketball scout analyzing player strengths and weaknesses factoring in the position of the player. Create a very detailed scouting report for the player using stats provided. In the report include strengths, weaknesses, detailed analysis, recommendations on how to improve , and NBA player(s) comparison."},
                        # {"role": "system", "content": "You are an expert basketball scout analyzing an opposing team. Given the team’s statistics, and playstyle, write a long, comprehensive, and professional scouting report. The report must include: 1. Team strengths – supported by specific stats and trends. 2. Team weaknesses – with clear statistical evidence. 3. Detailed analysis (25–30 sentences or more) – break down the team’s offensive and defensive tendencies, pace, shot selection, rebounding, transition habits,etc. 5. Strategic recommendations – how the opposing team should game-plan when playing them, both on offense against them and defense against them and make sure to be long and detailed in this section as well. Be analytical, strategic, and position-aware. **Do not include any internal reasoning, planning, or 'thinking aloud'. Only output the final scouting report directly.**"},
                        # {"role": "system", "content": "You are an expert basketball scout analyzing an opposing team. Given the team’s statistics, and playstyle, write a long, comprehensive, and professional scouting report. The report must include: 1. Team strengths – supported by specific stats and trends. 2. Team weaknesses – with clear statistical evidence. 3. Detailed analysis (25–30 sentences or more) – break down the team’s offensive and defensive tendencies, pace, shot selection, rebounding, transition habits,etc. 5. Strategic recommendations – how the opposing team should game-plan when playing them, both on offense against them and defense against them and make sure to be long and detailed in this section as well. Be analytical, strategic, and position-aware. **Do not include any internal reasoning, planning, or 'thinking aloud'. You will also be given the statistics of the highest usage players on the team. Give breakdowns for the top players on the team and how to stop them and play against them. Only output the final scouting report directly.**"},
                        {
          "role": "system",
          "content": (
            "You are an expert basketball scout analyzing an opposing team. Given the team’s statistics, playstyle, and the stats of their highest-usage players, write a long, comprehensive, and professional scouting report.\n\n"
            "Use the following **Markdown format** for a clean, structured report that can be displayed in Streamlit. Do NOT include any internal reasoning, planning, or XML tags like <think>. Only output the final formatted report.\n\n"
        
            "## 📊 Scouting Report: [Team Name]\n\n"
            
            "---\n\n"
        
            "### **1. Team Strengths**\n"
            "- List 3 to 5 of the team’s most dangerous strengths in bullet point format.\n"
            "- Support each strength with relevant **team statistics or trends** (e.g., 3P%, turnover margin, offensive efficiency).\n"
            "- Use precise language like 'excellent transition offense', 'high assist rate', or 'elite rim protection'.\n\n"
        
            "---\n\n"
        
            "### **2. Team Weaknesses**\n"
            "- Identify 2 to 4 key weaknesses with clear statistical backing.\n"
            "- Focus on areas such as defensive lapses, poor rebounding, low shooting efficiency, or foul issues.\n"
            "- Be analytical and constructive.\n\n"
        
            "---\n\n"
        
            "### **3. Team Style, Tendencies & Analytics**\n"
            "- Write 5 to 6 **long paragraphs** (at least 25–30 full sentences total).\n"
            "- Break down the team’s **offensive tendencies** (e.g., ball movement, PnR frequency, isolation rate, shot selection).\n"
            "- Discuss **defensive approach** (e.g., switch-heavy, pack line, full-court press, zone usage).\n"
            "- Mention **pace of play**, **transition offense**, **rebounding patterns**, and **foul tendencies**.\n"
            "- Be highly analytical and use team stats to justify observations.\n\n"
        
            "---\n\n"
        
            # "### **4. Strategic Recommendations**\n"
            # "- Provide **detailed and tactical guidance** for game-planning against this team.\n"
            # "- Split the section into:\n"
            # "  - **Offensive Strategy** (how to attack them)\n"
            # "  - **Defensive Strategy** (how to contain them)\n"
            # "- Be strategic and position-aware. Mention matchups, ball screen coverages, and help defense ideas.\n"
            # "- Use basketball terminology like 'hedge and recover', 'ice the screen', 'deny ball reversals', etc.\n"
            # "- Use bullet points or small paragraphs for clarity if needed.\n\n"
        
            "---\n\n"
        
            "### **5. Top Player Scouting & Game Plan**\n"
            "- Provide breakdowns for **3–5 of the team's highest-usage players**.\n"
            "- For each player:\n"
            "  - Start with a **bolded name** and position (e.g., **John Smith – PG**).\n"
            "  - Give a long, detailed playing style summary (e.g., 'shifty scoring guard who thrives in isolation').\n"
            # "  - Detailed, long recommendation **how to defend or exploit them** (e.g., force left, go under screens, attack on closeouts).\n"
            "- Be tactical, detailed, and focused on each player's impact.\n\n"
        
            "---"
          )
        },
                        {"role": "user", "content": player_summary}
                    ],
                    "max_tokens": 10000, 
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "top_k": 50,
                    "stream": True  
                }
        
                response = requests.post(API_URL, headers=HEADERS, json=payload, stream=True)
        
                if response.status_code == 200:
                    output_placeholder = st.empty()
                    full_report = ""
        
                    for line in response.iter_lines():
                        if line:
                            try:
                                data = line.decode('utf-8')
                                if data.startswith("data:"):  
                                    response_data = data.split("data:")[1].strip()
                                    
                                    if not response_data:
                                        continue
                                    
                                    chunk = json.loads(response_data)
                                    content = chunk["choices"][0]["delta"].get("content", "")
                                    
                                    full_report += content
        
                                    full_report = re.sub(r"<think>.*?</think>", "", full_report, flags=re.DOTALL).strip()
        
                                    output_placeholder.code(full_report, language="markdown")
        
                            except Exception as e:
                                st.error(f"Error processing stream: {e}")
                                break
                    return full_report
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
                    return None
            if st.button('Generate Scouting Report'):
                with st.spinner('Generating scouting report...'):
                    report = get_team_scouting_report(team_summary)
                    if report:
                        st.session_state['scouting_report'] = report
                        st.success("Scouting report generated!")
                    else:
                        st.error("Failed to generate scouting report.")
    c1,c2 = st.columns(2)
    shotdf['gameDate'] = pd.to_datetime(shotdf['gameDate'], format='%Y-%m-%d')
    
    # Aggregate shooting data by date
    shooting_over_time = shotdf.groupby('gameDate').agg({"success": ["sum", "count"]}).reset_index()
    shooting_over_time.columns = ['Date', 'Made', 'Total']
    
    # Calculate shooting percentage
    shooting_over_time['Percentage'] = round(shooting_over_time['Made'] / shooting_over_time['Total'] * 100,2)
    
    # Add a moving average to smooth out trends (optional)
    shooting_over_time['Moving Average (7 Days)'] = round(shooting_over_time['Percentage'].rolling(window=7, min_periods=1).mean(),2)
    
    # Create the line chart
    fig2 = px.line(
        shooting_over_time, 
        x='Date', 
        y='Percentage', 
        title="Shooting Percentage Over Time",
        labels={'Percentage': 'Shooting Percentage (%)', 'Date': 'Game Date'},  # Axis labels
        markers=True,  # Show markers for each data point
        hover_data={'Date': True, 'Percentage': True, 'Made': True, 'Total': True},  # Show extra info on hover
    )
    
    # Add a moving average line (optional)
    fig2.add_scatter(
        x=shooting_over_time['Date'], 
        y=shooting_over_time['Moving Average (7 Days)'], 
        mode='lines', 
        name='7-Day Moving Average',
        line=dict(color='red', dash='dash')
    )
    
    # Customize the layout for better aesthetics
    fig2.update_layout(
        title="Shooting Percentage Over Time",
        title_x=0,  # Center the title
        title_font=dict(size=20, family='Arial', color='white'),
        xaxis=dict(
            showgrid=True, 
            tickangle=45,  # Rotate x-axis labels for better readability
            tickformat='%b %d, %Y',  # Show date in a readable format
            title_font=dict(size=14, family='Arial', color='white'),
        ),
        yaxis=dict(
            title="Shooting Percentage (%)",
            title_font=dict(size=14, family='Arial', color='white'),
        ),
        showlegend=False,  # Show the legend for the moving average line
        margin=dict(l=40, r=40, t=50, b=40),  # Adjust margins for better spacing
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Set background color to white for a clean look
    )
    with c1:
        st.plotly_chart(fig2)
    
    # Create distance bins
    distance_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]  # adjusted max distance
    df_filtered = shotdf.dropna(subset=['shotDist'])
    
    df_filtered['distance_bin'] = pd.cut(df_filtered['shotDist'], bins=distance_bins)
    
    # Convert the 'distance_bin' to strings to make it serializable
    df_filtered['distance_bin'] = df_filtered['distance_bin'].astype(str)
    
    # Calculate shooting percentage by distance bin
    shooting_by_distance = df_filtered.groupby('distance_bin').agg({"success": ["sum", "count"]}).reset_index()
    
    # Flatten the MultiIndex columns
    shooting_by_distance.columns = ['Distance', 'Made', 'Total']
    
    # Calculate accuracy
    shooting_by_distance['Percentage'] = round(shooting_by_distance['Made'] / shooting_by_distance['Total'] * 100,2)
    
    # Plot the figure using Plotly with a new theme
    fig3 = px.bar(
        shooting_by_distance, 
        x='Percentage', 
        y='Distance', 
        orientation='h',  # horizontal bars
        title="Shooting Percentage by Distance",
        color='Percentage',  # Use color to represent shooting percentage
        color_continuous_scale='YlOrRd',  # A warm color palette (yellow to red)
        labels={'Percentage': 'Shooting %', 'Distance': 'Shot Distance (ft)'},
        text='Percentage',  # Show percentage on the bars
    )
    fig3.update_coloraxes(showscale=False) 
    
    # Customize the layout for better aesthetics
    fig3.update_layout(
        title_text="Shooting Percentage by Distance",
        title_x=0,  # Center the title
        title_font=dict(size=20, family='Arial', color='white'),
        xaxis_title='Shooting Percentage (%)',
        yaxis_title='Shot Distance (ft)',
        yaxis_tickangle=-45,  # Rotate y-axis labels for readability
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        plot_bgcolor='rgba(0, 0, 0, 0)', # Set background color to white for clarity
        margin=dict(l=20, r=20, t=40, b=20),  # Adjust margins to fit content
        showlegend=False  # Disable legend for this chart
    )
    
    # Add annotations to show the percentage on each bar
    fig3.update_traces(
        texttemplate='%{text:.1f}%',  # Format percentage to 1 decimal place
        textposition='inside',  # Position the percentage text inside the bars
        insidetextanchor='middle'
    )
    
    # Display the plot in Streamlit
    with c2:
        st.plotly_chart(fig3)
    co1, co2 = st.columns(2)
    shot_accuracy_by_type = shotdf.groupby('subType').agg({"success": ["sum", "count"]}).reset_index()
    
    # Flatten MultiIndex columns
    shot_accuracy_by_type.columns = ['Shot Type', 'Made', 'Total']
    
    # Calculate shooting percentage
    shot_accuracy_by_type['Percentage'] = shot_accuracy_by_type['Made'] / shot_accuracy_by_type['Total'] * 100
    
    # Create a bar plot with enhancements
    fig = px.bar(
        shot_accuracy_by_type, 
        x='Percentage', 
        y='Shot Type', 
        orientation='h',  # Horizontal bars for better readability
        title="Shooting Percentage by Shot Type",
        color='Percentage',  # Color bars based on shooting percentage
        color_continuous_scale='Viridis',  # Modern color scale (Viridis)
        labels={'Percentage': 'Shooting %', 'Shot Type': 'Shot Type'},
        text='Shot Type',  # Show percentage on bars
    )
    fig.update_coloraxes(showscale=False) 
    
    
    
    # Customize the layout to improve aesthetics
    fig.update_layout(
        title_text="Shooting Percentage by Shot Type",
        title_x=0,  # Center the title
        title_font=dict(size=20, family='Arial', color='white'),
        xaxis_title='Shooting Percentage (%)',
        yaxis_title='Shot Type',
        yaxis_tickangle=-45,  # Rotate y-axis labels for better readability
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Set background color to white
        margin=dict(l=20, r=20, t=40, b=40),  # Adjust margins for better fit
        showlegend=False,  # Disable legend for clarity
        bargap=0.1,  # Set the gap between bars (lower value = wider bars)
        bargroupgap=0.05,  # Set gap between bar groups (useful for grouped bars)
    )
    
    # Update traces to show percentage inside bars
    fig.update_traces(
        # texttemplate='%{text:.1f}%',  # Format text to 1 decimal place
        textposition='inside',  # Position the text inside the bars
        insidetextanchor='middle'  # Align text in the middle of bars
    )
    fig.update_layout(
        yaxis=dict(
            showticklabels=False,  # Hide the tick labels
            zeroline=False,        # Optionally, remove the zero line if you want a cleaner look
            showline=False         # Optionally, remove the axis line
        )
    )
    
    # Add hover information to show shot type, made count, and total shots
    fig.update_traces(
        hovertemplate='%{y}<br>%{customdata[0]}/%{customdata[1]}<br>%{x:.1f}%',
        customdata=shot_accuracy_by_type[['Made', 'Total']].values
    )
    with co1:
        st.plotly_chart(fig)
    
    shot_type_distribution = shotdf['subType'].value_counts().reset_index()
    shot_type_distribution.columns = ['Shot Type', 'Count']
    
    # Create a pie chart with enhancements
    fig = px.pie(
        shot_type_distribution, 
        names='Shot Type', 
        values='Count', 
        # title="Shot Type Distribution", 
        color='Shot Type',  # Color by shot type for better distinction
        color_discrete_sequence=px.colors.sequential.Plasma,  # Modern color scale
        hole=0.3,  # Create a donut chart (optional)
        # hover_data={'Shot Type': False, 'Count': True},  # Hover will show only 'Count', not 'Shot Type'
    )
    
    # Customize layout for better visual appeal
    fig.update_layout(
        title="Shot Type Distribution",
        title_x=0,  # Center the title
        title_y=1,
        title_font=dict(size=20, family='Arial', color='white'),
        showlegend=False,  # Show the legend
        legend_title='Shot Type',
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="bottom", 
            y=1.02, 
            xanchor="center", 
            x=0.5
        ),
        margin=dict(l=20, r=20, t=40, b=20),  # Adjust margins
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Set background color to white for a clean look
    )
    
    # Enhance hover info and show percentages on slices
    fig.update_traces(
        textinfo='percent+label',  # Display percentage and shot type
        pull=[0.1 if i == shot_type_distribution['Count'].idxmax() else 0 for i in range(len(shot_type_distribution))],  # Explode the max slice
        hovertemplate='Shot Type: %{label}<br>Count: %{value}'  # Detailed hover info
    )
    
    with co2:
        st.plotly_chart(fig)
    cl1, cl2 = st.columns(2)
    with cl2:

        # Create a new column for shot clock intervals (1-5, 6-10, etc.)
        shotdf['Shot Clock Interval'] = pd.cut(
            shotdf['shotClock'], 
            bins=[0, 5, 10, 15, 20, 25, 30], 
            labels=['1-5', '6-10', '11-15', '16-20', '21-25','26-30'], 
            right=False
        )
        
        # Group by the new intervals
        shot_accuracy_by_clock = shotdf.groupby('Shot Clock Interval').agg({"success": ["sum", "count"]}).reset_index()
        
        # Flatten MultiIndex columns
        shot_accuracy_by_clock.columns = ['Shot Clock Interval', 'Made', 'Total']
        
        # Calculate shooting percentage
        shot_accuracy_by_clock['Percentage'] = shot_accuracy_by_clock['Made'] / shot_accuracy_by_clock['Total'] * 100
        
        # Create a bar plot with enhancements
        fig = px.bar(
            shot_accuracy_by_clock, 
            x='Percentage', 
            y='Shot Clock Interval', 
            orientation='h',
            title="Shooting Accuracy by Shot Clock Time",
            color='Percentage', 
            color_continuous_scale='Cividis',  
            labels={'Percentage': 'Shooting %', 'Shot Clock Interval': 'Shot Clock Interval (s)'},
            text='Percentage',  
        )
        
        # Customize layout
        fig.update_layout(
            title_text="Shooting Accuracy by Shot Clock Intervals",
            title_x=0.5,  
            title_font=dict(size=20, family='Arial', color='white'),
            xaxis_title='Shooting Percentage (%)',
            yaxis_title='Shot Clock Interval (s)',
            yaxis_tickangle=0,
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            plot_bgcolor='rgba(0, 0, 0, 0)',  
            margin=dict(l=20, r=20, t=50, b=40),
            showlegend=False,  
            bargap=0.2,  
        )
        fig.update_coloraxes(showscale=False)
        
        # Update traces to show percentage inside bars
        fig.update_traces(
            texttemplate='%{text:.1f}%',  
            textposition='inside',  
            insidetextanchor='middle',
        )
        
        # Add hover information to show made shots and total attempts
        fig.update_traces(
            hovertemplate='Shot Clock Interval: %{y}<br>Made: %{customdata[0]} / Total: %{customdata[1]}<br>Shooting %: %{x:.1f}%',
            customdata=shot_accuracy_by_clock[['Made', 'Total']].values
        )
        
        st.plotly_chart(fig)
    with cl1:
        if filterby == 'Player':
            url = st.secrets["stats"]["playerteamshoturl"] + f'?teamid={teamid}&gender={genderselect}&competitionId={competitionid}&playerid={playerid}'
        
            # Send the GET request to the API
            response = requests.get(url)        
            data = response.json()
            shotdf = pd.DataFrame(data)
            shotdf['fullName'] = shotdf['playerId'].map(player_id_to_name)
            shotdf['assisterName'] = shotdf['assisterId'].map(player_id_to_name)
            shotdf = shotdf[shotdf['assisterName'] == playername]
            # Get assist distribution by assisterName
            assist_distribution = shotdf['fullName'].value_counts().reset_index()
            assist_distribution.columns = ['Assisted', 'Count']
            
            # Create a pie chart for assists
            fig = px.pie(
                assist_distribution, 
                names='Assisted', 
                values='Count', 
                color='Assisted',  
                color_discrete_sequence=px.colors.qualitative.Set3,  # New color scale
                hole=0.3,  # Donut chart
            )
            
            fig.update_layout(
                title="Assist Distribution",
                title_x=0.5,  # Center the title
                title_font=dict(size=20, family='Arial', color='white'),
                showlegend=False,  # Show legend
                legend_title='Assisted',
                legend=dict(
                    orientation="h",  # Horizontal legend
                    yanchor="bottom", 
                    y=1.02, 
                    xanchor="center", 
                    x=0.5
                ),
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
            )
            
            fig.update_traces(
                textinfo='percent+label',  
                hovertemplate='%{label}<br>Count: %{value}',  
            )
            
            st.plotly_chart(fig) 
        else:
            import matplotlib.colors as mcolors  # matplotlib has great color parsing

            def color_scale_rgba(base_color, norm_value, alpha=0.6):
                """
                Returns an RGBA color string scaled by norm_value (0 to 1).
                """
                r, g, b = mcolors.to_rgb(base_color)
                r_scaled = int(r * 255 * norm_value)
                g_scaled = int(g * 255 * norm_value)
                b_scaled = int(b * 255 * norm_value)
                return f'rgba({r_scaled}, {g_scaled}, {b_scaled}, {alpha})'

            import networkx as nx

            # df = pd.read_csv('TeamArticle.csv')

            base_color = color2
            # Group and filter assist counts
            assist_counts = df.groupby(['assisterName', 'fullName']).size().reset_index(name='assist_count')
            assist_counts = assist_counts[assist_counts['assist_count'] > 1]

            # Create graph
            G = nx.DiGraph()
            for _, row in assist_counts.iterrows():
                G.add_edge(row['assisterName'], row['fullName'], weight=row['assist_count'])

            # Layout
            pos = nx.circular_layout(G)
            for node in G.nodes:
                G.nodes[node]['pos'] = list(pos[node])

            # Normalize weights for color scale (optional)
            weights = [G[u][v]['weight'] for u, v in G.edges()]
            min_weight = min(weights)
            max_weight = max(weights)

            # Build edge traces (width & color = assist count)
            edge_traces = []
            for u, v in G.edges():
                x0, y0 = G.nodes[u]['pos']
                x1, y1 = G.nodes[v]['pos']
                weight = G[u][v]['weight']
                norm_color = (weight - min_weight) / (max_weight - min_weight + 1e-9)  # avoid divide by zero
                edge_traces.append(go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=weight*.5, color=color_scale_rgba(base_color, norm_color))
            ,  # red scale
                    hoverinfo='text',
                    text=f"{weight} assists"
                ))

            # Node trace
            node_x = []
            node_y = []
            names = []
            for node in G.nodes:
                x, y = G.nodes[node]['pos']
                node_x.append(x)
                node_y.append(y)
                names.append(node)

            hover_texts = []
            for node in G.nodes:
                total_assists = sum(G[node][nbr]['weight'] for nbr in G.successors(node))
                hover_texts.append(f"{node}: {total_assists} assists")

            assist_distribution = shotdf['assisterName'].value_counts().to_dict()

            # Generate hover labels for nodes using assist_distribution
            hover_texts = [f"{name}: {assist_distribution.get(name, 0)} assists" for name in names]

            # Updated node trace
            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                text=names,
                textposition="bottom center",
                hoverinfo='text',
                hovertext=hover_texts,
                marker=dict(
                    color=color1,
                    size=20,
                    line_width=2
                )
            )


            # Final interactive plot
            fig = go.Figure(
                data=edge_traces + [node_trace],
                layout=go.Layout(
                    title='Assist Network',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    annotations=[dict(
                        # text="Thicker & darker lines = more assists",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002
                    )],
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False)
                )
            )

            st.plotly_chart(fig)
