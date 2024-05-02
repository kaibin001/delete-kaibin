import streamlit as st
import pandas as pd
from pycaret.classification import *

# Load the model and data
model = load_model('model')

fight_events_path = "new_fight_detail_full.csv"
fight_events = pd.read_csv(fight_events_path)

upcoming_events_path = "upcoming_events.csv"
upcoming_events = pd.read_csv(upcoming_events_path)

fight_demo_path = "Test_cases.csv"
fight_demo = pd.read_csv(fight_demo_path)

st.title("UFC Fight Predictor")

# Upcoming fights display
upcoming_events['Matchup'] = upcoming_events['Fighter1'] + ' vs ' + upcoming_events['Fighter2']
st.sidebar.write("UFC 301", upcoming_events[['Matchup']])

fight_demo['Demo Matchup'] = fight_demo['Fighter1'] + ' vs ' + fight_demo['Fighter2']
st.sidebar.write("Matchup", fight_demo[['Demo Matchup']])

# Fighter Selection
fighter_names = sorted(fight_events['Fighter1'].dropna().unique())
fighter1 = st.selectbox("Select Fighter 1", [''] + fighter_names)
fighter1last5 = fight_events[(fight_events['Fighter1'] == fighter1)].rename(columns={'Fighter2': 'Fighter','Win/Loss (Fighter1)': 'Result'}).drop(columns=['Fighter1', 'Weight Class', 'Time']).head(5)
if fighter1:
    st.write(f"{fighter1} Last 5 Fights", fighter1last5, index=False)

fighter2 = st.selectbox("Select Fighter 2", [''] + [f for f in fighter_names if f != fighter1])
fighter2last5 = fight_events[(fight_events['Fighter1'] == fighter2)].rename(columns={'Fighter2': 'Fighter','Win/Loss (Fighter1)': 'Result'}).drop(columns=['Fighter1', 'Weight Class','Time']).head(5)
if fighter2:
    st.write(f"{fighter2} Last 5 Fights", fighter2last5, index=False)
    
# Prediction Button
if st.button("Predict Winner"):
    if not fighter1 or not fighter2:
        st.error("Please select both fighters.")
    else:
        
        st.write(fighter1, " VS ", fighter2)
        
        # Make predictions using the loaded model
        prediction = predict_model(model, data=fight_events[fight_events["Fighter1"]==fighter1])
        winner = f"Prediction: Fighter 1, {fighter1} Wins!" if prediction.iloc[0,-2] == "Win" else f"Prediction: Fighter 2, {fighter2}  Wins!"
        st.success(winner)
