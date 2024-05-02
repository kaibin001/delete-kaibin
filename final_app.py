import streamlit as st
import pandas as pd
import pickle
import os

# Load the model and data
home_directory = os.path.expanduser('~')
model_path = os.path.join(home_directory, 'Downloads', 'Capstone', 'latest_model.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

fight_events_path = os.path.join(home_directory, 'Downloads', 'Capstone', 'new_fight_detail_full.csv')
fight_events = pd.read_csv(fight_events_path)

upcoming_events_path = os.path.join(home_directory, 'Downloads', 'Capstone', 'upcoming_events.csv')
upcoming_events = pd.read_csv(upcoming_events_path)

def process_features(fighter1, fighter2):
    # Process features for the model
    features_f1 = fight_events.loc[fight_events['Fighter1'] == fighter1].iloc[:, 7:20].iloc[0:1, :]
    features_f2 = fight_events.loc[fight_events['Fighter2'] == fighter2].iloc[:, 20:].iloc[0:1, :]
    features = pd.concat([features_f1.reset_index(drop=True), features_f2.reset_index(drop=True)], axis=1)
    percentage_features = ['Win Rate (Fighter 1)', 'Str. Acc. (Fighter 1)', 'Str. Def (Fighter 1)', 'TD Acc. (Fighter 1)', 'TD Def. (Fighter 1)', 
                           'Win Rate (Fighter 2)', 'Str. Acc. (Fighter 2)', 'Str. Def (Fighter 2)', 'TD Acc. (Fighter 2)', 'TD Def. (Fighter 2)']
    for feature in percentage_features:
        features[feature] = features[feature].str.rstrip('%').astype('float') / 100
    return features

st.title("UFC Fight Predictor")

# Upcoming fights display
upcoming_events['Matchup'] = upcoming_events['Fighter1'] + ' vs ' + upcoming_events['Fighter2']
st.sidebar.write("Upcoming Events", upcoming_events[['Matchup']])



# Fighter Selection
fighter_names = sorted(fight_events['Fighter1'].dropna().unique())
fighter1 = st.selectbox("Select Fighter 1", [''] + fighter_names)
fighter1last5 = fight_events[(fight_events['Fighter1'] == fighter1)].rename(columns={'Fighter2': 'Fighter','Win/Loss (Fighter1)': 'Result'}).drop(columns=['Fighter1', 'Weight Class']).head(5)
if fighter1:
    st.write(f"{fighter1} Last 5 Fights", fighter1last5, index=False)

fighter2 = st.selectbox("Select Fighter 2", [''] + [f for f in fighter_names if f != fighter1])
fighter2last5 = fight_events[(fight_events['Fighter1'] == fighter2)].rename(columns={'Fighter2': 'Fighter','Win/Loss (Fighter1)': 'Result'}).drop(columns=['Fighter1', 'Weight Class']).head(5)
if fighter2:
    st.write(f"{fighter2} Last 5 Fights", fighter2last5, index=False)
    
# Prediction Button
if st.button("Predict Winner"):
    if not fighter1 or not fighter2:
        st.error("Please select both fighters.")
    else:
        # Assuming you have a function to process input features
        input_features = process_features(fighter1, fighter2)
        # st.write("Shape of the input features:", input_features.shape)  # Check the shape
        
        # Ensure input_features is two-dimensional
        if len(input_features.shape) == 3 and input_features.shape[0] == 1:
            input_features = input_features.reshape(1, -1)
        
        st.write(fighter1, " VS ", fighter2)
        prediction = model.predict(input_features)  # Make sure input_features is correctly shaped
        winner = f"Prediction: Fighter 1, {fighter1} Wins!" if prediction == 1 else f"Prediction: Fighter 2, {fighter2}  Wins!"
        st.success(winner)
