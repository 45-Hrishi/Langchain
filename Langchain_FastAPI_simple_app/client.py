import streamlit as st
import requests

def get_model_response(input_text,model):
    if model == "llama3":
        endpoint = "http://localhost:8000/llama3/invoke"
    else:
        endpoint = "http://localhost:8000/mistral/invoke"
    
    response = requests.post(endpoint,json={"player_name": input_text})
    if response.status_code == 200:
        return response.json()["output"]
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None
    
st.title('Langchain & Langserve demo')
input_text=st.text_input("Player Name")

options = ['llama3', 'mistral']
selected_option = st.selectbox("Select a model:", options)

if st.button("Get Response"):
    if input_text.strip():  # Ensure input is not empty
        response = get_model_response(input_text, selected_option)
        if response:
            st.write("Response:", response)
    else:
        st.warning("Please enter a valid player name.")