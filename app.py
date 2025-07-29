import pandas as pd
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import google.generativeai as genai
import time
import random

# pip install google-generativeai streamlit
# Load model Task 1
model_task1 = AutoModelForSequenceClassification.from_pretrained("bertweet-disaster-task1")
tokenizer_task1 = AutoTokenizer.from_pretrained("bertweet-disaster-task1")
model_task1.eval()

# Load model Task 2
model_task2 = AutoModelForSequenceClassification.from_pretrained("bertweet-disaster-task2")
tokenizer_task2 = AutoTokenizer.from_pretrained("bertweet-disaster-task2")
model_task2.eval()

# Load Gemini model for Task 3
model = genai.GenerativeModel('models/gemini-2.0-flash-thinking-exp')

# Task 1 - Dự đoán với mô hình Task 1
def predict_task1(tweet_text):
    inputs = tokenizer_task1(tweet_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model_task1(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    return prediction

# Task 2 - Dự đoán với mô hình Task 2 (Disaster type)
def predict_task2(tweet_text):
    inputs = tokenizer_task2(tweet_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model_task2(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    return prediction

# Task 3 (Gemini) - Dự đoán với mô hình Gemini (Disaster Info, Emergency Help, Emotion Sharing)
def create_prompt(tweet_text):
    return f"""
    You are an AI assistant trained in meteorology and disaster event analysis. Your task is to classify tweets based on their content as belonging to three labels: disaster_info, emergency_help, emotion_sharing.

    Label definitions:

    1. disaster_info (label=1): Tweets that provide information about natural or man-made disasters, including earthquakes, floods, wildfires, landslides.
    non-disaster_info (label=0): Tweets that do not provide information about natural or man-made disasters.
    Example: "In California, there is a very large wildfire, which has now spread due to strong winds."

    2. emergency_help (label=1): Tweets requesting help or aid caused by natural disasters. It can be a casual or urgent call for help.
    non-emergency_help (label=0): Tweets that are not related to requests for help.
    Example: "Typhoon Yagi has arrived in Hanoi, my friend was slashed in the shoulder by the corrugated iron roof at the Melia Hotel, it is bleeding, who will save my friend?"

    3. emotion_sharing (label=1): Tweets expressing emotions about natural disasters, such as panic, bewilderment, fear, joy, etc.
    non-emotion_sharing (label=0): Tweets that are not related to emotion sharing.

    The tweet to classify is provided between three backticks:

    ```
    {tweet_text}
    ```

    In your response, return only the following labels:
    - "disaster_info=1" or "disaster_info=0"
    - "emergency_help=1" or "emergency_help=0"
    - "emotion_sharing=1" or "emotion_sharing=0"
    """

def predict_task3(tweet_text):
    prompt = create_prompt(tweet_text)
    try:
        time.sleep(random.uniform(5, 7))  # Thời gian chờ 5 giây
        response = model.generate_content(prompt)
        prediction = response.text.strip()
        labels = {'p_disaster_info': 0, 'p_emergency_help': 0, 'p_emotion_sharing': 0, 'disaster_type': 'unknown'}  # Set default as 'unknown'

        if "disaster_info=1" in prediction:
            labels['p_disaster_info'] = 1
        if "emergency_help=1" in prediction:
            labels['p_emergency_help'] = 1
        if "emotion_sharing=1" in prediction:
            labels['p_emotion_sharing'] = 1

        # Extract disaster type from prediction
        if "flood" in prediction:
            labels['disaster_type'] = "flood"
        elif "earthquake" in prediction:
            labels['disaster_type'] = "earthquake"
        elif "typhoon" in prediction:
            labels['disaster_type'] = "typhoon"
        elif "wildfire" in prediction:
            labels['disaster_type'] = "wildfire"
        
        return labels
    except Exception as e:
        print(f"Error predicting for tweet: {tweet_text}\n{e}")
        return None

# Tạo pipeline cho cả ba mô hình
def pipeline_predict(tweet_text):
    # Bước 1: Dự đoán với Task 1
    pred_task1 = predict_task1(tweet_text)

    # Nếu Task 1 dự đoán là Non-disaster (0), trả về "Non-disaster"
    if pred_task1 == 0:
        return "Non-disaster tweet."
    
    # Bước 2: Nếu là Disaster, thực hiện Task 2
    pred_task2 = predict_task2(tweet_text) if pred_task1 == 1 else -1
    
    # Bước 3: Dự đoán với Task 3 (Gemini) cho tất cả dữ liệu
    task3_results = predict_task3(tweet_text)
    
    # Xử lý kết quả cho disaster_type dựa trên task 2
    disaster_type = "unknown"
    if pred_task2 == 0:
        disaster_type = "earthquake"
    elif pred_task2 == 1:
        disaster_type = "flood"
    elif pred_task2 == 2:
        disaster_type = "hurricane"
    elif pred_task2 == 3:
        disaster_type = "wildfire"
    
    # Định dạng kết quả
    result = f"Disaster: {pred_task1} - Disaster type: {disaster_type} - Sentiment: "
    
    # Kiểm tra các nhãn của Task 3
    if task3_results['p_disaster_info'] == 1:
        result += "Disaster Info"
    elif task3_results['p_emergency_help'] == 1:
        result += "Emergency Help"
    elif task3_results['p_emotion_sharing'] == 1:
        result += "Emotion Sharing"
    else:
        result += "Other"
    
    return result

# Giao diện Streamlit
st.title("Disaster Tweet Classification")

# Input text box for tweet
tweet_input = st.text_area("Enter tweet text:")

if st.button("Predict"):
    if tweet_input:
        # Run the pipeline and display result
        result = pipeline_predict(tweet_input)
        st.write(f"Prediction: {result}")
    else:
        st.write("Please enter a tweet.")
