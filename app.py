import streamlit as st
import pandas as pd
import numpy as np
import joblib

import streamlit.components.v1 as components

# Load model and scaler
model = joblib.load("xgb_sleep_quality_model.pkl")
scaler = joblib.load("scaler_sleep_quality.pkl")

# Streamlit app
st.set_page_config(page_title="Sleep Quality Predictor", layout="wide")


# Main Title
st.markdown("<h1 style='text-align: center; color: #6C3483;'>Sleep Quality Predictor</h1>", unsafe_allow_html=True)

# Input form
with st.form("sleep_form"):
    st.subheader("Enter your Health and Lifestyle Data")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 10, 100, 25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        sleep_duration = st.slider("Sleep Duration (hrs)", 0.0, 12.0, 7.0, 0.5)
        activity = st.slider("Physical Activity (mins/day)", 0, 180, 30)
        stress = st.slider("Stress Level (1–10)", 1, 10, 5)
        caffeine = st.slider("Caffeine Intake (cups/day)", 0, 10, 1)
        alcohol = st.slider("Alcohol Intake (units/day)", 0, 10, 0)

    with col2:
        smoker = st.selectbox("Do you smoke?", ["No", "Yes"])
        heart_rate = st.number_input("Heart Rate (bpm)", 40, 140, 70)
        screen_time = st.slider("Screen Time Before Bed (hrs)", 0.0, 10.0, 2.0, 0.5)
        bmi = st.number_input("BMI", 10.0, 50.0, 22.0)
        wake_consistency = st.selectbox("Wake-up Consistency", ["Regular", "Irregular"])
        env_score = st.slider("Sleep Environment Score (1–10)", 1, 10, 7)
        water = st.slider("Daily Water Intake (litres)", 0.0, 5.0, 2.0, 0.5)
        history = st.selectbox("Sleep Disorder History", ["No", "Yes"])  # last

    submitted = st.form_submit_button("Predict")


# Predict
if submitted:
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [1 if gender == "Male" else 0],
        'Sleep Duration (hrs)': [sleep_duration],
        'Physical Activity (mins/day)': [activity],
        'Stress Level (1–10)': [stress],
        'Caffeine Intake (cups/day)': [caffeine],
        'Alcohol Intake (units/day)': [alcohol],
        'Smoking': [1 if smoker == "Yes" else 0],
        'Heart Rate (bpm)': [heart_rate],
        'Screen Time Before Bed (hrs)': [screen_time],
        'Sleep Disorder History': [1 if history == "Yes" else 0],
        'BMI': [bmi],
        'Wake-up Consistency': [1 if wake_consistency == "Consistent" else 0],
        'Sleep Environment Score (1–10)': [env_score],
        'Daily Water Intake (litres)': [water]
    })

    # Scale inputs
    scaled_input = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_input)[0]
    label_map = {0: 'Poor', 1: 'Fair', 2: 'Good'}
    result = label_map[prediction]

    # Display result
    st.success(f" Predicted Sleep Quality :  {result}")




st.markdown("---")
st.subheader("Get Tips With Our AI Bot")

chatbot_html = """
<style>
  .chat-container {
    width: 100%;
    height: 450px;
    background: #ffffff;
    border: 1px solid #ddd;
    border-radius: 10px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    font-family: "Helvetica", "Arial", sans-serif;
  }
  .chat-header {
    background: #f0f2f6;
    padding: 12px;
    text-align: center;
    font-size: 18px;
    font-weight: bold;
  }
  .chat-box {
    flex: 1;
    padding: 12px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .message {
    padding: 8px 12px;
    border-radius: 8px;
    max-width: 75%;
    line-height: 1.4;
    font-size: 14px;
  }
  .user {
    background: #e8f0fe;
    align-self: flex-end;
  }
  .assistant {
    background: #f1f3f4;
    align-self: flex-start;
  }
  .input-area {
    display: flex;
    border-top: 1px solid #ddd;
    padding: 10px;
    background: #fafafa;
  }
  #user-input {
    flex: 1;
    padding: 8px;
    border: 1px solid #ccc;
    border-radius: 5px;
    font-size: 14px;
  }
  #send-btn {
    margin-left: 10px;
    padding: 8px 15px;
    background: #4a90e2;
    color: white;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-size: 14px;
  }
  #send-btn:hover {
    background: #357ab8;
  }
</style>

<div class="chat-container">
  <div class="chat-header"></div>
  <div id="chat-box" class="chat-box"></div>
  <div class="input-area">
    <input type="text" id="user-input" placeholder="Ask your question here..." />
    <button id="send-btn">Send</button>
  </div>
</div>

<script>
  const chatBox = document.getElementById("chat-box");
  const userInput = document.getElementById("user-input");
  const sendBtn = document.getElementById("send-btn");

  function appendMessage(role, content) {
    const msgDiv = document.createElement("div");
    msgDiv.classList.add("message", role);
    msgDiv.innerHTML = content.replace(/\\n/g, "<br>");
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  sendBtn.addEventListener("click", () => {
    const msg = userInput.value.trim();
    if (!msg) return;
    appendMessage("user", msg);
    userInput.value = "";

    // Extended predefined multi-tip responses
    setTimeout(() => {
      let reply = "🤔 Sorry, I can only answer questions about sleep, health, and relaxation. Try asking me about stress, insomnia, routine, or diet!";
      const m = msg.toLowerCase();

      if (m.includes("hi") || m.includes("hello") || m.includes("hey")) reply = "👋 Hi there! How can I help you with your sleep today?";
  else if (m.includes("stress")) reply = "🧘 Stress relief tips: <br>1. Practice deep breathing or meditation <br>2. Write thoughts in a journal before bed <br>3. Do light stretches to relax muscles.";
      else if (m.includes("screen")) reply = "📱 Screen time advice: <br>1. Avoid phones/TV 1 hour before bed <br>2. Use blue light filters if needed <br>3. Try reading a book instead.";
      else if (m.includes("insomnia")) reply = "💡 Insomnia management: <br>1. Stick to a consistent sleep schedule <br>2. Avoid caffeine/alcohol late in the day <br>3. Use your bed only for sleep, not work.";
      else if (m.includes("caffeine")) reply = "☕ Caffeine tips: <br>1. Avoid coffee after 2 PM <br>2. Switch to herbal tea in evenings <br>3. Watch out for hidden caffeine in sodas/energy drinks.";
      else if (m.includes("alcohol")) reply = "🍷 Alcohol & sleep: <br>1. Alcohol can cause lighter, disrupted sleep <br>2. Avoid drinking right before bed <br>3. Stay hydrated with water if you drink.";
      else if (m.includes("exercise")) reply = "🏃 Exercise & sleep: <br>1. Regular workouts improve sleep quality <br>2. Best time: morning or afternoon <br>3. Avoid intense exercise right before bed.";
      else if (m.includes("food") || m.includes("diet")) reply = "🥗 Food & sleep: <br>1. Avoid heavy/spicy meals late at night <br>2. A light snack (banana, warm milk) can help <br>3. Limit sugar before bedtime.";
      else if (m.includes("temperature") || m.includes("room")) reply = "🌡️ Ideal sleep environment: <br>1. Keep your room cool (18–22°C) <br>2. Use blackout curtains <br>3. Reduce noise with earplugs/white noise.";
      else if (m.includes("routine")) reply = "📅 Routine tips: <br>1. Sleep/wake up at the same time daily <br>2. Create a calming pre-bed routine <br>3. Avoid naps too late in the day.";
      else if (m.includes("water") || m.includes("drink")) reply = "💧 Hydration advice: <br>1. Drink water throughout the day <br>2. Avoid large amounts right before bed <br>3. Herbal teas (chamomile) can help sleep.";
      else if (m.includes("wake up") || m.includes("morning")) reply = "⏰ Morning energy tips: <br>1. Place your alarm across the room <br>2. Open curtains for sunlight immediately <br>3. Do light stretches to wake your body.";
      else if (m.includes("nap")) reply = "😴 Nap tips: <br>1. Keep naps under 30 minutes <br>2. Best nap time: 1–3 PM <br>3. Avoid napping too close to bedtime.";
      else if (m.includes("dream")) reply = "💭 Dreams & sleep: <br>1. Stress can trigger vivid dreams <br>2. A regular schedule supports healthy REM sleep <br>3. Avoid heavy food before bed to reduce nightmares.";
      else if (m.includes("mental health") || m.includes("anxiety")) reply = "❤️ Mental health & sleep: <br>1. Journaling before bed can reduce racing thoughts <br>2. Try mindfulness or gratitude practice <br>3. Seek help if anxiety regularly affects sleep.";
      else if (m.includes("light")) reply = "☀️ Light & sleep: <br>1. Get sunlight in the morning <br>2. Keep evenings dim to trigger melatonin <br>3. Use blackout curtains for deep sleep.";
      else if (m.includes("melatonin")) reply = "💊 Melatonin info: <br>1. Useful for jet lag or shift work <br>2. Take only under medical guidance <br>3. Best taken 30–60 mins before sleep.";
      else if (m.includes("shift work")) reply = "🌙 Shift work survival: <br>1. Keep a consistent sleep schedule <br>2. Use blackout curtains during the day <br>3. Limit caffeine towards end of shift.";
      else if (m.includes("music")) reply = "🎶 Sleep music: <br>1. Calm instrumental or white noise helps <br>2. Avoid loud/fast music <br>3. Nature sounds can also be relaxing.";
      else if (m.includes("relax")) reply = "🛀 Relaxation ideas: <br>1. Take a warm shower <br>2. Read a light book <br>3. Try guided meditation apps.";
      else if (m.includes("pillows") || m.includes("bed")) reply = "🛏️ Bed comfort: <br>1. Choose a supportive mattress <br>2. Replace old pillows regularly <br>3. Keep bedding clean & cozy.";
      else if (m.includes("bye") || m.includes("good night")) reply = "🌌 Good night! Sleep well and wake refreshed. 🌙";

      appendMessage("assistant", reply);
    }, 600);
  });
</script>
"""


components.html(chatbot_html, height=500, scrolling=False)

