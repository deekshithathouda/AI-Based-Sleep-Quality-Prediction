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
        history = st.selectbox("Sleep Disorder History", ["No", "Yes"])

    submitted = st.form_submit_button("Predict")

# Function to generate personalized suggestions
def generate_suggestions(user_inputs, prediction):
    suggestions = []
    
    # Sleep Duration Analysis
    if user_inputs['sleep_duration'] < 7:
        suggestions.append("💤 Increase Sleep Duration: Aim for 7-9 hours of sleep nightly. Consider going to bed 30 minutes earlier.")
    elif user_inputs['sleep_duration'] > 9:
        suggestions.append("💤 Optimize Sleep Duration: While 7-9 hours is ideal, excessive sleep can sometimes indicate underlying issues. Maintain consistent sleep patterns.")
    
    # Physical Activity
    if user_inputs['activity'] < 30:
        suggestions.append("🏃 Boost Physical Activity: Try to get at least 30 minutes of moderate exercise daily. Even a brisk walk can improve sleep quality.")
    elif user_inputs['activity'] > 120:
        suggestions.append("⏰ Time Your Workouts: Intense late-evening exercise might disrupt sleep. Consider finishing workouts 2-3 hours before bedtime.")
    
    # Stress Level
    if user_inputs['stress'] >= 7:
        suggestions.append("🧘 Manage Stress: Practice mindfulness, deep breathing, or meditation. Consider keeping a worry journal to clear your mind before bed.")
    
    # Caffeine Intake
    if user_inputs['caffeine'] >= 3:
        suggestions.append("☕ Reduce Caffeine: Limit to 1-2 cups daily and avoid caffeine after 2 PM. Try switching to herbal tea in the afternoon.")
    
    # Alcohol Consumption
    if user_inputs['alcohol'] >= 2:
        suggestions.append("🍷 Moderate Alcohol: Alcohol disrupts sleep architecture. Limit to 1 drink daily and avoid within 3 hours of bedtime.")
    
    # Smoking
    if user_inputs['smoker'] == "Yes":
        suggestions.append("🚭 Quit Smoking: Nicotine is a stimulant that interferes with sleep. Consider smoking cessation programs or nicotine replacement therapy.")
    
    # Screen Time
    if user_inputs['screen_time'] >= 2:
        suggestions.append("📱 Reduce Screen Time: Limit screen exposure 1 hour before bed. Use blue light filters or switch to reading a physical book.")
    
    # BMI Analysis
    if user_inputs['bmi'] >= 25:
        suggestions.append("⚖ Healthy Weight: Consider weight management through balanced diet and exercise, as excess weight can contribute to sleep apnea.")
    elif user_inputs['bmi'] < 18.5:
        suggestions.append("🍎 Nutrition Focus: Ensure adequate nutrition, as being underweight can also affect sleep quality and energy levels.")
    
    # Wake-up Consistency
    if user_inputs['wake_consistency'] == "Irregular":
        suggestions.append("⏰ Consistent Schedule: Try waking up at the same time every day, even on weekends. This regulates your body's internal clock.")
    
    # Sleep Environment
    if user_inputs['env_score'] <= 5:
        suggestions.append("🌙 Improve Sleep Environment: Optimize your bedroom for sleep - cool, dark, and quiet. Consider blackout curtains or white noise machines.")
    
    # Water Intake
    if user_inputs['water'] < 2:
        suggestions.append("💧 Increase Hydration: Aim for 2-3 liters of water daily, but reduce intake 1-2 hours before bed to minimize nighttime awakenings.")
    
    # Sleep Disorder History
    if user_inputs['history'] == "Yes":
        suggestions.append("👨‍⚕ Professional Consultation: Consider consulting a sleep specialist for ongoing sleep issues and proper diagnosis.")
    
    # Heart Rate
    if user_inputs['heart_rate'] > 100:
        suggestions.append("❤ Monitor Heart Rate: Elevated resting heart rate may indicate stress or other health issues. Practice relaxation techniques.")
    
    # Age-specific suggestions
    if user_inputs['age'] > 50:
        suggestions.append("👴 Age-Appropriate Routine: As we age, sleep patterns change. Maintain good sleep hygiene and consider shorter, more frequent rest periods if needed.")
    
    # Add motivational message based on prediction
    if prediction == "Good":
        suggestions.insert(0, "🎉 Excellent! Your sleep habits are great! Keep maintaining these healthy routines.")
    elif prediction == "Fair":
        suggestions.insert(0, "📈 Good foundation! With a few adjustments, you can achieve even better sleep quality.")
    else:  # Poor
        suggestions.insert(0, "🔄 Time for positive changes! Small improvements can make a big difference in your sleep quality.")
    
    return suggestions

# Predict and show suggestions
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
    st.success(f"Predicted Sleep Quality: {result}")
    
    # Generate and display personalized suggestions
    st.markdown("---")
    st.subheader("💡Improvement Suggestions")
    
    user_inputs = {
        'sleep_duration': sleep_duration,
        'activity': activity,
        'stress': stress,
        'caffeine': caffeine,
        'alcohol': alcohol,
        'smoker': smoker,
        'screen_time': screen_time,
        'bmi': bmi,
        'wake_consistency': wake_consistency,
        'env_score': env_score,
        'water': water,
        'history': history,
        'heart_rate': heart_rate,
        'age': age
    }
    
    suggestions = generate_suggestions(user_inputs, result)
    
    # Display suggestions in a nice format
    for i, suggestion in enumerate(suggestions, 1):
        st.markdown(f"{i}. {suggestion}")

# Removed "📊 Your Sleep Health Snapshot" section

st.markdown("---")
st.subheader("Get Tips With Our AI Bot")

# Chatbot HTML
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

    // Predefined tips
    setTimeout(() => {
      let reply = "🤔 Sorry, I can only answer questions about sleep, health, and relaxation. Try asking me about stress, insomnia, routine, or diet!";
      const m = msg.toLowerCase();

      if (m.includes("hi") || m.includes("hello") || m.includes("hey")) reply = "👋 Hi there! How can I help you with your sleep today?";
      else if (m.includes("stress")) reply = "🧘 Stress relief tips: <br>1. Practice deep breathing or meditation <br>2. Write thoughts in a journal before bed <br>3. Do light stretches to relax muscles.";
      else if (m.includes("screen")) reply = "📱 Screen time advice: <br>1. Avoid phones/TV 1 hour before bed <br>2. Use blue light filters if needed <br>3. Try reading a book instead.";
      else if (m.includes("insomnia")) reply = "💡 Insomnia management: <br>1. Stick to a consistent sleep schedule <br>2. Avoid caffeine/alcohol late in the day <br>3. Use your bed only for sleep, not work.";
      else if (m.includes("caffeine")) reply = "☕ Caffeine tips: <br>1. Avoid coffee after 2 PM <br>2. Switch to herbal tea in evenings <br>3. Watch out for hidden caffeine in sodas/energy drinks.";
      else if (m.includes("bye") || m.includes("good night")) reply = "🌌 Good night! Sleep well and wake refreshed. 🌙";

      appendMessage("assistant", reply);
    }, 600);
  });
</script>
"""

components.html(chatbot_html, height=500, scrolling=False)
