import streamlit as st
import pickle

# Load the vectorizer and model
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# Streamlit app
st.set_page_config(page_title="Spam Classifier", page_icon="📧", layout="centered")

st.title("📧 Email / SMS Spam Classifier")
st.write("This app uses a Machine Learning model to classify messages as **Spam** or **Ham (Not Spam)**.")

# Input box
message = st.text_area("✍️ Enter your message:")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("⚠️ Please enter a message to classify.")
    else:
        # Transform message
        vectorized_message = vectorizer.transform([message])

        # Prediction
        prediction = model.predict(vectorized_message)[0]

        # Result
        if prediction == 1:
            st.error("🚨 This looks like **SPAM**!")
        else:
            st.success("✅ This looks like **HAM** (Not Spam).")
