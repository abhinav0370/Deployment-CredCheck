# main.py

import streamlit as st
import os
from langdetect import detect
from audio_to_text import audio_to_text
from convert_to_english import translation
from cred_check import fake_news_detector
from claimbuster_check import check_claim
from top_headlines import fetch_headlines
from img_to_text import extract_text_from_image  # Import the image text extraction function
from video_to_audio import extract_audio  # Import the video to audio function

# Set page configuration for a modern layout
st.set_page_config(
    page_title="🔍 Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
custom_css = """
<style>
/* Update background color */
body {
    background-color: #f0f2f6;
}

/* Style for buttons */
.css-1emrehy.edgvbvh3 {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
}

/* Style for headers */
h1 {
    color: #333333;
    font-family: 'Arial', sans-serif;
}

/* Style for sidebar */
.sidebar .sidebar-content {
    background-color: #ffffff;
}

/* Remove default Streamlit footer and hamburger menu */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Title
st.title("🔍 Fake News Detector")

# Sidebar for Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose Input Method", ["Analyze Headline"])

def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False

# Define threshold for fake news classification
SCORE_THRESHOLD = 0.5

def classify_claim(score):
    return "🔴 Fake" if score < SCORE_THRESHOLD else "🟢 Real"

def classify_auth(is_fake):
    return "🔴 Fake" if is_fake else "🟢 Real"

if app_mode == "Analyze Headline":
    input_type = st.radio("Select Input Method", ("Text", "Audio", "Image", "Video"))  # Added "Video"

    if input_type == "Text":
        headline = st.text_input("📝 Enter the news headline:")
        if st.button("Analyze", key="analyze_text"):
            if headline:
                with st.spinner('Processing...'):
                    if not is_english(headline):
                        st.info("🔄 Translating to English...")
                        headline = translation(headline)
                        st.success(f"**Translated Text:** {headline}")
                    # Auth Detector
                    st.info("📊 Analyzing headline with CredCheck...")
                    auth_result = fake_news_detector(headline)
                    st.write(f"**CredCheck Classification:** {classify_auth(auth_result.get('is_fake', False))}")
                    
                    # ClaimBuster Check
                    st.info("🔍 Checking claim with ClaimBuster...")
                    claimbuster_result = check_claim(headline)
                    
                    if "error" in claimbuster_result:
                        st.error(claimbuster_result["error"])
                    else:
                        for result in claimbuster_result["results"]:
                            score = result["score"]
                            classification = classify_claim(score)
                            st.write(f"**Claim:** {result['text']}")
                            st.write(f"**Score:** {score}")
                            st.write(f"**ClaimBuster Classification:** {classification}")
                            st.markdown("---")
            else:
                st.error("⚠️ Please enter a headline.")

    elif input_type == "Audio":
        audio_file = st.file_uploader("🎤 Upload an audio file", type=["wav", "mp3", "flac"])
        if st.button("Analyze", key="analyze_audio"):
            if audio_file:
                with st.spinner('🎧 Processing audio...'):
                    # Save the uploaded audio file temporarily
                    temp_audio_path = "temp_audio.wav"
                    with open(temp_audio_path, "wb") as f:
                        f.write(audio_file.getbuffer())
                    # Convert audio to text
                    text = audio_to_text(temp_audio_path)
                    os.remove(temp_audio_path)
                    st.write(f"**Transcribed Text:** {text}")
                    if not is_english(text):
                        st.info("🔄 Translating to English...")
                        text = translation(text)
                        st.success(f"**Translated Text:** {text}")
                    # Auth Detector
                    st.info("📊 Analyzing headline with CredCheck...")
                    auth_result = fake_news_detector(text)
                    st.write(f"**CredCheck Classification:** {classify_auth(auth_result.get('is_fake', False))}")
                    
                    # ClaimBuster Check
                    st.info("🔍 Checking claim with ClaimBuster...")
                    claimbuster_result = check_claim(text)
                    
                    if "error" in claimbuster_result:
                        st.error(claimbuster_result["error"])
                    else:
                        for result in claimbuster_result["results"]:
                            score = result["score"]
                            classification = classify_claim(score)
                            st.write(f"**Claim:** {result['text']}")
                            st.write(f"**Score:** {score}")
                            st.write(f"**ClaimBuster Classification:** {classification}")
                            st.markdown("---")
            else:
                st.error("⚠️ Please upload an audio file.")

    elif input_type == "Image":  # Existing Image Input Section
        image_file = st.file_uploader("🖼️ Upload an image file", type=["png", "jpg", "jpeg", "tiff"])
        if st.button("Analyze", key="analyze_image"):
            if image_file:
                with st.spinner('🖼️ Processing image...'):
                    # Save the uploaded image file temporarily
                    temp_image_path = "temp_image"
                    with open(temp_image_path, "wb") as f:
                        f.write(image_file.getbuffer())
                    # Extract text from image
                    extracted_text = extract_text_from_image(temp_image_path)
                    os.remove(temp_image_path)
                    st.write(f"**Extracted Text:** {extracted_text}")
                    
                    if extracted_text.startswith("Error:"):
                        st.error(extracted_text)
                    else:
                        if not is_english(extracted_text):
                            st.info("🔄 Translating to English...")
                            extracted_text = translation(extracted_text)
                            st.success(f"**Translated Text:** {extracted_text}")
                        # Auth Detector
                        st.info("📊 Analyzing text with CredCheck...")
                        auth_result = fake_news_detector(extracted_text)
                        st.write(f"**CredCheck Classification:** {classify_auth(auth_result.get('is_fake', False))}")
                        
                        # ClaimBuster Check
                        st.info("🔍 Checking claim with ClaimBuster...")
                        claimbuster_result = check_claim(extracted_text)
                        
                        if "error" in claimbuster_result:
                            st.error(claimbuster_result["error"])
                        else:
                            for result in claimbuster_result["results"]:
                                score = result["score"]
                                classification = classify_claim(score)
                                st.write(f"**Claim:** {result['text']}")
                                st.write(f"**Score:** {score}")
                                st.write(f"**ClaimBuster Classification:** {classification}")
                                st.markdown("---")
            else:
                st.error("⚠️ Please upload an image file.")

    elif input_type == "Video":  # New Video Input Section
        video_file = st.file_uploader("🎥 Upload a video file", type=["mp4", "avi", "mov", "mkv"])
        if st.button("Analyze", key="analyze_video"):
            if video_file:
                with st.spinner('🎥 Processing video...'):
                    # Save the uploaded video file temporarily
                    temp_video_path = "temp_video.mp4"
                    with open(temp_video_path, "wb") as f:
                        f.write(video_file.getbuffer())
                    # Extract audio from video
                    extracted_audio_path = extract_audio(temp_video_path)
                    if extracted_audio_path.startswith("Error:"):
                        st.error(extracted_audio_path)
                        os.remove(temp_video_path)  # Ensure video file is removed even on error
                    else:
                        # Convert audio to text
                        text = audio_to_text(extracted_audio_path)
                        st.write(f"**Transcribed Text from Video:** {text}")
                        # Remove temporary files
                        os.remove(extracted_audio_path)
                        os.remove(temp_video_path)
                        if not is_english(text):
                            st.info("🔄 Translating to English...")
                            text = translation(text)
                            st.success(f"**Translated Text:** {text}")
                        # Auth Detector
                        st.info("📊 Analyzing text with CredCheck...")
                        auth_result = fake_news_detector(text)
                        st.write(f"**CredCheck Classification:** {classify_auth(auth_result.get('is_fake', False))}")
                        
                        # ClaimBuster Check
                        st.info("🔍 Checking claim with ClaimBuster...")
                        claimbuster_result = check_claim(text)
                        
                        if "error" in claimbuster_result:
                            st.error(claimbuster_result["error"])
                        else:
                            for result in claimbuster_result["results"]:
                                score = result["score"]
                                classification = classify_claim(score)
                                st.write(f"**Claim:** {result['text']}")
                                st.write(f"**Score:** {score}")
                                st.write(f"**ClaimBuster Classification:** {classification}")
                                st.markdown("---")
            else:
                st.error("⚠️ Please upload a video file.")

# Separate Top Headlines Section
st.header("📰 Real Time News Analysis")
if st.button("Fetch Top Headlines"):
    with st.spinner('📥 Fetching top headlines...'):
        headlines = fetch_headlines()
        if isinstance(headlines, list):
            st.write("Fetched headlines:")
            for idx, headline in enumerate(headlines, start=1):
                st.write(f"{idx}. {headline}")
            
            # Set number of headlines to analyze to ten
            num_to_analyze = min(10, len(headlines))
            headlines_to_analyze = headlines[:num_to_analyze]

            # Send each headline to ClaimBuster only
            with st.spinner('🔍 Analyzing headlines with ClaimBuster...'):
                for idx, headline in enumerate(headlines_to_analyze, start=1):
                    st.write(f"**Headline {idx}:** {headline}")
                    
                    # ClaimBuster Check
                    claimbuster_result = check_claim(headline)
                    if "error" in claimbuster_result:
                        st.error(f"Error analyzing headline {idx} with ClaimBuster: {claimbuster_result['error']}")
                    else:
                        for result in claimbuster_result["results"]:
                            score = result["score"]
                            classification = classify_claim(score)
                            st.write(f"**ClaimBuster Classification:** {classification}")
                    st.markdown("---")
        else:
            st.error(headlines.get("error", "⚠️ An error occurred."))

# Footer
st.markdown("---")
st.markdown("Developed by [Team Ignite](https://github.com/yourusername)")