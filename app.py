import streamlit as st
import assemblyai as ai
import matplotlib.pyplot as plt
from wordcloud import WordCloud

ai.settings.api_key = "3b448c6f3e994b86882c4a9dc6a5290a"

st.title("Customer Satisfaction from Audio Recording")

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3","wav"])

if uploaded_file is not None:
    audio_url = "./temp_audio.mp3" 
    with open(audio_url, "wb") as f:
        f.write(uploaded_file.read())

    config = ai.TranscriptionConfig(sentiment_analysis=True, auto_highlights=True)
    transcript = ai.Transcriber().transcribe(audio_url, config)

    positive_count = 1
    neutral_count = 1
    negative_count = 1

    positive_score = 0
    neutral_score = 0
    negative_score = 0

    for sentiment_result in transcript.sentiment_analysis:
        if sentiment_result.sentiment == ai.SentimentType.positive:
            positive_count += 1
            positive_score += sentiment_result.confidence
        elif sentiment_result.sentiment == ai.SentimentType.neutral:
            neutral_count += 1
            neutral_score += sentiment_result.confidence
        else:
            negative_count += 1
            negative_score += sentiment_result.confidence

    if positive_count > neutral_count and positive_count > negative_count:
        sentiment = "Positive"
        resultScore = positive_score / positive_count
    elif negative_count > neutral_count and negative_count > positive_count:
        sentiment = "Negative"
        resultScore = negative_score / negative_count
    else:
        sentiment = "Neutral"
        resultScore = neutral_score / neutral_count

    st.subheader("Sentiment Analysis:")
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Confidence Score: {resultScore:.2f}")


    st.subheader("Word Cloud of Highlighted Words:")
    highlights = []
    for result in transcript.auto_highlights.results:
        highlights.append(result.text)

    highlighted_text = " ".join(highlights)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(highlighted_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)
