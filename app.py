import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from textblob import TextBlob
from wordcloud import WordCloud

# Load Dataset
def load_data():
    df = pd.read_csv(r"C:\Users\Abhinaya\Downloads\youtube_trending_Analysis\youtube_trending_Analysis\US_youtube_trending_data.csv")
    df = df.drop_duplicates().dropna()
    df = df[df['view_count'] > 0]  # Remove invalid values
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    return df

# Sentiment Analysis
def sentiment_analysis(text):

    
    analysis = TextBlob(text)
    return 1 if analysis.sentiment.polarity > 0 else 0

# Category Analysis
def show_category_analysis(df):
    category_counts = df['categoryId'].value_counts()
    fig = px.bar(category_counts, x=category_counts.index, y=category_counts.values,
                 labels={'x': 'Category ID', 'y': 'Number of Videos'},
                 title="Most Popular Video Categories")
    st.plotly_chart(fig)

# Engagement Metrics Visualization
def show_engagement_metrics(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[['view_count', 'likes', 'comment_count']].corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Keyword Trend Analysis
def show_keyword_trends(df):
    text = ' '.join(df['tags'].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# Machine Learning-Based Trend Prediction
def predict_trend(df):
    df['sentiment'] = df['title'].apply(sentiment_analysis)
    
    features = df[['view_count', 'likes', 'comment_count', 'sentiment']]
    target = (df['view_count'] > df['view_count'].mean()).astype(int)  # 1 if trending, 0 otherwise
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    model = XGBClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    st.write(f"### Prediction Model Accuracy: {accuracy:.2f}")
    st.text(classification_report(y_test, predictions))
    return model

# Streamlit UI
st.title("📊 YouTube Trend Analysis Dashboard")
df = load_data()

st.sidebar.header("Select Analysis")
option = st.sidebar.selectbox("Choose an Analysis:",
                              ["Category Analysis", "Engagement Metrics", "Keyword Trends", "Trend Prediction"])

if option == "Category Analysis":
    show_category_analysis(df)
elif option == "Engagement Metrics":
    show_engagement_metrics(df)
elif option == "Keyword Trends":
    show_keyword_trends(df)
elif option == "Trend Prediction":
    model = predict_trend(df)