# ===== Full Streamlit Dashboard for Customer Reviews =====

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# ===== 1️⃣ Load Dataset =====
df = pd.read_csv(r"C:\Users\gold\Documents\python\project-5\reviews_clean.csv")  

# Convert date
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Optional: compute review length
df['review_length'] = df['review'].astype(str).apply(len)

# ===== 2️⃣ Title =====
st.title("Customer Sentiment Analysis Dashboard")
st.markdown("📊 Visualizing trends and insights from customer reviews")

# ===== 3️⃣ Key Question 1: Overall Sentiment =====
st.subheader("1️⃣ Overall Sentiment of User Reviews")
if 'sentiment' in df.columns:
    sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100
    st.write(sentiment_counts)
    st.bar_chart(sentiment_counts)

# ===== 4️⃣ Key Question 2: Sentiment by Rating =====
st.subheader("2️⃣ Sentiment vs Rating")
if 'rating' in df.columns and 'sentiment' in df.columns:
    pivot_rating = df.groupby(['rating','sentiment']).size().unstack(fill_value=0)
    st.write(pivot_rating)
    st.bar_chart(pivot_rating)

# ===== 5️⃣ Key Question 3: Keywords per Sentiment =====
st.subheader("3️⃣ Most Common Words per Sentiment")
if 'review' in df.columns and 'sentiment' in df.columns:
    sentiments = df['sentiment'].unique()
    for s in sentiments:
        st.markdown(f"**{s} Reviews**")
        text = " ".join(df[df['sentiment']==s]['review'].astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots(figsize=(10,4))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

# ===== 6️⃣ Key Question 4: Sentiment Over Time =====
st.subheader("4️⃣ Sentiment Change Over Time")
if 'date' in df.columns and 'sentiment' in df.columns:
    df_time = df.copy()
    df_time['month_year'] = df_time['date'].dt.to_period('M')
    time_pivot = df_time.groupby(['month_year','sentiment']).size().unstack(fill_value=0)
    st.line_chart(time_pivot)

# ===== 7️⃣ Key Question 5: Verified Users Sentiment =====
st.subheader("5️⃣ Verified vs Non-Verified Reviews Sentiment")
if 'verified_purchase' in df.columns and 'sentiment' in df.columns:
    verified_pivot = df.groupby(['verified_purchase','sentiment']).size().unstack(fill_value=0)
    st.write(verified_pivot)
    st.bar_chart(verified_pivot)

# ===== 6️⃣ Key Question 6: Review Length vs Sentiment =====
st.subheader("6️⃣ Review Length vs Sentiment")
if 'review_length' in df.columns and 'sentiment' in df.columns:
    fig, ax = plt.subplots(figsize=(10,5))
    sns.boxplot(data=df, x='sentiment', y='review_length', palette="Set3", ax=ax)
    ax.set_ylabel("Review Length (characters)")
    st.pyplot(fig)

# ===== 7️⃣ Key Question 7: Sentiment by Location =====
st.subheader("7️⃣ Sentiment by Location")
if 'location' in df.columns and 'sentiment' in df.columns:
    location_pivot = df.groupby(['location','sentiment']).size().unstack(fill_value=0)
    st.write(location_pivot.head(20))  # show top 20 locations
    st.bar_chart(location_pivot.head(20))

# ===== 8️⃣ Key Question 8: Sentiment by Platform =====
st.subheader("8️⃣ Sentiment Across Platforms")
if 'platform' in df.columns and 'sentiment' in df.columns:
    platform_pivot = df.groupby(['platform','sentiment']).size().unstack(fill_value=0)
    st.write(platform_pivot)
    st.bar_chart(platform_pivot)

# ===== 9️⃣ Key Question 9: ChatGPT Version Sentiment =====
st.subheader("9️⃣ ChatGPT Version vs Sentiment")
if 'version' in df.columns and 'sentiment' in df.columns:
    version_pivot = df.groupby(['version','sentiment']).size().unstack(fill_value=0)
    st.write(version_pivot)
    st.bar_chart(version_pivot)

# ===== 🔟 Key Question 10: Common Negative Feedback Themes =====
st.subheader("🔟 Most Common Negative Feedback Themes")
if 'sentiment' in df.columns and 'review' in df.columns:
    negative_text = " ".join(df[df['sentiment']=='Negative']['review'].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_text)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

