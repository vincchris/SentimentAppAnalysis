import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import Optional, List
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Scikit-learn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Sentiment Analysis Tool",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Sentiment Analysis dengan Multiple Algorithms")
st.markdown("Upload dataset CSV untuk melatih model sentiment analysis dan melihat evaluasi performanya")

# Sidebar for algorithm selection
st.sidebar.header("⚙️ Konfigurasi Model")

# Algorithm selection
algorithms = {
    "svm": "Support Vector Machine",
    "naive_bayes": "Naive Bayes",
    "random_forest": "Random Forest"
}

selected_algorithm = st.sidebar.selectbox(
    "Pilih Algoritma:",
    options=list(algorithms.keys()),
    format_func=lambda x: algorithms[x]
)

# Test size selection
test_size = st.sidebar.slider(
    "Test Size (%):",
    min_value=10,
    max_value=40,
    value=20,
    step=5
) / 100

# Text preprocessing function
def preprocess_text(text):
    """Basic text preprocessing"""
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

def map_score_to_sentiment(score):
    """Map numeric score to sentiment labels"""
    if pd.isna(score):
        return "NEUTRAL"

    if score >= 4:
        return "POSITIVE"
    elif score <= 2:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

def train_sklearn_model(X_train, y_train, algorithm):
    """Train scikit-learn models"""

    if algorithm == "svm":
        model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
            ('classifier', SVC(kernel='linear', probability=True, random_state=42))
        ])
    elif algorithm == "naive_bayes":
        model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
            ('classifier', MultinomialNB(alpha=1.0))
        ])
    elif algorithm == "random_forest":
        model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Train the model
    model.fit(X_train, y_train)
    return model

def create_confusion_matrix_plot(cm, labels):
    """Create confusion matrix heatmap"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    return fig

def create_metrics_chart(metrics):
    """Create metrics bar chart"""
    fig = go.Figure(data=[
        go.Bar(name='Metrics',
               x=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
               y=[metrics['accuracy'], metrics['precision'],
                  metrics['recall'], metrics['f1_score']],
               marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ])

    fig.update_layout(
        title='Model Performance Metrics',
        yaxis_title='Score',
        showlegend=False,
        height=400
    )

    return fig

def create_sentiment_distribution_plot(sentiments):
    """Create sentiment distribution pie chart"""
    sentiment_counts = pd.Series(sentiments).value_counts()

    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution",
        color_discrete_map={
            'POSITIVE': '#2ca02c',
            'NEGATIVE': '#d62728',
            'NEUTRAL': '#ff7f0e'
        }
    )

    return fig

# Main content
st.header("📁 Upload Dataset")

uploaded_file = st.file_uploader(
    "Pilih file CSV",
    type=['csv'],
    help="File CSV harus memiliki kolom 'content' dan 'score'"
)

if uploaded_file is not None:
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)

        # Display dataset info
        st.subheader("📋 Informasi Dataset")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())

        # Show sample data
        st.subheader("🔍 Sample Data")
        st.dataframe(df.head())

        # Validate required columns
        if "content" not in df.columns:
            st.error("❌ CSV harus memiliki kolom 'content'")
            st.stop()

        if "score" not in df.columns:
            st.error("❌ CSV harus memiliki kolom 'score' untuk training")
            st.stop()

        # Remove empty rows
        df_clean = df.dropna(subset=['content', 'score'])

        if len(df_clean) == 0:
            st.error("❌ Tidak ada data valid dalam CSV")
            st.stop()

        # Preprocess data
        texts = df_clean["content"].tolist()
        processed_texts = [preprocess_text(text) for text in texts]
        true_sentiments = df_clean["score"].apply(map_score_to_sentiment).tolist()

        # Check if we have enough data for each class
        sentiment_counts = pd.Series(true_sentiments).value_counts()

        st.subheader("📊 Distribusi Sentiment dalam Dataset")
        fig_dist = create_sentiment_distribution_plot(true_sentiments)
        st.plotly_chart(fig_dist, use_container_width=True)

        # Show sentiment counts
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Positive", sentiment_counts.get('POSITIVE', 0))
        with col2:
            st.metric("Negative", sentiment_counts.get('NEGATIVE', 0))
        with col3:
            st.metric("Neutral", sentiment_counts.get('NEUTRAL', 0))

        if len(sentiment_counts) < 2:
            st.error("❌ Dataset harus memiliki setidaknya 2 kelas sentiment yang berbeda")
            st.stop()

        if st.button("🚀 Latih Model", type="primary"):
            with st.spinner(f"Melatih model menggunakan {algorithms[selected_algorithm]}..."):
                try:
                    # Split data for training
                    X_train, X_test, y_train, y_test = train_test_split(
                        processed_texts, true_sentiments,
                        test_size=test_size,
                        random_state=42,
                        stratify=true_sentiments
                    )

                    # Train model
                    model = train_sklearn_model(X_train, y_train, selected_algorithm)

                    # Make predictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)

                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        y_test, y_pred, average='weighted', zero_division=0
                    )

                    # Store results in session state
                    st.session_state.model_results = {
                        'model': model,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'y_test': y_test,
                        'y_pred': y_pred,
                        'algorithm': selected_algorithm,
                        'test_size': test_size
                    }

                    st.success("✅ Model berhasil dilatih!")

                except Exception as e:
                    st.error(f"❌ Error saat melatih model: {str(e)}")

        # Display results if available
        if 'model_results' in st.session_state:
            results = st.session_state.model_results

            st.header("📈 Hasil Evaluasi Model")

            # Metrics overview
            st.subheader("🎯 Performance Metrics")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Accuracy",
                    f"{results['accuracy']:.4f}",
                    delta=f"{(results['accuracy']-0.5):.4f}" if results['accuracy'] > 0.5 else None
                )
            with col2:
                st.metric("Precision", f"{results['precision']:.4f}")
            with col3:
                st.metric("Recall", f"{results['recall']:.4f}")
            with col4:
                st.metric("F1-Score", f"{results['f1_score']:.4f}")

            # Metrics chart
            st.subheader("📊 Performance Visualization")
            metrics_data = {
                'accuracy': results['accuracy'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1_score': results['f1_score']
            }
            fig_metrics = create_metrics_chart(metrics_data)
            st.plotly_chart(fig_metrics, use_container_width=True)

            # Classification Report
            st.subheader("📋 Detailed Classification Report")
            report = classification_report(
                results['y_test'],
                results['y_pred'],
                output_dict=True
            )

            # Convert to DataFrame for better display
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(4))

            # Model info
            st.subheader("ℹ️ Model Information")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Algorithm:** {algorithms[results['algorithm']]}")
                st.info(f"**Test Size:** {int(results['test_size']*100)}%")
            with col2:
                st.info(f"**Training Samples:** {len(processed_texts) - len(results['y_test'])}")
                st.info(f"**Test Samples:** {len(results['y_test'])}")

            # Feature importance for Random Forest
            if results['algorithm'] == 'random_forest':
                st.subheader("🌟 Feature Importance (Top 20)")
                try:
                    # Get feature names and importance
                    feature_names = results['model']['tfidf'].get_feature_names_out()
                    importance = results['model']['classifier'].feature_importances_

                    # Create DataFrame and sort
                    feat_imp_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importance
                    }).sort_values('importance', ascending=False).head(20)

                    # Create horizontal bar chart
                    fig_feat = px.bar(
                        feat_imp_df,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title='Top 20 Most Important Features'
                    )
                    fig_feat.update_layout(height=600)
                    st.plotly_chart(fig_feat, use_container_width=True)

                except Exception as e:
                    st.warning(f"Tidak dapat menampilkan feature importance: {str(e)}")

    except Exception as e:
        st.error(f"❌ Error saat memproses file: {str(e)}")

else:
    st.info("👆 Upload file CSV untuk memulai analisis sentiment")

    # Show example data format
    st.subheader("📝 Format Data yang Diperlukan")
    example_data = {
        'content': [
            'Produk ini sangat bagus dan berkualitas',
            'Pelayanan buruk sekali, tidak memuaskan',
            'Biasa saja, tidak ada yang istimewa'
        ],
        'score': [5, 1, 3]
    }
    example_df = pd.DataFrame(example_data)
    st.dataframe(example_df)

    st.markdown("""
    **Penjelasan kolom:**
    - `content`: Teks yang akan dianalisis sentimentnya
    - `score`: Nilai numerik sentiment (1-2: Negative, 3: Neutral, 4-5: Positive)
    """)

# Footer
st.markdown("---")
st.markdown("📊 **Sentiment Analysis Tool** - Built with Streamlit")