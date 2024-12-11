import os
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from classification import classify_email  # Update the import path if necessary

# Set Streamlit configuration
def launch_app():
    st.set_page_config(
        page_title="Spam Email Classifier",
        page_icon="📧",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state for page navigation
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Home"

    # Sidebar navigation
    with st.sidebar:
        st.title("📧 Spam Email Classifier")
        page = st.radio(
            "Navigate:",
            ["Home", "Classify Email", "Insights"],
            key="sidebar_navigation",
            index=["Home", "Classify Email", "Insights"].index(st.session_state["current_page"]),
        )
        st.session_state["current_page"] = page

    # Page rendering based on the current page
    if st.session_state["current_page"] == "Home":
        # Enhanced Home page design
        st.markdown(
            """
            <div style="
                background-color: #007BFF;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                color: white;
                margin-bottom: 20px;">
                <h1 style="font-size: 40px; font-weight: bold;">Welcome to Spam Classifier</h1>
                <p style="font-size: 18px;">Detect spam emails instantly and efficiently!</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <p style="font-size: 18px; text-align: center;">
                Upload your email files or paste email content to classify them as Spam or Ham. Explore insightful visualizations
                and uncover patterns in your email content with this intuitive and sleek interface.
            </p>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Start Classifying Emails"):
            st.session_state["current_page"] = "Classify Email"

    elif st.session_state["current_page"] == "Classify Email":
        # Email classification logic
        st.title("📧 Classify Your Email")
        st.markdown("### Upload files or enter email content below.")

        # Drag-and-drop multiple file upload
        uploaded_files = st.file_uploader(
            "Upload text files (you can upload multiple files)", 
            type=["txt"], 
            accept_multiple_files=True
        )

        if uploaded_files:
            st.markdown("### Uploaded Files:")
            results = []
            for file in uploaded_files:
                email_content = file.read().decode("utf-8")
                result = classify_email(email_content)
                results.append((file.name, result))

            st.markdown("### Classification Results:")
            for file_name, classification in results:
                color = "red" if classification.lower() == "spam" else "green"
                st.markdown(
                    f'<p style="color: {color}; font-size: 18px;">{file_name}: {classification.upper()}</p>',
                    unsafe_allow_html=True,
                )
        else:
            email_input = st.text_area("Or paste your email content here:", height=200)

            if st.button("Classify"):
                if email_input.strip():
                    result = classify_email(email_input)
                    color = "red" if result.lower() == "spam" else "green"
                    st.markdown(
                        f'<p style="color: {color}; font-size: 24px; font-weight: bold;">The email is classified as: {result.upper()}</p>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning("Please provide email content to classify.")

    elif st.session_state["current_page"] == "Insights":
        # Visualization and insights page
        st.title("🔁 Email Classification Insights")
        st.markdown("### Explore the patterns in your email content!")

        # Word cloud visualization
        st.markdown("#### Word Cloud")
        example_text = (
            "Win now! Free prize. Offer limited. Congratulations! Free entry. "
            "Please respond urgently. Claim now! Exclusive deal. Immediate attention."
        )
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(example_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

        # Feature importance (placeholder for actual values)
        st.markdown("#### Feature Importance")
        st.bar_chart({"Feature": ["Offer", "Congratulations", "Limited"], "Importance": [0.8, 0.6, 0.4]})

        # Model performance metrics (example values)
        st.markdown("#### Model Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", "97%")
        col2.metric("Precision", "95%")
        col3.metric("Recall", "96%")

        st.markdown("Explore more by navigating to the **Classify Email** tab!")
