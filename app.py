import streamlit as st
import torch
import pickle
import pandas as pd

from transformers import AutoTokenizer
from model import TopicGenreModel


# ---------------------------------------------------
# Load Everything
# ---------------------------------------------------
@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        "roberta-base"
    )

    with open("mlb.pkl", "rb") as f:
        mlb = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    model = TopicGenreModel(
        num_topics=32,
        num_genres=len(mlb.classes_),
        vocab_size=8000
    )

    model.load_state_dict(
        torch.load(
            "checkpoint.pt",
            map_location=torch.device("cpu")
        )
    )

    model.eval()

    return model, tokenizer, mlb, vectorizer


model, tokenizer, mlb, vectorizer = load_model()


# ---------------------------------------------------
# UI
# ---------------------------------------------------
st.title("📚 Genre Prediction with Topic Explanations")

st.write(
    "Enter a book or movie description and the model "
    "will predict genres and explain them using topics."
)

summary = st.text_area(
    "Description",
    height=250
)

# ---------------------------------------------------
# Predict
# ---------------------------------------------------
if st.button("Predict"):

    if summary.strip() == "":
        st.warning("Please enter some text.")
        st.stop()

    encoding = tokenizer(
        summary,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():

        topic_dist, genre_logits, _, _, _ = model(
            encoding["input_ids"],
            encoding["attention_mask"]
        )

        probs = torch.sigmoid(
            genre_logits
        ).cpu().numpy()[0]

    topic_dist_np = topic_dist.cpu().numpy()[0]

    # ---------------------------------------------------
    # Genre Probabilities
    # ---------------------------------------------------
    st.subheader("🎯 Genre Probabilities")

    prob_df = pd.DataFrame({
        "Genre": mlb.classes_,
        "Probability": probs
    })

    prob_df = prob_df.sort_values(
        by="Probability",
        ascending=False
    )

    st.bar_chart(
        prob_df.set_index("Genre")
    )

    # ---------------------------------------------------
    # Predicted Genres
    # ---------------------------------------------------
    threshold = 0.5

    predicted_genres = []

    for i, p in enumerate(probs):

        if p > threshold:
            predicted_genres.append(
                mlb.classes_[i]
            )

    st.subheader("✅ Predicted Genres")

    if len(predicted_genres) == 0:
        st.write("No genre exceeded threshold.")
    else:
        st.write(predicted_genres)

    # ---------------------------------------------------
    # Topic Resources
    # ---------------------------------------------------
    beta = model.beta.detach().cpu().numpy()

    vocab = vectorizer.get_feature_names_out()

    classifier_weights = (
        model.classifier[1]
        .weight
        .detach()
        .cpu()
        .numpy()
    )

    # ---------------------------------------------------
    # Dominant Topics
    # ---------------------------------------------------
    st.subheader("📌 Dominant Topics")

    top_topics = topic_dist_np.argsort()[-5:][::-1]

    for t in top_topics:

        word_ids = (
            beta[t]
            .argsort()[-8:][::-1]
        )

        words = [
            vocab[j]
            for j in word_ids
        ]

        st.write(
            f"**Topic {t}**"
        )

        st.write(
            " • ".join(words)
        )

    # ---------------------------------------------------
    # Genre Explanations
    # ---------------------------------------------------
    st.subheader("🔍 Why These Genres?")

    for genre in predicted_genres:

        genre_idx = list(
            mlb.classes_
        ).index(genre)

        genre_prob = probs[genre_idx]

        with st.expander(
            f"{genre} ({genre_prob:.3f})",
            expanded=True
        ):

            topic_contrib = (
                topic_dist_np *
                classifier_weights[
                    genre_idx
                ]
            )

            top_topic_ids = (
                topic_contrib
                .argsort()[-3:][::-1]
            )

            for t in top_topic_ids:

                word_ids = (
                    beta[t]
                    .argsort()[-8:][::-1]
                )

                words = [
                    vocab[j]
                    for j in word_ids
                ]

                st.write(
                    f"**Topic {t}**"
                )

                st.write(
                    " • ".join(words)
                )