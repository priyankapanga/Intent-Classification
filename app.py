#dependencies 

import streamlit as st
import torch

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
model_name = "priyankapanga/intent-classification-distilbert"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#-----CSS-----#
page_bg = f"""
<style>
body {{
    background-color: #FFF2EF;
    font-family: "Arial Rounded MT Bold", sans-serif;
}}
.title {{
    font-size: 38px;
    text-align: center;
    background: linear-gradient(to right, #5D688A, #F7A5A5, #FFDBB6, #FFF2EF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: bold;
    margin-bottom: 20px;
}}
.sub {{
    text-align: center;
    font-size: 18px;
    color: #5D688A;
}}
.pred-card {{
    background-color: #FFDBB6;
    border-radius: 15px;
    padding: 20px;
    margin-top: 20px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
    color: #5D688A;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
}}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Loading model
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

# Labels
id2label = model.config.id2label 

# Content
st.markdown('<div class="title">Intent Classification Demo</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Enter a sentence and I’ll predict its intent! </div>', unsafe_allow_html=True)

text = st.text_input("Type here:")

# Displaying the prediction
if text:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    pred_id = np.argmax(probs)
    pred_label = id2label[pred_id]

    # Prediction box
    st.markdown(
        f'<div class="pred-card">Predicted Intent: {pred_label}</div>',
        unsafe_allow_html=True
    )

    # Horizontal bar chart
    df = pd.DataFrame({
        "Intent": [id2label[i] for i in range(len(probs))],
        "Confidence": probs
    }).sort_values("Confidence", ascending=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(df["Intent"], df["Confidence"], color=["#5D688A", "#F7A5A5", "#FFDBB6", "#FFF2EF"] * 2)
    ax.set_xlabel("Confidence")
    ax.set_xlim(0, 1)
    st.pyplot(fig)
