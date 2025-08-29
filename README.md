<p align="center">
  <img src="images/banner_version2" width="800"/>
</p>

# Intent Classification with DistilBERT

This project fine-tunes **DistilBERT**, a lightweight("Distilled") transformer model, to classify **user intents** from the [SNIPS dataset](https://huggingface.co/datasets/benayas/snips).
It also includes a **Streamlit web app** to test predictions interactively.

The 7 intents are:

- PlayMusic
- BookRestaurant
- GetWeather
- SearchScreeningEvent
- RateBook
- AddToPlaylist
- SearchCreativeWork

## Steps

- Preprocesses and cleans the SNIPS dataset (7 intents).
- Fine-tunes **DistilBERT** for text classification using HuggingFace’s Trainer API.
  -> **99% validation accuracy**.
- Visualizes results with a **confusion matrix**.
- Deploys an **interactive Streamlit app**, in which you can:
  -> Input a sentence → get predicted intent in real-time.
  -> Horizontal bar chart shows confidence across all 7 categories.

## Acknowledgements


- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [SNIPS Dataset](https://huggingface.co/datasets/benayas/snips)
- [Streamlit](https://streamlit.io/)

Thank you for reading!
