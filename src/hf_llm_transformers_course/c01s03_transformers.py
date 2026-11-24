# /// script
# dependencies = [
#     "marimo",
#     "transformers[torch, sentencepiece]==5.24.0",
# ]
# requires-python = ">=3.15"
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Transformers, What Can They Do?")

with app.setup:
    import marimo as mo
    from transformers import pipeline


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Transformers, what can they do?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - Run this notebook in sandbox mode. From project root:

        ```bash
        marimo edit --sandbox --watch "src/hf_llm_transformers_course/c01s03_transformers.py"
        ```

    - The notebook contains examples in addition to those from the course.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Sentiment Analysis

    Assigns a sentiment and confidence score to the text.
    """)
    return


@app.cell
def _():
    classifier_sentiment = pipeline("sentiment-analysis")
    classifier_sentiment(
        "I've been waiting for a HuggingFace course my whole life."
    )
    return (classifier_sentiment,)


@app.cell
def _(classifier_sentiment):
    classifier_sentiment(
        [
            "I've been waiting for a HuggingFace course my whole life.",
            "I hate this so much!",
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Zero Shot Classification

    Accepts labels to which to assign the text. Returns the labels with a confidence score corresponding to each.
    """)
    return


@app.cell
def _():
    classifier_zero_shot = pipeline("zero-shot-classification")
    classifier_zero_shot(
        "This is a course about the Transformers library.",
        candidate_labels=["education", "politics", "business"],
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Text Generation

    Autocompletes a given prompt.
    """)
    return


@app.cell
def _():
    text_generator = pipeline(task="text-generation", model="distilgpt2")
    text_generator(
        "In this course, we will teach you how to",
        max_length=30,
        num_return_sequences=2,
    )
    return


if __name__ == "__main__":
    app.run()
