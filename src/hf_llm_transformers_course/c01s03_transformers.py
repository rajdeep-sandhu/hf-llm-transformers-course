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
    - Results are cached to disk where possible for expensive computations. The first run will, however, take longer.
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
    sentiment_classifier = pipeline("sentiment-analysis")

    with mo.persistent_cache(name="classifier_sentiment_1"):
        sentiment_classifier(
            "I've been waiting for a HuggingFace course my whole life."
        )
    return (sentiment_classifier,)


@app.cell
def _(sentiment_classifier):
    with mo.persistent_cache(name="classifier_sentiment_2"):
        sentiment_classifier(
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

    with mo.persistent_cache(name="classifier_zero_shot"):
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

    with mo.persistent_cache(name="text_generator"):
        text_generator(
            "In this course, we will teach you how to",
            max_length=30,
            num_return_sequences=2,
        )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Fill Mask

    Predicts missing words in a sentence.
    """)
    return


@app.cell
def _():
    unmasker = pipeline(task="fill-mask")

    with mo.persistent_cache(name="unmasker"):
        unmasker("This course will teach you all about <mask> models.", top_k=2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Named Entity Recognition

    Identifies entities like persons, organisations or locations in a sentence.
    """)
    return


@app.cell
def _():
    ner = pipeline(task="ner", aggregation_strategy="simple")

    with mo.persistent_cache(name="ner"):
        ner("My name is Mickey Mouse and I work at Disney World in Florida.")
    return


if __name__ == "__main__":
    app.run()
