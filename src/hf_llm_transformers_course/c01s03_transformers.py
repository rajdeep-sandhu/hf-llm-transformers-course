"""Chapter 1, Section 3: Transformers, What Can They Do?"""

# /// script
# dependencies = [
#     "marimo",
#     "transformers[torch, sentencepiece]==5.24.0",
# ]
# requires-python = ">=3.15"
# ///

# ruff: noqa: E501

import marimo

__generated_with = "0.26.0"
app = marimo.App(width="full", app_title="Transformers, What Can They Do?")

with app.setup:
    import marimo as mo
    from transformers import pipeline


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Transformers, what can they do?
    """)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - Run this notebook in sandbox mode. From project root:

        ```bash
        marimo edit --sandbox --watch "src/hf_llm_transformers_course/c01s03_transformers.py"
        ```

    - The notebook contains examples in addition to those from the course.
    - Results are cached to disk where possible for expensive computations.
    - The first run will, however, take longer due to model downloads and initial computation.
    """)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Sentiment Analysis

    Assigns a sentiment and confidence score to the text.
    """)


@app.function
@mo.persistent_cache
def classify_sentiment(text: str | list[str]):
    """Classify sentiment for provided text.

    Args:
        text (str | list[str]): Text to classsify
    """
    # pyrefly: ignore[no-matching-overload]
    sentiment_classifier = pipeline("sentiment-analysis")

    result = sentiment_classifier(text)

    return result


@app.cell
def _():
    classify_sentiment(
        text="I've been waiting for a HuggingFace course my whole life."
    )


@app.cell
def _():
    classify_sentiment(
        text=[
            "I've been waiting for a HuggingFace course my whole life.",
            "I hate this so much!",
        ]
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Zero Shot Classification

    Accepts labels to which to assign the text. Returns the labels with a confidence score corresponding to each.
    """)


@app.cell
def _():
    zero_shot_classifier = pipeline("zero-shot-classification")

    with mo.persistent_cache(name="zero_shot_classifier"):
        zero_shot_classification: dict[str, object] = zero_shot_classifier(
            "This is a course about the Transformers library.",
            candidate_labels=["education", "politics", "business"],
        )
    return (zero_shot_classification,)


@app.cell
def _(zero_shot_classification: dict[str, object]):
    zero_shot_classification


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Text Generation

    Autocompletes a given prompt.

    - The output is not cached, as this is meant to demonstrate different text generated on each run.
    """)


@app.cell
def _():
    text_generator = pipeline(task="text-generation", model="distilgpt2")

    generated_text: list[dict[str, str]] = text_generator(
        "In this course, we will teach you how to",
        max_length=30,
        num_return_sequences=2,
    )
    return (generated_text,)


@app.cell
def _(generated_text: list[dict[str, str]]):
    generated_text


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Fill Mask

    Predicts missing words in a sentence.
    """)


@app.cell
def _():
    unmasker = pipeline(task="fill-mask")

    with mo.persistent_cache(name="unmasker"):
        fill_predictions: list[dict[str, object]] = unmasker(
            "This course will teach you all about <mask> models.", top_k=2
        )
    return (fill_predictions,)


@app.cell
def _(fill_predictions: list[dict[str, object]]):
    fill_predictions


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Named Entity Recognition

    Identifies entities like persons, organisations or locations in a sentence.
    """)


@app.cell
def _():
    # pyrefly: ignore[no-matching-overload]
    entity_recognizer = pipeline(task="ner", aggregation_strategy="simple")

    with mo.persistent_cache(name="ner"):
        named_entities: list[dict[str, object]] = entity_recognizer(
            "My name is Mickey Mouse and I work at Disney World in Florida."
        )
    return (named_entities,)


@app.cell
def _(named_entities: list[dict[str, object]]):
    named_entities


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Extractive Question Answering

    Extracts answers to a question from a given context.

    - Text generation is used with an instruction-tuned chat model, as the `"question-answering"` task has been removed in `transformers` v5.
    """)


@app.cell
def _():
    question_answerer = pipeline(
        task="text-generation", model="Qwen/Qwen2.5-0.5B-Instruct"
    )

    context: str = (
        "My name is Mickey Mouse and I work at Disney World in Florida."
    )
    question: str = "Where do I work?"

    question_messages: list[dict[str, str]] = [
        {
            "role": "user",
            "content": (
                "Answer the question using only the provided context.\n"
                f"Context: {context}\n"
                f"Question: {question}"
            ),
        }
    ]
    return question_answerer, question_messages


@app.cell
def _(question_answerer, question_messages: list[dict[str, str]]):
    with mo.persistent_cache(name="question_answerer"):
        answer: list[dict[str, str]] = question_answerer(question_messages)
    return (answer,)


@app.cell
def _(answer: list[dict[str, str]]):
    answer


if __name__ == "__main__":
    app.run()
