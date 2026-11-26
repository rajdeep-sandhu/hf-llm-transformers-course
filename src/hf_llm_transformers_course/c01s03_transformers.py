"""Chapter 1, Section 3: Transformers, What Can They Do?"""

# /// script
# dependencies = [
#     "marimo",
#     "torch==2.15.0",
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
    import torch
    from transformers import (
        AutoModelForQuestionAnswering,
        AutoTokenizer,
        pipeline,
    )


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

    - **Text generation** is used with an instruction-tuned chat model, as the `"question-answering"` task has been removed in `transformers` v5.
    - Alternatively `AutoModelForQuestionAnswering` can be used.
    """)


@app.cell
def _():
    context: str = (
        "My name is Mickey Mouse and I work at Disney World in Florida."
    )
    question: str = "Where do I work?"
    return context, question


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question Answering using Text Generation
    """)


@app.function
@mo.persistent_cache
def generative_question_answerer(
    question: str, context: str
) -> list[dict[str, str]]:
    """Answer a question using text generation.

    Args:
        context (str): Context from which to extract the answer
        question (str): The question asked
    """
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    text_generator = pipeline(
        task="text-generation", model=model_name
    )

    message: list[dict[str, str]] = [
        {
            "role": "user",
            "content": (
                "Answer the question using only the provided context.\n"
                f"Context: {context}\n"
                f"Question: {question}"
            ),
        }
    ]

    answer: list[dict[str, str]] = text_generator(message)

    return answer


@app.cell
def _(context: str, question: str):
    generative_question_answerer(question=question, context=context)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Question Answering using `AutoModelForQuestionAnswering`

    **Limitations**

    - The score may differ from the original `"question-answering"` pipeline due to differences in special-token handling.
    - This does not implement batching, sliding context windows or unanswerable question handling.
    - The confidence score represents the model's estimated probability for the answer span, not a calibrated probability that the answer is correct.
    """)


@app.function
@mo.persistent_cache
def automodel_question_answerer(
    question: str, context: str, max_answer_length: int
) -> dict[str, object]:
    """Answer a question using AutoModelForQuestionAnswering.

    Args:
        context (str): Context from which to extract the answer
        question (str): The question asked
        max_answer_length (int): Maximum length for answer
    """
    model_name: str = "distilbert/distilbert-base-cased-distilled-squad"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    inputs = tokenizer(
        question, context, return_tensors="pt", return_offsets_mapping=True
    )

    offsets = inputs.pop("offset_mapping")[0]

    context_mask = torch.tensor(
        [seq_id == 1 for seq_id in inputs.sequence_ids()]
    )

    with torch.inference_mode():
        outputs = model(**inputs)

    # exclude question and special tokens
    start_logits = outputs.start_logits[0].masked_fill(
        ~context_mask, -torch.inf
    )
    end_logits = outputs.end_logits[0].masked_fill(
        ~context_mask, -torch.inf
    )

    # convert logits to probabilities
    start_probs = torch.softmax(start_logits, dim=0)
    end_probs = torch.softmax(end_logits, dim=0)

    # calculate scores for possible answer spans
    scores = torch.outer(start_probs, end_probs)

    # exclude invalid and excessively long spans
    scores = scores.triu().tril(max_answer_length - 1)

    # select the highest scoring span
    start_token, end_token = divmod(
        scores.argmax().item(), scores.shape[1]
    )

    start = offsets[start_token][0].item()
    end = offsets[end_token][1].item()

    return {
        "score": scores[start_token, end_token].item(),
        "start": start,
        "end": end,
        "answer": context[start:end],
    }


@app.cell
def _(context: str, question: str):
    automodel_question_answerer(
        question=question, context=context, max_answer_length=15
    )


if __name__ == "__main__":
    app.run()
