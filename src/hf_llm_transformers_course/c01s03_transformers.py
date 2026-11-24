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


if __name__ == "__main__":
    app.run()
