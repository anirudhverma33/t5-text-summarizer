# Dialogue Summarisation with T5

Fine-tuning `t5-small` on the [SAMSum](https://huggingface.co/datasets/Samsung/samsum) dialogue-summarisation dataset with the Hugging Face `Trainer`, served through a small FastAPI web app.

## What's here

| Path | Contents |
|---|---|
| `text_summarizer.ipynb` | Main training run (CUDA): preprocessing, tokenisation, fine-tuning, saving, test generation |
| `first_text_summarizer.ipynb` | Earlier run of the same pipeline |
| `text_summarizer_app/` | FastAPI app (`app.py`) with a Jinja2 front end (`index.html`) |

## Pipeline

1. **Data**: SAMSum train/validation CSVs, randomly subsampled to 4,000 training and 500 validation dialogues (`random_state=42`).
2. **Cleaning**: newline/whitespace normalisation, HTML-tag removal, lower-casing.
3. **Tokenisation**: `T5Tokenizer`, inputs padded/truncated to 512 tokens, summaries to 150.
4. **Fine-tuning**: `T5ForConditionalGeneration` (`t5-small`), 6 epochs, batch size 8, weight decay 0.01, 500 warm-up steps, evaluation and checkpoint each epoch.
5. **Inference**: beam search (4 beams, max 150 tokens).

## Training log (main run)

Validation loss per epoch, from the trainer state of the main run:

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Validation loss | 0.381 | 0.359 | 0.355 | 0.351 | 0.350 | 0.350 |

Average training loss over the run: 0.913 (3,000 steps). Padding positions in the labels are not masked out (`-100`), so these losses are not comparable to published SAMSum numbers. ROUGE has not been computed yet.

## Running the app

The fine-tuned weights (~230 MB) are not committed. Run `text_summarizer.ipynb` to produce `saved_summary_model/`, then copy it into `text_summarizer_app/`:

```bash
pip install -r requirements.txt
cd text_summarizer_app
uvicorn app:app --reload
```

Open http://127.0.0.1:8000, or call the API directly:

```bash
curl -X POST http://127.0.0.1:8000/summarize/ \
  -H "Content-Type: application/json" \
  -d '{"dialogue": "Amanda: I baked cookies. Do you want some? Jerry: Sure!"}'
```

## Data

The SAMSum CSVs are not included. Download them from the Hugging Face Hub and place `samsum-train.csv` and `samsum-validation.csv` next to the notebook. SAMSum is released under CC BY-NC-ND 4.0.
