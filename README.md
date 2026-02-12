# bert-emotion-classifier

PyTorch BERT fine-tuning project for emotion classification using `shreyaspullehf/emotion_dataset_100k`.

## Main Project
- Code and docs: `bert_emotion_project/`
- Project README: `bert_emotion_project/README.md`

## Quick Run
```bash
uv sync
uv run python bert_emotion_project/scripts/phase1_data_prep.py
uv run python bert_emotion_project/scripts/phase4_train.py
uv run python -m bert_emotion_project.src.inference
```
