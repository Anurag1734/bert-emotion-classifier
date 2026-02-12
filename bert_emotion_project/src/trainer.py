"""
Training orchestration module.

Purpose:
    - Coordinate training loop, validation, early stopping, checkpointing.

Responsibilities:
    - Provide `Trainer` class with train() method.
    - Handle device placement, gradient clipping, early stopping, model checkpointing.
    - Persist training logs to CSV.
    - Save config snapshot with best model.
"""

import time
import torch
import torch.nn as nn
import csv
import json
from pathlib import Path
from bert_emotion_project.src.evaluator import compute_metrics


class Trainer:
    """Full training orchestrator with validation, early stopping, checkpointing."""

    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, scheduler, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.best_val_macro_f1 = -float('inf')
        self.patience = config['training']['early_stopping_patience']
        self.patience_counter = 0
        self.best_epoch = -1

        # Prepare reports directory and CSV log
        self.reports_dir = Path(self.config['paths']['reports'])
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.reports_dir / "training_log.csv"

        # Overwrite log file at each new run
        with open(self.log_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "accuracy",
                "macro_f1",
                "weighted_f1"
            ])

    def train_epoch(self):
        self.model.train()
        train_loss = 0.0

        for batch in self.train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['max_grad_norm']
            )
            self.optimizer.step()
            self.scheduler.step()

            train_loss += loss.item()

        return train_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(self.val_loader)
        metrics = compute_metrics(all_preds, all_labels)

        return val_loss, metrics

    def evaluate_test(self):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                logits = self.model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        metrics = compute_metrics(all_preds, all_labels)
        return metrics

    def train(self):
        start_time = time.time()
        num_epochs = self.config['training']['num_epochs']

        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss, val_metrics = self.validate()

            val_acc = val_metrics['accuracy']
            val_macro_f1 = val_metrics['macro_f1']
            val_weighted_f1 = val_metrics['weighted_f1']

            # Log to CSV (append per epoch)
            with open(self.log_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    train_loss,
                    val_loss,
                    val_acc,
                    val_macro_f1,
                    val_weighted_f1
                ])

            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_acc:.4f}")
            print(f"  Val Macro F1: {val_macro_f1:.4f}")
            print(f"  Val Weighted F1: {val_weighted_f1:.4f}")

            if val_macro_f1 > self.best_val_macro_f1:
                self.best_val_macro_f1 = val_macro_f1
                self.best_epoch = epoch + 1
                self.patience_counter = 0
                self._save_checkpoint()
                print(f"  ✓ Best model saved (Macro F1: {val_macro_f1:.4f})")
            else:
                self.patience_counter += 1
                print(f"  Patience: {self.patience_counter}/{self.patience}")
                if self.patience_counter >= self.patience:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
                    break
            print()

        elapsed_time = time.time() - start_time
        print(f"Training completed in {elapsed_time:.2f} seconds.")
        print(f"Best epoch: {self.best_epoch}\n")

        self._load_checkpoint()
        test_metrics = self.evaluate_test()

        print("\nReproducibility Reminder:")
        print("Run: pip freeze > requirements_frozen.txt\n")

        return test_metrics, elapsed_time

    def _save_checkpoint(self):
        models_dir = Path(self.config['paths']['models'])
        models_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = models_dir / 'best_model.pt'
        config_path = models_dir / 'best_model_config.json'

        torch.save(self.model.state_dict(), checkpoint_path)

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4)

    def _load_checkpoint(self):
        models_dir = Path(self.config['paths']['models'])
        checkpoint_path = models_dir / 'best_model.pt'
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )
        self.model.to(self.device)
