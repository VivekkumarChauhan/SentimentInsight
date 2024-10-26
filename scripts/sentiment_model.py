from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import pandas as pd

df = pd.read_csv('data/cleaned_data.csv')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_data(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

encodings = tokenize_data(df['cleaned_text'].tolist())
labels = [1 if score > 0 else (0 if score == 0 else -1) for score in df['sentiment_score']]
dataset = SentimentDataset(encodings, labels)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
training_args = TrainingArguments(output_dir='./results', num_train_epochs=3, per_device_train_batch_size=8)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()
model.save_pretrained("models/sentiment_model.pt")
