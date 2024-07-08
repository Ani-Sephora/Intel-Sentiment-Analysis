#IMPORTING DATASET

import pandas as pd
from sklearn.model_selection import train_test_split

from google.colab import files
uploaded = files.upload()

# Define the file path
file_path = 'final_dataset.csv'

# Load the dataset
df = pd.read_csv(file_path)

# Split the dataset into train and eval sets
train_texts, eval_texts, train_labels, eval_labels = train_test_split(
    df['body'].tolist(), df['sentiment'].tolist(), test_size=0.2, random_state=42
)

#TRAING A MODEL
!pip install transformers[torch] -U
!pip install accelerate -U


import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification, DebertaTokenizer, DebertaForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch

# Load the small dataset
df = pd.read_csv('/content/final_dataset.csv')

# Map ratings to sentiment
def map_sentiment(rating):
    if rating in [4, 5]:
        return 2  # Positive
    elif rating == 3:
        return 1  # Neutral
    else:
        return 0  # Negative

df['sentiment'] = df['rating'].apply(map_sentiment)

# Adjust the column names as per the dataset
train_texts, eval_texts, train_labels, eval_labels = train_test_split(
    df['body'].tolist(), df['sentiment'].tolist(), test_size=0.2, random_state=42
)

# Ensure all entries are strings
train_texts = [str(text) for text in train_texts]
eval_texts = [str(text) for text in eval_texts]

class ReviewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Define the fine-tuning function
def fine_tune_model(model, tokenizer, train_texts, eval_texts, train_labels, eval_labels, model_name):
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    eval_encodings = tokenizer(eval_texts, truncation=True, padding=True)
    train_dataset = ReviewsDataset(train_encodings, train_labels)
    eval_dataset = ReviewsDataset(eval_encodings, eval_labels)

    training_args = TrainingArguments(
        output_dir=f'./results/{model_name}',
        num_train_epochs=3,
        per_device_train_batch_size=4,  # Use a smaller batch size due to the small dataset
        per_device_eval_batch_size=4,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir=f'./logs/{model_name}',
        logging_steps=5,
        evaluation_strategy="steps",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    model.save_pretrained(f'./{model_name}')
    tokenizer.save_pretrained(f'./{model_name}')

# Fine-tune BERT
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
fine_tune_model(bert_model, bert_tokenizer, train_texts, eval_texts, train_labels, eval_labels, 'bert')

# Fine-tune RoBERTa
roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
fine_tune_model(roberta_model, roberta_tokenizer, train_texts, eval_texts, train_labels, eval_labels, 'roberta')

# Fine-tune DistilBERT
distilbert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
fine_tune_model(distilbert_model, distilbert_tokenizer, train_texts, eval_texts, train_labels, eval_labels, 'distilbert')

# Fine-tune DeBERTa
deberta_model = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-base', num_labels=3)
deberta_tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
fine_tune_model(deberta_model, deberta_tokenizer, train_texts, eval_texts, train_labels, eval_labels, 'deberta')

# Load the fine-tuned models
bert_model = BertForSequenceClassification.from_pretrained('./bert')
bert_tokenizer = BertTokenizer.from_pretrained('./bert')

roberta_model = RobertaForSequenceClassification.from_pretrained('./roberta')
roberta_tokenizer = RobertaTokenizer.from_pretrained('./roberta')

distilbert_model = DistilBertForSequenceClassification.from_pretrained('./distilbert')
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('./distilbert')

deberta_model = DebertaForSequenceClassification.from_pretrained('./deberta')
deberta_tokenizer = DebertaTokenizer.from_pretrained('./deberta')

# Ensemble prediction function
def predict_sentiment(review_text):
    bert_inputs = bert_tokenizer(review_text, return_tensors='pt', truncation=True, padding=True)
    roberta_inputs = roberta_tokenizer(review_text, return_tensors='pt', truncation=True, padding=True)
    distilbert_inputs = distilbert_tokenizer(review_text, return_tensors='pt', truncation=True, padding=True)
    deberta_inputs = deberta_tokenizer(review_text, return_tensors='pt', truncation=True, padding=True)

    bert_outputs = bert_model(**bert_inputs)
    roberta_outputs = roberta_model(**roberta_inputs)
    distilbert_outputs = distilbert_model(**distilbert_inputs)
    deberta_outputs = deberta_model(**deberta_inputs)

    bert_probs = torch.nn.functional.softmax(bert_outputs.logits, dim=-1)
    roberta_probs = torch.nn.functional.softmax(roberta_outputs.logits, dim=-1)
    distilbert_probs = torch.nn.functional.softmax(distilbert_outputs.logits, dim=-1)
    deberta_probs = torch.nn.functional.softmax(deberta_outputs.logits, dim=-1)

    avg_probs = (bert_probs + roberta_probs + distilbert_probs + deberta_probs) / 4
    sentiment = torch.argmax(avg_probs, dim=1).item()

    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return sentiment_map[sentiment]

# Save the ensemble model
import pickle

ensemble_model = {
    "bert_model": bert_model,
    "bert_tokenizer": bert_tokenizer,
    "roberta_model": roberta_model,
    "roberta_tokenizer": roberta_tokenizer,
    "distilbert_model": distilbert_model,
    "distilbert_tokenizer": distilbert_tokenizer,
    "deberta_model": deberta_model,
    "deberta_tokenizer": deberta_tokenizer,
    "predict_sentiment": predict_sentiment,
}

with open('4techs_model.pkl', 'wb') as f:
    pickle.dump(ensemble_model, f)

print("Ensemble model saved as 4techs_model.pkl")

from google.colab import files
files.download('/content/4techs_model.pkl')


#PREDICION
import pickle
import torch

# Load the ensemble model
with open('4techs_model.pkl', 'rb') as f:
    ensemble_model = pickle.load(f)

# Example review
review_text = "worst product ever"
predicted_sentiment = ensemble_model["predict_sentiment"](review_text)
print(f"Predicted Sentiment: {predicted_sentiment}")


