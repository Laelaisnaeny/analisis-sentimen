import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch

# Load train and test datasets without header
train_data_path = 'cleaned_train_set.tsv'
test_data_path = 'cleaned_test_set.tsv'

train_df = pd.read_csv(train_data_path, sep='\t', header=None, names=['text', 'label'])
test_df = pd.read_csv(test_data_path, sep='\t', header=None, names=['text', 'label'])

# Map string labels to integers
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
train_df['label'] = train_df['label'].map(label_map)
test_df['label'] = test_df['label'].map(label_map)

# Display the first few rows of the dataframes
print("Train set:")
print(train_df.head())
print("\nTest set:")
print(test_df.head())

# Convert dataframes to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Tokenize the datasets
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=64)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./sentiment-model')
tokenizer.save_pretrained('./sentiment-model')
