#Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

import torch #PyTorch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW 
from torch.utils.data import DataLoader, Dataset
import evaluate #Calculate metrics like accuracy

#Read File & EDA
#Load Data
data = pd.read_csv("F:\\New\\5. Sentiment Analysis\\Patient Feedback Data - Kaggle\\doctorReviews.csv")

#Renaming the columns
data.columns = ["PatientID", "Review", "Label", "Tag"]

#Read data
data.sample(n=5)

#Shape
data.shape

#Check for null values
data.info()

#Convert Review to String Datatype
data["Review"] = data["Review"].astype(str)

#Split the data into Train and Test data
train, test = train_test_split(data, test_size=0.2, random_state=40)

#Shape of Train and Test
print("Train:", train.shape)
print("Test:", test.shape)

#Extracts text and labels for training and testing splits
train_reviews = train["Review"].astype(str)
train_labels = train["Label"]
test_reviews = test["Review"].astype(str)
test_labels = test["Label"]

train_reviews = train_reviews.astype(str).tolist()
test_reviews = test_reviews.astype(str).tolist()

#BERT Tokenizer and Model
#Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') #Loads a pre-trained BERT tokenizer (bert-base-uncased), which splits text into tokens compatible with the BERT model.

""" input_ids: Tokenized IDs of the text, including special tokens:

[101] = [CLS] (start of the sentence)
[102] = [SEP] (end of the sentence)
attention_mask: Indicates which tokens are real (1) and which are padding (0)"""

#Loads a pre-trained BERT model for binary classification (num_labels=2)
model_pr = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

""" bert-base-uncased:

Pre-trained BERT model with 12 transformer layers.
Processes input_ids and computes contextual embeddings for each token.
The [CLS] token’s embedding is used as the representation of the entire input sequence."""

""" Classification Head:

A fully connected layer is applied to the [CLS] token’s embedding
Outputs logits: raw scores for each class (positive and negative sentiment)"""

#Custom Dataset Class

class PatientReviewDataset(Dataset):
    #Initializes the dataset with tokenized encodings and corresponding labels
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    #Returns the number of examples in the dataset
    def __len__(self):
        #Use the minimum length to avoid indexing issues
        return min(len(self.encodings["input_ids"]), len(self.labels))

    #Retrieves a single example at a given index as a dictionary containing: input_ids, attention_mask, and labels
    def __getitem__(self, idx):
        # Ensure the index is within bounds
        if idx >= len(self.labels):
            raise IndexError(f"Index {idx} out of bounds for labels of size {len(self.labels)}")
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels.iloc[idx] if isinstance(self.labels, pd.Series) else self.labels[idx])
        return item

#Tokenization
#Tokenizes the text data
train_encodings = tokenizer(train_reviews, truncation=True, padding=True, max_length=128) 
test_encodings = tokenizer(test_reviews, truncation=True, padding=True, max_length=128)

""" 
truncation=True: Truncates text longer than 128 tokens

padding=True: Pads shorter text to 128 tokens"""

#Dataset and DataLoader
#Creates IMDbDataset objects for training and testing data
train_dataset = PatientReviewDataset(train_encodings, train_labels)
test_dataset = PatientReviewDataset(test_encodings, test_labels)

#Wraps datasets in DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

"""
batch_size=16: Each batch contains 16 examples

shuffle=True: Shuffles training data"""

#Optimizer
#Configures the optimizer with model parameters and a learning rate of 2e-5
optimizer = AdamW(model_pr.parameters(), lr=2e-5)

#Device Configuration
#Moves the model to GPU (cuda) if available; otherwise, uses CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_pr.to(device)

#Training Loop
#Trains the model for 3 epochs
for epoch in range(3): 
    model_pr.train()
    total_loss = 0

print(f"Train Encodings: {len(train_encodings['input_ids'])}")
print(f"Train Labels: {len(train_labels)}")

train_labels = train_labels.reset_index(drop=True)

for batch in train_loader:
    optimizer.zero_grad()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    outputs = model_pr(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    total_loss += loss.item()
    loss.backward()
    optimizer.step()

for batch in train_loader:
    optimizer.zero_grad()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    outputs = model_pr(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    total_loss += loss.item()
    loss.backward()
    optimizer.step()

"""
For each batch:

Clears gradients: optimizer.zero_grad()

Processes inputs: input_ids, attention_mask, and labels

Computes loss

Backpropagates gradients: loss.backward()

Updates model parameters: optimizer.step()"""

#Prints the average loss after each epoch
print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

#Evaluation
#Loads the accuracy matrix
accuracy_metric = evaluate.load("accuracy")

#Sets the model to evaluation mode and initializes storage for predictions and labels
model_pr.eval()
predictions = []
references = []

for batch in test_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    with torch.no_grad():
        outputs = model_pr(input_ids, attention_mask=attention_mask)
    preds = torch.argmax(outputs.logits, dim=-1)
    predictions.extend(preds.cpu().numpy())
    references.extend(labels.cpu().numpy())

"""
For each batch in the test set:

Moves inputs to the appropriate device.

Predicts logits without computing gradients: torch.no_grad().

Converts logits to predictions: torch.argmax().

Stores predictions and labels."""

#Computes and prints the test set accuracy
accuracy = accuracy_metric.compute(predictions=predictions, references=references)
print(f"Test Accuracy: {accuracy['accuracy']}")

#Save Model
model_pr.save_pretrained('sentiment_model_patient_reviews')
tokenizer.save_pretrained('sentiment_model_patient_reviews')

