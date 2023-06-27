import warnings
warnings.filterwarnings('ignore')
from config import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import os
import pandas as pd
import numpy as np
from nlp import Dataset

import argparse
os.environ["TOKENIZERS_PARALLELISM"] = "false"


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument('--predict',dest="predict", action='store_true')

parser.set_defaults(train=False)
parser.set_defaults(predict=True)

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
print("**"*20)
print(" "*20)



if __name__ == '__main__':
    if args.train:

       # Load the tokenizer and model

        model = AutoModelForSequenceClassification.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
        print(" "*20)
        print("Pre-trained model loaded")
        print("**"*20)


        def load_dataset():
            pos_file = os.path.join(data_path, pos_corpus)
            neg_file = os.path.join(data_path, neg_corpus)

            pos_sents = []
            with open(pos_file, 'r', encoding='utf-8') as f:
                for sent in f:
                    pos_sents.append(sent.replace('\n', ''))

            neg_sents = []
            with open(neg_file, 'r', encoding='utf-8') as f:
                for sent in f:
                    neg_sents.append(sent.replace('\n', ''))

            balance_len = min(len(pos_sents), len(neg_sents))

            pos_df = pd.DataFrame(pos_sents, columns=['text'])
            pos_df['polarity'] = 1
            pos_df = pos_df[:balance_len]

            neg_df = pd.DataFrame(neg_sents, columns=['text'])
            neg_df['polarity'] = 0
            neg_df = neg_df[:balance_len]

            return pd.concat([pos_df, neg_df]).reset_index(drop=True)


        print('Loading dataset...')

        dataset = load_dataset()

        print('Dataset size ', len(dataset))

        X_train, X_val, y_train, y_val = train_test_split(dataset['text'], dataset['polarity'], test_size=0.2, random_state=2020)
        testset = pd.read_csv(test_path)
        X_test = testset['title_seg']
        X_test = X_test.dropna()

        # Concatenate train data and test data
        all_text = np.concatenate([X_train.values, X_val.values,X_test.values])

        # Encode our concatenated data
        encoded_text = [tokenizer.encode(sent, add_special_tokens=True) for sent in all_text]

        # Find the maximum length
        max_len = max([len(sent) for sent in encoded_text])
        print('Max length: ', max_len)


        # Tokenize the text data
        encoded_train = tokenizer.batch_encode_plus(
            X_train.tolist(),
            add_special_tokens=True,
            truncation=True,
            padding="longest",
            max_length=max_len,
            return_tensors="pt"
            #,return_attention_mask=False
        )

        encoded_val = tokenizer.batch_encode_plus(
            X_val.tolist(),
            add_special_tokens=True,
            truncation=True,
            padding="longest",
            max_length=max_len,
            return_tensors="pt"
            #,return_attention_mask=False
        )

        encoded_test = tokenizer.batch_encode_plus(
            X_test.tolist(),
            add_special_tokens=True,
            truncation=True,
            padding="longest",
            max_length=max_len,
            return_tensors="pt"
            #,return_attention_mask=False
        )

        # Create TensorDataset for train and validation sets
        train_dataset = TensorDataset(encoded_train["input_ids"], encoded_train["attention_mask"], torch.tensor(y_train.tolist()))
        val_dataset = TensorDataset(encoded_val["input_ids"],encoded_val["attention_mask"],  torch.tensor(y_val.tolist()))
        #test_dataset = TensorDataset(encoded_val["input_ids"],encoded_val["attention_mask"])
        # Define batch size and number of workers
        batch_size = 8
        num_workers = 2

        # Create train and validation data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)




        # Define the optimizer, loss function, and device
        optimizer = Adam(model.parameters(), lr=1e-5)
        criterion = nn.CrossEntropyLoss()

        def train_epoch(model, trainloader, optimizer, criterion, device):
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            progress_bar = tqdm(train_loader, desc="Training", leave=False)

            for batch in progress_bar:
                input_ids, attention_mask, labels = (t.to(device) for t in batch)

                optimizer.zero_grad()

                outputs = model(input_ids=input_ids, return_dict=True)
                logits = outputs.logits
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(logits, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                progress_bar.set_postfix({"Loss": total_loss / (len(progress_bar) + 1e-8), "Accuracy": correct / total})

            return total_loss / len(trainloader), correct / total

        def evaluate(model, valloader, criterion, device):
            model.eval()
            total_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in valloader:
                    input_ids, attention_mask, labels = (t.to(device) for t in batch)

                    outputs = model(input_ids=input_ids, return_dict=True)
                    logits = outputs.logits
                    loss = criterion(logits, labels)

                    total_loss += loss.item()
                    _, predicted = torch.max(logits, dim=1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

            return total_loss / len(valloader), correct / total


        model = model.to(device)
        for epoch in range(3):
            train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

            print(f"Epoch {epoch+1}/{3}")
            print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")
            print()
            torch.save(model, model_path)

        print("Training finished.")


    elif args.predict:

            testset = pd.read_csv(test_path)
            X_test = testset['title_seg']
            X_test = X_test.dropna()


            encoded_test = tokenizer.batch_encode_plus(
                X_test.tolist(),
                add_special_tokens=True,
                truncation=True,
                padding="longest",
                max_length=266,
                return_tensors="pt"
                #,return_attention_mask=False
            )


            test_dataset = TensorDataset(encoded_test["input_ids"],encoded_test["attention_mask"])

            batch_size = 8
            num_workers = 2
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            model = torch.load(model_path)
            # Set the model to evaluation mode
            model.eval()

            # List to store the predictions
            predictions = []
            predictions_prob = []
            # Iterate over the test set and make predictions
            with torch.no_grad():
                progress_bar = tqdm(test_loader, desc="Predicting", leave=False)
                for batch in progress_bar:
                    input_ids, attention_mask = batch  # Modify this based on your input format
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)

                    # Forward pass
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=1)

                    #prob = torch.argmax(probabilities, dim=1)

                    # Get the predicted labels (class with maximum probability)
                    _, predicted_labels = torch.max(probabilities, dim=1)
                    predictions_prob.extend(_.cpu().numpy())

                    # Append the predicted labels to the list
                    predictions.extend(predicted_labels.cpu().tolist())  # Move to CPU and convert to a Python list



            # Convert the concatenated tensor to a pandas DataFrame
            bert_prediction = pd.DataFrame()
            bert_prediction['text'] = X_test.values
            bert_prediction['prediction'] = predictions
            bert_prediction['prob'] = predictions_prob
            # Save the DataFrame to a CSV file
            bert_prediction.to_csv(pred_path, index=False)
            print("Prediction success.")





