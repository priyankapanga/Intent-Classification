
#installing dependencies
import re #regex library for preprocessing
import pandas as pd
from datasets import load_dataset #dataset library from HuggingFace
from sklearn.model_selection import train_test_split #to split up data into training, validation, and testing
import torch #For deep learning tensorts, datasets, and training
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import random

#loading datasets and creating pandas dataframes
dataset = load_dataset("benayas/snips") #loading the SNIPS dataset
df_train = pd.DataFrame(dataset["train"]) #the train label column from the SNIPS dataset
df_test= pd.DataFrame(dataset["test"]) #the test label column from the SNIPS dataset



#pre-processing & removing noise 
def pre_process(text):
    text = text.lower() #all to lowercase
    text = re.sub(r"[^a-z0-9\s]", "", text) #removing punctuation and special characters
    text = re.sub(r"\s+", " ", text).strip() #remove spaces
    return text

df_train['text'] = df_train['text'].apply(pre_process)
df_test['text']= df_test['text'].apply(pre_process)
#now all data is cleaned. 


#Splitting into training and validation:
train_df, validation_df = train_test_split(df_train, test_size=0.1, random_state=50, stratify=df_train['category'])
#train_df is 90% and val_df(validation) is 10%. 



#encoding labels 
labels = train_df['category'].unique().tolist()
label_to_id = {label: i for i, label in enumerate(labels)} # convert string labels (category names) to integers 
id_to_label = {i: label for label, i in label_to_id.items()} #map integers back to strings so HuggingFace knows label names for later


#converting the category labels(strings) into number IDs for training. 
train_labels = train_df['category'].map(label_to_id).tolist()
validation_labels = validation_df['category'].map(label_to_id).tolist()
test_labels = df_test['category'].map(label_to_id).tolist()


#Tokenization
from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


train_encodings = tokenizer(list(train_df['text']), padding=True, truncation=True, max_length=64)
val_encodings = tokenizer(list(validation_df['text']), padding=True, truncation=True, max_length=64)
test_encodings = tokenizer(list(df_test['text']), padding=True, truncation=True, max_length=64)



#creating a pytorch datset
class SNIPSDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


train_dataset = SNIPSDataset(train_encodings, train_labels)
val_dataset = SNIPSDataset(val_encodings, validation_labels)
test_dataset = SNIPSDataset(test_encodings, test_labels)



#setting up model 

from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=len(labels),
    id_to_label=id_to_label,
    label_to_id=label_to_id
)




#Training set-up
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="no",
    learning_rate=2e-5,
    logging_dir='./logs',
    logging_steps=50,
    seed=42
)

#trainer api 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

trainer.train()

trainer.save_model("model")
tokenizer.save_pretrained("model")


#------------------------------------------------------------------------------------#
#EVALUATION


#evaluation + confusion matrix
val_preds = trainer.predict(val_dataset)
val_pred_labels = val_preds.predictions.argmax(-1) #picks label with highest probability
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Validation Accuracy:", accuracy_score(validation_labels, val_pred_labels))
print(classification_report(validation_labels, val_pred_labels, target_names=labels))


test_preds = trainer.predict(test_dataset)
test_pred_labels = test_preds.predictions.argmax(-1)
cm = confusion_matrix(test_labels, test_pred_labels, labels=list(range(len(labels))))


plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
