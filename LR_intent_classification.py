
#setting up dependencies 

import re
from datasets import load_dataset
import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


#loading dataset

dataset=load_dataset("benayas/snips")

#Converting data into pandas dataframes
df_train = pd.DataFrame(dataset["train"])
df_test = pd.DataFrame(dataset["test"])

print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)

#pre-processing: 

def preprocess_text(text):
    text=text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text) 
    text = re.sub(r"\s+", " ", text).strip()  
    return text


df_train['text'] = df_train['text'].apply(preprocess_text)
df_test['text'] = df_test['text'].apply(preprocess_text)

#Validation split: 

from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df_train, test_size=0.1, random_state=50)

print("Train size: ", len(train_df))
print("Validation size: ", len(val_df))

X_train, y_train = train_df["text"], train_df["category"]
X_val, y_val = val_df["text"], val_df["category"]
X_test, y_test = df_test["text"], df_test["category"]

vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

# Implementating logistic regression
new_model = LogisticRegression(max_iter=200)
new_model.fit(X_train_tfidf, y_train)

# Validation results
y_val_pred = new_model.predict(X_val_tfidf)
print("\nValidation Results:")
print("Accuracy:", accuracy_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

# Testing
y_test_pred = new_model.predict(X_test_tfidf)
print("\nTest Results:")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))




######### CONFUSION MATRIX 

import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.metrics import confusion_matrix

labels = y_test.unique()

# Calculating confusion matrix
cm = confusion_matrix(y_test, y_test_pred, labels=labels)

# Display confusion matrix
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression SNIPS")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()
