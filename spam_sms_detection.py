"""Spam SMS Detection. Usage: python spam_sms_detection.py [spam.csv]"""
import sys, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "spam.csv", encoding="latin-1")
df = df.iloc[:, :2]; df.columns = ["label", "text"]
df = df.dropna().drop_duplicates()
y = (df["label"].str.lower() == "spam").astype(int)
Xtr, Xte, ytr, yte = train_test_split(df["text"], y, test_size=0.2, stratify=y, random_state=42)
vec = TfidfVectorizer(stop_words="english", lowercase=True, ngram_range=(1, 2), sublinear_tf=True)
Xtr_v, Xte_v = vec.fit_transform(Xtr), vec.transform(Xte)
models = {"Naive Bayes": MultinomialNB(), "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
          "SVM": LinearSVC(class_weight="balanced")}
for n, m in models.items():
    m.fit(Xtr_v, ytr); p = m.predict(Xte_v)
    print(f"\n=== {n} ===  Acc: {accuracy_score(yte,p):.4f}")
    print(classification_report(yte, p, target_names=["ham","spam"], digits=4))
m = models["SVM"]
for s in ["Congratulations! You won a free prize, call now", "Hey, are we meeting at 5?"]:
    print(f"'{s}' ->", "spam" if m.predict(vec.transform([s]))[0] else "ham")
