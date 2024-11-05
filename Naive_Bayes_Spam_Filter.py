import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.multiclass import unique_labels


df = pd.read_csv('https://raw.githubusercontent.com/AiDevNepal/ai-saturdays-workshop-8/master/data/spam.csv')

df['target'] = np.where(df['target']=='spam',1, 0)

X_train, X_test, y_train, y_test = train_test_split(df['text'],df['target'], random_state = 5)

# Vectorizer butun kelimelere sutun olusturur
vectorizer = CountVectorizer(ngram_range =(1,2)).fit(X_train)
X_train_vectorized = vectorizer.transform(X_train)
X_train_vectorized.toarray().shape


vectorizer.get_feature_names_out()

model = MultinomialNB(alpha = 0.1)
model.fit(X_train_vectorized, y_train)

predictions = model.predict(vectorizer.transform(X_test))
print("Accuracy:", 100 * sum(predictions == y_test) / len(predictions), '%')

# ex
model.predict(vectorizer.transform(
    [
        "Thank you, ABC. Can you also share your LinkedIn profile? As you are a good at programming at pyhthon, would be willing to see your personal/college projects.",
        "Hi y’all, We have a Job Openings in the positions of software engineer, IT officer at ABC Company.Kindly, send us your resume and the cover letter as soon as possible if you think you are an eligible candidate and meet the criteria.",
        "Dear ABC, Congratulations! You have been selected as a SOftware Developer at XYZ Company. We were really happy to see your enthusiasm for this vision and mission. We are impressed with your background and we think you would make an excellent addition to the team.",
        "Congrats! you won iphone 10 for free",
        "congratulations, you became today's lucky winner",
        "1-month unlimited calls offer Activate now",
        "Ram wants your phone number",
    ])
            )      

threshold = 0.95
y_pred = (model.predict_proba(vectorizer.transform(X_test))[:, 1] > threshold).astype('float')
confusion_matrix(y_test, y_pred)










