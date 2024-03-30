from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder


class LogisticSGDModel:

    def __init__(self, max_iter=1000, tol=1e-4):
        self.max_iter = max_iter
        self.tol = tol
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.model = SGDClassifier(loss='log_loss', max_iter=max_iter, tol=tol)

    def fit(self, X, y):
        X_tfidf = self.tfidf_vectorizer.fit_transform(X)
        self.model.fit(X_tfidf, y)

    def predict(self, new_texts):
        new_text_tfidf = self.tfidf_vectorizer.transform(new_texts)
        return self.model.predict(new_text_tfidf)
