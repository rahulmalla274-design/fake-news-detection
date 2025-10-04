import os
import threading
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

try:
    import nltk
    from nltk.corpus import stopwords
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False

MODEL_FILENAME = 'fake_news_model.pkl'



def load_dataset_try_auto():
    """Try to auto-load dataset from common files (True.csv, Fake.csv) or prompt user."""
    if os.path.exists('True.csv') and os.path.exists('Fake.csv'):
        try:
            df_true = pd.read_csv('True.csv')
            df_fake = pd.read_csv('Fake.csv')
            df_true['label'] = 'real'
            df_fake['label'] = 'fake'
            df = pd.concat([df_true, df_fake], ignore_index=True)
            return df
        except Exception:
            pass


    for fname in ['news.csv', 'dataset.csv', 'combined.csv']:
        if os.path.exists(fname):
            try:
                return pd.read_csv(fname)
            except Exception:
                pass

    return None


def prepare_dataframe(df):
    """Return dataframe with a 'text' column and 'label' column if available.
    Tries to combine title + text if present.
    """
    df = df.copy()
    text_cols = [c for c in df.columns if c.lower() in ('text', 'content', 'article')]
    title_cols = [c for c in df.columns if c.lower() in ('title', 'headline')]

    if not text_cols and not title_cols:

        candidate = None
        for c in df.columns:
            if df[c].dtype == 'object':
                candidate = c
                break
        if candidate is None:
            raise ValueError('No text-like column found in dataframe')
        df['text'] = df[candidate].astype(str)
    else:
        parts = []
        if title_cols:
            parts.append(df[title_cols[0]].astype(str))
        if text_cols:
            parts.append(df[text_cols[0]].astype(str))
        df['text'] = parts[0] if len(parts) == 1 else (parts[0] + '. ' + parts[1])

    if 'label' in df.columns:
        df['label'] = df['label'].astype(str)
    else:

        inferred = None
        for c in df.columns:
            if c.lower() in ('truth', 'target'):
                inferred = c
                break
        if inferred:
            df['label'] = df[inferred].astype(str)

    return df[['text'] + (['label'] if 'label' in df.columns else [])]


def build_pipeline():
    stop_words = 'english'
    if NLTK_AVAILABLE:
        try:
            sw = set(stopwords.words('english'))
            stop_words = sw
        except Exception:
            stop_words = 'english'

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_df=0.9, min_df=3, ngram_range=(1,2), stop_words=stop_words)),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    return pipe


def train_model(df, gui_callback=None):
    """Train model on dataframe with text + label. Returns trained pipeline and metrics."""
    df = prepare_dataframe(df)
    if 'label' not in df.columns:
        raise ValueError('No label column available for training')

    X = df['text'].fillna('')
    y = df['label'].fillna('')


    y = y.str.lower().map(lambda s: 'fake' if 'fake' in s or '0' == s or 'f' == s else 'real')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = build_pipeline()

    if gui_callback:
        gui_callback('Fitting model...')
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)

    if gui_callback:
        gui_callback(f'Training finished. Accuracy: {acc:.4f}')

    return pipeline, acc, report



class FakeNewsGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Fake News Detector')
        self.geometry('850x600')


        self.pipeline = None
        self.dataset = None


        ctrl = tk.Frame(self)
        ctrl.pack(fill=tk.X, padx=8, pady=6)

        self.btn_load_auto = tk.Button(ctrl, text='Auto-load dataset', command=self.auto_load)
        self.btn_load_auto.pack(side=tk.LEFT, padx=4)

        self.btn_open_csv = tk.Button(ctrl, text='Open CSV', command=self.open_csv)
        self.btn_open_csv.pack(side=tk.LEFT, padx=4)

        self.btn_train = tk.Button(ctrl, text='Train Model', command=self.train_model_thread)
        self.btn_train.pack(side=tk.LEFT, padx=4)

        self.btn_load_model = tk.Button(ctrl, text='Load Model', command=self.load_model)
        self.btn_load_model.pack(side=tk.LEFT, padx=4)

        self.btn_save_model = tk.Button(ctrl, text='Save Model', command=self.save_model)
        self.btn_save_model.pack(side=tk.LEFT, padx=4)

        self.btn_bulk = tk.Button(ctrl, text='Bulk Predict CSV', command=self.bulk_predict_csv)
        self.btn_bulk.pack(side=tk.LEFT, padx=4)


        mid = tk.Frame(self)
        mid.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        left = tk.Frame(mid)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(left, text='Enter news title + body (or paste article):').pack(anchor='w')
        self.txt_input = scrolledtext.ScrolledText(left, height=10)
        self.txt_input.pack(fill=tk.BOTH, expand=True)

        self.btn_predict = tk.Button(left, text='Predict', command=self.predict_text)
        self.btn_predict.pack(pady=6)

        right = tk.Frame(mid)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(right, text='Prediction Result:').pack(anchor='w')
        self.lbl_result = tk.Label(right, text='No model loaded', font=('Arial', 14), fg='blue')
        self.lbl_result.pack(anchor='w', pady=4)

        tk.Label(right, text='Probability (class probabilities):').pack(anchor='w')
        self.txt_probs = tk.Text(right, height=6)
        self.txt_probs.pack(fill=tk.BOTH, expand=True)


        tk.Label(self, text='Log / Messages:').pack(anchor='w', padx=8)
        self.log = scrolledtext.ScrolledText(self, height=8)
        self.log.pack(fill=tk.BOTH, expand=False, padx=8, pady=6)
        self.log.config(state=tk.DISABLED)


        if os.path.exists(MODEL_FILENAME):
            try:
                self.pipeline = pickle.load(open(MODEL_FILENAME, 'rb'))
                self.log_message(f'Loaded saved model from {MODEL_FILENAME}')
                self.lbl_result.config(text='Model loaded (from file)')
            except Exception as e:
                self.log_message(f'Could not load existing model: {e}')

    def log_message(self, msg):
        self.log.config(state=tk.NORMAL)
        self.log.insert(tk.END, msg + '\n')
        self.log.see(tk.END)
        self.log.config(state=tk.DISABLED)

    def auto_load(self):
        df = load_dataset_try_auto()
        if df is None:
            messagebox.showinfo('Auto-load', 'No auto-detectable dataset found (True.csv & Fake.csv). Use Open CSV.')
            return
        try:
            self.dataset = prepare_dataframe(df)
            self.log_message('Auto-loaded dataset with {} rows'.format(len(self.dataset)))
            messagebox.showinfo('Auto-load', f'Loaded dataset with {len(self.dataset)} rows')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to prepare dataset: {e}')

    def open_csv(self):
        path = filedialog.askopenfilename(filetypes=[('CSV files', '*.csv'), ('All files', '*.*')])
        if not path:
            return
        try:
            df = pd.read_csv(path)
            self.dataset = prepare_dataframe(df)
            self.log_message(f'Opened CSV: {path} ({len(self.dataset)} rows)')
            messagebox.showinfo('CSV Open', f'Loaded {len(self.dataset)} rows')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to open CSV: {e}')

    def train_model_thread(self):

        if self.dataset is None:
            messagebox.showwarning('No dataset', 'No dataset loaded. Use Auto-load or Open CSV first.')
            return
        t = threading.Thread(target=self._train_model)
        t.daemon = True
        t.start()

    def _train_model(self):
        try:
            self.set_buttons_state('disabled')
            self.log_message('Starting training...')
            pipeline, acc, report = train_model(self.dataset, gui_callback=self.log_message)
            self.pipeline = pipeline
            self.log_message(f'Accuracy on test set: {acc:.4f}')
            self.log_message('Classification report:\n' + report)
            self.lbl_result.config(text=f'Model trained (acc {acc:.3f})')
        except Exception as e:
            self.log_message('Training failed: ' + str(e))
            messagebox.showerror('Training error', str(e))
        finally:
            self.set_buttons_state('normal')

    def set_buttons_state(self, state):
        for btn in [self.btn_load_auto, self.btn_open_csv, self.btn_train, self.btn_load_model, self.btn_save_model, self.btn_bulk]:
            try:
                btn.config(state=state)
            except Exception:
                pass

    def predict_text(self):
        txt = self.txt_input.get('1.0', tk.END).strip()
        if not txt:
            messagebox.showwarning('Input needed', 'Please paste or type news text to predict.')
            return
        if self.pipeline is None:
            messagebox.showwarning('No model', 'No model loaded or trained.')
            return
        try:
            pred = self.pipeline.predict([txt])[0]
            probs = None
            try:
                probs = self.pipeline.predict_proba([txt])[0]
                classes = self.pipeline.classes_
                prob_text = '\n'.join([f'{c}: {p:.4f}' for c, p in zip(classes, probs)])
            except Exception:
                prob_text = 'Probabilities not available for this model.'

            self.lbl_result.config(text=f'Prediction: {pred.upper()}')
            self.txt_probs.delete('1.0', tk.END)
            self.txt_probs.insert(tk.END, prob_text)
            self.log_message('Predicted: ' + pred)
        except Exception as e:
            messagebox.showerror('Prediction error', str(e))

    def save_model(self):
        if self.pipeline is None:
            messagebox.showwarning('No model', 'No model to save.')
            return
        path = filedialog.asksaveasfilename(defaultextension='.pkl', filetypes=[('Pickle', '*.pkl')], initialfile=MODEL_FILENAME)
        if not path:
            return
        try:
            with open(path, 'wb') as f:
                pickle.dump(self.pipeline, f)
            self.log_message('Saved model to ' + path)
            messagebox.showinfo('Saved', 'Model saved successfully')
        except Exception as e:
            messagebox.showerror('Save error', str(e))

    def load_model(self):
        path = filedialog.askopenfilename(filetypes=[('Pickle', '*.pkl'), ('All files', '*.*')])
        if not path:
            return
        try:
            self.pipeline = pickle.load(open(path, 'rb'))
            self.log_message('Loaded model from ' + path)
            self.lbl_result.config(text='Model loaded')
            messagebox.showinfo('Model loaded', 'Model loaded successfully')
        except Exception as e:
            messagebox.showerror('Load error', str(e))

    def bulk_predict_csv(self):
        if self.pipeline is None:
            messagebox.showwarning('No model', 'Load or train a model first for bulk predictions.')
            return
        path = filedialog.askopenfilename(filetypes=[('CSV files', '*.csv'), ('All files', '*.*')])
        if not path:
            return
        try:
            df = pd.read_csv(path)
            dfp = prepare_dataframe(df)
            texts = dfp['text'].fillna('').tolist()
            preds = self.pipeline.predict(texts)
            try:
                probs = self.pipeline.predict_proba(texts)
            except Exception:
                probs = None

            df['predicted_label'] = preds
            if probs is not None:

                df['pred_prob'] = [max(p) for p in probs]

            out_path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV', '*.csv')], initialfile='predictions.csv')
            if out_path:
                df.to_csv(out_path, index=False)
                messagebox.showinfo('Saved', f'Saved predictions to {out_path}')
                self.log_message('Saved bulk predictions to ' + out_path)
        except Exception as e:
            messagebox.showerror('Bulk predict error', str(e))


if __name__ == '__main__':
    app = FakeNewsGUI()
    app.mainloop()
