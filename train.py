import joblib, pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

DATA_XLSX = Path('CSINTSY MCO2 Dataset (ID 572-624 Corrected).xlsx')  # REPLACE WITH FILENAME OF DATASET

def map_label_from_row(row):
    raw = None
    if 'is_correct' in row and ('is_correct' in globals() or True):
        try:
            is_correct = row.get('is_correct')
        except:
            is_correct = True
    else:
        is_correct = True
    if (pd.notna(row.get('corrected_label')) and (str(is_correct).strip().lower() in ['false','0','no','F'])):
        raw = row.get('corrected_label')
    else:
        raw = row.get('label') # Backup incase is_correct is empty (not accurate)
    if pd.isna(raw):
        return 'OTH'
    s = str(raw).strip().upper()
    if s in ('NUM','SYM','EXPR','ABB','UNK'):
        return 'OTH'
    if s == 'CS':
        return 'FIL'
    if s.startswith('ENG'):
        return 'ENG'
    if s.startswith('FIL'):
        return 'FIL'
    if s.startswith('NE'):
        return 'OTH'
    return 'OTH'

def manual_features_list(tokens):
    feats = []
    for t in tokens:
        t = str(t)
        f_has_digit = int(any(ch.isdigit() for ch in t))
        f_has_hyphen = int('-' in t)
        f_is_title = int(t.istitle())
        f_len = len(t)
        f_has_nonalpha = int(any(not ch.isalpha() for ch in t))
        vowels = sum(1 for ch in t.lower() if ch in 'aeiou')
        f_vowel_ratio = vowels / f_len if f_len>0 else 0.0
        suf_fil = int(any(t.lower().endswith(s) for s in [
            'hin','han','pin','an','in','ka','ng']))
        suf_eng = int(any(t.lower().endswith(s) for s in [
            'ment','able','ing','ion','ed','ly'
            ]))
        pref_fil = int(any(t.lower().startswith(s) for s in [
            'mag','nag','pag','tag','pa','ma','na','maka','makipag','maki',
            'paki','ipag','ika','nagpa','magpa','pinaka','pinag','taga','tiga'
            ]))
        pref_eng = int(any(t.lower().startswith(s) for s in [
            'un','re','in','im','ir','il','dis','non','over','mis','sub','pre',
            'inter','trans','super','semi','anti','de','en','em','be','fore','out','under'
            ]))
        feats.append([f_has_digit, f_has_hyphen, f_is_title, f_len, f_has_nonalpha, f_vowel_ratio, suf_fil, suf_eng, pref_fil, pref_eng])
    return np.array(feats, dtype=float)

def main():
    df = pd.read_excel(DATA_XLSX)
    df_proc = df.copy()
    if 'word' not in df_proc.columns:
        raise RuntimeError('Expected column "word" in dataset.')
    df_proc['label_mapped'] = df_proc.apply(map_label_from_row, axis=1)
    df_proc['token'] = df_proc['word'].astype(str)
    df_proc = df_proc[~df_proc['token'].str.strip().eq('')].copy()

    vec = CountVectorizer(analyzer='char_wb', ngram_range=(2,4))
    X_ngrams = vec.fit_transform(df_proc['token'].str.lower().tolist())
    X_manual = csr_matrix(manual_features_list(df_proc['token'].tolist()))
    X = hstack([X_ngrams, X_manual])
    y = df_proc['label_mapped'].values

    # train/val/test split 70-15-15
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    val_size_rel = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size_rel, random_state=42, stratify=y_temp)

    clf = MultinomialNB(alpha=1.0)
    clf.fit(X_train, y_train)

    # scores
    train_acc = clf.score(X_train, y_train)
    val_acc = clf.score(X_val, y_val)
    test_acc = clf.score(X_test, y_test)

    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}\n")

    # detailed diagnostics for validation
    y_val_pred = clf.predict(X_val)
    labels = ['FIL', 'ENG', 'OTH']  # order for reporting
    print("Validation performance:")
    print(classification_report(y_val, y_val_pred, labels=labels, target_names=labels, digits=2))
    cm_val = confusion_matrix(y_val, y_val_pred, labels=labels)
    print("Validation confusion matrix (rows=true, cols=pred):")
    print(cm_val, "\n")

    # detailed diagnostics for test
    y_test_pred = clf.predict(X_test)
    print("Test performance:")
    print(classification_report(y_test, y_test_pred, labels=labels, target_names=labels, digits=2))
    cm_test = confusion_matrix(y_test, y_test_pred, labels=labels)
    print("Test confusion matrix (rows=true, cols=pred):")
    print(cm_test, "\n")

    # save model and vectorizer
    joblib.dump({'clf':clf,'vec':vec,'manual_feature_names':['has_digit','has_hyphen','is_title','len','has_nonalpha','vowel_ratio','suf_fil','suf_eng']}, 'model.joblib')

if __name__=='__main__':
    main()
