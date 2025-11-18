from typing import List
import os
import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

# Feature list
MANUAL_FEATURE_NAMES = [
    'has_digit',      # token contains any digit
    'has_hyphen',     # token contains hyphen
    'is_title',       # capitalized tokens
    'len',            # token length
    'has_nonalpha',   # punctuation/symbols
    'vowel_ratio',    # vowel density
    'suf_fil',        # Filipino affixes
    'suf_eng',        # English affixes
    'pref_fil',       # Filipino prefixes
    'pref_eng',       # English prefixes
    'func_fil',       # Filipino function words
    'func_eng',       # English function words
    'eng_letters'    # presence of letters c, x, j, z, q, f
]

def extract_feature_map(token: str):
    """
    Produce a dictionary of the manual features for a single token.
    Returned values are either bool (for presence flags) or numeric (len, ratio).
    """
    t = "" if token is None else str(token)

    # Feature computations
    f_has_digit = any(ch.isdigit() for ch in t)
    f_has_hyphen = '-' in t
    f_is_title = t.istitle()
    f_len = len(t)
    f_has_nonalpha = any(not ch.isalpha() for ch in t)
    vowels = sum(1 for ch in t.lower() if ch in 'aeiou')
    f_vowel_ratio = (vowels / f_len) if f_len > 0 else 0.0
    suf_fil = any(t.lower().endswith(s) for s in [
        'hin','han','pin','an','in','ka','ng'
    ])
    suf_eng = any(t.lower().endswith(s) for s in [
        'ing','ed','ion','ment','ly','able','ness','ful','less','est','er'
    ])
    pref_fil = any(t.lower().startswith(s) for s in [
        'mag','nag','pag','tag','pa','ma','na',
        'maka','makipag','maki','paki','ipag','ika',
        'nagpa','magpa','pinaka','pinag','taga','tiga'
    ])
    pref_eng = any(t.lower().startswith(s) for s in [
        'un','re','in','im','ir','il','dis','non','over','mis','sub','pre',
        'inter','trans','super','semi','anti','de','en','em','be','fore','out','under'
    ])
    func_fil = any(t.lower() == s for s in [
        'ang', 'mga', 'para', 'sa', 'kapag', 'habang', 'kasi', 'dahil', 'ako', 
        'siya', 'ka', 'ikaw', 'sila', 'tayo', 'nila'
    ])
    func_eng = any(t.lower() == s for s in [
        'i','you','she','they','them','he','it','we','does','did','can','will','cannot','is',
        'are','have','and','or','else','but','if','then'
    ])
    eng_letters = any(s in t.lower() for s in ['c','x','j','z','q','f'])

    return {
        'has_digit': f_has_digit,
        'has_hyphen': f_has_hyphen,
        'is_title': f_is_title,
        'len': f_len,
        'has_nonalpha': f_has_nonalpha,
        'vowel_ratio': f_vowel_ratio,
        'suf_fil': suf_fil,
        'suf_eng': suf_eng,
        'pref_fil': pref_fil,
        'pref_eng': pref_eng,
        'func_fil': func_fil,
        'func_eng': func_eng,
        'eng_letters': eng_letters
    }

def manual_features_array(tokens: List[str]):
    rows = []
    for t in tokens:
        fmap = extract_feature_map(t)
        row = []
        for name in MANUAL_FEATURE_NAMES:
            v = fmap[name]
            row.append(int(v) if isinstance(v, bool) else float(v))
        rows.append(row)
    return np.array(rows, dtype=float)

# OTH rules (updated)
def quick_oth(token: str) -> bool:
    t = "" if token is None else str(token)
    if t == '':
        return True
    if all(not ch.isalnum() for ch in t):     # punctuation-only, emojis, symbols
        return True
    if any(ch.isdigit() for ch in t):         # numeric tokens (years, amounts, ids)
        return True
    return False

SHORT_ENG_WORDS = {
    'do','am','a','an','the','in','on','at','of','for','with','has'
}
# ENG short words 
def quick_eng(token: str) -> bool:
    return token.lower() in SHORT_ENG_WORDS

BASE_DIR = os.path.dirname(__file__)
MODEL_FILE = os.path.join(BASE_DIR, 'model.joblib')

if os.path.exists(MODEL_FILE):
    _model_bundle = joblib.load(MODEL_FILE)
else:
    raise FileNotFoundError(f"model.joblib not found in {BASE_DIR}. Train the model using train.py and save model.joblib before using pinoybot.")

if not isinstance(_model_bundle, dict) or 'clf' not in _model_bundle:
    raise RuntimeError("model.joblib must be a dict containing at least the key 'clf' and preferably 'vec'.")

_clf = _model_bundle['clf']
_vec = _model_bundle.get('vec')


# Main Function
def tag_language(tokens: List[str]) -> List[str]:
    if tokens is None:
        return []

    tokens = ["" if t is None else str(t) for t in tokens]

    results = [None] * len(tokens)
    to_predict_tokens = []
    to_predict_indices = []
    for i, t in enumerate(tokens):
        if quick_oth(t):
            results[i] = 'OTH'
        elif quick_eng(t):
            results[i] = 'ENG'
        else:
            to_predict_indices.append(i)
            to_predict_tokens.append(t)

    if not to_predict_tokens:
        return results

    # 1. Transform with saved character n-gram vectorizer
    if _vec is not None:
        tokens_lower = [t.lower() for t in to_predict_tokens]
        X_ngrams = _vec.transform(tokens_lower)
    else:
        X_ngrams = None

    # 2. Manual numeric features
    X_manual = csr_matrix(manual_features_array(to_predict_tokens))

    # 3. Combine feature blocks
    if X_ngrams is not None:
        try:
            X = hstack([X_ngrams, X_manual])
        except Exception:
            X = np.hstack([X_ngrams.toarray(), X_manual.toarray()])
    else:
        X = X_manual

    # 4. Predict with the classifier
    try:
        preds = _clf.predict(X)
    except Exception:
        X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        preds = _clf.predict(X_dense)


    for idx, p in zip(to_predict_indices, preds):
        results[idx] = p if p in ('ENG', 'FIL', 'OTH') else 'OTH'

    return results

if __name__ == "__main__":
    # Input (to be changed to accept user input)
    sample = ['i','xxx','she','they','them','he','it','we','do','does','did','can','will','cannot','is',
       'am','are','have','has','a','an','the','and','or','else','but','if','then','in','on','at','of','for','with']
    print("Tokens:", sample)
    print("Tags :", tag_language(sample))
