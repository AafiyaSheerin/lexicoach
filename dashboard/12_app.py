import os
import pickle
import re
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag as nltk_pos_tag
from nltk.corpus import wordnet

# wordfreq: used to filter obscure/archaic WordNet synonyms
try:
    from wordfreq import word_frequency as _wf
    def _word_freq(w): return _wf(w, 'en')
except ImportError:
    # Graceful fallback if wordfreq not available — pass all candidates
    def _word_freq(w): return 1.0

# Minimum frequency a WordNet synonym must have to be accepted.
# 1e-5 ≈ everyday words ("achieve", "obtain", "demonstrate").
# Rare words like "ilk", "clip" (≈time), "scat" score well below this.
_WF_THRESHOLD = 1e-5

# Ensure all required NLTK resources are downloaded to prevent LookupErrors
import nltk
for res, path in [('stopwords', 'corpora/stopwords'), 
                  ('wordnet', 'corpora/wordnet'), 
                  ('punkt', 'tokenizers/punkt'), 
                  ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
                  ('omw-1.4', 'corpora/omw-1.4')]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(res, quiet=True)

# ══════════════════════════════════════════════════════════════
# dashboard/12_app.py
# LexiCoach Streamlit App (With SWAYAM-style Login & Real Google OAuth)
# ══════════════════════════════════════════════════════════════

# To support both new and old versions of streamlit query parameters:
def get_query_params():
    try:
        return st.query_params
    except AttributeError:
        return st.experimental_get_query_params()

def set_query_params(params):
    try:
        st.query_params.clear()
        for k, v in params.items():
            st.query_params[k] = v
    except AttributeError:
        st.experimental_set_query_params(**{k: [v] for k, v in params.items()})

# Load Google OAuth Config
def load_oauth_config():
    try:
        if os.path.exists("oauth_config.json"):
            with open("oauth_config.json", "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_oauth_config(client_id, client_secret):
    try:
        with open("oauth_config.json", "w") as f:
            json.dump({"client_id": client_id, "client_secret": client_secret}, f)
        return True
    except Exception:
        return False

oauth_config = load_oauth_config()
client_id = oauth_config.get("client_id")
client_secret = oauth_config.get("client_secret")

# Router using URL query parameters
params = get_query_params()
redirect_uri = "http://localhost:8501/"

# 1. Real Google OAuth redirect callback handler
if "code" in params:
    auth_code = params["code"]
    if client_id and client_secret:
        try:
            token_response = requests.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": auth_code,
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code"
                }
            )
            if token_response.status_code == 200:
                tokens = token_response.json()
                id_token = tokens.get("id_token")
                # Fetch profile details securely using TokenInfo API
                user_info_resp = requests.get(f"https://oauth2.googleapis.com/tokeninfo?id_token={id_token}")
                if user_info_resp.status_code == 200:
                    user_info = user_info_resp.json()
                    st.session_state['logged_in'] = True
                    st.session_state['user_email'] = user_info.get("email")
                    st.session_state['user_name'] = user_info.get("name", "Google User")
                    set_query_params({})
                    st.rerun()
                else:
                    st.error(f"Failed to fetch user info: {user_info_resp.text}")
            else:
                st.error(f"Failed to exchange auth code: {token_response.text}")
        except Exception as e:
            st.error(f"OAuth 2.0 Error: {str(e)}")
    else:
        st.error("Received authorization code but Google OAuth credentials are not configured.")

# 2. Simulated/Mock Redirect back from Choose an Account page
if params.get("logged_in") == "true" and "email" in params:
    st.session_state['logged_in'] = True
    st.session_state['user_email'] = params["email"]
    set_query_params({})
    st.rerun()

# 3. Choose an Account full-screen simulation page (renders when page=mock_google_auth)
if params.get("page") == "mock_google_auth":
    custom_email = params.get("email", "aafeeya.sheerin@gmail.com")
    name_parts = custom_email.split('@')[0].replace('.', ' ').title()
    first_char = name_parts[0] if name_parts else 'G'
    
    st.markdown(f"""<style>
/* Hide standard Streamlit UI elements for a distraction-free screen */
[data-testid="stSidebar"], [data-testid="stHeader"] {{
    display: none !important;
}}
[data-testid="stAppViewBlockContainer"] {{
    padding: 0 !important;
    max-width: 100% !important;
    background-color: #FFFFFF !important;
}}
.google-login-container {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    background-color: #FFFFFF;
    font-family: 'Roboto', 'Arial', sans-serif;
    color: #202124;
    width: 100%;
}}
.google-card {{
    width: 450px;
    background: #FFFFFF;
    border: 1px solid #DADCE0;
    border-radius: 8px;
    padding: 40px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
}}
.google-logo {{
    display: flex;
    justify-content: center;
    margin-bottom: 16px;
}}
.google-title {{
    font-size: 24px;
    font-weight: 400;
    line-height: 32px;
    text-align: center;
    margin: 0 0 8px 0 !important;
    color: #202124 !important;
}}
.google-subtitle {{
    font-size: 16px;
    font-weight: 400;
    line-height: 24px;
    text-align: center;
    margin: 0 0 24px 0 !important;
    color: #202124 !important;
}}
.account-list {{
    display: flex;
    flex-direction: column;
    border-top: 1px solid #DADCE0;
}}
.account-row-link {{
    text-decoration: none !important;
    color: inherit !important;
}}
.account-row {{
    display: flex;
    align-items: center;
    padding: 14px 0;
    border-bottom: 1px solid #DADCE0;
    cursor: pointer;
    transition: background-color 0.15s ease;
}}
.account-row:hover {{
    background-color: #F8F9FA;
}}
.avatar {{
    width: 32px;
    height: 32px;
    border-radius: 50%;
    color: #FFFFFF;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 500;
    font-size: 15px;
    margin-right: 12px;
}}
.info-block {{
    display: flex;
    flex-direction: column;
    text-align: left;
}}
.name {{
    font-size: 14px;
    font-weight: 500;
    color: #3C4043 !important;
}}
.email {{
    font-size: 12px;
    color: #5F6368 !important;
    margin-top: 2px;
}}
.google-footer {{
    display: flex;
    justify-content: space-between;
    width: 450px;
    margin-top: 24px;
    font-size: 12px;
    color: #757575;
    padding: 0 8px;
}}
.google-footer a {{
    color: #757575;
    text-decoration: none;
}}
</style>

<div class="google-login-container">
    <div class="google-card">
        <div class="google-logo">
            <svg width="24" height="24" viewBox="0 0 24 24">
                <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22c-.62-.62-1.07-1.37-1.2-2.12z"/>
                <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"/>
            </svg>
        </div>
        <h1 class="google-title">Choose an account</h1>
        <p class="google-subtitle">to continue to <span style="color: #4A148C; font-weight: bold;">LexiCoach</span></p>
        
        <div class="account-list">
            <a href="?logged_in=true&email={custom_email}" target="_parent" class="account-row-link">
                <div class="account-row">
                    <div class="avatar" style="background-color: #1A73E8;">{first_char}</div>
                    <div class="info-block">
                        <div class="name">{name_parts}</div>
                        <div class="email">{custom_email}</div>
                    </div>
                </div>
            </a>
            <a href="?logged_in=true&email=student@lexicoach.edu" target="_parent" class="account-row-link">
                <div class="account-row">
                    <div class="avatar" style="background-color: #E8710A;">S</div>
                    <div class="info-block">
                        <div class="name">Lexi Student</div>
                        <div class="email">student@lexicoach.edu</div>
                    </div>
                </div>
            </a>
            <a href="?logged_in=true&email=professor@lexicoach.edu" target="_parent" class="account-row-link">
                <div class="account-row">
                    <div class="avatar" style="background-color: #137333;">P</div>
                    <div class="info-block">
                        <div class="name">Lexi Professor</div>
                        <div class="email">professor@lexicoach.edu</div>
                    </div>
                </div>
            </a>
        </div>
    </div>
    <div class="google-footer">
        <div>English (United States)</div>
        <div style="display: flex; gap: 15px;">
            <a href="#">Help</a>
            <a href="#">Privacy</a>
            <a href="#">Terms</a>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
    st.stop()


# Set Page Config
st.set_page_config(
    page_title="LexiCoach — Writing Improvement System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Bright Theme CSS
st.markdown("""
<style>
    /* Main Background Gradient: White, Light Purple, Light Sky Blue */
    body, [data-testid="stAppViewContainer"] {
        background-color: #FFFFFF;
        background-image: linear-gradient(135deg, #F3E5F5 0%, #E3F2FD 100%);
        color: #000000 !important;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    
    /* Input field borders and text color for normal fields */
    .stTextArea textarea {
        color: #000000 !important;
        background-color: #FFFFFF !important;
        border: 1px solid #B0BEC5 !important;
        border-radius: 8px !important;
    }
    
    /* Text overrides to black */
    h1, h2, h3, h4, h5, h6, p, span, label, li {
        color: #000000 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #F8F9FA !important;
        border-right: 1px solid #E1BEE7 !important;
    }
    
    /* Premium SWAYAM style card styling (stVerticalBlockBorderWrapper wrapper override) */
    [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stVerticalBlockBorderWrapper"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #E0E0E0 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        padding: 30px !important;
        max-width: 480px !important;
        margin: auto !important;
    }
    
    /* SWAYAM style floating labels inside stTextInput */
    div[data-testid="stTextInput"] {
        position: relative;
        padding-top: 10px !important;
        margin-bottom: 20px !important;
    }
    div[data-testid="stTextInput"] label {
        position: absolute;
        top: 0px !important;
        left: 10px !important;
        background-color: #FFFFFF !important;
        padding: 0 4px !important;
        font-size: 13px !important;
        color: #757575 !important;
        z-index: 99 !important;
        font-weight: 500 !important;
    }
    div[data-testid="stTextInput"] input {
        border: 1px solid #CFD8DC !important;
        border-radius: 4px !important;
        padding: 10px 12px !important;
        background-color: #FFFFFF !important;
        color: #000000 !important;
        height: 48px !important;
        font-size: 15px !important;
    }
    
    /* Social sign in buttons and icons */
    .google-btn:hover {
        opacity: 0.9;
    }
    
    /* SWAYAM Cancel and Sign In buttons */
    button[data-testid="stBaseButton-secondary"], button[kind="secondary"], .stButton button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: #1976D2 !important;
        border: 1px solid #1976D2 !important;
        font-weight: 600 !important;
        border-radius: 4px !important;
        padding: 8px 16px !important;
        width: 100% !important;
        height: 40px !important;
        transition: background-color 0.2s !important;
    }
    button[data-testid="stBaseButton-secondary"]:hover, button[kind="secondary"]:hover {
        background-color: #F5F5F5 !important;
    }
    
    button[data-testid="stBaseButton-primary"], button[kind="primary"], .stButton button[kind="primary"] {
        background-color: #1976D2 !important;
        color: #FFFFFF !important;
        border: 1px solid #1976D2 !important;
        font-weight: 600 !important;
        border-radius: 4px !important;
        padding: 8px 16px !important;
        width: 100% !important;
        height: 40px !important;
        transition: background-color 0.2s !important;
    }
    button[data-testid="stBaseButton-primary"]:hover, button[kind="primary"]:hover {
        background-color: #1565C0 !important;
    }
    
    /* Cloudflare turnstile mockup styling */
    .cloudflare-spinner {
        border: 3px solid #f3f3f3;
        border-top: 3px solid #F38020;
        border-radius: 50%;
        width: 20px;
        height: 20px;
        animation: spin 1s linear infinite;
        display: inline-block;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Tag Highlights */
    .highlight-informal {
        background-color: #FFEBEE;
        border: 1.5px solid #EF5350;
        border-radius: 4px;
        padding: 2px 6px;
        font-weight: 600;
        color: #C62828 !important;
        cursor: pointer;
    }
    .highlight-academic {
        background-color: #E8F5E9;
        border: 1.5px solid #66BB6A;
        border-radius: 4px;
        padding: 2px 6px;
        font-weight: 600;
        color: #2E7D32 !important;
    }
    .highlight-impact {
        background-color: #FFF8E1;
        border: 2px dashed #FFCA28;
        border-radius: 4px;
        padding: 3px 8px;
        font-weight: 700;
        color: #F57F17 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL CHKPNT ─────────────────────────────────────────
@st.cache_resource
def load_lexicoach_model():
    if os.path.exists("model.pkl"):
        with open("model.pkl", "rb") as f:
            return pickle.load(f)
    return None

model_data = load_lexicoach_model()
lemmatizer = WordNetLemmatizer()

# Fallback dictionaries for academic replacements
ACADEMIC_REPLACEMENTS = {
    "use": ["utilize", "employ", "harness", "apply"],
    "uses": ["utilizes", "employs", "harnesses", "applies"],
    "used": ["utilized", "employed", "harnessed", "applied"],
    "using": ["utilizing", "employing", "harnessing", "applying"],
    "get": ["obtain", "acquire", "retrieve", "procure"],
    "gets": ["obtains", "acquires", "retrieves"],
    "got": ["obtained", "acquired", "retrieved"],
    "getting": ["obtaining", "acquiring", "retrieving"],
    "show": ["demonstrate", "illustrate", "exhibit", "indicate"],
    "shows": ["demonstrates", "illustrates", "exhibits", "indicates"],
    "showed": ["demonstrated", "illustrated", "exhibited"],
    "showing": ["demonstrating", "illustrating", "exhibiting"],
    "make": ["generate", "construct", "formulate", "produce"],
    "makes": ["generates", "constructs", "formulates", "produces"],
    "made": ["generated", "constructed", "formulated", "produced"],
    "making": ["generating", "constructing", "formulating"],
    "help": ["facilitate", "assist", "expedite", "support"],
    "helps": ["facilitates", "assists", "expedites", "supports"],
    "helped": ["facilitated", "assisted", "expedited"],
    "helping": ["facilitating", "assisting", "expediting"],
    "need": ["require", "necessitate", "demand"],
    "needs": ["requires", "necessitates", "demands"],
    "needed": ["required", "necessitated", "demanded"],
    "start": ["commence", "initiate", "embark"],
    "starts": ["commences", "initiates"],
    "started": ["commenced", "initiated"],
    "end": ["conclude", "terminate"],
    "ends": ["concludes", "terminates"],
    "ended": ["concluded", "terminated"],
    "good": ["proficient", "exemplary", "superior", "beneficial"],
    "bad": ["adverse", "detrimental", "deficient", "substandard"],
    "big": ["substantial", "significant", "considerable", "prominent"],
    "large": ["substantial", "extensive", "considerable"],
    "small": ["minimal", "negligible", "marginal"],
    "better": ["superior", "enhanced", "preferable"],
    "best": ["optimal", "paramount", "supreme"],
    "results": ["outcomes", "findings", "consequences"],
    "result": ["outcome", "finding", "consequence"],
    "methods": ["methodologies", "approaches", "procedures"],
    "method": ["methodology", "approach", "procedure"],
    "kids": ["children", "adolescents", "youth"],
    "things": ["elements", "phenomena", "components"],
    "thing": ["element", "phenomenon", "component"],
    "ways": ["methodologies", "channels", "approaches"],
    "way": ["methodology", "channel", "approach"],
    "idea": ["concept", "notion", "hypothesis"],
    "ideas": ["concepts", "notions", "hypotheses"],
    "problem": ["challenge", "constraint", "adversity"],
    "problems": ["challenges", "constraints", "adversities"],
    "teacher": ["educator", "instructor", "mentor"],
    "teachers": ["educators", "instructors"],
    "student": ["learner", "scholar"],
    "students": ["learners", "scholars"],
    "improve": ["enhance", "ameliorate", "refine"],
    "improves": ["enhances", "ameliorates", "refines"],
    "improved": ["enhanced", "ameliorated", "refined"],
    "improving": ["enhancing", "ameliorating", "refining"]
}

# ── INFERENCE HELPER FUNCTIONS ────────────────────────────────
def clean_text_inf(text):
    text = str(text)
    text = re.sub(r'@[A-Z]+\d*', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def is_valid_word_inf(word):
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    if word in stop_words: return False
    if len(word) < 3: return False
    if len(word) > 25: return False
    if not word.isalpha(): return False
    return True

def count_syllables_inf(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    prev_vow = False
    for ch in word:
        is_v = ch in vowels
        if is_v and not prev_vow:
            count += 1
        prev_vow = is_v
    if word.endswith('e') and count > 1:
        count -= 1
    return max(1, count)

pos_cache_inf = {}
def get_pos_score_inf(word):
    w = word.lower()
    if w in pos_cache_inf:
        return pos_cache_inf[w]
    try:
        tag = nltk_pos_tag([w])[0][1]
        if tag.startswith('NN'): score = 1.0
        elif tag.startswith('JJ'): score = 1.0
        elif tag.startswith('VB'): score = 0.5
        elif tag.startswith('RB'): score = 0.3
        else: score = 0.2
    except Exception:
        score = 0.5
    pos_cache_inf[w] = score
    return score

def consonant_density_inf(word):
    vowels = set('aeiou')
    consonants = [c for c in word.lower() if c.isalpha() and c not in vowels]
    return min(1.0, len(consonants) / max(len(word), 1))

def unique_char_ratio_inf(word):
    return min(1.0, len(set(word.lower())) / max(len(word), 1))

# Latin prefixes & suffixes (from feature extractor)
LATIN_PREFIXES_INF = ("inter","intra","trans","pre","post","sub","super","extra","ultra","semi","anti","counter","non","over","under","out","dis","mis","re","un","co","com","con","col","cor","pro","per","para","meta","poly","mono","micro","macro","hyper","hypo","auto","bio","geo","neo","pseudo","quasi","multi","omni","uni","bi","tri","quad","penta","hexa","deca","cent","kilo","mega","giga","tele","photo","thermo","electro","hydro","aero","chrono","demo","ethn","gen")
FORMAL_SUFFIXES_INF = ('tion','sion','ity','ance','ence','ment','ness','ism','ive','ous','ize','ise','ate','ify','fy','ology','ography','ometry')
TOP500_COMMON_INF = set(["time","year","people","way","day","man","woman","child","world","life","hand","part","place","case","week","company","system","program","question","work","government","number","night","point","home","water","room","mother","area","money","story","fact","month","lot","right","study","book","eye","job","word","business","issue","side","kind","head","house","service","friend","father","power","hour","game","line","end","among","off","always","state","once","book","hear","body","drive","took","play","move","live","believe","hold","bring","happen","write","provide","sit","stand","lose","pay","meet","include","continue","set","learn","change","lead","understand","watch","follow","stop","create","speak","read","spend","grow","open","walk","win","offer","remember","love","consider","appear","buy","wait","serve","die","send","expect","build","stay","fall","cut","reach","kill","remain","suggest","raise","pass","sell","require","report","decide","pull","break","wish","pick","carry","explain","face","approach","allow","become","begin","call","come","feel","gave","give","gone","got","keep","knew","know","let","long","look","made","make","need","next","nothing","often","old","only","other","over","own","place","put","said","same","saw","says","see","seen","show","small","still","take","taken","tell","thing","think","thought","told","took","turn","under","upon","used","want","well","went","were","when","where","while","whose","within","without","year","years","young","able","about","actually","after","again","ago","almost","along","already","although","another","around","away","back","because","been","before","being","between","both","brought","came","come","done","down","during","even","ever","every","everyone","everything","few","find","first","found","four","from","full","further","gave","give","going","great","group","hard","having","here","high","him","himself","however","important","instead","into","its","itself","just","large","last","later","left","less","like","likely","little","local","made","make","many","might","much","must","near","never","new","next","night","none","nor","nothing","now","number","off","once","only","open","our","out","over","own","part","past","people","place","plan","point","possible","problem","public","real","right","run","same","several","should","since","small","some","something","sometimes","soon","such","sure","take","than","their","them","then","there","these","they","thing","this","those","three","through","together","too","toward","turn","two","under","until","very","view","want","water","whether","which","while","whole","whose","wide","will","with","within","would","write"])

def extract_word_feats_inf(word, word_freq, total_freq):
    w = word.lower()
    freq = word_freq.get(w, 0)
    f_freq = min(1.0, np.log1p(freq) / np.log1p(total_freq))
    f_syl = min(1.0, count_syllables_inf(w) / 6.0)
    f_len = min(1.0, len(w) / 20.0)
    f_suff = 1.0 if w.endswith(FORMAL_SUFFIXES_INF) else 0.0
    f_pos = get_pos_score_inf(w)
    f_com = 1.0 if w in TOP500_COMMON_INF else 0.0
    f_lat = 1.0 if any(w.startswith(p) for p in LATIN_PREFIXES_INF) else 0.0
    f_cons = consonant_density_inf(w)
    f_uniq = unique_char_ratio_inf(w)
    return [f_freq, f_syl, f_len, f_suff, f_pos, f_com, f_lat, f_cons, f_uniq]

def extract_features_with_context_inf(words, idx, word_freq, total_freq):
    word_feats = extract_word_feats_inf(words[idx], word_freq, total_freq)
    neighbours = []
    for offset in [-2, -1, 1, 2]:
        ni = idx + offset
        if 0 <= ni < len(words):
            nw = words[ni].lower()
            freq = word_freq.get(nw, 0)
            neighbours.append(np.log1p(freq) / np.log1p(total_freq))
    f_ctx = float(np.mean(neighbours)) if neighbours else 0.5
    return word_feats + [f_ctx]

def forward_inference(X, weights, biases, gamma, beta, running_mean, running_var, activation, use_batchnorm):
    a = X
    L = len(weights) + 1
    for l in range(L - 2):
        W = weights[l]
        b = biases[l]
        z = a @ W + b
        
        if use_batchnorm:
            mean = running_mean[l]
            var = running_var[l]
            z_hat = (z - mean) / np.sqrt(var + 1e-5)
            z_bn = gamma[l] * z_hat + beta[l]
            pre_activation = z_bn
        else:
            pre_activation = z
            
        if activation == 'relu':
            a = np.maximum(0, pre_activation)
        else:
            a = 1.0 / (1.0 + np.exp(-np.clip(pre_activation, -500, 500)))
            
    W = weights[-1]
    b = biases[-1]
    z = a @ W + b
    a_out = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    return a_out


# ── SESSION STATE INITIALIZATION ──────────────────────────────
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'user_email' not in st.session_state:
    st.session_state['user_email'] = ""
if 'show_google_emails' not in st.session_state:
    st.session_state['show_google_emails'] = False

# ── LOGIN VIEW ────────────────────────────────────────────────
if not st.session_state['logged_in']:
    # Compute the Google authorization URL dynamically
    if client_id and client_secret:
        google_auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}&scope=openid%20email%20profile&state=google_oauth"
    else:
        # Fallback simulator URL
        google_auth_url = "?page=mock_google_auth&email=aafeeya.sheerin@gmail.com"

    st.markdown("<br><br>", unsafe_allow_html=True)
    
    _, col_mid, _ = st.columns([1.2, 2, 1.2])
    
    with col_mid:
        with st.container(border=True):
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 20px;">
                <h2 style="margin: 0; font-size: 22px; font-weight: 700; color: #000000; font-family: 'Inter', sans-serif;">Sign in to LexiCoach</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Google Sign In button
            st.markdown(f"""
            <div style="margin-bottom: 15px; text-align: center;">
                <a href="{google_auth_url}" target="_parent" class="google-btn" style="text-decoration: none; display: block; width: 100%;">
                    <div style="display: flex; align-items: center; justify-content: center; gap: 10px; border: 1px solid #CFD8DC; border-radius: 4px; padding: 10px 16px; background-color: #FFFFFF; cursor: pointer; transition: background-color 0.2s; box-sizing: border-box; width: 100%;">
                        <svg width="18" height="18" viewBox="0 0 24 24">
                            <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                            <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                            <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22c-.62-.62-1.07-1.37-1.2-2.12z"/>
                            <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"/>
                        </svg>
                        <span style="color: #3C4043; font-weight: 500; font-size: 14px; font-family: Roboto, Arial, sans-serif;">Sign in with Google</span>
                    </div>
                </a>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<div style='text-align: center; margin: 15px 0 25px 0; font-size: 13px; color: #757575; border-bottom: 1px solid #EEEEEE; line-height: 0.1em;'><span style='background:#fff; padding:0 10px;'>Or continue with LexiCoach Account</span></div>", unsafe_allow_html=True)
            
            # Email & Password Fields
            email_input = st.text_input("Email", placeholder="Enter your email", key="login_email_box")
            password_input = st.text_input("Password", type="password", placeholder="Enter your password", key="login_pwd_box")
            
            # Forgot password link
            st.markdown("<div style='text-align: right; font-size: 13px; margin-top: -10px; margin-bottom: 20px;'><a href='#' style='color: #1976D2; text-decoration: none; font-weight: 500;'>Forgot your password?</a></div>", unsafe_allow_html=True)
            
            # Cancel and Sign In action buttons side-by-side
            col_can, col_si = st.columns(2)
            with col_can:
                if st.button("Cancel", use_container_width=True, key="cancel_btn"):
                    st.session_state['login_email_box'] = ""
                    st.session_state['login_pwd_box'] = ""
            with col_si:
                if st.button("Sign In", type="primary", use_container_width=True, key="signin_btn"):
                    if email_input and len(password_input) >= 6:
                        st.session_state['user_email'] = email_input
                        st.session_state['logged_in'] = True
                        st.rerun()
                    else:
                        st.error("Please enter a valid email and a password of at least 6 characters.")
            
            # Sign Up Text
            st.markdown("<div style='text-align: center; font-size: 14px; margin-top: 20px;'>Need a LexiCoach account? <a href='#' style='color: #1976D2; text-decoration: none; font-weight: 600;'>Sign Up</a></div>", unsafe_allow_html=True)
        
        # Collapsible Google OAuth Config & Custom Simulator
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🛠️ Google Sign-In Setup & Simulator", expanded=False):
            st.markdown("""
            **To configure Google Sign-In (Real Browser-based OAuth):**
            1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
            2. Enable the **OAuth Consent Screen** for your project.
            3. Create an **OAuth 2.0 Client ID** (Web application type).
            4. Add the redirect URI: `http://localhost:8501/`.
            5. Paste your Client ID and Client Secret below and save.
            """)
            
            saved_client_id = client_id if client_id else ""
            saved_client_secret = client_secret if client_secret else ""
            
            cid = st.text_input("Google Client ID", value=saved_client_id, placeholder="Paste your Client ID here", key="setup_client_id")
            csec = st.text_input("Google Client Secret", value=saved_client_secret, type="password", placeholder="Paste your Client Secret here", key="setup_client_secret")
            
            if st.button("Save Google OAuth Config"):
                if cid and csec:
                    if save_oauth_config(cid, csec):
                        st.success("Configuration saved successfully! Reloading...")
                        st.rerun()
                    else:
                        st.error("Failed to write oauth_config.json. Check workspace permissions.")
                else:
                    st.error("Please fill in both Client ID and Client Secret.")
            
            st.markdown("---")
            st.markdown("**Or, simulate Google Login using a custom email:**")
            sim_email = st.text_input("Gmail address to simulate", value="aafeeya.sheerin@gmail.com", placeholder="e.g. yourname@gmail.com", key="sim_email_input")
            if st.button("Simulate Browser Google Login"):
                sim_auth_url = f"?page=mock_google_auth&email={sim_email}"
                components.html(f"<script>window.parent.location.href = '{sim_auth_url}';</script>", height=0)
                st.stop()
        
        # Footer (Divider, Download App, Follow Us)
        st.markdown("""
        <div style="margin-top: 30px; text-align: center; width: 100%;">
            <hr style="border: 0; border-top: 1px solid #E0E0E0; margin: 20px 0; width: 100%;" />
            <div style="display: flex; justify-content: space-around; max-width: 480px; margin: auto; padding: 10px 0;">
                <div style="text-align: left;">
                    <div style="font-size: 12px; color: #757575; font-weight: 600; margin-bottom: 8px;">Download App</div>
                    <a href="https://play.google.com/store" target="_blank" style="text-decoration: none;">
                        <div style="background-color: #000000; border-radius: 4px; padding: 6px 12px; display: flex; align-items: center; gap: 8px; border: 1px solid #333;">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="#FFFFFF">
                                <path d="M5 3.25c-.27 0-.52.11-.7.31L12.5 12l-8.2 8.44c.18.2.43.31.7.31.22 0 .43-.07.61-.2l12.44-7.25c.57-.33.95-.97.95-1.7 0-.73-.38-1.37-.95-1.7L6.61 3.45c-.18-.13-.39-.2-.61-.2z"/>
                            </svg>
                            <div style="text-align: left; line-height: 1;">
                                <span style="font-size: 8px; color: #AAAAAA; display: block;">GET IT ON</span>
                                <span style="font-size: 12px; color: #FFFFFF; font-weight: 700; font-family: 'Inter', sans-serif;">Google Play</span>
                            </div>
                        </div>
                    </a>
                </div>
                <div style="text-align: left;">
                    <div style="font-size: 12px; color: #757575; font-weight: 600; margin-bottom: 8px;">Follow Us</div>
                    <div style="display: flex; gap: 12px;">
                        <a href="https://facebook.com" target="_blank" style="text-decoration: none;">
                            <div style="width: 32px; height: 32px; border-radius: 50%; background-color: #1877F2; display: flex; align-items: center; justify-content: center;">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="#FFFFFF">
                                    <path d="M22 12c0-5.52-4.48-10-10-10S2 6.48 2 12c0 4.84 3.44 8.87 8 9.8V15H8v-3h2V9.5C10 7.57 11.57 6 13.5 6H16v3h-2c-.55 0-1 .45-1 1v2h3v3h-3v6.8c4.56-.93 8-4.96 8-9.8z"/>
                                </svg>
                            </div>
                        </a>
                        <a href="https://instagram.com" target="_blank" style="text-decoration: none;">
                            <div style="width: 32px; height: 32px; border-radius: 50%; background: radial-gradient(circle at 30% 107%, #fdf497 0%, #fdf497 5%, #fd5949 45%,#d6249f 60%,#285AEB 90%); display: flex; align-items: center; justify-content: center;">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="#FFFFFF">
                                    <path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zM12 0C8.741 0 8.333.014 7.053.072 2.695.272.273 2.69.073 7.051C.014 8.333 0 8.741 0 12c0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98 1.281.058 1.689.072 4.948.072 3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98C15.668.014 15.259 0 12 0zm0 5.838a6.162 6.162 0 1 0 0 12.324 6.162 6.162 0 0 0 0-12.324zM12 16a4 4 0 1 1 0-8 4 4 0 0 1 0 8zm6.406-11.845a1.44 1.44 0 1 0 0 2.881 1.44 1.44 0 0 0 0-2.881z"/>
                                </svg>
                            </div>
                        </a>
                        <a href="https://twitter.com" target="_blank" style="text-decoration: none;">
                            <div style="width: 32px; height: 32px; border-radius: 50%; background-color: #000000; display: flex; align-items: center; justify-content: center;">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="#FFFFFF">
                                    <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
                                </svg>
                            </div>
                        </a>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── LOGGED IN MAIN APP VIEW ───────────────────────────────────
else:
    # Sidebar
    st.sidebar.markdown(f"""
    <div style="text-align: center; padding: 15px 0;">
        <span style="font-size: 45px;">🎓</span>
        <h3 style="margin: 10px 0 5px 0; color: #4A148C; font-weight: 700;">LexiCoach</h3>
        <p style="margin: 0; font-size: 12px; color: #616161;">Active User: <b>{st.session_state['user_email']}</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    if st.sidebar.button("Logout", type="primary", use_container_width=True):
        st.session_state['logged_in'] = False
        st.session_state['user_email'] = ""
        st.rerun()
        
    st.sidebar.markdown("""
    <div style="margin-top: 30px; font-size: 12px; color: #757575; line-height: 1.4;">
        <b>Theoretical Grounding:</b><br>
        Surfacing every correction causes high cognitive load. Surfacing a <i>single</i> high-impact substitution helps learners focus and retain better academic vocabulary choices.
    </div>
    """, unsafe_allow_html=True)
    
    # Title & Header
    st.markdown("""
    <div style="background-color: #FFFFFF; border: 1px solid #E1BEE7; border-radius: 10px; padding: 20px; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(106, 27, 154, 0.04);">
        <h1 style="margin: 0; color: #4A148C; font-size: 28px; font-weight: 700;">LexiCoach Writing Scorer & Essay Optimizer</h1>
        <p style="margin: 5px 0 0 0; color: #616161; font-size: 14px;">Surfacing single highest-impact academic substitutions powered by a custom-trained MLP neural network.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2 = st.tabs(["✍️ Essay Optimizer", "📖 About LexiCoach"])
    
    # ── TAB 1: ESSAY OPTIMIZER ────────────────────────────────
    with tab1:
        if model_data is None:
            st.error("Model weights checkpoint not found. Please train the model first by running `python 04_mlp.py`.")
        else:
            st.markdown("### Paste your passage below to optimize vocabulary:")
            essay_input = st.text_area(
                "Essay Passage",
                value="Students need to use good methods to get better results",
                height=150,
                label_visibility="collapsed"
            )
            
            if st.button("Analyze Essay & Optimize", type="primary"):
                # Run Optimization
                cleaned_text = clean_text_inf(essay_input)
                raw_words = cleaned_text.split()
                
                # Check valid words
                valid_words = []
                valid_indices = []
                for idx, w in enumerate(raw_words):
                    if is_valid_word_inf(w):
                        valid_words.append(w)
                        valid_indices.append(idx)
                        
                if not valid_words:
                    st.warning("No valid vocabulary words found in the passage (excluding short words and stopwords).")
                else:
                    word_freq = model_data['word_freq']
                    total_freq = model_data['total_freq']
                    weights = model_data['weights']
                    biases = model_data['biases']
                    gamma = model_data['gamma']
                    beta = model_data['beta']
                    running_mean = model_data['running_mean']
                    running_var = model_data['running_var']
                    activation = model_data['activation']
                    use_batchnorm = model_data['use_batchnorm']
                    threshold = model_data['threshold']
                    
                    # Comprehensive curated academic replacement dictionary
                    # Covers slang, contractions, colloquialisms and informal terms
                    ACADEMIC_REPLACEMENTS = {
                        # Degree/manner adverbs
                        "kinda": ["somewhat", "rather", "moderately", "to some extent"],
                        "sorta": ["somewhat", "to some degree", "relatively"],
                        "pretty": ["considerably", "relatively", "notably"],
                        "super": ["exceptionally", "remarkably", "exceedingly"],
                        "totally": ["entirely", "completely", "thoroughly"],
                        "really": ["genuinely", "significantly", "substantially"],
                        "very": ["considerably", "greatly", "substantially"],
                        "lots": ["numerous", "a substantial number of", "considerable"],
                        "way": ["considerably", "substantially", "markedly"],
                        "tons": ["numerous", "a significant quantity of"],
                        # Contractions / intent verbs
                        "wanna": ["wish to", "desire to", "intend to"],
                        "gonna": ["going to", "preparing to", "intending to"],
                        "gotta": ["must", "need to", "am required to"],
                        "hafta": ["must", "am obligated to", "need to"],
                        "lemme": ["allow me to", "permit me to"],
                        "gimme": ["provide me with", "supply"],
                        # Social / emotional words
                        "chill": ["composed", "relaxed", "amenable", "informal"],
                        "chilling": ["relaxing", "socializing", "unwinding"],
                        "vibe": ["atmosphere", "ambiance", "mood", "tenor"],
                        "vibes": ["atmosphere", "ambiance", "disposition"],
                        "cool": ["commendable", "impressive", "favorable"],
                        "awesome": ["remarkable", "exceptional", "outstanding"],
                        "amazing": ["remarkable", "extraordinary", "impressive"],
                        "great": ["significant", "substantial", "considerable"],
                        "nice": ["favorable", "commendable", "agreeable"],
                        "bad": ["unfavorable", "detrimental", "adverse"],
                        "good": ["beneficial", "favorable", "advantageous"],
                        "fun": ["enjoyable", "engaging", "stimulating"],
                        "sad": ["unfortunate", "regrettable", "disheartening"],
                        "mad": ["frustrated", "agitated", "displeased"],
                        "crazy": ["extraordinary", "unprecedented", "remarkable"],
                        "stuff": ["materials", "elements", "components"],
                        "thing": ["aspect", "element", "factor"],
                        "things": ["aspects", "elements", "factors"],
                        "a lot": ["considerably", "substantially", "significantly"],
                        "alot": ["considerably", "substantially", "significantly"],
                        "ok": ["acceptable", "adequate", "satisfactory"],
                        "okay": ["acceptable", "adequate", "satisfactory"],
                        "big": ["substantial", "significant", "considerable"],
                        "small": ["minimal", "limited", "negligible"],
                        "hard": ["challenging", "demanding", "arduous"],
                        "easy": ["straightforward", "manageable", "uncomplicated"],
                        "use": ["employ", "utilize", "apply"],
                        "used": ["employed", "utilized", "applied"],
                        "using": ["employing", "utilizing", "applying"],
                        "get": ["obtain", "acquire", "attain"],
                        "got": ["obtained", "acquired", "attained"],
                        "show": ["demonstrate", "illustrate", "indicate"],
                        "make": ["create", "produce", "generate"],
                        "do": ["perform", "execute", "conduct"],
                        "put": ["place", "position", "insert"],
                        "look": ["examine", "observe", "investigate"],
                        "looks": ["appears", "seems", "presents as"],
                        "think": ["believe", "contend", "assert"],
                        "thought": ["believed", "contended", "concluded"],
                        "know": ["understand", "recognize", "comprehend"],
                        "knew": ["understood", "recognized", "comprehended"],
                        "talk": ["discuss", "communicate", "deliberate"],
                        "talked": ["discussed", "communicated", "deliberated"],
                        "tell": ["inform", "advise", "communicate"],
                        "said": ["stated", "asserted", "indicated"],
                        "says": ["states", "asserts", "contends"],
                        "went": ["proceeded", "advanced", "progressed"],
                        "go": ["proceed", "advance", "progress"],
                        "come": ["arrive", "approach", "emerge"],
                        "came": ["arrived", "emerged", "resulted in"],
                        # Youth slang
                        "lit": ["exceptional", "outstanding", "remarkable"],
                        "dope": ["excellent", "superior", "distinguished"],
                        "fire": ["outstanding", "exceptional", "impressive"],
                        "sick": ["remarkable", "exceptional", "impressive"],
                        "lol": ["notably", "it is worth noting"],
                        "omg": ["remarkably", "notably"],
                        "btw": ["furthermore", "additionally", "incidentally"],
                        "tbh": ["candidly", "frankly", "in truth"],
                        "imo": ["in my assessment", "from my perspective"],
                        "fyi": ["for reference", "it should be noted that"],
                        "aka": ["also known as", "referred to as"],
                        "asap": ["promptly", "expeditiously", "at the earliest opportunity"],
                        # People / social relationships
                        "friends": ["companions", "peers", "associates", "colleagues"],
                        "friend": ["companion", "peer", "associate", "colleague"],
                        "buddy": ["companion", "associate", "colleague"],
                        "buddies": ["companions", "peers", "associates"],
                        "guys": ["individuals", "peers", "participants"],
                        "folks": ["individuals", "people", "members"],
                        "kids": ["students", "children", "individuals"],
                        # Food / informal nouns
                        "snacking": ["consuming", "eating", "partaking"],
                        "snack": ["refreshment", "light meal"],
                        "food": ["sustenance", "nourishment", "nutrition"],
                        # Informal verbs
                        "cracking": ["generating", "producing", "exchanging"],
                        "laughing": ["expressing amusement", "engaging jovially"],
                        "hanging": ["spending time", "socialising", "congregating"],
                        "pumped": ["motivated", "enthused", "energised"],
                        "binge": ["prolonged engagement", "extended session", "intensive viewing"],
                        "binging": ["extensively consuming", "immersive viewing"],
                        "overthink": ["over-analyse", "excessively deliberate"],
                        "overthinking": ["over-analysing", "excessive deliberation"],
                        "stress": ["anxiety", "pressure", "concern"],
                        "stressing": ["experiencing anxiety", "under pressure"],
                        # Time expressions
                        "tomorrow": ["the following day", "subsequently"],
                        # Discourse markers
                        "honestly": ["candidly", "frankly", "in truth"],
                        "basically": ["fundamentally", "essentially", "primarily"],
                        "literally": ["precisely", "in actuality", "indeed"],
                        "actually": ["in reality", "in fact", "indeed"],
                        "obviously": ["evidently", "clearly", "manifestly"],
                        "clearly": ["evidently", "demonstrably", "unambiguously"],
                    }

                    # Pure slang words that are force-scored as 100% informal
                    # (only actual slang/contractions — NOT neutral everyday words)
                    FORCE_SLANG_WORDS = {
                        "kinda", "sorta", "wanna", "gonna", "gotta", "hafta",
                        "lemme", "gimme", "lol", "omg", "btw", "tbh", "imo",
                        "fyi", "aka", "asap", "lit", "dope", "vibe", "vibes",
                        "chill", "chilling", "binge", "binging", "pumped",
                        "alot", "ok", "okay", "super", "totally",
                    }

                    # Words that should NEVER be flagged as informal —
                    # neutral/standard English words the MLP may misclassify
                    NEUTRAL_WHITELIST = {
                        "friends", "friend", "family", "school", "college",
                        "university", "teacher", "student", "day", "week",
                        "month", "year", "time", "life", "world", "place",
                        "home", "house", "people", "person", "man", "woman",
                        "child", "children", "morning", "evening", "night",
                        "water", "food", "work", "job", "city", "country",
                        "money", "health", "body", "mind", "heart", "hand",
                        "book", "word", "name", "number", "part", "side",
                        "point", "fact", "idea", "area", "case", "group",
                        "problem", "question", "answer", "result", "example",
                        "information", "research", "study", "data", "evidence",
                        "analysis", "process", "method", "system", "structure",
                        "level", "rate", "difference", "effect", "impact",
                        "role", "function", "value", "purpose", "reason",
                        "street", "food", "price", "moment", "evening",
                    }

                    scores = []
                    for idx in valid_indices:
                        w_lower = raw_words[idx].lower()
                        if w_lower in NEUTRAL_WHITELIST:
                            # Never flag neutral words as informal
                            prob_replace = 0.0
                        elif w_lower in FORCE_SLANG_WORDS:
                            prob_replace = 1.0  # Force maximum informal score for slang
                        else:
                            feats = extract_features_with_context_inf(raw_words, idx, word_freq, total_freq)
                            X_feat = np.array(feats).reshape(1, -1)
                            prob_replace = forward_inference(X_feat, weights, biases, gamma, beta, running_mean, running_var, activation, use_batchnorm)[0, 0]
                        scores.append((raw_words[idx], idx, prob_replace))
                        
                    # Sort scores to identify highest-impact replacements (higher prob = more informal/replaceable)
                    informal_scores = sorted([s for s in scores if s[2] >= threshold], key=lambda x: x[2], reverse=True)
                    
                    # Highlight words in the text
                    highlighted_html = ""
                    highlighted_dict = {s[1]: s[2] for s in scores}
                    
                    # Find highest-impact word index
                    highest_impact_idx = informal_scores[0][1] if informal_scores else -1
                    
                    for idx, word in enumerate(raw_words):
                        if idx in highlighted_dict:
                            prob = highlighted_dict[idx]
                            if idx == highest_impact_idx:
                                highlighted_html += f" <span class='highlight-impact' title='Highest Impact Scorer: {prob:.2f}'>{word}</span>"
                            elif prob >= threshold:
                                highlighted_html += f" <span class='highlight-informal' title='Informal Scorer: {prob:.2f}'>{word}</span>"
                            else:
                                highlighted_html += f" <span class='highlight-academic' title='Academic Scorer: {1.0 - prob:.2f}'>{word}</span>"
                        else:
                            highlighted_html += f" {word}"
                            
                    st.markdown("#### Analyzed Passage:")
                    st.markdown(f"<div style='background-color: #FAFAFA; padding: 20px; border-radius: 8px; border: 1px solid #CFD8DC; line-height: 1.8; font-size: 16px;'>{highlighted_html}</div>", unsafe_allow_html=True)
                    
                    # Display Legend
                    st.markdown("""
                    <div style="display: flex; gap: 20px; margin-top: 10px; font-size: 13px;">
                        <div><span class='highlight-academic'>Academic (Keep)</span></div>
                        <div><span class='highlight-informal'>Informal (Consider Replacing)</span></div>
                        <div><span class='highlight-impact'>Highest Impact Intervention</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")

                    # ── ALL INFORMAL WORDS REPLACEMENTS SECTION ──────────────────
                    if informal_scores:
                        st.markdown("### 🎯 Academic Upgrade — All Informal Words")
                        st.markdown(
                            "<p style='color:#616161; font-size:14px; margin-top:-10px;'>"
                            "Every informal word detected is listed below, ranked by impact (highest first). "
                            "Replace them all to maximise academic register.</p>",
                            unsafe_allow_html=True
                        )

                        # Helper: get best replacements for a word
                        def get_replacements_for(word, word_idx):
                            lw = word.lower()
                            lem = lemmatizer.lemmatize(lw)
                            reps = (
                                ACADEMIC_REPLACEMENTS.get(lw)
                                or ACADEMIC_REPLACEMENTS.get(lem)
                                or []
                            )
                            # WordNet fallback — with frequency filtering to exclude
                            # archaic/obscure synonyms (e.g. 'ilk', 'scat', 'clip', 'daytime')
                            wn_candidates = []
                            for syn in wordnet.synsets(word):
                                for lm in syn.lemmas():
                                    cand = lm.name().replace('_', ' ').lower()
                                    if cand == lw or not cand.isalpha() or ' ' in cand:
                                        continue
                                    # ── Frequency gate: discard rare/archaic words ──
                                    if _word_freq(cand) < _WF_THRESHOLD:
                                        continue
                                    # ── Accept only if MLP or academic wordlist confirms it ──
                                    if cand in model_data['academic_words']:
                                        wn_candidates.append((cand, _word_freq(cand)))
                                    else:
                                        tw = raw_words.copy()
                                        tw[word_idx] = cand
                                        f2 = extract_features_with_context_inf(tw, word_idx, word_freq, total_freq)
                                        p2 = forward_inference(np.array(f2).reshape(1, -1), weights, biases, gamma, beta, running_mean, running_var, activation, use_batchnorm)[0, 0]
                                        if p2 < threshold:
                                            wn_candidates.append((cand, _word_freq(cand)))
                            # Sort by descending frequency so most natural word comes first
                            wn_candidates.sort(key=lambda x: x[1], reverse=True)
                            wn_syns = [c for c, _ in wn_candidates]
                            return list(dict.fromkeys(reps + wn_syns))[:4]

                        # Build the best-replacement map for all informal words
                        best_replacement_map = {}  # idx -> best replacement word
                        all_rows_html = ""

                        for rank, (inf_word, inf_idx, inf_prob) in enumerate(informal_scores, 1):
                            reps = get_replacements_for(inf_word, inf_idx)
                            best_rep = reps[0] if reps else None
                            if best_rep:
                                best_replacement_map[inf_idx] = best_rep

                            # Score best replacement improvement
                            if best_rep:
                                tw = raw_words.copy()
                                tw[inf_idx] = best_rep
                                f2 = extract_features_with_context_inf(tw, inf_idx, word_freq, total_freq)
                                p2 = forward_inference(np.array(f2).reshape(1, -1), weights, biases, gamma, beta, running_mean, running_var, activation, use_batchnorm)[0, 0]
                                improvement = (inf_prob - p2) * 100
                            else:
                                improvement = 0.0

                            # Badge colour: gold for #1, else gradient green
                            rank_badge = (
                                "<span style='background:#FFD700;color:#3E2000;font-weight:700;"
                                "border-radius:50%;width:26px;height:26px;display:inline-flex;"
                                "align-items:center;justify-content:center;font-size:13px;'>★</span>"
                                if rank == 1 else
                                f"<span style='background:#E0E0E0;color:#424242;font-weight:700;"
                                f"border-radius:50%;width:26px;height:26px;display:inline-flex;"
                                f"align-items:center;justify-content:center;font-size:12px;'>{rank}</span>"
                            )

                            # Build alternatives chips
                            if reps:
                                chips = "".join(
                                    f"<span style='background:#E8F5E9;color:#2E7D32;border:1px solid #A5D6A7;"
                                    f"border-radius:20px;padding:4px 12px;margin:3px 4px 3px 0;"
                                    f"font-size:13px;font-weight:600;display:inline-block;'>{r}</span>"
                                    for r in reps
                                )
                            else:
                                chips = "<span style='color:#9E9E9E;font-size:13px;font-style:italic;'>No curated replacement — consider manual review</span>"

                            improvement_badge = (
                                f"<span style='color:#2E7D32;font-weight:700;'>+{improvement:.1f}%</span>"
                                if improvement > 0 else
                                "<span style='color:#9E9E9E;'>—</span>"
                            )

                            row_bg = "#FFFDE7" if rank == 1 else ("#FAFAFA" if rank % 2 == 0 else "#FFFFFF")
                            all_rows_html += f"""
                            <div style="background:{row_bg};border:1px solid #E0E0E0;border-radius:10px;
                                        padding:14px 18px;margin-bottom:10px;display:flex;
                                        align-items:flex-start;gap:14px;">
                                <div style="flex-shrink:0;margin-top:2px;">{rank_badge}</div>
                                <div style="flex:1;">
                                    <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
                                        <span style="background:#FFEBEE;color:#C62828;border:1px solid #FFCDD2;
                                                     border-radius:6px;padding:4px 14px;font-size:16px;
                                                     font-weight:800;">✗ {inf_word}</span>
                                        <span style="color:#9E9E9E;font-size:18px;">→</span>
                                        <span style="font-size:13px;color:#616161;">
                                            Informal score: <b>{inf_prob:.0%}</b> &nbsp;|&nbsp; 
                                            Improvement: {improvement_badge}
                                        </span>
                                    </div>
                                    <div>{chips}</div>
                                </div>
                            </div>
                            """

                        st.markdown(all_rows_html, unsafe_allow_html=True)

                        # ── FULLY OPTIMIZED SENTENCE ─────────────────────────────
                        st.markdown("---")
                        st.markdown("### ✅ Fully Optimized Passage")
                        st.markdown(
                            "<p style='color:#616161;font-size:14px;margin-top:-10px;'>"
                            "All informal words replaced with the best academic alternative.</p>",
                            unsafe_allow_html=True
                        )

                        optimized_words = raw_words.copy()
                        for inf_word, inf_idx, _ in informal_scores:
                            if inf_idx in best_replacement_map:
                                optimized_words[inf_idx] = (
                                    f"<span style='color:#1B5E20;font-weight:700;"
                                    f"background:#E8F5E9;border-radius:4px;padding:1px 5px;'>"
                                    f"{best_replacement_map[inf_idx]}</span>"
                                )

                        st.markdown(
                            f"<div style='background:#F9FBE7;border-left:5px solid #8BC34A;"
                            f"border-radius:8px;padding:18px 22px;font-size:16px;line-height:1.9;"
                            f"color:#212121;box-shadow:0 2px 8px rgba(0,0,0,0.06);'>"
                            f"{' '.join(optimized_words)}</div>",
                            unsafe_allow_html=True
                        )

                    else:
                        st.success("🎉 Excellent! No informal words detected. The passage is highly academic.")

    # ── TAB 2: ABOUT LEXICOACH ────────────────────────────────
    with tab2:
        st.markdown("### 📖 LexiCoach: Cognitive Load Theory in Vocabulary Optimization")
        st.markdown("""
        #### The Scientific Gap
        Grammar checkers and correction tools (such as Grammarly or Microsoft Editor) operate on the **flood model** — they highlight and suggest corrections for every single stylistic, structural, and vocabulary error simultaneously. 
        
        Educational psychology shows that displaying a massive list of corrections imposes a heavy **cognitive load** on learners, causing frustration, choice fatigue, and poor learning retention.
        
        #### The LexiCoach Paradigm
        LexiCoach treats writing improvement as a **ranked optimization problem**:
        1. It scores every word in a passage to identify potential informal vocabulary.
        2. Instead of surfacing all changes, it uses its custom-trained Multi-Layer Perceptron to select the **single highest-impact intervention**.
        3. Surfacing one highly-optimized substitution per feedback cycle allows deep concentration, leading to significantly higher learning retention.
        
        #### Dataset & Core Mathematics
        The underlying neural classifier was trained on a combined corpus of **14,806 essays** from ASAP and IELTS Kaggle datasets. 
        Features are computed for each word using purely linguistic parameters (syllables, prefix/suffix register, consonant clustering density, and context frequencies).
        
        No third-party neural engines (such as TensorFlow or PyTorch) are used at inference or training time. All gradients, weights initialization, backpropagation, and optimizations are calculated using pure **NumPy matrix operations**.
        """)
