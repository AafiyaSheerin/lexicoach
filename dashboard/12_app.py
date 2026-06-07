import os
import pickle
import re
import json
import hashlib
import requests
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag as nltk_pos_tag
from nltk.corpus import wordnet

# ── SUPABASE ──────────────────────────────────────────────────
from supabase import create_client, Client

def get_supabase() -> Client:
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def signup_user(email: str, password: str):
    try:
        sb = get_supabase()
        existing = sb.table("users").select("email").eq("email", email).execute()
        if existing.data:
            return False, "An account with this email already exists. Please sign in."
        sb.table("users").insert({
            "email": email,
            "password_hash": hash_password(password)
        }).execute()
        return True, "Account created successfully! You can now sign in."
    except Exception as e:
        return False, f"Signup failed: {str(e)}"

def login_user(email: str, password: str):
    try:
        sb = get_supabase()
        result = sb.table("users").select("email, password_hash").eq("email", email).execute()
        if not result.data:
            return False, "No account found with this email. Please sign up first."
        user = result.data[0]
        if user["password_hash"] != hash_password(password):
            return False, "Incorrect password. Please try again."
        return True, "Login successful!"
    except Exception as e:
        return False, f"Login failed: {str(e)}"

def reset_password(email: str):
    """Send password reset email via Gmail SMTP"""
    import smtplib
    import secrets
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    try:
        sb = get_supabase()
        # Check if user exists
        existing = sb.table("users").select("email, password_hash").eq("email", email).execute()
        if not existing.data:
            return False, "No account found with this email."
        if existing.data[0].get("password_hash") == "GOOGLE_AUTH":
            return False, "This account uses Google Sign-In. Please sign in with Google instead."

        # Generate a reset token and store it
        token = secrets.token_urlsafe(32)
        sb.table("users").update({"reset_token": token}).eq("email", email).execute()

        # Build reset link
        try:
            host = st.context.headers.get("host", "localhost:8501")
            if "localhost" in host:
                base_url = f"http://{host}"
            else:
                base_url = f"https://{host}"
        except Exception:
            base_url = "http://localhost:8501"

        reset_link = f"{base_url}/?reset_token={token}&email={email}"

        # Send email via Gmail SMTP
        sender = st.secrets["email"]["sender"]
        app_password = st.secrets["email"]["app_password"]

        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Reset your LexiCoach password"
        msg["From"] = sender
        msg["To"] = email

        html_body = f"""
        <div style="font-family: Arial, sans-serif; max-width: 500px; margin: auto; padding: 30px; border: 1px solid #E0E0E0; border-radius: 8px;">
            <h2 style="color: #4A148C;">🎓 LexiCoach Password Reset</h2>
            <p>Hi there,</p>
            <p>You requested a password reset for your LexiCoach account (<b>{email}</b>).</p>
            <p>Click the button below to reset your password:</p>
            <a href="{reset_link}" style="display:inline-block;background:#1976D2;color:#FFFFFF;padding:12px 24px;border-radius:4px;text-decoration:none;font-weight:600;margin:16px 0;">Reset Password</a>
            <p style="color:#757575;font-size:12px;">If you didn't request this, ignore this email. The link expires in 1 hour.</p>
            <hr style="border:0;border-top:1px solid #E0E0E0;margin:20px 0;"/>
            <p style="color:#757575;font-size:12px;">LexiCoach — AI-powered Writing Improvement System</p>
        </div>
        """

        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, app_password)
            server.sendmail(sender, email, msg.as_string())

        return True, "Password reset email sent! Check your inbox."
    except Exception as e:
        return False, f"Failed to send reset email: {str(e)}"

def update_password(email: str, token: str, new_password: str):
    """Update password if token matches"""
    try:
        sb = get_supabase()
        result = sb.table("users").select("email, reset_token").eq("email", email).execute()
        if not result.data:
            return False, "Invalid reset link."
        if result.data[0].get("reset_token") != token:
            return False, "Invalid or expired reset token."
        # Update password and clear token
        sb.table("users").update({
            "password_hash": hash_password(new_password),
            "reset_token": None
        }).eq("email", email).execute()
        return True, "Password updated successfully! You can now sign in."
    except Exception as e:
        return False, f"Failed to update password: {str(e)}"

# ── WORDFREQ ──────────────────────────────────────────────────
try:
    from wordfreq import word_frequency as _wf
    def _word_freq(w): return _wf(w, 'en')
except ImportError:
    def _word_freq(w): return 1.0

_WF_THRESHOLD = 1e-5

# ── NLTK ──────────────────────────────────────────────────────
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

# ── PAGE CONFIG (must be first Streamlit call) ─────────────────
st.set_page_config(
    page_title="LexiCoach — Writing Improvement System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── MODULE-LEVEL CONSTANTS ─────────────────────────────────────
ACADEMIC_REPLACEMENTS = {
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
    "wanna": ["wish to", "desire to", "intend to"],
    "gonna": ["going to", "preparing to", "intending to"],
    "gotta": ["must", "need to", "am required to"],
    "hafta": ["must", "am obligated to", "need to"],
    "lemme": ["allow me to", "permit me to"],
    "gimme": ["provide me with", "supply"],
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
    "friends": ["companions", "peers", "associates", "colleagues"],
    "friend": ["companion", "peer", "associate", "colleague"],
    "buddy": ["companion", "associate", "colleague"],
    "buddies": ["companions", "peers", "associates"],
    "guys": ["individuals", "peers", "participants"],
    "folks": ["individuals", "people", "members"],
    "kids": ["students", "children", "individuals"],
    "snacking": ["consuming", "eating", "partaking"],
    "snack": ["refreshment", "light meal"],
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
    "tomorrow": ["the following day", "subsequently"],
    "honestly": ["candidly", "frankly", "in truth"],
    "basically": ["fundamentally", "essentially", "primarily"],
    "literally": ["precisely", "in actuality", "indeed"],
    "actually": ["in reality", "in fact", "indeed"],
    "obviously": ["evidently", "clearly", "manifestly"],
    "clearly": ["evidently", "demonstrably", "unambiguously"],
}

FORCE_SLANG_WORDS = {
    "kinda", "sorta", "wanna", "gonna", "gotta", "hafta",
    "lemme", "gimme", "lol", "omg", "btw", "tbh", "imo",
    "fyi", "aka", "asap", "lit", "dope", "vibe", "vibes",
    "chill", "chilling", "binge", "binging", "pumped",
    "alot", "ok", "okay", "super", "totally",
}

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

# ── QUERY PARAMS ──────────────────────────────────────────────
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

def get_param(params, key, default=""):
    val = params.get(key, default)
    if isinstance(val, list):
        return val[0] if val else default
    return val if val is not None else default

# ── OAUTH CONFIG ──────────────────────────────────────────────
def load_oauth_config():
    try:
        cid = st.secrets["google_oauth"]["client_id"]
        csec = st.secrets["google_oauth"]["client_secret"]
        if cid and csec:
            return {"client_id": cid, "client_secret": csec}
    except Exception:
        pass
    try:
        if os.path.exists("oauth_config.json"):
            with open("oauth_config.json", "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

oauth_config = load_oauth_config()
client_id = oauth_config.get("client_id")
client_secret = oauth_config.get("client_secret")

params = get_query_params()

def get_redirect_uri():
    try:
        host = st.context.headers.get("host", "localhost:8501")
        if "localhost" in host:
            return f"http://{host}/"
        else:
            return f"https://{host}/"
    except Exception:
        return "http://localhost:8501/"

redirect_uri = get_redirect_uri()

# ── GOOGLE OAUTH CALLBACK ─────────────────────────────────────
if get_param(params, "code"):
    auth_code = get_param(params, "code")
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
                user_info_resp = requests.get(f"https://oauth2.googleapis.com/tokeninfo?id_token={id_token}")
                if user_info_resp.status_code == 200:
                    user_info = user_info_resp.json()
                    email = user_info.get("email")
                    try:
                        sb = get_supabase()
                        existing = sb.table("users").select("email").eq("email", email).execute()
                        if not existing.data:
                            sb.table("users").insert({"email": email, "password_hash": "GOOGLE_AUTH"}).execute()
                    except Exception:
                        pass
                    st.session_state['logged_in'] = True
                    st.session_state['user_email'] = email
                    st.session_state['user_name'] = user_info.get("name", "Google User")
                    set_query_params({})
                    st.rerun()
                else:
                    st.error("Failed to fetch user info from Google.")
            else:
                st.error("Failed to exchange auth code with Google.")
        except Exception as e:
            st.error(f"OAuth 2.0 Error: {str(e)}")

# ── MOCK GOOGLE CALLBACK ──────────────────────────────────────
if get_param(params, "logged_in") == "true" and get_param(params, "email"):
    st.session_state['logged_in'] = True
    st.session_state['user_email'] = get_param(params, "email")
    set_query_params({})
    st.rerun()

# ── MOCK GOOGLE PAGE ──────────────────────────────────────────
if get_param(params, "page") == "mock_google_auth":
    custom_email = get_param(params, "email", "aafeeya.sheerin@gmail.com")
    name_parts = custom_email.split('@')[0].replace('.', ' ').title()
    first_char = name_parts[0] if name_parts else 'G'
    st.markdown(f"""<style>
[data-testid="stSidebar"], [data-testid="stHeader"] {{ display: none !important; }}
[data-testid="stAppViewBlockContainer"] {{ padding: 0 !important; max-width: 100% !important; background-color: #FFFFFF !important; }}
.google-login-container {{ display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; background-color: #FFFFFF; font-family: 'Roboto', 'Arial', sans-serif; }}
.google-card {{ width: 450px; background: #FFFFFF; border: 1px solid #DADCE0; border-radius: 8px; padding: 40px; box-sizing: border-box; }}
.account-row {{ display: flex; align-items: center; padding: 14px 0; border-bottom: 1px solid #DADCE0; cursor: pointer; }}
.account-row:hover {{ background-color: #F8F9FA; }}
.avatar {{ width: 32px; height: 32px; border-radius: 50%; color: #FFFFFF; display: flex; align-items: center; justify-content: center; font-weight: 500; font-size: 15px; margin-right: 12px; }}
</style>
<div class="google-login-container">
    <div class="google-card">
        <h1 style="font-size:24px;font-weight:400;text-align:center;color:#202124;">Choose an account</h1>
        <p style="text-align:center;color:#202124;">to continue to <b style="color:#4A148C;">LexiCoach</b></p>
        <a href="?logged_in=true&email={custom_email}" target="_top" style="text-decoration:none;">
            <div class="account-row">
                <div class="avatar" style="background-color:#1A73E8;">{first_char}</div>
                <div>
                    <div style="font-size:14px;font-weight:500;color:#3C4043;">{name_parts}</div>
                    <div style="font-size:12px;color:#5F6368;">{custom_email}</div>
                </div>
            </div>
        </a>
    </div>
</div>""", unsafe_allow_html=True)
    st.stop()

# ── PASSWORD RESET PAGE ───────────────────────────────────────
if get_param(params, "reset_token") and get_param(params, "email"):
    reset_token = get_param(params, "reset_token")
    reset_email = get_param(params, "email")

    st.markdown("<br><br>", unsafe_allow_html=True)
    _, col_mid, _ = st.columns([1.2, 2, 1.2])
    with col_mid:
        with st.container(border=True):
            st.markdown("""
            <div style="text-align:center;margin-bottom:20px;">
                <h2 style="margin:0;font-size:22px;font-weight:700;color:#000000;">🔐 Reset Your Password</h2>
            </div>""", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;color:#616161;'>Resetting password for <b>{reset_email}</b></p>", unsafe_allow_html=True)
            new_pwd = st.text_input("New Password", type="password", placeholder="Min 6 characters", key="new_pwd")
            confirm_pwd = st.text_input("Confirm New Password", type="password", placeholder="Re-enter password", key="confirm_pwd")
            if st.button("Update Password", type="primary", use_container_width=True):
                if len(new_pwd) < 6:
                    st.error("Password must be at least 6 characters.")
                elif new_pwd != confirm_pwd:
                    st.error("Passwords do not match.")
                else:
                    with st.spinner("Updating password..."):
                        success, msg = update_password(reset_email, reset_token, new_pwd)
                    if success:
                        st.success(msg)
                        set_query_params({})
                        st.rerun()
                    else:
                        st.error(msg)
    st.stop()


st.markdown("""
<style>
    body, [data-testid="stAppViewContainer"] {
        background-color: #FFFFFF;
        background-image: linear-gradient(135deg, #F3E5F5 0%, #E3F2FD 100%);
        color: #000000 !important;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    .stTextArea textarea {
        color: #000000 !important;
        background-color: #FFFFFF !important;
        border: 1px solid #B0BEC5 !important;
        border-radius: 8px !important;
    }
    h1, h2, h3, h4, h5, h6, p, span, label, li { color: #000000 !important; }
    [data-testid="stSidebar"] {
        background-color: #F8F9FA !important;
        border-right: 1px solid #E1BEE7 !important;
    }
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
    div[data-testid="stTextInput"] { position: relative; padding-top: 10px !important; margin-bottom: 20px !important; }
    div[data-testid="stTextInput"] label {
        position: absolute; top: 0px !important; left: 10px !important;
        background-color: #FFFFFF !important; padding: 0 4px !important;
        font-size: 13px !important; color: #757575 !important; z-index: 99 !important; font-weight: 500 !important;
    }
    div[data-testid="stTextInput"] input {
        border: 1px solid #CFD8DC !important; border-radius: 4px !important;
        padding: 10px 12px !important; background-color: #FFFFFF !important;
        color: #000000 !important; height: 48px !important; font-size: 15px !important;
    }
    button[data-testid="stBaseButton-secondary"] {
        background-color: #FFFFFF !important; color: #1976D2 !important;
        border: 1px solid #1976D2 !important; font-weight: 600 !important;
        border-radius: 4px !important; width: 100% !important; height: 40px !important;
    }
    button[data-testid="stBaseButton-primary"] {
        background-color: #1976D2 !important; color: #FFFFFF !important;
        border: 1px solid #1976D2 !important; font-weight: 600 !important;
        border-radius: 4px !important; width: 100% !important; height: 40px !important;
    }
    .highlight-informal {
        background-color: #FFEBEE; border: 1.5px solid #EF5350;
        border-radius: 4px; padding: 2px 6px; font-weight: 600; color: #C62828 !important;
    }
    .highlight-academic {
        background-color: #E8F5E9; border: 1.5px solid #66BB6A;
        border-radius: 4px; padding: 2px 6px; font-weight: 600; color: #2E7D32 !important;
    }
    .highlight-impact {
        background-color: #FFF8E1; border: 2px dashed #FFCA28;
        border-radius: 4px; padding: 3px 8px; font-weight: 700; color: #F57F17 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ────────────────────────────────────────────────
@st.cache_resource
def load_lexicoach_model():
    model_path = os.path.join(os.path.dirname(__file__), "..", "model.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

model_data = load_lexicoach_model()
lemmatizer = WordNetLemmatizer()

# ── INFERENCE HELPERS ─────────────────────────────────────────
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

LATIN_PREFIXES_INF = ("inter","intra","trans","pre","post","sub","super","extra","ultra","semi","anti","counter","non","over","under","out","dis","mis","re","un","co","com","con","col","cor","pro","per","para","meta","poly","mono","micro","macro","hyper","hypo","auto","bio","geo","neo","pseudo","quasi","multi","omni","uni","bi","tri","quad","penta","hexa","deca","cent","kilo","mega","giga","tele","photo","thermo","electro","hydro","aero","chrono","demo","ethn","gen")
FORMAL_SUFFIXES_INF = ('tion','sion','ity','ance','ence','ment','ness','ism','ive','ous','ize','ise','ate','ify','fy','ology','ography','ometry')
TOP500_COMMON_INF = set(["time","year","people","way","day","man","woman","child","world","life","hand","part","place","case","week","company","system","program","question","work","government","number","night","point","home","water","room","mother","area","money","story","fact","month","lot","right","study","book","eye","job","word","business","issue","side","kind","head","house","service","friend","father","power","hour","game","line","end","among","off","always","state","once","hear","body","drive","took","play","move","live","believe","hold","bring","happen","write","provide","sit","stand","lose","pay","meet","include","continue","set","learn","change","lead","understand","watch","follow","stop","create","speak","read","spend","grow","open","walk","win","offer","remember","love","consider","appear","buy","wait","serve","die","send","expect","build","stay","fall","cut","reach","kill","remain","suggest","raise","pass","sell","require","report","decide","pull","break","wish","pick","carry","explain","face","approach","allow","become","begin","call","come","feel","gave","give","gone","got","keep","knew","know","let","long","look","made","make","need","next","nothing","often","old","only","other","over","own","place","put","said","same","saw","says","see","seen","show","small","still","take","taken","tell","thing","think","thought","told","took","turn","under","upon","used","want","well","went","were","when","where","while","whose","within","without","year","years","young"])

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
        W = weights[l]; b = biases[l]
        z = a @ W + b
        if use_batchnorm:
            z_hat = (z - running_mean[l]) / np.sqrt(running_var[l] + 1e-5)
            pre_activation = gamma[l] * z_hat + beta[l]
        else:
            pre_activation = z
        if activation == 'relu':
            a = np.maximum(0, pre_activation)
        else:
            a = 1.0 / (1.0 + np.exp(-np.clip(pre_activation, -500, 500)))
    z = a @ weights[-1] + biases[-1]
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

# ── SESSION STATE ─────────────────────────────────────────────
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'user_email' not in st.session_state:
    st.session_state['user_email'] = ""
if 'show_signup' not in st.session_state:
    st.session_state['show_signup'] = False
if 'show_forgot' not in st.session_state:
    st.session_state['show_forgot'] = False

# ── LOGIN / SIGNUP VIEW ───────────────────────────────────────
if not st.session_state['logged_in']:
    if client_id and client_secret:
        google_auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}&scope=openid%20email%20profile&state=google_oauth"
    else:
        google_auth_url = "?page=mock_google_auth&email=aafeeya.sheerin@gmail.com"

    st.markdown("<br><br>", unsafe_allow_html=True)
    _, col_mid, _ = st.columns([1.2, 2, 1.2])

    with col_mid:
        with st.container(border=True):
            if not st.session_state['show_signup']:
                # ── SIGN IN PAGE ──
                st.markdown("""
                <div style="text-align:center;margin-bottom:20px;">
                    <h2 style="margin:0;font-size:22px;font-weight:700;color:#000000;">Sign in to LexiCoach</h2>
                </div>""", unsafe_allow_html=True)

                # Google Sign In button
                st.link_button("🔵  Sign in with Google", url=google_auth_url, use_container_width=True)

                st.markdown("<div style='text-align:center;margin:15px 0 25px 0;font-size:13px;color:#757575;border-bottom:1px solid #EEEEEE;line-height:0.1em;'><span style='background:#fff;padding:0 10px;'>Or continue with LexiCoach Account</span></div>", unsafe_allow_html=True)

                email_input = st.text_input("Email", placeholder="Enter your email", key="login_email_box")
                password_input = st.text_input("Password", type="password", placeholder="Enter your password", key="login_pwd_box")

                col_fp, _ = st.columns([1, 1])
                with col_fp:
                    if st.button("Forgot your password?", key="forgot_btn", help="Send a reset link to your email"):
                        st.session_state['show_forgot'] = not st.session_state.get('show_forgot', False)

                if st.session_state.get('show_forgot', False):
                    with st.container():
                        st.markdown("<div style='background:#E3F2FD;border-radius:8px;padding:12px;margin-bottom:10px;'>", unsafe_allow_html=True)
                        reset_email = st.text_input("Enter your email to reset password", key="reset_email_input", placeholder="your@email.com")
                        if st.button("Send Reset Link", type="primary", key="send_reset_btn"):
                            if reset_email and "@" in reset_email:
                                with st.spinner("Sending reset email..."):
                                    success, msg = reset_password(reset_email)
                                if success:
                                    st.success(msg)
                                    st.session_state['show_forgot'] = False
                                else:
                                    st.error(msg)
                            else:
                                st.error("Please enter a valid email address.")
                        st.markdown("</div>", unsafe_allow_html=True)

                col_can, col_si = st.columns(2)
                with col_can:
                    if st.button("Cancel", use_container_width=True, key="cancel_btn"):
                        st.session_state['login_email_box'] = ""
                        st.session_state['login_pwd_box'] = ""
                with col_si:
                    if st.button("Sign In", type="primary", use_container_width=True, key="signin_btn"):
                        if email_input and len(password_input) >= 6:
                            success, msg = login_user(email_input, password_input)
                            if success:
                                st.session_state['user_email'] = email_input
                                st.session_state['logged_in'] = True
                                st.rerun()
                            else:
                                st.error(msg)
                        else:
                            st.error("Please enter a valid email and password (min 6 characters).")

                st.markdown("<div style='text-align:center;font-size:14px;margin-top:20px;color:#000000;'>Need a LexiCoach account?</div>", unsafe_allow_html=True)
                if st.button("✏️ Sign Up", use_container_width=True, key="goto_signup_btn"):
                    st.session_state['show_signup'] = True
                    st.rerun()

            else:
                # ── SIGN UP PAGE ──
                st.markdown("""
                <div style="text-align:center;margin-bottom:20px;">
                    <h2 style="margin:0;font-size:22px;font-weight:700;color:#000000;">Create LexiCoach Account</h2>
                </div>""", unsafe_allow_html=True)

                su_email = st.text_input("Email", placeholder="Enter your email", key="signup_email")
                su_password = st.text_input("Password", type="password", placeholder="Min 6 characters", key="signup_pwd")
                su_confirm = st.text_input("Confirm Password", type="password", placeholder="Re-enter your password", key="signup_confirm")

                col_back, col_create = st.columns(2)
                with col_back:
                    if st.button("← Back", use_container_width=True, key="back_to_signin"):
                        st.session_state['show_signup'] = False
                        st.rerun()
                with col_create:
                    if st.button("Create Account", type="primary", use_container_width=True, key="create_account_btn"):
                        if not su_email or "@" not in su_email:
                            st.error("Please enter a valid email address.")
                        elif len(su_password) < 6:
                            st.error("Password must be at least 6 characters.")
                        elif su_password != su_confirm:
                            st.error("Passwords do not match.")
                        else:
                            with st.spinner("Creating your account..."):
                                success, msg = signup_user(su_email, su_password)
                            if success:
                                st.success(msg)
                                st.session_state['show_signup'] = False
                                st.rerun()
                            else:
                                st.error(msg)

                st.markdown("<div style='text-align:center;margin:15px 0;font-size:13px;color:#757575;'>— or —</div>", unsafe_allow_html=True)
                st.link_button("🔵  Sign up with Google instead", url=google_auth_url, use_container_width=True)

        # Footer
        st.markdown("""
        <div style="margin-top:30px;text-align:center;width:100%;">
            <hr style="border:0;border-top:1px solid #E0E0E0;margin:20px 0;width:100%;"/>
            <div style="display:flex;justify-content:space-around;max-width:480px;margin:auto;padding:10px 0;">
                <div style="text-align:left;">
                    <div style="font-size:12px;color:#757575;font-weight:600;margin-bottom:8px;">Download App</div>
                    <a href="https://play.google.com/store" target="_blank" style="text-decoration:none;">
                        <div style="background-color:#000000;border-radius:4px;padding:6px 12px;display:flex;align-items:center;gap:8px;border:1px solid #333;">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="#FFFFFF"><path d="M5 3.25c-.27 0-.52.11-.7.31L12.5 12l-8.2 8.44c.18.2.43.31.7.31.22 0 .43-.07.61-.2l12.44-7.25c.57-.33.95-.97.95-1.7 0-.73-.38-1.37-.95-1.7L6.61 3.45c-.18-.13-.39-.2-.61-.2z"/></svg>
                            <div style="text-align:left;line-height:1;">
                                <span style="font-size:8px;color:#AAAAAA;display:block;">GET IT ON</span>
                                <span style="font-size:12px;color:#FFFFFF;font-weight:700;">Google Play</span>
                            </div>
                        </div>
                    </a>
                </div>
                <div style="text-align:left;">
                    <div style="font-size:12px;color:#757575;font-weight:600;margin-bottom:8px;">Follow Us</div>
                    <div style="display:flex;gap:12px;">
                        <a href="https://facebook.com" target="_blank" style="text-decoration:none;">
                            <div style="width:32px;height:32px;border-radius:50%;background-color:#1877F2;display:flex;align-items:center;justify-content:center;">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="#FFFFFF"><path d="M22 12c0-5.52-4.48-10-10-10S2 6.48 2 12c0 4.84 3.44 8.87 8 9.8V15H8v-3h2V9.5C10 7.57 11.57 6 13.5 6H16v3h-2c-.55 0-1 .45-1 1v2h3v3h-3v6.8c4.56-.93 8-4.96 8-9.8z"/></svg>
                            </div>
                        </a>
                        <a href="https://instagram.com" target="_blank" style="text-decoration:none;">
                            <div style="width:32px;height:32px;border-radius:50%;background:radial-gradient(circle at 30% 107%,#fdf497 0%,#fdf497 5%,#fd5949 45%,#d6249f 60%,#285AEB 90%);display:flex;align-items:center;justify-content:center;">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="#FFFFFF"><path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zM12 0C8.741 0 8.333.014 7.053.072 2.695.272.273 2.69.073 7.051C.014 8.333 0 8.741 0 12c0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98 1.281.058 1.689.072 4.948.072 3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98C15.668.014 15.259 0 12 0zm0 5.838a6.162 6.162 0 1 0 0 12.324 6.162 6.162 0 0 0 0-12.324zM12 16a4 4 0 1 1 0-8 4 4 0 0 1 0 8zm6.406-11.845a1.44 1.44 0 1 0 0 2.881 1.44 1.44 0 0 0 0-2.881z"/></svg>
                            </div>
                        </a>
                        <a href="https://twitter.com" target="_blank" style="text-decoration:none;">
                            <div style="width:32px;height:32px;border-radius:50%;background-color:#000000;display:flex;align-items:center;justify-content:center;">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="#FFFFFF"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>
                            </div>
                        </a>
                    </div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

# ── LOGGED IN MAIN APP ────────────────────────────────────────
else:
    st.sidebar.markdown(f"""
    <div style="text-align:center;padding:15px 0;">
        <span style="font-size:45px;">🎓</span>
        <h3 style="margin:10px 0 5px 0;color:#4A148C;font-weight:700;">LexiCoach</h3>
        <p style="margin:0;font-size:12px;color:#616161;">Active User: <b>{st.session_state['user_email']}</b></p>
    </div>""", unsafe_allow_html=True)

    st.sidebar.markdown("---")

    if st.sidebar.button("Logout", type="primary", use_container_width=True):
        st.session_state['logged_in'] = False
        st.session_state['user_email'] = ""
        st.rerun()

    st.sidebar.markdown("""
    <div style="margin-top:30px;font-size:12px;color:#757575;line-height:1.4;">
        <b>Theoretical Grounding:</b><br>
        Surfacing every correction causes high cognitive load. Surfacing a <i>single</i> high-impact substitution helps learners focus and retain better academic vocabulary choices.
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color:#FFFFFF;border:1px solid #E1BEE7;border-radius:10px;padding:20px;margin-bottom:25px;box-shadow:0 4px 15px rgba(106,27,154,0.04);">
        <h1 style="margin:0;color:#4A148C;font-size:28px;font-weight:700;">LexiCoach Writing Scorer & Essay Optimizer</h1>
        <p style="margin:5px 0 0 0;color:#616161;font-size:14px;">Surfacing single highest-impact academic substitutions powered by a custom-trained MLP neural network.</p>
    </div>""", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["✍️ Essay Optimizer", "📖 About LexiCoach"])

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
                try:
                    word_freq = model_data.get('word_freq', {})
                    total_freq = model_data.get('total_freq', 1)
                    weights = model_data.get('weights')
                    biases = model_data.get('biases')
                    gamma = model_data.get('gamma')
                    beta = model_data.get('beta')
                    running_mean = model_data.get('running_mean')
                    running_var = model_data.get('running_var')
                    activation = model_data.get('activation', 'relu')
                    use_batchnorm = model_data.get('use_batchnorm', False)
                    threshold = model_data.get('threshold', 0.5)
                    academic_words = model_data.get('academic_words', set())

                    if weights is None or biases is None:
                        st.error("Model file is missing required keys. Please retrain the model.")
                        st.stop()
                except Exception as e:
                    st.error(f"Failed to load model data: {e}")
                    st.stop()

                cleaned_text = clean_text_inf(essay_input)
                raw_words = cleaned_text.split()

                valid_words = []
                valid_indices = []
                for idx, w in enumerate(raw_words):
                    if is_valid_word_inf(w):
                        valid_words.append(w)
                        valid_indices.append(idx)

                if not valid_words:
                    st.warning("No valid vocabulary words found in the passage (excluding short words and stopwords).")
                else:
                    with st.spinner("Analyzing passage and scoring vocabulary..."):
                        scores = []
                        for idx in valid_indices:
                            w_lower = raw_words[idx].lower()
                            if w_lower in NEUTRAL_WHITELIST:
                                prob_replace = 0.0
                            elif w_lower in FORCE_SLANG_WORDS:
                                prob_replace = 1.0
                            else:
                                feats = extract_features_with_context_inf(raw_words, idx, word_freq, total_freq)
                                X_feat = np.array(feats).reshape(1, -1)
                                prob_replace = forward_inference(X_feat, weights, biases, gamma, beta, running_mean, running_var, activation, use_batchnorm)[0, 0]
                            scores.append((raw_words[idx], idx, prob_replace))

                    informal_scores = sorted([s for s in scores if s[2] >= threshold], key=lambda x: x[2], reverse=True)

                    highlighted_html = ""
                    highlighted_dict = {s[1]: s[2] for s in scores}
                    highest_impact_idx = informal_scores[0][1] if informal_scores else -1

                    for idx, word in enumerate(raw_words):
                        if idx in highlighted_dict:
                            prob = highlighted_dict[idx]
                            if idx == highest_impact_idx:
                                highlighted_html += f" <span class='highlight-impact' title='Highest Impact: {prob:.2f}'>{word}</span>"
                            elif prob >= threshold:
                                highlighted_html += f" <span class='highlight-informal' title='Informal: {prob:.2f}'>{word}</span>"
                            else:
                                highlighted_html += f" <span class='highlight-academic' title='Academic: {1.0-prob:.2f}'>{word}</span>"
                        else:
                            highlighted_html += f" {word}"

                    st.markdown("#### Analyzed Passage:")
                    st.markdown(f"<div style='background-color:#FAFAFA;padding:20px;border-radius:8px;border:1px solid #CFD8DC;line-height:1.8;font-size:16px;'>{highlighted_html}</div>", unsafe_allow_html=True)
                    st.markdown("""
                    <div style="display:flex;gap:20px;margin-top:10px;font-size:13px;">
                        <div><span class='highlight-academic'>Academic (Keep)</span></div>
                        <div><span class='highlight-informal'>Informal (Consider Replacing)</span></div>
                        <div><span class='highlight-impact'>Highest Impact Intervention</span></div>
                    </div>""", unsafe_allow_html=True)
                    st.markdown("---")

                    if informal_scores:
                        st.markdown("### 🎯 Academic Upgrade — All Informal Words")
                        st.markdown("<p style='color:#616161;font-size:14px;margin-top:-10px;'>Every informal word detected is listed below, ranked by impact (highest first).</p>", unsafe_allow_html=True)

                        def get_replacements_for(word, word_idx):
                            lw = word.lower()
                            lem = lemmatizer.lemmatize(lw)
                            reps = ACADEMIC_REPLACEMENTS.get(lw) or ACADEMIC_REPLACEMENTS.get(lem) or []
                            wn_candidates = []
                            for syn in wordnet.synsets(word):
                                for lm in syn.lemmas():
                                    cand = lm.name().replace('_', ' ').lower()
                                    if cand == lw or not cand.isalpha() or ' ' in cand: continue
                                    if _word_freq(cand) < _WF_THRESHOLD: continue
                                    if cand in academic_words:
                                        wn_candidates.append((cand, _word_freq(cand)))
                                    else:
                                        tw = raw_words.copy(); tw[word_idx] = cand
                                        f2 = extract_features_with_context_inf(tw, word_idx, word_freq, total_freq)
                                        p2 = forward_inference(np.array(f2).reshape(1,-1), weights, biases, gamma, beta, running_mean, running_var, activation, use_batchnorm)[0,0]
                                        if p2 < threshold:
                                            wn_candidates.append((cand, _word_freq(cand)))
                            wn_candidates.sort(key=lambda x: x[1], reverse=True)
                            return list(dict.fromkeys(reps + [c for c,_ in wn_candidates]))[:4]

                        best_replacement_map = {}
                        all_rows_html = ""

                        for rank, (inf_word, inf_idx, inf_prob) in enumerate(informal_scores, 1):
                            reps = get_replacements_for(inf_word, inf_idx)
                            best_rep = reps[0] if reps else None
                            if best_rep: best_replacement_map[inf_idx] = best_rep

                            if best_rep:
                                tw = raw_words.copy(); tw[inf_idx] = best_rep
                                f2 = extract_features_with_context_inf(tw, inf_idx, word_freq, total_freq)
                                p2 = forward_inference(np.array(f2).reshape(1,-1), weights, biases, gamma, beta, running_mean, running_var, activation, use_batchnorm)[0,0]
                                improvement = (inf_prob - p2) * 100
                            else:
                                improvement = 0.0

                            rank_badge = (
                                "<span style='background:#FFD700;color:#3E2000;font-weight:700;border-radius:50%;width:26px;height:26px;display:inline-flex;align-items:center;justify-content:center;font-size:13px;'>★</span>"
                                if rank == 1 else
                                f"<span style='background:#E0E0E0;color:#424242;font-weight:700;border-radius:50%;width:26px;height:26px;display:inline-flex;align-items:center;justify-content:center;font-size:12px;'>{rank}</span>"
                            )
                            chips = "".join(f"<span style='background:#E8F5E9;color:#2E7D32;border:1px solid #A5D6A7;border-radius:20px;padding:4px 12px;margin:3px 4px 3px 0;font-size:13px;font-weight:600;display:inline-block;'>{r}</span>" for r in reps) if reps else "<span style='color:#9E9E9E;font-size:13px;font-style:italic;'>No curated replacement — consider manual review</span>"
                            improvement_badge = f"<span style='color:#2E7D32;font-weight:700;'>+{improvement:.1f}%</span>" if improvement > 0 else "<span style='color:#9E9E9E;'>—</span>"
                            row_bg = "#FFFDE7" if rank == 1 else ("#FAFAFA" if rank % 2 == 0 else "#FFFFFF")
                            all_rows_html += f"""
                            <div style="background:{row_bg};border:1px solid #E0E0E0;border-radius:10px;padding:14px 18px;margin-bottom:10px;display:flex;align-items:flex-start;gap:14px;">
                                <div style="flex-shrink:0;margin-top:2px;">{rank_badge}</div>
                                <div style="flex:1;">
                                    <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
                                        <span style="background:#FFEBEE;color:#C62828;border:1px solid #FFCDD2;border-radius:6px;padding:4px 14px;font-size:16px;font-weight:800;">✗ {inf_word}</span>
                                        <span style="color:#9E9E9E;font-size:18px;">→</span>
                                        <span style="font-size:13px;color:#616161;">Informal score: <b>{inf_prob:.0%}</b> &nbsp;|&nbsp; Improvement: {improvement_badge}</span>
                                    </div>
                                    <div>{chips}</div>
                                </div>
                            </div>"""

                        st.markdown(all_rows_html, unsafe_allow_html=True)
                        st.markdown("---")
                        st.markdown("### ✅ Fully Optimized Passage")
                        st.markdown("<p style='color:#616161;font-size:14px;margin-top:-10px;'>All informal words replaced with the best academic alternative.</p>", unsafe_allow_html=True)

                        optimized_words = raw_words.copy()
                        for _, inf_idx, _ in informal_scores:
                            if inf_idx in best_replacement_map:
                                optimized_words[inf_idx] = f"<span style='color:#1B5E20;font-weight:700;background:#E8F5E9;border-radius:4px;padding:1px 5px;'>{best_replacement_map[inf_idx]}</span>"

                        st.markdown(f"<div style='background:#F9FBE7;border-left:5px solid #8BC34A;border-radius:8px;padding:18px 22px;font-size:16px;line-height:1.9;color:#212121;box-shadow:0 2px 8px rgba(0,0,0,0.06);'>{' '.join(optimized_words)}</div>", unsafe_allow_html=True)
                    else:
                        st.success("🎉 Excellent! No informal words detected. The passage is highly academic.")

    with tab2:
        st.markdown("### 📖 LexiCoach: Cognitive Load Theory in Vocabulary Optimization")
        st.markdown("""
        #### The Scientific Gap
        Grammar checkers and correction tools operate on the **flood model** — highlighting every error simultaneously, imposing heavy **cognitive load** on learners.

        #### The LexiCoach Paradigm
        LexiCoach treats writing improvement as a **ranked optimization problem**:
        1. Scores every word to identify informal vocabulary.
        2. Uses a custom-trained MLP to select the **single highest-impact intervention**.
        3. Focused substitution enables deeper learning retention.

        #### Dataset & Core Mathematics
        Trained on **14,806 essays** from ASAP and IELTS datasets. All gradients, backpropagation, and optimizations use pure **NumPy matrix operations** — no TensorFlow or PyTorch.
        """)