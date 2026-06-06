
import os
import re
import pickle
import random
import numpy as np
import pandas as pd
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag as nltk_pos_tag
from sklearn.model_selection import train_test_split

os.makedirs("results/plots", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ══════════════════════════════════════════════════════════════
# data/02_feature_extractor.py
# LexiCoach — Data Cleaning + Feature Extraction (v2 — 10 features)
#
# Features per word:
#   1. frequency         — log-normalized corpus frequency
#   2. syllables         — phonological complexity
#   3. length            — longer = more formal
#   4. suffix            — formal suffix patterns
#   5. pos_score         — POS tag formality (noun/adj=1, verb=0.5, other=0)
#   6. is_common         — in top-500 most frequent English words
#   7. latin_prefix      — Latin/Greek academic prefix
#   8. consonant_density — consonant cluster ratio (formal words are denser)
#   9. unique_char_ratio — unique chars / length (complexity proxy)
#  10. context_score     — avg frequency of ±2 surrounding words
#
# f_acad removed — directly encoded the label (data leakage).
# Labels from word lists. Classes balanced 50/50 by undersampling.
# Word frequency built on TRAIN split only — no leakage.
# ══════════════════════════════════════════════════════════════

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("=" * 60)
print("LexiCoach — Feature Extraction Pipeline (10 features)")
print("=" * 60)

lemmatizer = WordNetLemmatizer()

# ══════════════════════════════════════════════════════════════
# WORD LISTS
# ══════════════════════════════════════════════════════════════

INFORMAL_WORDS = set([
    "use","uses","used","using",
    "get","gets","got","gotten","getting",
    "show","shows","showed","shown","showing",
    "make","makes","made","making",
    "help","helps","helped","helping",
    "need","needs","needed","needing",
    "start","starts","started","starting",
    "end","ends","ended","ending",
    "think","thinks","thought","thinking",
    "find","finds","found","finding",
    "look","looks","looked","looking",
    "try","tries","tried","trying",
    "give","gives","gave","given","giving",
    "talk","talks","talked","talking",
    "keep","keeps","kept","keeping",
    "build","builds","built","building",
    "change","changes","changed","changing",
    "check","checks","checked","checking",
    "create","creates","created","creating",
    "improve","improves","improved","improving",
    "see","sees","saw","seen","seeing",
    "say","says","said","saying",
    "ask","asks","asked","asking",
    "go","goes","went","gone","going",
    "know","knows","knew","known","knowing",
    "buy","buys","bought","buying",
    "work","works","worked","working",
    "good","bad","big","large","small",
    "fast","slow","hard","easy","clear",
    "new","old","real","main","next",
    "better","best","wrong","correct",
    "important","different","same",
    "very","really","also","but","so",
    "maybe","usually","often","always",
    "mainly","mostly","basically","generally",
    "kids","things","thing","ways","way",
    "idea","ideas","goal","goals",
    "plan","plans","result","results",
    "problem","problems","part","parts",
    "reason","reasons","example","examples",
    "area","areas","topic","topics",
    "method","methods","answer","answers",
    "group","groups","person","people",
    "teacher","teachers","student","students",
    "book","books","info","lot","lots",
    "nice","great","cool","okay","fine",
    "stuff","kind","sort","pretty","quite",
    "rather","just","still","already",
    "around","put","come","came",
    "tell","told","feel","felt",
    "move","moved","turn","turned",
    "call","called","play","played",
    "run","ran","live","lived",
    "leave","left","bring","brought",
    "hold","held","stand","stood",
    "lose","lost","set","sets",
])

ACADEMIC_WORDS = set([
    "analyze","analysis","analytical","analyzing","analyzed",
    "approach","approaches","approached","approaching",
    "assess","assessment","assessing","assessed",
    "assume","assumption","assuming","assumed",
    "demonstrate","demonstration","demonstrating","demonstrated",
    "evaluate","evaluation","evaluating","evaluated",
    "establish","establishment","establishing","established",
    "facilitate","facilitation","facilitating","facilitated",
    "generate","generation","generating","generated",
    "identify","identification","identifying","identified",
    "implement","implementation","implementing","implemented",
    "indicate","indication","indicating","indicated",
    "investigate","investigation","investigating","investigated",
    "maintain","maintenance","maintaining","maintained",
    "obtain","obtaining","obtained",
    "participate","participation","participating","participated",
    "perceive","perception","perceiving","perceived",
    "require","requirement","requiring","required",
    "respond","response","responding","responded",
    "utilize","utilization","utilizing","utilized",
    "acknowledge","acknowledging","acknowledged",
    "acquire","acquisition","acquiring","acquired",
    "adapt","adaptation","adapting","adapted",
    "adequate","adequately","adequacy",
    "allocate","allocation","allocating","allocated",
    "anticipate","anticipation","anticipating","anticipated",
    "attribute","attribution","attributing","attributed",
    "clarify","clarification","clarifying","clarified",
    "collaborate","collaboration","collaborating","collaborated",
    "commence","commencement","commencing","commenced",
    "communicate","communication","communicating","communicated",
    "comprise","comprising","comprised",
    "concentrate","concentration","concentrating","concentrated",
    "conclude","conclusion","concluding","concluded",
    "conduct","conducting","conducted",
    "confirm","confirmation","confirming","confirmed",
    "construct","construction","constructing","constructed",
    "contribute","contribution","contributing","contributed",
    "coordinate","coordination","coordinating","coordinated",
    "define","definition","defining","defined",
    "derive","derivation","deriving","derived",
    "determine","determination","determining","determined",
    "distribute","distribution","distributing","distributed",
    "emphasize","emphasis","emphasizing","emphasized",
    "enhance","enhancement","enhancing","enhanced",
    "ensure","ensuring","ensured",
    "examine","examination","examining","examined",
    "formulate","formulation","formulating","formulated",
    "hypothesize","hypothesis","hypothesizing",
    "incorporate","incorporation","incorporating","incorporated",
    "initiate","initiation","initiating","initiated",
    "interpret","interpretation","interpreting","interpreted",
    "modify","modification","modifying","modified",
    "monitor","monitoring","monitored",
    "observe","observation","observing","observed",
    "optimize","optimization","optimizing","optimized",
    "propose","proposal","proposing","proposed",
    "provide","provision","providing","provided",
    "recommend","recommendation","recommending","recommended",
    "represent","representation","representing","represented",
    "resolve","resolution","resolving","resolved",
    "select","selection","selecting","selected",
    "significant","significance","significantly",
    "specify","specification","specifying","specified",
    "substantial","substantially","substantiate",
    "sufficient","sufficiently","sufficiency",
    "summarize","summary","summarizing","summarized",
    "support","supporting","supported",
    "theoretical","theory","theorize",
    "transform","transformation","transforming","transformed",
    "validate","validation","validating","validated",
    "verify","verification","verifying","verified",
    "fundamental","fundamentally","fundamentals",
    "comprehensive","comprehensively","comprehension",
    "consequently","consequence","consequential",
    "furthermore","moreover","nevertheless","nonetheless",
    "however","therefore","thus","hence","whereby",
    "methodology","methodological","methodologies",
    "objective","objectively","objectives",
    "perspective","perspectives",
    "phenomenon","phenomena","phenomenal",
    "preliminary","preliminarily",
    "subsequent","subsequently",
    "component","components","constituent","constituents",
    "correlation","correlate","correlating","correlated",
    "criterion","criteria","critical","critically",
    "dimension","dimensions","dimensional",
    "empirical","empirically","empiricism",
    "equivalent","equivalence","equivalently",
    "explicit","explicitly","explicitness",
    "implicit","implicitly","implication","implications",
    "inherent","inherently",
    "innovative","innovation","innovatively",
    "integral","integrally","integration","integrate",
    "paradigm","paradigms","paradigmatic",
    "parameter","parameters","parametric",
    "rational","rationale","rationalize",
    "relevant","relevance","relevantly",
    "rigorous","rigorously","rigor",
    "systematic","systematically","systematize",
    "transparent","transparency","transparently",
    "valid","validity","validly",
    "variable","variables","variability","variation",
    "coherent","coherence","coherently",
    "concise","concisely","conciseness",
    "consistent","consistency","consistently",
    "constraint","constraints","constrain",
    "context","contextual","contextually",
    "diverse","diversity","diversify",
    "dynamic","dynamically","dynamics",
    "framework","frameworks",
    "mechanism","mechanisms","mechanistic",
    "outcome","outcomes",
    "policy","policies",
    "potential","potentially","potentiality",
    "principle","principles","principled",
    "procedure","procedures","procedural",
    "process","processes","processing","processed",
    "regulate","regulation","regulations","regulatory",
    "restrict","restriction","restrictions","restrictive",
    "strategy","strategies","strategic","strategically",
    "structure","structures","structural","structurally",
    "substitute","substitution","substituting","substituted",
    "sustain","sustainability","sustainable","sustainably",
    "transfer","transference","transferring","transferred",
    "appropriate","appropriately","appropriateness",
    "capacity","capacities","capable","capability",
    "circumstance","circumstances","circumstantial",
    "complexity","complex","complexly",
    "concept","concepts","conceptual","conceptually",
    "conventional","conventionally","convention",
    "evidence","evidential","evidently",
    "factor","factors","factorial",
    "individual","individuals","individually",
    "insight","insights","insightful",
    "instance","instances",
    "literature","literary","literate",
    "model","models","modeling","modeled",
    "notion","notions","notional",
    "novel","novelty","novelties",
    "pattern","patterns","patterned",
    "population","populations",
    "proportion","proportional","proportionally",
    "rapid","rapidly","rapidity",
    "recover","recovery","recovering","recovered",
    "reference","references","referencing","referenced",
    "relation","relations","relational",
    "resource","resources","resourceful",
    "scale","scales","scaling","scaled",
    "simulate","simulation","simulating","simulated",
    "source","sources","sourcing","sourced",
    "standard","standards","standardize","standardized",
    "technique","techniques","technical","technically",
    "trend","trends","trending","trended",
    "unique","uniquely","uniqueness",
    "value","values","valuation",
    "impact","impactful","impacting","impacted",
    "data","dataset","datasets",
    "domain","domains","discipline","disciplines",
    "feature","features","featured",
    "network","networks","networked","networking",
    "phase","phases","phased","phasing",
    "region","regions","regional",
    "role","roles",
    "sector","sectors","sectoral",
    "series","serial",
    "scope","scoping","scoped",
    "abstract","abstraction","abstractly",
    "accumulate","accumulation","accumulated",
    "benefit","benefits","beneficial","benefiting",
    "category","categories","categorize","categorized",
    "challenge","challenges","challenging","challenged",
    "character","characteristic","characteristics",
    "cognitive","cognition","cognitively",
    "compile","compilation","compiling","compiled",
    "conscious","consciousness","consciously",
    "debate","debates","debating","debated",
    "decline","declining","declined",
    "deduce","deduction","deducing","deduced",
    "design","designing","designed",
    "detect","detection","detecting","detected",
    "develop","development","developing","developed",
    "distinct","distinction","distinctive","distinctly",
    "document","documentation","documenting","documented",
    "economic","economy","economically","economies",
    "effective","effectively","effectiveness","effect",
    "efficient","efficiency","efficiently",
    "emerge","emergence","emerging","emerged",
    "enable","enabling","enabled",
    "engage","engagement","engaging","engaged",
    "environment","environmental","environmentally",
    "estimate","estimation","estimating","estimated",
    "ethical","ethics","ethically",
    "evolve","evolution","evolving","evolved",
    "expand","expansion","expanding","expanded",
    "expose","exposure","exposing","exposed",
    "extend","extension","extending","extended",
    "extract","extraction","extracting","extracted",
    "focus","focusing","focused",
    "formal","formally","formality",
    "global","globally","globalization",
    "govern","governance","governing","governed",
    "influence","influencing","influenced",
    "inform","information","informing","informed",
    "interact","interaction","interacting","interacted",
    "introduce","introduction","introducing","introduced",
    "involve","involvement","involving","involved",
    "justify","justification","justifying","justified",
    "knowledge","knowledgeable",
    "language","linguistic","linguistically",
    "learn","learning","learned",
    "measure","measurement","measuring","measured",
    "occur","occurrence","occurring","occurred",
    "outline","outlining","outlined",
    "overcome","overcoming","overcame",
    "perform","performance","performing","performed",
    "positive","positively","positivity",
    "predict","prediction","predicting","predicted",
    "present","presentation","presenting","presented",
    "priority","prioritize","prioritizing","prioritized",
    "professional","professionally","professionalism",
    "promote","promotion","promoting","promoted",
    "prove","proof","proving","proved","proven",
    "pursue","pursuit","pursuing","pursued",
    "recognize","recognition","recognizing","recognized",
    "reduce","reduction","reducing","reduced",
    "reflect","reflection","reflecting","reflected",
    "reinforce","reinforcement","reinforcing","reinforced",
    "relate","relation","relating","related",
    "rely","reliance","relying","relied",
    "report","reporting","reported",
    "research","researcher","researching","researched",
    "reveal","revelation","revealing","revealed",
    "review","reviewing","reviewed",
    "secure","security","securing","secured",
    "seek","seeking","sought",
    "separate","separation","separating","separated",
    "society","social","socially","societal",
    "solve","solution","solving","solved",
    "stable","stability","stabilize","stabilized",
    "suggest","suggestion","suggesting","suggested",
    "survey","surveying","surveyed",
    "target","targeting","targeted",
    "test","testing","tested",
    "traditional","traditionally","tradition",
    "understand","understanding","understood",
])

STOPWORDS = set([
    "the","a","an","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could",
    "should","may","might","shall","can","i","you","he","she",
    "it","we","they","me","him","her","us","them","my","your",
    "his","its","our","their","this","that","these","those",
    "what","which","who","when","where","and","or","not","no",
    "in","on","at","to","for","of","with","by","from","as",
    "into","after","before","since","all","both","each","every",
    "more","most","than","then","if","about","up","out",
    "over","between","through","during","without","within",
    "along","across","behind","beyond","plus","except",
    "down","off","near","upon","per","via","vs",
])

# Top-500 most common English words (informality marker)
TOP500_COMMON = set([
    "time","year","people","way","day","man","woman","child",
    "world","life","hand","part","place","case","week","company",
    "system","program","question","work","government","number",
    "night","point","home","water","room","mother","area","money",
    "story","fact","month","lot","right","study","book","eye",
    "job","word","business","issue","side","kind","head","house",
    "service","friend","father","power","hour","game","line",
    "end","among","off","always","state","once","book","hear",
    "body","drive","took","play","move","live","believe","hold",
    "bring","happen","write","provide","sit","stand","lose","pay",
    "meet","include","continue","set","learn","change","lead",
    "understand","watch","follow","stop","create","speak","read",
    "spend","grow","open","walk","win","offer","remember","love",
    "consider","appear","buy","wait","serve","die","send","expect",
    "build","stay","fall","cut","reach","kill","remain","suggest",
    "raise","pass","sell","require","report","decide","pull",
    "break","wish","pick","carry","explain","face","approach",
    "allow","become","begin","call","come","feel","gave","give",
    "gone","got","keep","knew","know","let","long","look","made",
    "make","need","next","nothing","often","old","only","other",
    "over","own","place","put","said","same","saw","says","see",
    "seen","show","small","still","take","taken","tell","thing",
    "think","thought","told","took","turn","under","upon","used",
    "want","well","went","were","when","where","while","whose",
    "within","without","year","years","young","able","about",
    "actually","after","again","ago","almost","along","already",
    "although","another","around","away","back","because","been",
    "before","being","between","both","brought","came","come",
    "done","down","during","even","ever","every","everyone",
    "everything","few","find","first","found","four","from",
    "full","further","gave","give","going","great","group",
    "hard","having","here","high","him","himself","however",
    "important","instead","into","its","itself","just","large",
    "last","later","left","less","like","likely","little","local",
    "made","make","many","might","much","must","near","never",
    "new","next","night","none","nor","nothing","now","number",
    "off","once","only","open","our","out","over","own","part",
    "past","people","place","plan","point","possible","problem",
    "public","real","right","run","same","several","should",
    "since","small","some","something","sometimes","soon","such",
    "sure","take","than","their","them","then","there","these",
    "they","thing","this","those","three","through","together",
    "too","toward","turn","two","under","until","very","view",
    "want","water","whether","which","while","whole","whose",
    "wide","will","with","within","would","write",
])

# Latin/Greek academic prefixes
LATIN_GREEK_PREFIXES = (
    "inter","intra","trans","pre","post","sub","super","extra",
    "ultra","semi","anti","counter","non","over","under","out",
    "dis","mis","re","un","co","com","con","col","cor",
    "pro","per","para","meta","poly","mono","micro","macro",
    "hyper","hypo","auto","bio","geo","neo","pseudo","quasi",
    "multi","omni","uni","bi","tri","quad","penta","hexa",
    "deca","cent","kilo","mega","giga","tele","photo","thermo",
    "electro","hydro","aero","chrono","demo","ethn","gen",
)

# Formal suffixes
FORMAL_SUFFIXES = (
    'tion','sion','ity','ance','ence',
    'ment','ness','ism','ive','ous',
    'ize','ise','ate','ify','fy',
    'ology','ography','ometry',
)


# ══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════

def clean_text(text):
    text = str(text)
    text = re.sub(r'@[A-Z]+\d*', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def is_valid_word(word):
    if word in STOPWORDS:                        return False
    if len(word) < 3:                            return False
    if len(word) > 25:                           return False
    if not word.isalpha():                       return False
    if len(word) < 6 and len(set(word)) <= 2:   return False
    return True


def count_syllables(word):
    word     = word.lower()
    count    = 0
    vowels   = "aeiouy"
    prev_vow = False
    for ch in word:
        is_v = ch in vowels
        if is_v and not prev_vow:
            count += 1
        prev_vow = is_v
    if word.endswith('e') and count > 1:
        count -= 1
    return max(1, count)


pos_cache = {}

def get_pos_score(word):
    """
    POS tag formality score with caching.
    Nouns and adjectives are more formal than verbs,
    verbs more formal than other POS.
    Returns float in [0, 1].
    """
    w = word.lower()
    if w in pos_cache:
        return pos_cache[w]
    try:
        tag = nltk_pos_tag([w])[0][1]
        if tag.startswith('NN'):   score = 1.0   # noun
        elif tag.startswith('JJ'): score = 1.0   # adjective
        elif tag.startswith('VB'): score = 0.5   # verb
        elif tag.startswith('RB'): score = 0.3   # adverb
        else:                      score = 0.2
    except Exception:
        score = 0.5
    pos_cache[w] = score
    return score


def consonant_density(word):
    """
    Ratio of consonant clusters to word length.
    Formal/academic words tend to have denser consonant clusters.
    E.g. 'strength', 'infrastructure', 'acknowledge'.
    """
    vowels = set('aeiou')
    consonants = [c for c in word.lower() if c.isalpha() and c not in vowels]
    return min(1.0, len(consonants) / max(len(word), 1))


def unique_char_ratio(word):
    """
    Ratio of unique characters to word length.
    More varied character usage → more complex word.
    """
    return min(1.0, len(set(word.lower())) / max(len(word), 1))


# ══════════════════════════════════════════════════════════════
# STEP 1: LOAD AND CLEAN
# ══════════════════════════════════════════════════════════════
print("\n[1] Loading and cleaning datasets...")

asap = pd.read_csv(
    'data/training_set_rel3.tsv',
    sep='\t', encoding='latin-1'
)[['essay','domain1_score','essay_set']]

ielts = pd.read_csv(
    'data/ielts_writing_dataset.csv',
    encoding='utf-8'
)[['Essay','Lexical_Resource','Overall']]

asap  = asap.dropna(subset=['essay','domain1_score'])
ielts = ielts.dropna(subset=['Essay','Overall','Lexical_Resource'])
asap  = asap.drop_duplicates(subset=['essay'])
ielts = ielts.drop_duplicates(subset=['Essay'])

asap['wc']  = asap['essay'].str.split().str.len()
ielts['wc'] = ielts['Essay'].str.split().str.len()
asap  = asap[asap['wc']  >= 30].drop(columns=['wc'])
ielts = ielts[ielts['wc'] >= 30].drop(columns=['wc'])

asap['clean']  = asap['essay'].apply(clean_text)
ielts['clean'] = ielts['Essay'].apply(clean_text)

print(f"  ASAP  after cleaning : {len(asap):,}")
print(f"  IELTS after cleaning : {len(ielts):,}")


# ══════════════════════════════════════════════════════════════
# STEP 2: TRAIN/TEST SPLIT FIRST (no leakage)
# ══════════════════════════════════════════════════════════════
print("\n[2] Splitting train/test before frequency analysis...")

asap_tr, asap_te   = train_test_split(
    asap,  test_size=0.2, random_state=SEED,
    stratify=asap['essay_set']
)
ielts_tr, ielts_te = train_test_split(
    ielts, test_size=0.2, random_state=SEED
)

print(f"  ASAP  train: {len(asap_tr):,}  test: {len(asap_te):,}")
print(f"  IELTS train: {len(ielts_tr):,}  test: {len(ielts_te):,}")


# ══════════════════════════════════════════════════════════════
# STEP 3: WORD FREQUENCY MAP — TRAIN ONLY
# ══════════════════════════════════════════════════════════════
print("\n[3] Building word frequency map from train only...")

train_texts = (
    asap_tr['clean'].tolist() +
    ielts_tr['clean'].tolist()
)
all_words  = re.findall(r'\b[a-z]+\b', " ".join(train_texts))
word_freq  = Counter(all_words)
total_freq = sum(word_freq.values())

print(f"  Total tokens  : {total_freq:,}")
print(f"  Unique words  : {len(word_freq):,}")


# ══════════════════════════════════════════════════════════════
# STEP 4: FEATURE EXTRACTION — 10 FEATURES
# ══════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    'frequency',        # F1 — log-normalized corpus freq
    'syllables',        # F2 — phonological complexity
    'length',           # F3 — word length proxy
    'suffix',           # F4 — formal suffix (-tion, -ity ...)
    'pos_score',        # F5 — POS formality signal
    'is_common',        # F6 — in top-500 common words (informal)
    'latin_prefix',     # F7 — Latin/Greek academic prefix
    'consonant_density',# F8 — consonant cluster ratio
    'unique_char_ratio',# F9 — character diversity
    'context_score',    # F10 — avg frequency of ±2 context words
]


def extract_word_features(word):
    """Extract 9 word-level features (context added separately)."""
    w = word.lower()

    # F1: Log-normalized corpus frequency — high = informal
    freq   = word_freq.get(w, 0)
    f_freq = min(1.0, np.log1p(freq) / np.log1p(total_freq))

    # F2: Syllable count — more syllables = more academic
    f_syl  = min(1.0, count_syllables(w) / 6.0)

    # F3: Word length — longer = more formal
    f_len  = min(1.0, len(w) / 20.0)

    # F4: Formal suffix
    f_suff = 1.0 if w.endswith(FORMAL_SUFFIXES) else 0.0

    # F5: POS score — noun/adj = 1.0, verb = 0.5, other = 0.2
    f_pos  = get_pos_score(w)

    # F6: Is common word — top-500 → likely informal
    f_com  = 1.0 if w in TOP500_COMMON else 0.0

    # F7: Latin/Greek prefix — academic vocabulary marker
    f_lat  = 1.0 if any(w.startswith(p) for p in LATIN_GREEK_PREFIXES) else 0.0

    # F8: Consonant density — formal words are phonologically dense
    f_cons = consonant_density(w)

    # F9: Unique character ratio — complexity proxy
    f_uniq = unique_char_ratio(w)

    return [f_freq, f_syl, f_len, f_suff, f_pos,
            f_com, f_lat, f_cons, f_uniq]


def extract_features_with_context(words, idx):
    """
    10 features: 9 word-level + 1 context score.
    Context score = avg corpus frequency of ±2 surrounding words.
    Formal words appear near other formal words.
    """
    word_feats = extract_word_features(words[idx])

    # F10: context score — avg frequency of ±2 neighbours
    neighbours = []
    for offset in [-2, -1, 1, 2]:
        ni = idx + offset
        if 0 <= ni < len(words):
            nw   = words[ni].lower()
            freq = word_freq.get(nw, 0)
            neighbours.append(np.log1p(freq) / np.log1p(total_freq))
    f_ctx = float(np.mean(neighbours)) if neighbours else 0.5

    return word_feats + [f_ctx]


# ══════════════════════════════════════════════════════════════
# STEP 5: BALANCED SAMPLE EXTRACTION
# ══════════════════════════════════════════════════════════════
print("\n[4] Extracting balanced word samples (10 features)...")
print("  Label 1 -> INFORMAL_WORDS -> replace")
print("  Label 0 -> ACADEMIC_WORDS -> keep")


def extract_samples(texts, split_name):
    X_pos, y_pos = [], []
    X_neg, y_neg = [], []
    skipped      = 0

    for text in texts:
        words   = clean_text(text).split()
        valid_idx = [
            i for i, w in enumerate(words)
            if is_valid_word(w)
        ]
        if not valid_idx:
            continue

        n_sample  = min(len(valid_idx), 30)
        sampled   = random.sample(valid_idx, n_sample)

        for idx in sampled:
            word = words[idx]
            lemm = lemmatizer.lemmatize(word)

            if word in INFORMAL_WORDS or lemm in INFORMAL_WORDS:
                feats = extract_features_with_context(words, idx)
                X_pos.append(feats)
                y_pos.append(1)
            elif word in ACADEMIC_WORDS or lemm in ACADEMIC_WORDS:
                feats = extract_features_with_context(words, idx)
                X_neg.append(feats)
                y_neg.append(0)
            else:
                skipped += 1

    min_count = min(len(X_pos), len(X_neg))
    if min_count == 0:
        print(f"  {split_name}: WARNING — one class empty")
        return [], []

    pos_idx = random.sample(range(len(X_pos)), min_count)
    neg_idx = random.sample(range(len(X_neg)), min_count)

    X = ([X_pos[i] for i in pos_idx] +
         [X_neg[i] for i in neg_idx])
    y = [1] * min_count + [0] * min_count

    print(f"  {split_name}: {len(X):,} balanced samples "
          f"({min_count:,} per class | {skipped:,} skipped)")
    return X, y


X_tr_a, y_tr_a = extract_samples(asap_tr['essay'].tolist(),  "ASAP  train")
X_tr_i, y_tr_i = extract_samples(ielts_tr['Essay'].tolist(), "IELTS train")
X_te_a, y_te_a = extract_samples(asap_te['essay'].tolist(),  "ASAP  test ")
X_te_i, y_te_i = extract_samples(ielts_te['Essay'].tolist(), "IELTS test ")

X_train = np.array(X_tr_a + X_tr_i, dtype=np.float32)
y_train = np.array(y_tr_a + y_tr_i, dtype=np.int32)
X_test  = np.array(X_te_a + X_te_i, dtype=np.float32)
y_test  = np.array(y_te_a + y_te_i, dtype=np.int32)

# Shuffle
tr_idx  = np.random.permutation(len(X_train))
te_idx  = np.random.permutation(len(X_test))
X_train = X_train[tr_idx];  y_train = y_train[tr_idx]
X_test  = X_test[te_idx];   y_test  = y_test[te_idx]


# ══════════════════════════════════════════════════════════════
# STEP 6: CLASS BALANCE VERIFICATION
# ══════════════════════════════════════════════════════════════
print("\n[5] Class balance verification...")
tr_pos = (y_train == 1).sum()
tr_neg = (y_train == 0).sum()
te_pos = (y_test  == 1).sum()
te_neg = (y_test  == 0).sum()

print(f"  Train label 0 (keep)    : {tr_neg:,} "
      f"({tr_neg/len(y_train)*100:.1f}%)")
print(f"  Train label 1 (replace) : {tr_pos:,} "
      f"({tr_pos/len(y_train)*100:.1f}%)")
print(f"  Test  label 0 (keep)    : {te_neg:,} "
      f"({te_neg/len(y_test)*100:.1f}%)")
print(f"  Test  label 1 (replace) : {te_pos:,} "
      f"({te_pos/len(y_test)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════
# STEP 7: FEATURE STATISTICS
# ══════════════════════════════════════════════════════════════
print("\n[6] Feature statistics (train set)...")
print(f"  {'Feature':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print(f"  {'-'*52}")
for i, name in enumerate(FEATURE_NAMES):
    col = X_train[:, i]
    print(f"  {name:<20} {col.mean():>8.4f} {col.std():>8.4f} "
          f"{col.min():>8.4f} {col.max():>8.4f}")


# ══════════════════════════════════════════════════════════════
# STEP 8: SAVE
# ══════════════════════════════════════════════════════════════
print("\n[7] Saving processed data...")

with open('data/processed_data.pkl', 'wb') as f:
    pickle.dump({
        'X_train':        X_train,
        'y_train':        y_train,
        'X_test':         X_test,
        'y_test':         y_test,
        'feature_names':  FEATURE_NAMES,
        'word_freq':      word_freq,
        'total_freq':     total_freq,
        'informal_words': INFORMAL_WORDS,
        'academic_words': ACADEMIC_WORDS,
        'seed':           SEED,
    }, f)

print("  Saved -> data/processed_data.pkl")

print("\n" + "=" * 60)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 60)
print(f"  Train samples  : {len(X_train):,}")
print(f"  Test samples   : {len(X_test):,}")
print(f"  Features       : {X_train.shape[1]}  (upgraded from 4)")
print(f"  Feature names  :")
for i, n in enumerate(FEATURE_NAMES, 1):
    print(f"    {i:2}. {n}")
print(f"\n  f_acad removed - no label leakage")
print(f"  Context window - +/-2 words included")
print(f"  POS tagging    - NLTK per-word")
print(f"  Class balance  : 50/50 undersampled")
print(f"\n  NEXT: python 03_perceptron.py")
print("=" * 60)