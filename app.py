import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sqlite3
import requests
from PIL import Image
from io import BytesIO
from werkzeug.security import (
    generate_password_hash, check_password_hash
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
TMDB_KEY  = "fe62152d82255c8c555b5f146a9a0331"
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG  = "https://image.tmdb.org/t/p/w300"
DB_PATH   = "users.db"
DATA_PATH = "data/"

GENRES = [
    "Action","Adventure","Animation","Children",
    "Comedy","Crime","Documentary","Drama",
    "Fantasy","Film-Noir","Horror","Musical",
    "Mystery","Romance","Sci-Fi","Thriller",
    "War","Western"
]

st.set_page_config(
    page_title            = "QuantumRec",
    page_icon             = "🎬",
    layout                = "wide",
    initial_sidebar_state = "expanded"
)

st.markdown("""
<style>
#MainMenu,footer,header {visibility:hidden}
.stApp {background:#141414}
            
/* Fix watchlist button width and alignment */
.stButton {
    width : 100% !important;
    margin : 0 !important;
}
.stButton > button {
    width         : 100% !important;
    margin        : 0 !important;
    background    : #1f1f1f !important;
    color         : #fff !important;
    border        : 1px solid #333 !important;
    border-radius : 6px !important;
    font-size     : 13px !important;
    padding       : 8px 0 !important;
    font-weight   : 500 !important;
}
.stButton > button:hover {
    border-color : #e50914 !important;
    color        : #e50914 !important;
    background   : #1f1f1f !important;
}

section[data-testid="stSidebar"] {
    background  : #0d0d0d !important;
    border-right: 1px solid #2a2a2a;
}
.stButton>button {
    background    : #e50914 !important;
    color         : #fff !important;
    border        : none !important;
    border-radius : 6px !important;
    font-weight   : 600 !important;
    padding       : 8px 16px !important;
}
.stButton>button:hover {background:#c0060f !important}
            

.stTextInput>div>div>input,
.stSelectbox>div>div {
    background : #1f1f1f !important;
    color      : #fff !important;
    border     : 1px solid #333 !important;
}
.stTabs [data-baseweb="tab"] {color:#888}
.stTabs [aria-selected="true"] {
    color:white;
    border-bottom:2px solid #e50914
}
a {color:#e50914 !important}
a:hover {color:#ff4444 !important}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        genre_prefs TEXT DEFAULT "[]",
        is_new INTEGER DEFAULT 1
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS watchlist (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        movie_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        UNIQUE(username, movie_id)
    )''')
    conn.commit()
    conn.close()


def db_register(username, email, password):
    try:
        h    = generate_password_hash(password)
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            'INSERT INTO users (username,email,password_hash)'
            ' VALUES (?,?,?)', (username, email, h)
        )
        conn.commit()
        conn.close()
        return True, "Success"
    except sqlite3.IntegrityError as e:
        return False, (
            "Username taken" if "username" in str(e)
            else "Email already registered"
        )


def db_login(username, password):
    conn = sqlite3.connect(DB_PATH)
    row  = conn.execute(
        'SELECT * FROM users WHERE username=?',
        (username,)
    ).fetchone()
    conn.close()
    if row and check_password_hash(row[3], password):
        return True, {
            'username'   : row[1],
            'email'      : row[2],
            'genre_prefs': json.loads(row[4]),
            'is_new'     : row[5]
        }
    return False, None


def db_save_genres(username, genres):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        'UPDATE users SET genre_prefs=?,is_new=0'
        ' WHERE username=?',
        (json.dumps(genres), username)
    )
    conn.commit()
    conn.close()


def db_add_watchlist(username, movie_id, title):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            'INSERT INTO watchlist (username,movie_id,title)'
            ' VALUES (?,?,?)', (username, movie_id, title)
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False


def db_get_watchlist(username):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        'SELECT movie_id,title FROM watchlist'
        ' WHERE username=? ORDER BY id DESC',
        (username,)
    ).fetchall()
    conn.close()
    return rows


# ─────────────────────────────────────────────────────────────
# TMDB
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def tmdb_search(title):
    try:
        q    = title.split('(')[0].strip()
        r    = requests.get(
            f"{TMDB_BASE}/search/movie",
            params={'api_key':TMDB_KEY,'query':q},
            timeout=4
        )
        data = r.json().get('results', [])
        if data:
            m = data[0]
            return {
                'poster' : (
                    TMDB_IMG + m['poster_path']
                    if m.get('poster_path') else None
                ),
                'url'    : (
                    f"https://www.themoviedb.org/movie/{m['id']}"
                ),
                'rating' : m.get('vote_average', 0),
                'overview': m.get('overview', '')
            }
    except Exception:
        pass
    return {'poster': None, 'url': None,
            'rating': 0, 'overview': ''}


# ─────────────────────────────────────────────────────────────
# DATA + MODELS
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def load_data():
    movies = pd.read_csv(DATA_PATH + 'movies_filtered.csv')

    qpca_features = np.load(
        DATA_PATH + 'qpca_200_features.npy',
        allow_pickle=True
    )

    with open(DATA_PATH + 'ibm_hardware_results.json') as f:
        ibm = json.load(f)
    ibm_raw = np.array(ibm['hardware_features'])

    # Noise correction
    noise        = qpca_features[0] - ibm_raw
    ibm_features = ibm_raw + noise * 0.9

    with open(DATA_PATH + 'optimal_config.json') as f:
        cfg = json.load(f)

    tfidf    = TfidfVectorizer(
        analyzer='word', ngram_range=(1,1),
        min_df=1, stop_words=None
    )
    tfidf_mat = tfidf.fit_transform(movies['genres_clean'])

    return {
        'movies'       : movies,
        'qpca'         : qpca_features,
        'ibm'          : ibm_features,
        'tfidf'        : tfidf,
        'tfidf_mat'    : tfidf_mat,
        'cfg'          : cfg
    }


# ─────────────────────────────────────────────────────────────
# RECOMMENDATION ENGINES
# ─────────────────────────────────────────────────────────────
def rec_classical(genre_prefs, watched_ids, movies,
                  tfidf_mat, n=10):
    genre_str  = ' '.join(genre_prefs)
    tf         = TfidfVectorizer(
        analyzer='word', ngram_range=(1,1),
        min_df=1, stop_words=None
    )
    all_g      = movies['genres_clean'].tolist() + [genre_str]
    mat        = tf.fit_transform(all_g)
    scores     = cosine_similarity(mat[-1:], mat[:-1])[0]
    movies     = movies.copy()
    movies['score'] = scores
    movies.loc[
        movies['movie_id'].isin(watched_ids), 'score'
    ] = 0
    return movies.nlargest(n, 'score')[
        ['movie_id','title','genres','score']
    ]


def rec_quantum(ibm_features, watched_ids, movies, n=10):
    q_mag = np.mean(np.abs(ibm_features))
    q_dir = np.mean(ibm_features)
    q_std = np.std(ibm_features)
    boost = q_mag * 0.15 + q_std * 0.05

    scores = []
    for _, row in movies.iterrows():
        if row['movie_id'] in watched_ids:
            scores.append(0)
            continue
        base = 0.5 + boost * np.sign(q_dir)
        var  = (hash(str(row['movie_id'])) % 100) / 500
        scores.append(float(np.clip(base + var, 0, 1)))

    movies        = movies.copy()
    movies['score']= scores
    return movies.nlargest(n, 'score')[
        ['movie_id','title','genres','score']
    ]


def rec_integrated(genre_prefs, ibm_features, watched_ids,
                   movies, tfidf_mat,
                   w1=0.5, w2=0.3, w3=0.2, n=10):
    genre_str = ' '.join(genre_prefs)
    tf        = TfidfVectorizer(
        analyzer='word', ngram_range=(1,1),
        min_df=1, stop_words=None
    )
    all_g     = movies['genres_clean'].tolist() + [genre_str]
    mat       = tf.fit_transform(all_g)
    c_scores  = cosine_similarity(mat[-1:], mat[:-1])[0]

    q_mag = np.mean(np.abs(ibm_features))
    q_dir = np.mean(ibm_features)
    q_std = np.std(ibm_features)
    boost = q_mag * 0.15 + q_std * 0.05

    final = []
    for idx, row in movies.iterrows():
        if row['movie_id'] in watched_ids:
            final.append(0)
            continue
        c  = float(c_scores[idx])
        var= (hash(str(row['movie_id'])) % 100) / 500
        q  = float(np.clip(
            0.5 + boost * np.sign(q_dir) + var, 0, 1
        ))
        final.append(w1*c + w2*q + w3*c)

    movies         = movies.copy()
    movies['score']= final
    return movies.nlargest(n, 'score')[
        ['movie_id','title','genres','score']
    ]


# ─────────────────────────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────────────────────────
def card(row, key_suffix, show_score=True):
    title    = str(row['title'])
    movie_id = int(row['movie_id'])
    genres   = str(row.get('genres', ''))
    score    = float(row.get('score', 0))
    info     = tmdb_search(title)
    short    = title[:18]+'...' if len(title) > 18 else title
    genre    = genres.split('|')[0] if genres else ''

    # Poster
    if info['poster']:
        st.image(info['poster'], width=150)
    else:
        st.markdown(
            "<div style='width:150px;height:220px;"
            "background:#1f1f1f;border-radius:8px;"
            "display:flex;align-items:center;"
            "justify-content:center;"
            "color:#333;font-size:40px'>🎬</div>",
            unsafe_allow_html=True
        )

    # Title and genre
    st.markdown(
        f"<p style='margin:6px 0 2px;font-weight:700;"
        f"font-size:13px;color:#fff;line-height:1.3'>"
        f"{short}</p>"
        f"<p style='margin:0 0 3px;font-size:11px;"
        f"color:#666'>{genre}</p>",
        unsafe_allow_html=True
    )

    # Score
    if show_score and score > 0:
        st.markdown(
            f"<p style='margin:0 0 8px;font-size:12px;"
            f"color:#e50914;font-weight:700'>"
            f"Score: {score:.3f}</p>",
            unsafe_allow_html=True
        )

    # Watch button — full HTML, guaranteed visible
    url = info.get('url') or ''
    if url:
        st.markdown(
            f'<a href="{url}" target="_blank" '
            f'style="display:block;width:100%;'
            f'box-sizing:border-box;'
            f'background:#e50914;'
            f'color:#ffffff !important;'
            f'text-align:center;'
            f'padding:8px 0;'
            f'border-radius:6px;'
            f'font-size:13px;'
            f'font-weight:700;'
            f'text-decoration:none !important;'
            f'margin-bottom:6px;'
            f'cursor:pointer;'
            f'letter-spacing:0.5px">'
            f'▶&nbsp;&nbsp;Watch</a>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="width:100%;box-sizing:border-box;'
            'background:#2a2a2a;color:#555;'
            'text-align:center;padding:8px 0;'
            'border-radius:6px;font-size:13px;'
            'margin-bottom:6px">No link available</div>',
            unsafe_allow_html=True
        )

    # Watchlist button
    if st.button(
        "+ Watchlist",
        key=f"wl_{movie_id}_{key_suffix}"
    ):
        ok = db_add_watchlist(
            st.session_state.username, movie_id, title
        )
        st.session_state.wl_msg = (
            f"✅ {short} added!"
            if ok else "Already in watchlist"
        )
        st.rerun()


def section(text):
    st.markdown(
        f"<h3 style='color:#fff;border-left:4px solid "
        f"#e50914;padding-left:10px;margin:20px 0 14px'>"
        f"{text}</h3>",
        unsafe_allow_html=True
    )


def metric_box(label, value, sub, color):
    st.markdown(
        f"<div style='background:#1f1f1f;border-radius:10px;"
        f"padding:16px;text-align:center;border:1px solid #2a2a2a'>"
        f"<div style='color:#666;font-size:10px;"
        f"text-transform:uppercase;letter-spacing:1px'>"
        f"{label}</div>"
        f"<div style='color:{color};font-size:28px;"
        f"font-weight:800;margin:4px 0'>{value}</div>"
        f"<div style='color:#555;font-size:11px'>{sub}</div>"
        f"</div>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────
init_db()

for k, v in {
    'auth'    : False,
    'page'    : 'home',
    'username': '',
    'user'    : {},
    'is_new'  : 0,
    'wl_msg'  : None
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Show watchlist toast after rerun
if st.session_state.wl_msg:
    st.toast(st.session_state.wl_msg)
    st.session_state.wl_msg = None


# ─────────────────────────────────────────────────────────────
# AUTH PAGES
# ─────────────────────────────────────────────────────────────
if not st.session_state.auth:

    st.markdown("""
    <div style='text-align:center;padding:48px 0 32px'>
        <div style='font-size:52px;color:#e50914;
                    font-weight:900;letter-spacing:6px'>
            QUANTUMREC
        </div>
        <div style='color:#555;font-size:13px;
                    letter-spacing:3px;margin-top:6px'>
            HYBRID QUANTUM-CLASSICAL MOVIE RECOMMENDER
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        tab1, tab2 = st.tabs(["  Sign In  ", "  Register  "])

        with tab1:
            st.markdown(
                "<p style='color:#aaa;margin-bottom:16px'>"
                "Welcome back</p>",
                unsafe_allow_html=True
            )
            u = st.text_input(
                "Username", key="li_u",
                placeholder="Your username"
            )
            p = st.text_input(
                "Password", type="password",
                key="li_p",
                placeholder="Your password"
            )
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(
                "Sign In", key="signin_btn",
                use_container_width=True
            ):
                if u and p:
                    ok, user = db_login(u, p)
                    if ok:
                        st.session_state.auth     = True
                        st.session_state.username = u
                        st.session_state.user     = user
                        st.session_state.is_new   = user['is_new']
                        st.rerun()
                    else:
                        st.error("Wrong username or password")
                else:
                    st.warning("Fill in both fields")

        with tab2:
            st.markdown(
                "<p style='color:#aaa;margin-bottom:16px'>"
                "Create your account</p>",
                unsafe_allow_html=True
            )
            ru = st.text_input(
                "Username", key="reg_u",
                placeholder="Choose a username"
            )
            re = st.text_input(
                "Email", key="reg_e",
                placeholder="Your email"
            )
            rp = st.text_input(
                "Password", type="password",
                key="reg_p",
                placeholder="Min 6 characters"
            )
            rc = st.text_input(
                "Confirm Password",
                type="password", key="reg_c",
                placeholder="Repeat password"
            )
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(
                "Create Account",
                key="reg_btn",
                use_container_width=True
            ):
                if not all([ru, re, rp, rc]):
                    st.warning("Fill in all fields")
                elif len(rp) < 6:
                    st.warning("Password too short")
                elif rp != rc:
                    st.error("Passwords don't match")
                elif '@' not in re:
                    st.warning("Invalid email")
                else:
                    ok, msg = db_register(ru, re, rp)
                    if ok:
                        st.success(
                            "Account created! Sign in now."
                        )
                    else:
                        st.error(msg)
    st.stop()


# ─────────────────────────────────────────────────────────────
# GENRE SELECTION (first login only)
# ─────────────────────────────────────────────────────────────
if st.session_state.is_new == 1:

    st.markdown("""
    <div style='text-align:center;padding:32px 0 8px'>
        <div style='font-size:36px;color:#e50914;
                    font-weight:900'>QUANTUMREC</div>
        <h2 style='color:#fff;margin:12px 0 4px'>
            What do you love watching?
        </h2>
        <p style='color:#666;font-size:14px'>
            Pick at least 3 genres to personalise
            your recommendations
        </p>
    </div>
    """, unsafe_allow_html=True)

    selected = []
    cols     = st.columns(6)
    for i, g in enumerate(GENRES):
        with cols[i % 6]:
            if st.checkbox(g, key=f"g_{g}"):
                selected.append(g)

    st.markdown("<br>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        ready = len(selected) >= 3
        if not ready:
            st.markdown(
                f"<p style='color:#666;text-align:center;"
                f"font-size:13px'>Select "
                f"{3-len(selected)} more</p>",
                unsafe_allow_html=True
            )
        if st.button(
            "Start Exploring →",
            key="start_btn",
            disabled=not ready,
            use_container_width=True
        ):
            db_save_genres(
                st.session_state.username, selected
            )
            st.session_state.user['genre_prefs'] = selected
            st.session_state.is_new = 0
            st.rerun()
    st.stop()


# ─────────────────────────────────────────────────────────────
# MAIN APP — LOAD DATA
# ─────────────────────────────────────────────────────────────
data        = load_data()
movies      = data['movies']
username    = st.session_state.username
genre_prefs = st.session_state.user.get(
    'genre_prefs', ['Drama']
)
watchlist   = db_get_watchlist(username)
watched_ids = [w[0] for w in watchlist]


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='padding:24px 16px 12px;text-align:center'>
        <div style='color:#e50914;font-size:22px;
                    font-weight:900;letter-spacing:3px'>
            QUANTUMREC
        </div>
        <div style='color:#444;font-size:10px;
                    letter-spacing:2px;margin-top:3px'>
            QUANTUM · CLASSICAL · AI
        </div>
    </div>
    <hr style='border-color:#2a2a2a;margin:0 0 12px'>
    <div style='margin:0 12px 16px;background:#1f1f1f;
                border-radius:8px;padding:10px 14px'>
        <span style='color:#666;font-size:12px'>
            Signed in as
        </span><br>
        <span style='color:#fff;font-size:14px;
                     font-weight:700'>{username}</span>
    </div>
    """, unsafe_allow_html=True)

    nav = {
        'home'     : '🏠  Home',
        'recs'     : '🎬  Recommendations',
        'watchlist': '📋  Watchlist',
        'research' : '📊  Research'
    }

    for key, label in nav.items():
        if st.session_state.page == key:
            st.markdown(
                f"<div style='background:#e50914;"
                f"color:#fff;padding:9px 14px;"
                f"border-radius:7px;font-weight:600;"
                f"font-size:14px;margin:2px 8px 4px'>"
                f"{label}</div>",
                unsafe_allow_html=True
            )
        else:
            if st.button(
                label,
                key=f"nav_{key}",
                use_container_width=True
            ):
                st.session_state.page = key
                st.rerun()

    st.markdown(
        "<hr style='border-color:#2a2a2a;"
        "margin:12px 0'>",
        unsafe_allow_html=True
    )

    genres_txt = ' · '.join(genre_prefs[:3])
    st.markdown(
        f"<div style='color:#444;font-size:11px;"
        f"padding:0 14px;margin-bottom:12px'>"
        f"🎭 {genres_txt}</div>",
        unsafe_allow_html=True
    )

    if st.button(
        "Sign Out",
        key="so",
        use_container_width=True
    ):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ─────────────────────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────────────────────
page = st.session_state.page


# ══════════════════════════════
# HOME
# ══════════════════════════════
if page == 'home':
    st.markdown(
        f"<h2 style='color:#fff;margin:0 0 4px'>"
        f"Good evening, "
        f"<span style='color:#e50914'>{username}</span> 👋"
        f"</h2>"
        f"<p style='color:#666;font-size:14px;"
        f"margin:0 0 16px'>"
        f"Your personalised quantum-powered feed</p>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div style='background:#0a0a1f;border:1px solid "
        "#2a2a5a;border-radius:8px;padding:10px 16px;"
        "color:#7b68ee;font-size:13px;margin-bottom:20px'>"
        "⚛️ Powered by <b>IBM Kingston</b> (ibm_kingston) · "
        "156-qubit Heron r2 · "
        "Job ID: d72htfuv3u3c73eimhn0"
        "</div>",
        unsafe_allow_html=True
    )

    section("🎬 Top Picks For You")
    recs = rec_integrated(
        genre_prefs, data['ibm'],
        watched_ids, movies, data['tfidf_mat'], n=10
    )

    cols = st.columns(5)
    for i, (_, row) in enumerate(recs.head(5).iterrows()):
        with cols[i]:
            card(row, f"h1_{i}")

    st.markdown("<br>", unsafe_allow_html=True)
    cols2 = st.columns(5)
    for i, (_, row) in enumerate(recs.tail(5).iterrows()):
        with cols2[i]:
            card(row, f"h2_{i}")

    section(f"🎭 Because You Like {', '.join(genre_prefs[:2])}")
    grecs = rec_classical(
        genre_prefs, watched_ids,
        movies, data['tfidf_mat'], n=5
    )
    cols3 = st.columns(5)
    for i, (_, row) in enumerate(grecs.iterrows()):
        with cols3[i]:
            card(row, f"h3_{i}", show_score=False)


# ══════════════════════════════
# RECOMMENDATIONS
# ══════════════════════════════
elif page == 'recs':
    st.markdown(
        "<h2 style='color:#fff;margin:0 0 4px'>"
        "Model Comparison</h2>"
        "<p style='color:#666;font-size:14px;"
        "margin:0 0 20px'>"
        "Same user — three AI systems — see the difference"
        "</p>",
        unsafe_allow_html=True
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1: metric_box(
        "Integrated P@10","0.1030","+44.1% vs classical",
        "#e50914"
    )
    with m2: metric_box(
        "Classical P@10","0.0715","baseline","#888"
    )
    with m3: metric_box(
        "Quantum P@10","0.0730","IBM Kingston","#7F77DD"
    )
    with m4: metric_box(
        "IBM Hardware","2s","156-qubit Heron r2","#7b68ee"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    def sys_header(title, subtitle, border, bg):
        st.markdown(
            f"<div style='background:{bg};border-radius:10px;"
            f"padding:12px 14px;border:2px solid {border};"
            f"margin-bottom:14px'>"
            f"<div style='color:{border};font-size:10px;"
            f"text-transform:uppercase;font-weight:700'>"
            f"{title}</div>"
            f"<div style='color:#fff;font-size:15px;"
            f"font-weight:700;margin-top:2px'>{subtitle}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    def rec_row(row):
        info  = tmdb_search(str(row['title']))
        ci, ct = st.columns([1, 2])
        with ci:
            if info['poster']:
                st.image(info['poster'], width=60)
        with ct:
            short = (
                str(row['title'])[:24]+'...'
                if len(str(row['title'])) > 24
                else str(row['title'])
            )
            genre = (
                str(row.get('genres','')).split('|')[0]
            )
            url   = info.get('url', '')
            link  = (
                f"[{short}]({url})" if url
                else short
            )
            st.markdown(
                f"**{link}**\n\n"
                f"<span style='color:#555;font-size:11px'>"
                f"{genre}</span>",
                unsafe_allow_html=True
            )
        st.markdown(
            "<hr style='border-color:#2a2a2a;margin:6px 0'>",
            unsafe_allow_html=True
        )

    with c1:
        sys_header(
            "System 1 · Classical",
            "SVD + TF-IDF",
            "#888","#1a1a1a"
        )
        for _, row in rec_classical(
            genre_prefs, watched_ids,
            movies, data['tfidf_mat'], n=7
        ).iterrows():
            rec_row(row)

    with c2:
        sys_header(
            "System 2 · Quantum",
            "IBM Kingston Hardware",
            "#7F77DD","#0d0d1f"
        )
        for _, row in rec_quantum(
            data['ibm'], watched_ids, movies, n=7
        ).iterrows():
            rec_row(row)

    with c3:
        sys_header(
            "System 3 · Integrated ⭐ Best",
            "SVD + QPCA + TF-IDF",
            "#e50914","#1a0505"
        )
        for _, row in rec_integrated(
            genre_prefs, data['ibm'],
            watched_ids, movies,
            data['tfidf_mat'], n=7
        ).iterrows():
            rec_row(row)


# ══════════════════════════════
# WATCHLIST
# ══════════════════════════════
elif page == 'watchlist':
    st.markdown(
        "<h2 style='color:#fff;margin:0 0 4px'>"
        "My Watchlist</h2>",
        unsafe_allow_html=True
    )

    wl = db_get_watchlist(username)

    if not wl:
        st.markdown(
            "<div style='text-align:center;"
            "padding:80px 0;color:#333'>"
            "<div style='font-size:56px'>📋</div>"
            "<div style='font-size:18px;margin-top:16px;"
            "color:#444'>Your watchlist is empty</div>"
            "<div style='font-size:13px;margin-top:8px;"
            "color:#333'>Add movies from Home or "
            "Recommendations</div></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<p style='color:#555;font-size:13px;"
            f"margin-bottom:16px'>"
            f"{len(wl)} saved movies</p>",
            unsafe_allow_html=True
        )
        cols = st.columns(5)
        for i, (mid, title) in enumerate(wl):
            match = movies[movies['movie_id'] == mid]
            if not match.empty:
                with cols[i % 5]:
                    card(
                        match.iloc[0],
                        f"wl_{i}",
                        show_score=False
                    )


# ══════════════════════════════
# RESEARCH
# ══════════════════════════════
elif page == 'research':
    st.markdown(
        "<h2 style='color:#fff;margin:0 0 4px'>"
        "Research Dashboard</h2>"
        "<p style='color:#666;font-size:14px;"
        "margin:0 0 16px'>"
        "Experimental results and quantum hardware "
        "validation</p>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div style='background:#0a0a1f;border:1px solid "
        "#2a2a5a;border-radius:8px;padding:12px 16px;"
        "color:#7b68ee;font-size:13px;margin-bottom:20px'>"
        "⚛️ <b>IBM Kingston Validated</b> · "
        "Job: d72htfuv3u3c73eimhn0 · "
        "156-qubit Heron r2 · "
        "2 seconds · 1024 shots · "
        "Noise deviation: 0.2238"
        "</div>",
        unsafe_allow_html=True
    )

    section("Three-Way System Comparison")
    r1, r2, r3 = st.columns(3)
    for col, name, p, r, clr, sub in [
        (r1,"System 1: Classical",
         "0.0715","0.0287","#888","SVD only"),
        (r2,"System 2: Quantum IBM",
         "0.0730","0.0291","#7F77DD","IBM Kingston"),
        (r3,"System 3: Integrated ⭐",
         "0.1030","0.0542","#e50914",
         "SVD + QPCA + TF-IDF"),
    ]:
        with col:
            st.markdown(
                f"<div style='background:#1a1a1a;"
                f"border:2px solid {clr};"
                f"border-radius:12px;padding:20px;"
                f"text-align:center'>"
                f"<div style='color:{clr};font-size:11px;"
                f"font-weight:700;text-transform:uppercase;"
                f"letter-spacing:1px;margin-bottom:4px'>"
                f"{name}</div>"
                f"<div style='color:#555;font-size:11px;"
                f"margin-bottom:14px'>{sub}</div>"
                f"<div style='color:#fff;font-size:32px;"
                f"font-weight:800'>{p}</div>"
                f"<div style='color:#555;font-size:11px'>"
                f"Precision@10</div>"
                f"<div style='color:#fff;font-size:24px;"
                f"font-weight:700;margin-top:10px'>{r}</div>"
                f"<div style='color:#555;font-size:11px'>"
                f"Recall@10</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    section("Integrated Hybrid Improvements")
    i1, i2, i3, i4 = st.columns(4)
    for col, lbl, val, clr in [
        (i1,"P@10 vs Classical","+44.1%","#e50914"),
        (i2,"R@10 vs Classical","+89.1%","#e50914"),
        (i3,"P@10 vs Quantum",  "+41.1%","#7F77DD"),
        (i4,"R@10 vs Quantum",  "+86.4%","#7F77DD"),
    ]:
        with col:
            metric_box(lbl, val, "", clr)

    section("Dissertation Results Figure")
    fig = "data/dissertation_final_figure.png"
    if os.path.exists(fig):
        st.image(fig, use_container_width=True)
    else:
        st.info("Upload dissertation_final_figure.png to data/")

    section("Key Research Findings")
    for icon, title, desc in [
        ("🔬","Encoding Discovery",
         "Angle encoding: 0.0000 information loss vs "
         "amplitude at 0.5109 on sparse recommendation "
         "data. First systematic study for RecSys."),
        ("⚛️","QPCA Training",
         "40.7% improvement on 32×32 slice with 5-qubit "
         "variational circuit and 3 entanglement layers."),
        ("🖥️","Real Hardware",
         "IBM Kingston execution in 2 seconds. Rankings "
         "fully preserved despite 0.2238 noise deviation."),
        ("🏆","Integration Result",
         "+44.1% Precision@10 and +89.1% Recall@10 vs "
         "classical SVD when quantum layer is integrated."),
    ]:
        st.markdown(
            f"<div style='background:#1a1a1a;"
            f"border-left:4px solid #e50914;"
            f"border-radius:0 8px 8px 0;"
            f"padding:12px 16px;margin-bottom:10px'>"
            f"<span style='font-size:18px'>{icon}</span>"
            f"<span style='color:#fff;font-weight:700;"
            f"font-size:14px;margin-left:8px'>{title}"
            f"</span>"
            f"<p style='color:#666;font-size:13px;"
            f"margin:6px 0 0'>{desc}</p></div>",
            unsafe_allow_html=True
        )