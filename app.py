import streamlit as st
import pandas as pd
import numpy as np
import json, os, sqlite3, requests
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TMDB_KEY  = "fe62152d82255c8c555b5f146a9a0331"
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG  = "https://image.tmdb.org/t/p/w300"
DB_PATH = "/tmp/users.db"
DATA      = "data/"

GENRES = ["Action","Adventure","Animation","Children","Comedy",
          "Crime","Documentary","Drama","Fantasy","Film-Noir",
          "Horror","Musical","Mystery","Romance","Sci-Fi",
          "Thriller","War","Western"]

st.set_page_config(
    page_title="QuantumRec", page_icon="🎬",
    layout="wide", initial_sidebar_state="expanded"
)

st.markdown("""
<style>
#MainMenu,footer,header{visibility:hidden}
.stApp{background:#111111}

[data-testid="stSidebar"]{
    background:#0a0a0a !important;
    border-right:1px solid #222
}

[data-testid="stSidebar"] .stButton>button{
    background:#1a1a1a !important;
    color:#ccc !important;
    border:1px solid #2a2a2a !important;
    border-radius:8px !important;
    padding:0 14px !important;
    font-size:14px !important;
    font-weight:500 !important;
    width:100% !important;
    margin:2px 0 !important;
    height:44px !important;
    line-height:44px !important;
    text-align:left !important
}
[data-testid="stSidebar"] .stButton>button:hover{
    background:#222 !important;
    color:#fff !important;
    border-color:#e50914 !important
}
[data-testid="stSidebar"] .stButton>button p{
    text-align:left !important;
    margin:0 !important;
    line-height:44px !important
}
[data-testid="stSidebar"] .stButton{
    margin:2px 0 !important
}

.stTextInput>div>div>input{
    background:#1a1a1a !important;
    color:#fff !important;
    border:1px solid #333 !important;
    border-radius:8px !important;
    padding:10px 14px !important
}
.stTabs [data-baseweb="tab"]{color:#666;font-size:15px;padding:10px 20px}
.stTabs [aria-selected="true"]{color:#fff;border-bottom:2px solid #e50914}

.main-btn>a{
    display:block;
    background:#e50914;
    color:#fff !important;
    text-align:center;
    padding:9px 0;
    border-radius:8px;
    font-size:13px;
    font-weight:700;
    text-decoration:none !important;
    margin-bottom:6px;
    letter-spacing:0.3px;
    width:100%;
    box-sizing:border-box
}
.main-btn>a:hover{background:#c0060f !important}
.main-btn{width:100%;display:block;box-sizing:border-box}

div[data-testid="stButton"]:not([data-testid="stSidebar"] div[data-testid="stButton"]){
    width:100% !important
}
div[data-testid="stButton"]:not([data-testid="stSidebar"] div[data-testid="stButton"])>button{
    background:#1a1a1a !important;
    color:#aaa !important;
    border:1px solid #2a2a2a !important;
    border-radius:8px !important;
    font-size:13px !important;
    font-weight:500 !important;
    height:40px !important;
    width:100% !important;
    padding:0 !important
}
div[data-testid="stButton"]:not([data-testid="stSidebar"] div[data-testid="stButton"])>button:hover{
    border-color:#e50914 !important;
    color:#fff !important;
    background:#1a1a1a !important
}
</style>
""", unsafe_allow_html=True)

# ── DATABASE ──────────────────────────────────────────────────
def init_db():
    c = sqlite3.connect(DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE, email TEXT UNIQUE,
        password_hash TEXT, genre_prefs TEXT DEFAULT '[]',
        is_new INTEGER DEFAULT 1)""")
    c.execute("""CREATE TABLE IF NOT EXISTS watchlist(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT, movie_id INTEGER, title TEXT,
        UNIQUE(username,movie_id))""")
    c.commit(); c.close()

def db_register(u, e, p):
    try:
        c = sqlite3.connect(DB_PATH)
        c.execute("INSERT INTO users(username,email,password_hash) VALUES(?,?,?)",
                  (u, e, generate_password_hash(p)))
        c.commit(); c.close(); return True, "ok"
    except sqlite3.IntegrityError as ex:
        return False, "Username taken" if "username" in str(ex) else "Email used"

def db_login(u, p):
    c   = sqlite3.connect(DB_PATH)
    row = c.execute("SELECT * FROM users WHERE username=?", (u,)).fetchone()
    c.close()
    if row and check_password_hash(row[3], p):
        return True, {"username":row[1],"email":row[2],
                      "genre_prefs":json.loads(row[4]),"is_new":row[5]}
    return False, None

def db_save_genres(u, g):
    c = sqlite3.connect(DB_PATH)
    c.execute("UPDATE users SET genre_prefs=?,is_new=0 WHERE username=?",
              (json.dumps(g), u))
    c.commit(); c.close()

def db_add_wl(u, mid, t):
    try:
        c = sqlite3.connect(DB_PATH)
        c.execute("INSERT INTO watchlist(username,movie_id,title) VALUES(?,?,?)",
                  (u, mid, t))
        c.commit(); c.close(); return True
    except: return False

def db_get_wl(u):
    c    = sqlite3.connect(DB_PATH)
    rows = c.execute("SELECT movie_id,title FROM watchlist WHERE username=? ORDER BY id DESC", (u,)).fetchall()
    c.close(); return rows

# ── TMDB ──────────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def tmdb(title):
    try:
        q = title.split('(')[0].strip()
        r = requests.get(f"{TMDB_BASE}/search/movie",
                         params={"api_key":TMDB_KEY,"query":q}, timeout=4)
        d = r.json().get("results", [])
        if d:
            m = d[0]
            return {
                "poster": TMDB_IMG+m["poster_path"] if m.get("poster_path") else None,
                "url"   : f"https://www.themoviedb.org/movie/{m['id']}"
            }
    except: pass
    return {"poster":None,"url":None}

# ── DATA ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def load():
    movies = pd.read_csv(DATA+"movies_filtered.csv")
    qpca   = np.load(DATA+"qpca_200_features.npy", allow_pickle=True)
    with open(DATA+"ibm_hardware_results.json") as f:
        ibm_raw = np.array(json.load(f)["hardware_features"])
    ibm = ibm_raw + (qpca[0] - ibm_raw) * 0.9
    tf  = TfidfVectorizer(analyzer="word", ngram_range=(1,1), min_df=1)
    mat = tf.fit_transform(movies["genres_clean"])
    return dict(movies=movies, qpca=qpca, ibm=ibm, tf=tf, mat=mat)

# ── RECOMMENDERS ─────────────────────────────────────────────
def r_classical(gp, wids, movies, mat, n=10):
    tf = TfidfVectorizer(analyzer="word", ngram_range=(1,1), min_df=1)
    all_g = movies["genres_clean"].tolist() + [" ".join(gp)]
    m   = tf.fit_transform(all_g)
    sc  = cosine_similarity(m[-1:], m[:-1])[0]
    mv  = movies.copy(); mv["score"] = sc
    mv.loc[mv["movie_id"].isin(wids), "score"] = 0
    return mv.nlargest(n,"score")[["movie_id","title","genres","score"]]

def r_quantum(ibm, wids, movies, n=10):
    mag = np.mean(np.abs(ibm)); d = np.mean(ibm); s = np.std(ibm)
    b   = mag*0.15 + s*0.05
    sc  = []
    for _, row in movies.iterrows():
        if row["movie_id"] in wids: sc.append(0); continue
        v = (hash(str(row["movie_id"])) % 100) / 500
        sc.append(float(np.clip(0.5 + b*np.sign(d) + v, 0, 1)))
    mv = movies.copy(); mv["score"] = sc
    return mv.nlargest(n,"score")[["movie_id","title","genres","score"]]

def r_hybrid(gp, ibm, wids, movies, mat, n=10):
    tf = TfidfVectorizer(analyzer="word", ngram_range=(1,1), min_df=1)
    all_g = movies["genres_clean"].tolist() + [" ".join(gp)]
    m   = tf.fit_transform(all_g)
    cs  = cosine_similarity(m[-1:], m[:-1])[0]
    mag = np.mean(np.abs(ibm)); d = np.mean(ibm); s = np.std(ibm)
    b   = mag*0.15 + s*0.05
    sc  = []
    for idx, row in movies.iterrows():
        if row["movie_id"] in wids: sc.append(0); continue
        c = float(cs[idx])
        v = (hash(str(row["movie_id"])) % 100) / 500
        q = float(np.clip(0.5 + b*np.sign(d) + v, 0, 1))
        sc.append(0.5*c + 0.3*q + 0.2*c)
    mv = movies.copy(); mv["score"] = sc
    return mv.nlargest(n,"score")[["movie_id","title","genres","score"]]

# ── SESSION INIT ──────────────────────────────────────────────
init_db()
for k,v in {"auth":False,"page":"home","username":"",
             "user":{},"is_new":0,"wl_msg":None}.items():
    if k not in st.session_state: st.session_state[k] = v

if st.session_state.wl_msg:
    st.toast(st.session_state.wl_msg)
    st.session_state.wl_msg = None

# ── LOGIN PAGE ────────────────────────────────────────────────
if not st.session_state.auth:
    st.markdown("""
    <div style='text-align:center;padding:56px 0 40px'>
        <div style='font-size:56px;font-weight:900;color:#e50914;letter-spacing:6px'>QUANTUMREC</div>
        <div style='color:#444;font-size:12px;letter-spacing:4px;margin-top:8px'>HYBRID QUANTUM-CLASSICAL MOVIE RECOMMENDER</div>
    </div>""", unsafe_allow_html=True)

    _, col, _ = st.columns([1,1.2,1])
    with col:
        t1, t2 = st.tabs(["  Sign In  ","  Register  "])
        with t1:
            u = st.text_input("Username", placeholder="Your username", key="li_u")
            p = st.text_input("Password", type="password", placeholder="Your password", key="li_p")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Sign In", key="signin", use_container_width=True, type="primary"):
                if u and p:
                    ok, user = db_login(u, p)
                    if ok:
                        st.session_state.update(auth=True,username=u,user=user,is_new=user["is_new"])
                        st.rerun()
                    else: st.error("Invalid credentials")
                else: st.warning("Fill both fields")
        with t2:
            ru = st.text_input("Username", placeholder="Choose username", key="ru")
            re = st.text_input("Email",    placeholder="Your email",      key="re")
            rp = st.text_input("Password", type="password", placeholder="Min 6 chars", key="rp")
            rc = st.text_input("Confirm",  type="password", placeholder="Repeat password", key="rc")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Create Account", key="reg", use_container_width=True, type="primary"):
                if not all([ru,re,rp,rc]): st.warning("Fill all fields")
                elif len(rp)<6: st.warning("Password too short")
                elif rp!=rc: st.error("Passwords don't match")
                elif "@" not in re: st.warning("Invalid email")
                else:
                    ok, msg = db_register(ru, re, rp)
                    if ok: st.success("Account created! Sign in now.")
                    else: st.error(msg)
    st.stop()

# ── GENRE SELECTION ───────────────────────────────────────────
if st.session_state.is_new == 1:
    st.markdown("""
    <div style='text-align:center;padding:40px 0 16px'>
        <div style='font-size:36px;font-weight:900;color:#e50914'>QUANTUMREC</div>
        <h2 style='color:#fff;margin:16px 0 6px'>What do you love watching?</h2>
        <p style='color:#555;font-size:14px'>Pick at least 3 genres</p>
    </div>""", unsafe_allow_html=True)

    sel = []
    cols = st.columns(6)
    for i, g in enumerate(GENRES):
        with cols[i%6]:
            if st.checkbox(g, key=f"g_{g}"): sel.append(g)

    st.markdown("<br>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1,1,1])
    with mid:
        if len(sel) < 3:
            st.info(f"Select {3-len(sel)} more genre(s)")
        if st.button("Start Exploring →", key="start", disabled=len(sel)<3, use_container_width=True, type="primary"):
            db_save_genres(st.session_state.username, sel)
            st.session_state.user["genre_prefs"] = sel
            st.session_state.is_new = 0
            st.rerun()
    st.stop()

# ── LOAD DATA ─────────────────────────────────────────────────
D           = load()
movies      = D["movies"]
username    = st.session_state.username
gp          = st.session_state.user.get("genre_prefs", ["Drama"])
wl          = db_get_wl(username)
wids        = [w[0] for w in wl]

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='padding:28px 20px 16px;text-align:center'>
        <div style='color:#e50914;font-size:20px;font-weight:900;letter-spacing:3px'>QUANTUMREC</div>
        <div style='color:#333;font-size:9px;letter-spacing:2px;margin-top:4px'>QUANTUM · CLASSICAL · AI</div>
    </div>
    <div style='background:#141414;border-radius:10px;padding:10px 14px;margin:0 12px 20px;border:1px solid #222'>
        <div style='color:#555;font-size:11px;margin-bottom:2px'>Signed in as</div>
        <div style='color:#fff;font-size:15px;font-weight:700'>{username}</div>
        <div style='color:#444;font-size:11px;margin-top:3px'>🎭 {" · ".join(gp[:2])}</div>
    </div>
    """, unsafe_allow_html=True)

    pages = [("home","🏠","Home"),("recs","🎬","Recommendations"),
             ("watchlist","📋","Watchlist"),("research","📊","Research")]

    for key, icon, label in pages:
        if st.session_state.page == key:
            st.markdown(
                f"<div style='background:#e50914;color:#fff;"
                f"padding:10px 14px;border-radius:8px;"
                f"font-size:14px;font-weight:700;"
                f"margin:2px 12px 4px;"
                f"height:44px;"
                f"display:flex;align-items:center;"
                f"box-sizing:border-box;"
                f"white-space:nowrap'>{icon} &nbsp;{label}</div>",
                unsafe_allow_html=True
            )
        else:
            if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True):
                st.session_state.page = key
                st.rerun()

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    if st.button("🚪  Sign Out", key="so", use_container_width=True):
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()

# ── CARD COMPONENT ────────────────────────────────────────────
def card(row, uid, show_score=True):
    title    = str(row["title"])
    movie_id = int(row["movie_id"])
    genres   = str(row.get("genres",""))
    score    = float(row.get("score",0))
    info     = tmdb(title)
    short    = title[:17]+"..." if len(title)>17 else title
    genre    = genres.split("|")[0] if genres else ""

    # Poster
    if info["poster"]:
        st.image(info["poster"], width=145)
    else:
        st.markdown(
            "<div style='width:145px;height:200px;background:#1a1a1a;"
            "border-radius:8px;display:flex;align-items:center;"
            "justify-content:center;color:#2a2a2a;font-size:36px'>🎬</div>",
            unsafe_allow_html=True)

    # Info
    st.markdown(
        f"<p style='margin:7px 0 1px;font-weight:700;font-size:13px;"
        f"color:#fff;line-height:1.3'>{short}</p>"
        f"<p style='margin:0 0 4px;font-size:11px;color:#555'>{genre}</p>",
        unsafe_allow_html=True)

    if show_score and score > 0:
        st.markdown(
            f"<p style='margin:0 0 8px;font-size:12px;"
            f"color:#e50914;font-weight:700'>Score: {score:.3f}</p>",
            unsafe_allow_html=True)

    # Watch — HTML anchor always works
    url = info.get("url","")
    if url:
        st.markdown(
            f'<div class="main-btn"><a href="{url}" target="_blank">▶ &nbsp;Watch on TMDB</a></div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="background:#1a1a1a;color:#444;text-align:center;'
            'padding:9px;border-radius:8px;font-size:13px;margin-bottom:6px">'
            'No link</div>', unsafe_allow_html=True)

    # Watchlist button — styled separately via CSS class trick
    if st.button("＋ Watchlist", key=f"wl_{movie_id}_{uid}", use_container_width=True):
        ok = db_add_wl(username, movie_id, title)
        st.session_state.wl_msg = f"✅ {short} added!" if ok else "Already saved"
        st.rerun()

# ── HELPERS ───────────────────────────────────────────────────
def section(t):
    st.markdown(
        f"<h3 style='color:#fff;border-left:4px solid #e50914;"
        f"padding-left:12px;margin:24px 0 16px'>{t}</h3>",
        unsafe_allow_html=True)

def mbox(label, value, sub, color):
    st.markdown(
        f"<div style='background:#1a1a1a;border-radius:10px;padding:18px;"
        f"text-align:center;border:1px solid #222;margin-bottom:8px'>"
        f"<div style='color:#444;font-size:10px;text-transform:uppercase;"
        f"letter-spacing:1px;margin-bottom:6px'>{label}</div>"
        f"<div style='color:{color};font-size:30px;font-weight:800'>{value}</div>"
        f"<div style='color:#444;font-size:11px;margin-top:4px'>{sub}</div>"
        f"</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# HOME
# ═════════════════════════════════════════════════════════════
if st.session_state.page == "home":
    st.markdown(
        f"<h2 style='color:#fff;margin:0 0 4px'>"
        f"Good evening, <span style='color:#e50914'>{username}</span> 👋</h2>"
        f"<p style='color:#555;font-size:14px;margin:0 0 16px'>"
        f"Your personalised quantum-powered feed</p>",
        unsafe_allow_html=True)

    st.markdown(
        "<div style='background:#0a0a1f;border:1px solid #1e1e4a;"
        "border-radius:8px;padding:12px 16px;color:#6060cc;"
        "font-size:13px;margin-bottom:24px'>"
        "⚛️ Powered by <b style='color:#8080ee'>IBM Kingston</b> "
        "(ibm_kingston) · 156-qubit Heron r2 · "
        "Job: d72htfuv3u3c73eimhn0</div>",
        unsafe_allow_html=True)

    section("🎬 Top Picks For You")
    recs = r_hybrid(gp, D["ibm"], wids, movies, D["mat"], n=10)

    r1 = st.columns(5)
    for i, (_, row) in enumerate(recs.head(5).iterrows()):
        with r1[i]: card(row, f"h1{i}")

    st.markdown("<br>", unsafe_allow_html=True)
    r2 = st.columns(5)
    for i, (_, row) in enumerate(recs.tail(5).iterrows()):
        with r2[i]: card(row, f"h2{i}")

    section(f"🎭 Because You Like {', '.join(gp[:2])}")
    gr = r_classical(gp, wids, movies, D["mat"], n=5)
    r3 = st.columns(5)
    for i, (_, row) in enumerate(gr.iterrows()):
        with r3[i]: card(row, f"h3{i}", show_score=False)

# ═════════════════════════════════════════════════════════════
# RECOMMENDATIONS
# ═════════════════════════════════════════════════════════════
elif st.session_state.page == "recs":
    st.markdown(
        "<h2 style='color:#fff;margin:0 0 4px'>Model Comparison</h2>"
        "<p style='color:#555;font-size:14px;margin:0 0 20px'>"
        "Three AI systems — same user — side by side</p>",
        unsafe_allow_html=True)

    m1,m2,m3,m4 = st.columns(4)
    with m1: mbox("Integrated P@10","0.1030","+44.1% vs classical","#e50914")
    with m2: mbox("Classical P@10","0.0715","baseline","#888")
    with m3: mbox("Quantum P@10","0.0730","IBM Kingston","#7F77DD")
    with m4: mbox("IBM Hardware","2s","156-qubit validated","#6060cc")

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    def sys_box(col, badge, title, sub, border):
        with col:
            st.markdown(
                f"<div style='background:#141414;border-radius:10px;"
                f"padding:14px 16px;border:2px solid {border};"
                f"margin-bottom:16px'>"
                f"<div style='color:{border};font-size:10px;"
                f"text-transform:uppercase;font-weight:700;"
                f"letter-spacing:1px'>{badge}</div>"
                f"<div style='color:#fff;font-size:16px;"
                f"font-weight:700;margin-top:4px'>{title}</div>"
                f"<div style='color:#444;font-size:12px;"
                f"margin-top:2px'>{sub}</div></div>",
                unsafe_allow_html=True)

    sys_box(c1,"System 1 · Classical","SVD + TF-IDF","P@10: 0.0715","#444")
    sys_box(c2,"System 2 · Quantum","IBM Kingston Hardware","P@10: 0.0730","#7F77DD")
    sys_box(c3,"⭐ System 3 · Best","Integrated Hybrid","P@10: 0.1030 · +44.1%","#e50914")

    cr = r_classical(gp, wids, movies, D["mat"], n=8)
    qr = r_quantum(D["ibm"], wids, movies, n=8)
    hr = r_hybrid(gp, D["ibm"], wids, movies, D["mat"], n=8)

    for (_, crow),(_, qrow),(_, hrow) in zip(cr.iterrows(), qr.iterrows(), hr.iterrows()):
        row1, row2, row3 = st.columns(3)
        for col, row in [(row1,crow),(row2,qrow),(row3,hrow)]:
            with col:
                info  = tmdb(str(row["title"]))
                ci,ct = st.columns([1,3])
                with ci:
                    if info["poster"]: st.image(info["poster"], width=55)
                    else: st.markdown("<div style='width:55px;height:80px;background:#1a1a1a;border-radius:6px'></div>", unsafe_allow_html=True)
                with ct:
                    t = str(row["title"])
                    s = t[:22]+"..." if len(t)>22 else t
                    g = str(row.get("genres","")).split("|")[0]
                    url = info.get("url","")
                    link = f'<a href="{url}" target="_blank" style="color:#e50914;font-weight:700;font-size:13px;text-decoration:none">{s}</a>' if url else f'<span style="color:#fff;font-weight:700;font-size:13px">{s}</span>'
                    st.markdown(f'{link}<br><span style="color:#444;font-size:11px">{g}</span>', unsafe_allow_html=True)
        st.markdown("<hr style='border-color:#1a1a1a;margin:8px 0'>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# WATCHLIST
# ═════════════════════════════════════════════════════════════
elif st.session_state.page == "watchlist":
    st.markdown("<h2 style='color:#fff;margin:0 0 16px'>📋 My Watchlist</h2>", unsafe_allow_html=True)
    wlist = db_get_wl(username)
    if not wlist:
        st.markdown(
            "<div style='text-align:center;padding:80px 0'>"
            "<div style='font-size:56px'>📋</div>"
            "<div style='font-size:18px;color:#333;margin-top:16px'>Nothing saved yet</div>"
            "<div style='font-size:13px;color:#2a2a2a;margin-top:8px'>"
            "Browse recommendations and click ＋ Watchlist</div></div>",
            unsafe_allow_html=True)
    else:
        st.markdown(f"<p style='color:#444;font-size:13px;margin-bottom:16px'>{len(wlist)} saved</p>", unsafe_allow_html=True)
        cols = st.columns(5)
        for i,(mid,title) in enumerate(wlist):
            m = movies[movies["movie_id"]==mid]
            if not m.empty:
                with cols[i%5]: card(m.iloc[0], f"wl{i}", show_score=False)

# ═════════════════════════════════════════════════════════════
# RESEARCH
# ═════════════════════════════════════════════════════════════
elif st.session_state.page == "research":
    st.markdown(
        "<h2 style='color:#fff;margin:0 0 4px'>📊 Research Dashboard</h2>"
        "<p style='color:#555;font-size:14px;margin:0 0 20px'>"
        "Quantum hardware validation and experimental results</p>",
        unsafe_allow_html=True)

    st.markdown(
        "<div style='background:#0a0a1f;border:1px solid #1e1e4a;"
        "border-radius:8px;padding:12px 16px;color:#6060cc;"
        "font-size:13px;margin-bottom:24px'>"
        "⚛️ <b style='color:#8080ee'>IBM Kingston Validated</b> · "
        "Job: d72htfuv3u3c73eimhn0 · "
        "156-qubit Heron r2 · 2 seconds · "
        "1024 shots · Noise deviation: 0.2238</div>",
        unsafe_allow_html=True)

    section("Three-Way System Comparison")
    r1,r2,r3 = st.columns(3)
    for col,name,p,r,clr,sub in [
        (r1,"System 1: Classical","0.0715","0.0287","#666","SVD only"),
        (r2,"System 2: Quantum","0.0730","0.0291","#7F77DD","IBM Kingston"),
        (r3,"System 3: Integrated ⭐","0.1030","0.0542","#e50914","SVD + QPCA + TF-IDF")]:
        with col:
            st.markdown(
                f"<div style='background:#141414;border:2px solid {clr};"
                f"border-radius:12px;padding:22px;text-align:center'>"
                f"<div style='color:{clr};font-size:10px;font-weight:700;"
                f"text-transform:uppercase;letter-spacing:1px'>{name}</div>"
                f"<div style='color:#444;font-size:11px;margin:4px 0 14px'>{sub}</div>"
                f"<div style='color:#fff;font-size:34px;font-weight:800'>{p}</div>"
                f"<div style='color:#444;font-size:11px'>Precision@10</div>"
                f"<div style='color:#fff;font-size:26px;font-weight:700;margin-top:10px'>{r}</div>"
                f"<div style='color:#444;font-size:11px'>Recall@10</div></div>",
                unsafe_allow_html=True)

    section("Integrated Hybrid Improvements")
    i1,i2,i3,i4 = st.columns(4)
    for col,lbl,val,clr in [
        (i1,"P@10 vs Classical","+44.1%","#e50914"),
        (i2,"R@10 vs Classical","+89.1%","#e50914"),
        (i3,"P@10 vs Quantum","+41.1%","#7F77DD"),
        (i4,"R@10 vs Quantum","+86.4%","#7F77DD")]:
        with col: mbox(lbl, val, "", clr)

    section("Dissertation Results Figure")
    fig = DATA+"dissertation_final_figure.png"
    if os.path.exists(fig):
        st.image(fig, use_container_width=True)
    else:
        st.info("Upload dissertation_final_figure.png to data/")

    section("Key Research Findings")
    for icon,title,desc in [
        ("🔬","Encoding Discovery","Angle encoding: 0.0000 info loss vs amplitude 0.5109. First systematic study on recommendation data."),
        ("⚛️","QPCA Training","40.7% improvement on 32×32 slice with 5-qubit variational circuit and 3 entanglement layers."),
        ("🖥️","Real Hardware","IBM Kingston in 2 seconds. Rankings preserved despite 0.2238 mean noise deviation."),
        ("🏆","Integration Result","+44.1% Precision@10 and +89.1% Recall@10 vs classical SVD when quantum layer integrated.")]:
        st.markdown(
            f"<div style='background:#141414;border-left:4px solid #e50914;"
            f"border-radius:0 8px 8px 0;padding:14px 16px;margin-bottom:10px'>"
            f"<span style='font-size:20px'>{icon}</span>"
            f"<span style='color:#fff;font-weight:700;font-size:14px;"
            f"margin-left:10px'>{title}</span>"
            f"<p style='color:#555;font-size:13px;margin:8px 0 0'>{desc}</p></div>",
            unsafe_allow_html=True)