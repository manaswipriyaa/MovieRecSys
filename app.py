import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os, re, requests, urllib.parse

st.set_page_config(page_title="CineMatch", layout="wide", initial_sidebar_state="collapsed")

# ── query-param router ────────────────────────────────────────────
qp = st.query_params.to_dict()
if "movie" in qp:
    st.session_state["movie"] = urllib.parse.unquote(qp["movie"])
    st.session_state["prev"]  = qp.get("prev", "home")
    st.query_params.clear(); st.rerun()
if "nav" in qp:
    dest = qp["nav"]
    st.session_state["page"]  = "home" if dest == "logo" else (dest if dest in ("home","recs","watchlist") else "home")
    st.session_state["movie"] = None
    st.query_params.clear(); st.rerun()

for k, v in {"page":"home","prev":"home","movie":None,"genres":[],
             "recs":None,"watchlist":[],"rec_mode":"genre","search_q":""}.items():
    if k not in st.session_state: st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────
# CSS — visual styles only, NO layout padding (done via st.columns)
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=Outfit:wght@300;400;500;600&display=swap');

:root {
  --bg:#08090e; --surf:#0e0f18; --surf2:#141520;
  --bdr:rgba(255,255,255,0.07);
  --gold:#c9a96e; --gold2:#e8c992; --gdim:rgba(201,169,110,0.11);
  --txt:#eaeaf5; --muted:#5a5a72; --subtle:#2a2a3e;
}
*, *::before, *::after { box-sizing:border-box; margin:0; padding:0; }
html, body, [class*="css"] { font-family:'Outfit',sans-serif; -webkit-font-smoothing:antialiased; }
.stApp { background:var(--bg) !important; color:var(--txt); }
.block-container { padding:0 !important; max-width:100% !important; }
section[data-testid="stMain"] > div { padding:0 !important; }
div[data-testid="stSidebar"], footer, header { display:none !important; }
.stVerticalBlock { gap:0 !important; }

::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--subtle); border-radius:3px; }

/* NAV */
.nav {
  position:sticky; top:0; z-index:9999;
  background:rgba(8,9,14,0.97);
  backdrop-filter:blur(18px); -webkit-backdrop-filter:blur(18px);
  border-bottom:1px solid var(--bdr);
}
.nav-inner {
  padding:0 48px; height:62px;
  display:flex; align-items:center; gap:0;
}
.nav-logo {
  font-family:'Playfair Display',serif; font-weight:900;
  font-size:1.05rem; letter-spacing:5px; text-transform:uppercase;
  color:var(--gold); white-space:nowrap; flex-shrink:0; margin-right:40px;
  text-decoration:none; cursor:pointer;
}
.nav-sep { width:1px; height:18px; background:var(--bdr); flex-shrink:0; margin-right:6px; }
.nav-links { display:flex; align-items:center; gap:2px; }
.nav-link {
  font-size:0.7rem; font-weight:500; letter-spacing:2px; text-transform:uppercase;
  color:var(--muted); text-decoration:none; padding:8px 18px; border-radius:5px;
  white-space:nowrap; transition:color .15s, background .15s; cursor:pointer;
}
.nav-link:hover { color:var(--gold); background:rgba(201,169,110,0.07); }
.nav a, .nav-logo, .nav-link, .nav-cta { text-decoration:none !important; }
.nav a:hover, .nav-logo:hover, .nav-link:hover, .nav-cta:hover { text-decoration:none !important; }
.nav-link.active { color:var(--gold2); }
.nav-badge {
  display:inline-block; background:var(--gold); color:#08090e;
  font-size:0.55rem; font-weight:700; border-radius:8px;
  padding:1px 7px; margin-left:5px; vertical-align:middle;
}
.nav-cta {
  background:var(--gold); color:#08090e !important;
  font-family:'Outfit',sans-serif; font-size:0.7rem; font-weight:700;
  letter-spacing:1.5px; text-transform:uppercase; text-decoration:none;
  padding:8px 20px; border-radius:6px; white-space:nowrap; flex-shrink:0;
  transition:opacity .15s; cursor:pointer;
}
.nav-cta:hover { opacity:0.85; }

/* HERO */
.hero {
  background:linear-gradient(155deg,#0c0b1c 0%,#0f0c1f 55%,#080912 100%);
  border-bottom:1px solid var(--bdr); position:relative; overflow:hidden;
}
.hero::before {
  content:''; position:absolute; top:-80px; right:0;
  width:400px; height:400px; border-radius:50%; pointer-events:none;
  background:radial-gradient(circle,rgba(201,169,110,.08) 0%,transparent 65%);
}
.hero-body { padding:52px 0 44px; }
.hero-eye { font-size:.58rem; letter-spacing:4px; text-transform:uppercase; color:var(--gold); font-weight:500; margin-bottom:14px; }
.hero-h { font-family:'Playfair Display',serif; font-weight:900; font-size:clamp(2rem,3.5vw,3rem); color:#fff; line-height:1.1; margin-bottom:14px; }
.hero-h em { font-style:italic; color:var(--gold); }
.hero-p { font-size:.88rem; color:var(--muted); line-height:1.75; max-width:420px; font-weight:300; }

/* BUTTONS */
.stButton > button {
  background:var(--gold) !important; color:#08090e !important;
  border:none !important; border-radius:6px !important;
  font-family:'Outfit',sans-serif !important; font-size:.72rem !important;
  font-weight:600 !important; letter-spacing:1.5px !important;
  text-transform:uppercase !important; padding:10px 26px !important;
  box-shadow:0 4px 16px rgba(201,169,110,.2) !important;
  width:auto !important; min-width:0 !important; white-space:nowrap !important;
  transition:opacity .15s !important; transform:none !important;
}
.stButton > button:hover { opacity:.85 !important; transform:none !important; }
.btn-ghost .stButton > button {
  background:transparent !important; color:var(--muted) !important;
  border:1px solid rgba(255,255,255,.13) !important; box-shadow:none !important;
  text-transform:none !important; letter-spacing:1px !important;
  font-size:.72rem !important; padding:9px 22px !important;
}
.btn-ghost .stButton > button:hover { color:var(--txt) !important; border-color:rgba(255,255,255,.28) !important; }
.btn-danger .stButton > button {
  background:transparent !important; color:#e05c5c !important;
  border:1px solid rgba(224,92,92,.28) !important; box-shadow:none !important;
  font-size:.72rem !important; padding:9px 22px !important;
  text-transform:none !important; letter-spacing:.5px !important;
}
.stDownloadButton > button {
  background:transparent !important; color:var(--muted) !important;
  border:1px solid rgba(255,255,255,.1) !important; box-shadow:none !important;
  font-size:.7rem !important; padding:8px 18px !important;
}

/* DIVIDER */
.divider { height:1px; background:var(--bdr); }

/* SECTION LABELS */
.sec-eye { font-size:.58rem; letter-spacing:3.5px; text-transform:uppercase; color:var(--muted); font-weight:500; margin-bottom:5px; }
.sec-h { font-family:'Playfair Display',serif; font-weight:700; font-size:1.22rem; color:#fff; margin-bottom:22px; letter-spacing:-.3px; }
.count { font-size:.6rem; color:var(--muted); letter-spacing:2px; font-weight:500; margin:14px 0 18px; }

/* MOVIE CARD */
a.mc { display:block; text-decoration:none; color:inherit; }
.mcard {
  border-radius:8px; overflow:hidden; background:var(--surf); border:1px solid var(--bdr);
  transition:transform .22s cubic-bezier(.22,.68,0,1.2), box-shadow .22s, border-color .2s;
  position:relative; cursor:pointer;
}
a.mc:hover .mcard {
  transform:translateY(-7px) scale(1.012);
  box-shadow:0 20px 50px rgba(0,0,0,.8), 0 0 0 1px rgba(201,169,110,.22);
  border-color:rgba(201,169,110,.3);
}
.mcard-img { width:100%; aspect-ratio:2/3; object-fit:cover; display:block; }
.mcard-ph { width:100%; aspect-ratio:2/3; background:linear-gradient(160deg,var(--surf),var(--surf2)); display:flex; align-items:center; justify-content:center; font-size:2rem; color:var(--muted); }
.mcard-ov {
  position:absolute; inset:0;
  background:linear-gradient(to top,rgba(8,9,14,.97) 0%,rgba(8,9,14,.4) 36%,transparent 66%);
  opacity:0; transition:opacity .22s;
  display:flex; flex-direction:column; justify-content:flex-end; padding:12px 10px 11px;
}
a.mc:hover .mcard-ov { opacity:1; }
.mcard-ov-t { font-family:'Playfair Display',serif; font-size:.73rem; font-weight:700; color:#fff; line-height:1.2; margin-bottom:2px; }
.mcard-ov-s { font-size:.59rem; color:var(--gold); }
.mcard-body { padding:9px 10px 10px; }
.mcard-title { font-size:.71rem; font-weight:500; color:#c0c0d8; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-bottom:2px; }
.mcard-genre { font-size:.59rem; color:var(--muted); }

/* BOTTOM BANNER */


/* DETAIL */
.det-backdrop { width:100%; height:210px; position:relative; overflow:hidden; }
.det-backdrop img { width:100%; height:100%; object-fit:cover; opacity:.18; display:block; }
.det-fade { position:absolute; inset:0; background:linear-gradient(to top,var(--bg) 0%,transparent 55%); }
.det-flex { display:flex; gap:40px; align-items:flex-start; margin-top:-72px; position:relative; z-index:2; }
.det-poster { width:175px; min-width:175px; border-radius:10px; box-shadow:0 24px 60px rgba(0,0,0,.88); display:block; }
.det-poster-ph { width:175px; min-width:175px; height:262px; border-radius:10px; background:var(--surf2); display:flex; align-items:center; justify-content:center; font-size:2.8rem; }
.det-info { flex:1; padding-top:0; }
.det-title { font-family:'Playfair Display',serif; font-weight:900; font-size:clamp(1.5rem,3vw,2.5rem); color:#fff; line-height:1.08; margin-bottom:12px; }
.det-meta { display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin-bottom:13px; }
.det-rating { background:rgba(201,169,110,.12); color:var(--gold2); font-size:.73rem; font-weight:600; padding:3px 12px; border-radius:5px; border:1px solid rgba(201,169,110,.22); }
.det-year, .det-rt { font-size:.77rem; color:var(--muted); }
.det-tagline { font-style:italic; color:var(--muted); font-size:.83rem; margin-bottom:14px; }
.det-ov { font-size:.87rem; color:#7878a0; line-height:1.8; max-width:600px; margin-bottom:20px; font-weight:300; }
.det-pill { display:inline-block; background:rgba(255,255,255,.04); color:#aaa; font-size:.67rem; padding:3px 11px; border-radius:4px; border:1px solid rgba(255,255,255,.07); margin-right:5px; margin-bottom:5px; }
.trailer-a { display:inline-flex; align-items:center; gap:8px; background:rgba(224,92,92,.1); color:#ef8080; font-size:.73rem; font-weight:600; padding:9px 20px; border-radius:6px; border:1px solid rgba(224,92,92,.2); text-decoration:none; }
.trailer-a:hover { background:rgba(224,92,92,.18); }
.sub-h { font-family:'Playfair Display',serif; font-weight:700; font-size:.95rem; color:#fff; margin-bottom:16px; margin-top:28px; }
.sub-divider { height:1px; background:var(--bdr); margin:0 0 26px 0; }
.providers { display:flex; flex-wrap:wrap; gap:10px; }
.prov { background:var(--surf); border-radius:9px; border:1px solid var(--bdr); padding:12px 14px; display:flex; flex-direction:column; align-items:center; gap:5px; min-width:84px; text-decoration:none; transition:border-color .15s, transform .15s; }
.prov:hover { border-color:rgba(201,169,110,.32); transform:translateY(-2px); }
.prov img { width:40px; height:40px; border-radius:7px; object-fit:cover; }
.prov-n { font-size:.61rem; color:#888; text-align:center; }
.prov-t { font-size:.55rem; color:var(--muted); }
.cast-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(84px,1fr)); gap:9px; }
.cast-card { background:var(--surf); border-radius:7px; overflow:hidden; border:1px solid var(--bdr); }
.cast-card:hover { border-color:rgba(201,169,110,.22); }
.cast-img { width:100%; aspect-ratio:2/3; object-fit:cover; display:block; }
.cast-ph { width:100%; aspect-ratio:2/3; background:var(--surf2); display:flex; align-items:center; justify-content:center; font-size:1.3rem; }
.cast-name { font-size:.63rem; font-weight:600; color:#bbb; padding:5px 6px 2px; }
.cast-char { font-size:.56rem; color:var(--muted); padding:0 6px 7px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }

/* REC CARDS */
a.rc { display:block; text-decoration:none; color:inherit; margin-bottom:10px; }
.rec-card { background:var(--surf); border-radius:10px; border:1px solid var(--bdr); padding:16px 18px; display:flex; gap:14px; align-items:flex-start; transition:border-color .18s, transform .18s, box-shadow .18s; cursor:pointer; }
a.rc:hover .rec-card { border-color:rgba(201,169,110,.28); transform:translateX(3px); box-shadow:0 4px 22px rgba(0,0,0,.4); }
.rec-num { font-family:'Playfair Display',serif; font-size:1.4rem; font-weight:900; color:rgba(201,169,110,.13); min-width:34px; line-height:1; text-align:right; flex-shrink:0; }
.rec-poster { width:50px; height:75px; border-radius:6px; object-fit:cover; flex-shrink:0; }
.rec-poster-ph { width:50px; height:75px; border-radius:6px; background:var(--surf2); display:flex; align-items:center; justify-content:center; font-size:1.2rem; flex-shrink:0; }
.rec-body { flex:1; min-width:0; }
.rec-title { font-size:.86rem; font-weight:600; color:var(--txt); margin-bottom:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.rec-pills { display:flex; flex-wrap:wrap; gap:3px; margin-bottom:8px; }
.rec-pill { background:var(--gdim); color:var(--gold2); font-size:.59rem; padding:2px 7px; border-radius:3px; border:1px solid rgba(201,169,110,.14); font-weight:500; }
.rec-bar-bg { height:2px; background:var(--subtle); border-radius:2px; margin-bottom:4px; }
.rec-bar { height:2px; background:linear-gradient(90deg,var(--gold),var(--gold2)); border-radius:2px; }
.rec-score { font-size:.64rem; color:var(--gold); font-weight:500; }
.rec-hint { font-size:.6rem; color:var(--muted); margin-top:3px; }
.chart-panel { background:var(--surf); border:1px solid var(--bdr); border-radius:10px; padding:20px; }

/* WATCHLIST EMPTY */
.wl-empty { text-align:center; padding:72px 40px; color:var(--muted); }
.wl-empty-icon { font-size:2.8rem; margin-bottom:14px; }
.wl-empty-h { font-family:'Playfair Display',serif; font-size:1.2rem; color:#fff; margin-bottom:8px; }

/* INPUTS */
.stTextInput input { background:var(--surf) !important; color:var(--txt) !important; border:1px solid rgba(255,255,255,.09) !important; border-radius:7px !important; font-size:.85rem !important; }
.stTextInput input:focus { border-color:rgba(201,169,110,.4) !important; box-shadow:0 0 0 2px rgba(201,169,110,.07) !important; }
div[data-baseweb="select"] > div { background:var(--surf) !important; border:1px solid rgba(255,255,255,.09) !important; border-radius:7px !important; }
.stCheckbox label { background:var(--surf) !important; border:1px solid var(--bdr) !important; border-radius:5px !important; padding:5px 12px !important; font-size:.69rem !important; font-weight:500 !important; color:var(--muted) !important; cursor:pointer !important; white-space:nowrap !important; }
.stCheckbox label:hover { border-color:rgba(201,169,110,.35) !important; color:var(--gold2) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
TMDB_KEY = "e27947a52a97677530fdd9e476e8b17c"
IMG_BASE  = "https://image.tmdb.org/t/p"
GENRE_ICON = {
    'Action':'💥','Adventure':'🗺️','Animation':'🎨','Comedy':'😄',
    'Crime':'🕵️','Documentary':'📽️','Drama':'🎭','Fantasy':'🧙',
    'Horror':'👻','Musical':'🎵','Mystery':'🔍','Romance':'❤️',
    'Sci-Fi':'👽','Thriller':'🔪','War':'⚔️','Western':'🤠','Children':'🧒',
}
PROV_MAP = {
    'Netflix':'https://www.netflix.com/search?q=','Amazon Prime Video':'https://www.amazon.com/s?k=',
    'Prime Video':'https://www.amazon.com/s?k=','Disney+':'https://www.disneyplus.com/search/',
    'Hotstar':'https://www.hotstar.com/in/search?q=','Disney+ Hotstar':'https://www.hotstar.com/in/search?q=',
    'Apple TV+':'https://tv.apple.com/search?term=','Hulu':'https://www.hulu.com/search?q=',
    'Max':'https://www.max.com/search?q=','HBO Max':'https://www.max.com/search?q=',
    'Peacock':'https://www.peacocktv.com/search?q=','Paramount+':'https://www.paramountplus.com/search/',
    'Zee5':'https://www.zee5.com/search/result/','SonyLIV':'https://www.sonyliv.com/search?q=',
    'Jio Cinema':'https://www.jiocinema.com/search/',
}
def prov_href(n,t,fb=''): b=PROV_MAP.get(n,''); return (b+urllib.parse.quote(t)) if b else (fb or '#')
def card_href(t,prev): return f"?movie={urllib.parse.quote(t)}&prev={prev}"
def nav_href(pg): return f"?nav={pg}"

# ─────────────────────────────────────────────────────────────────
# MARGIN HELPER — the ONLY reliable spacing in Streamlit layout=wide
# Uses side-spacer columns; immune to CSS overrides.
# ─────────────────────────────────────────────────────────────────
def C():
    """Return the centred content column, surrounded by spacers."""
    _, col, _ = st.columns([1, 18, 1])
    return col

# ─────────────────────────────────────────────────────────────────
# TMDB & DATA
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def tmdb_search(title):
    clean = re.sub(r'\s*\(\d{4}\)\s*$','',title).strip()
    try:
        r = requests.get("https://api.themoviedb.org/3/search/movie",
                         params={"api_key":TMDB_KEY,"query":clean},timeout=5)
        res = r.json().get('results',[]) if r.status_code==200 else []
        wp  = [x for x in res if x.get('poster_path')]
        return wp[0] if wp else (res[0] if res else None)
    except: return None

@st.cache_data(show_spinner=False)
def tmdb_details(title):
    sr = tmdb_search(title)
    if not sr: return None
    try:
        r = requests.get(f"https://api.themoviedb.org/3/movie/{sr['id']}",
                         params={"api_key":TMDB_KEY,"append_to_response":"credits,watch/providers,videos"},
                         timeout=6)
        return r.json() if r.status_code==200 else None
    except: return None

@st.cache_data(show_spinner=False)
def poster_url(title):
    r = tmdb_search(title)
    return f"{IMG_BASE}/w300{r['poster_path']}" if r and r.get('poster_path') else None

@st.cache_data
def load_data():
    rp,mp = os.path.join("data","ratings.csv"),os.path.join("data","movies.csv")
    if not os.path.exists(rp) or not os.path.exists(mp): return None,None
    rd=pd.read_csv(rp); md=pd.read_csv(mp); md['genres']=md['genres'].fillna('')
    return rd,md

def genre_recs(picked,mdf,n=12):
    tfidf=TfidfVectorizer(token_pattern=r'[^|]+'); tfidf.fit(mdf['genres'])
    cos=cosine_similarity(tfidf.transform(['|'.join(picked)]),tfidf.transform(mdf['genres'])).flatten()
    def cov(g): return len(set(picked)&{x.strip() for x in g.split('|')})/len(picked)
    comb=0.6*cos+0.4*mdf['genres'].apply(cov).values; mx=comb.max()
    if mx>0: comb/=mx
    idx=comb.argsort()[::-1][:n]
    return pd.DataFrame([{'title':mdf.iloc[i]['title'],'genres':mdf.iloc[i]['genres'],'score':round(float(comb[i]),4)} for i in idx])

def search_recs(query,mdf,n=12):
    df=mdf.copy(); df['_t']=df['title'].fillna('')+' '+df['genres'].fillna('').str.replace('|',' ')
    tfidf=TfidfVectorizer(ngram_range=(1,2),max_features=30000); mat=tfidf.fit_transform(df['_t'])
    sc=cosine_similarity(tfidf.transform([query]),mat).flatten(); mx=sc.max()
    if mx>0: sc/=mx
    idx=sc.argsort()[::-1][:n]
    return pd.DataFrame([{'title':df.iloc[i]['title'],'genres':df.iloc[i]['genres'],'score':round(float(sc[i]),4)} for i in idx if sc[i]>0])

ratings_df, movies_df = load_data()
all_genres = sorted({g.strip() for gs in (movies_df['genres'] if movies_df is not None else [])
                     for g in gs.split('|') if g.strip() not in ('','(no genres listed)')})

# ─────────────────────────────────────────────────────────────────
# NAV - KEY CHANGES HERE: Using <span> with onclick instead of <a> with href
# ─────────────────────────────────────────────────────────────────
p   = st.session_state.page
wlc = len(st.session_state.watchlist)
badge = f'<span class="nav-badge">{wlc}</span>' if wlc else ''
def nc(pid): return "nav-link active" if p==pid else "nav-link"

st.markdown(f"""
<div class="nav">
  <div class="nav-inner">
    <span class="nav-logo" onclick="window.location.href='{nav_href('logo')}'">CineMatch</span>
    <div class="nav-sep"></div>
    <nav class="nav-links">
      <span class="{nc('home')}" onclick="window.location.href='{nav_href('home')}'">Browse</span>
      <span class="{nc('recs')}" onclick="window.location.href='{nav_href('recs')}'">For You</span>
      <span class="{nc('watchlist')}" onclick="window.location.href='{nav_href('watchlist')}'">Watchlist{badge}</span>
    </nav>
    <div style="flex:1;"></div>
    <span class="nav-cta" onclick="window.location.href='{nav_href('recs')}'">Get Recommendations</span>
  </div>
</div>
""", unsafe_allow_html=True)

if ratings_df is None:
    st.error("Data files not found. Add data/ratings.csv and data/movies.csv."); st.stop()

# ... rest of the code continues exactly as before ...
