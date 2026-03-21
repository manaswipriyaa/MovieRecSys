 import streamlit as st
 import pandas as pd
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from sklearn.feature_extraction.text import TfidfVectorizer
 from sklearn.metrics.pairwise import cosine_similarity
 import html
 import os, re, requests, urllib.parse
 
 st.set_page_config(page_title="CineMatch", layout="wide", initial_sidebar_state="collapsed")
 
 # ─────────────────────────────────────────────────────────────────
 # QUERY-PARAM ROUTER  (handles nav clicks + poster clicks)
 # ─────────────────────────────────────────────────────────────────
 qp = st.query_params.to_dict()
 
 if "movie" in qp:
     st.session_state["movie"] = urllib.parse.unquote(qp["movie"])
     st.session_state["prev"]  = qp.get("prev", "home")
     st.query_params.clear()
     st.rerun()
 
 if "nav" in qp:
     dest = qp["nav"]
     if dest in ("home", "recs", "watchlist"):
         st.session_state["page"]  = dest
         st.session_state["movie"] = None
     if dest == "logo":
         st.session_state["page"]  = "home"
         st.session_state["movie"] = None
     st.query_params.clear()
     st.rerun()
@@ -47,62 +48,143 @@ for k, v in {"page": "home", "prev": "home", "movie": None,
 st.markdown("""
 <style>
 @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=Outfit:wght@300;400;500;600&display=swap');
 
 :root {
   --bg:      #08090e;
   --surf:    #0e0f18;
   --surf2:   #141520;
   --bdr:     rgba(255,255,255,0.07);
   --gold:    #c9a96e;
   --gold2:   #e8c992;
   --gdim:    rgba(201,169,110,0.11);
   --txt:     #eaeaf5;
   --muted:   #5a5a72;
   --subtle:  #2a2a3e;
 }
 
 *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
 html, body, [class*="css"] {
   font-family: 'Outfit', sans-serif;
   -webkit-font-smoothing: antialiased;
 }
 .stApp { background: var(--bg) !important; color: var(--txt); }
 
 /* Strip all Streamlit padding */
-.block-container                           { padding: 0 !important; max-width: 100% !important; }
-section[data-testid="stMain"] > div       { padding: 0 !important; }
+.block-container                           { padding: 0 0 3rem !important; max-width: 100% !important; }
+section[data-testid="stMain"] > div       { padding: 0 24px 32px !important; }
 div[data-testid="stSidebar"], footer, header { display: none !important; }
 div[data-testid="stVerticalBlockBorderWrapper"] > div { padding: 0 !important; }
 .stVerticalBlock                           { gap: 0 !important; }
 /* remove gap that appears above first element */
 .stMainBlockContainer > div:first-child   { margin-top: 0 !important; }
 
 ::-webkit-scrollbar { width: 5px; }
 ::-webkit-scrollbar-track { background: var(--bg); }
 ::-webkit-scrollbar-thumb { background: var(--subtle); border-radius: 3px; }
 
+
+.content-section {
+  max-width: 1200px;
+  margin: 0 auto;
+  padding: 32px 80px 0;
+}
+.content-section.tight { padding-top: 24px; }
+.content-section.flush { padding-top: 0; }
+.toolbar-card, .banner-card {
+  max-width: 1200px;
+  margin: 28px auto 0;
+  padding: 0 80px;
+}
+.banner-card {
+  background: linear-gradient(120deg, #100e24, #17102a);
+  border: 1px solid rgba(201,169,110,0.12);
+  border-radius: 18px;
+  padding-top: 28px;
+  padding-bottom: 28px;
+  box-shadow: 0 20px 40px rgba(0,0,0,0.25);
+}
+.banner-title {
+  font-family: 'Playfair Display', serif;
+  font-size: 1.35rem;
+  font-weight: 700;
+  color: #fff;
+  margin-bottom: 8px;
+}
+.banner-subtitle {
+  font-size: 0.85rem;
+  color: var(--muted);
+  line-height: 1.7;
+  max-width: 520px;
+}
+.action-link {
+  display: inline-flex;
+  align-items: center;
+  justify-content: center;
+  min-height: 44px;
+  padding: 0 20px;
+  border-radius: 10px;
+  background: var(--gold);
+  color: #08090e !important;
+  text-decoration: none !important;
+  font-size: 0.75rem;
+  font-weight: 700;
+  letter-spacing: 1.2px;
+  text-transform: uppercase;
+  border: 1px solid transparent;
+  box-shadow: 0 10px 24px rgba(201,169,110,0.18);
+  transition: transform 0.18s ease, box-shadow 0.18s ease, opacity 0.18s ease;
+}
+.action-link:hover {
+  transform: translateY(-1px);
+  box-shadow: 0 14px 28px rgba(201,169,110,0.22);
+  opacity: 0.95;
+}
+.action-link.ghost {
+  background: transparent;
+  color: var(--txt) !important;
+  border-color: rgba(255,255,255,0.14);
+  box-shadow: none;
+}
+.action-link.ghost:hover {
+  border-color: rgba(201,169,110,0.28);
+  color: var(--gold2) !important;
+}
+.link-stack {
+  display: flex;
+  flex-wrap: wrap;
+  gap: 12px;
+  align-items: center;
+}
+@media (max-width: 900px) {
+  section[data-testid="stMain"] > div { padding: 0 14px 24px !important; }
+  .hero-body, .det-top, .det-body, .sub-sec, .content-section, .toolbar-card, .banner-card { padding-left: 20px !important; padding-right: 20px !important; }
+  .nav-inner { padding: 0 20px; }
+  .det-flex { flex-direction: column; margin-top: -40px; }
+  .det-info { padding-top: 0; }
+}
+
 /* ═══════════════════════════════════════
    PAGE SHELL  — centred, breathing room
 ═══════════════════════════════════════ */
 .shell {
   max-width: 1280px;
   margin: 0 auto;
   padding: 0 60px;
 }
 
 /* ═══════════════════════════════════════
    NAV  — 100% HTML, always visible
 ═══════════════════════════════════════ */
 .nav {
   position: sticky;
   top: 0;
   z-index: 9999;
   background: rgba(8,9,14,0.97);
   backdrop-filter: blur(18px);
   -webkit-backdrop-filter: blur(18px);
   border-bottom: 1px solid var(--bdr);
 }
 .nav-inner {
   max-width: 1200px;
   margin: 0 auto;
   padding: 0 40px;
@@ -418,50 +500,64 @@ PROV_MAP = {
     'Amazon Prime Video':'https://www.amazon.com/s?k=',
     'Prime Video':'https://www.amazon.com/s?k=',
     'Disney+':'https://www.disneyplus.com/search/',
     'Hotstar':'https://www.hotstar.com/in/search?q=',
     'Disney+ Hotstar':'https://www.hotstar.com/in/search?q=',
     'Apple TV+':'https://tv.apple.com/search?term=',
     'Hulu':'https://www.hulu.com/search?q=',
     'Max':'https://www.max.com/search?q=',
     'HBO Max':'https://www.max.com/search?q=',
     'Peacock':'https://www.peacocktv.com/search?q=',
     'Paramount+':'https://www.paramountplus.com/search/',
     'Zee5':'https://www.zee5.com/search/result/',
     'SonyLIV':'https://www.sonyliv.com/search?q=',
     'Jio Cinema':'https://www.jiocinema.com/search/',
 }
 def prov_href(name, title, fb=''):
     b = PROV_MAP.get(name, '')
     return (b + urllib.parse.quote(title)) if b else (fb or '#')
 
 def card_href(title, prev):
     return f"?movie={urllib.parse.quote(title)}&prev={prev}"
 
 def nav_href(page):
     return f"?nav={page}"
 
+
+def safe_html_text(value):
+    return html.escape(str(value or '')).replace("\n", '<br>')
+
+def safe_html_attr(value):
+    return html.escape(str(value or ''), quote=True)
+
+def render_link_button(label, href, variant='primary'):
+    cls = 'action-link' if variant == 'primary' else 'action-link ghost'
+    st.markdown(
+        f'<a class="{cls}" href="{href}" target="_self">{safe_html_text(label)}</a>',
+        unsafe_allow_html=True,
+    )
+
 # ─────────────────────────────────────────────────────────────────
 # TMDB
 # ─────────────────────────────────────────────────────────────────
 @st.cache_data(show_spinner=False)
 def tmdb_search(title):
     clean = re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip()
     try:
         r = requests.get("https://api.themoviedb.org/3/search/movie",
                          params={"api_key": TMDB_KEY, "query": clean}, timeout=5)
         res = r.json().get('results', []) if r.status_code == 200 else []
         wp  = [x for x in res if x.get('poster_path')]
         return wp[0] if wp else (res[0] if res else None)
     except: return None
 
 @st.cache_data(show_spinner=False)
 def tmdb_details(title):
     sr = tmdb_search(title)
     if not sr: return None
     try:
         r = requests.get(f"https://api.themoviedb.org/3/movie/{sr['id']}",
                          params={"api_key": TMDB_KEY,
                                  "append_to_response": "credits,watch/providers,videos"},
                          timeout=6)
         return r.json() if r.status_code == 200 else None
     except: return None
@@ -516,364 +612,362 @@ def search_recs(query, movies_df, n=12):
 # ─────────────────────────────────────────────────────────────────
 # LOAD DATA
 # ─────────────────────────────────────────────────────────────────
 ratings_df, movies_df = load_data()
 
 all_genres = []
 if movies_df is not None:
     all_genres = sorted({g.strip() for gs in movies_df['genres']
                          for g in gs.split('|')
                          if g.strip() not in ('', '(no genres listed)')})
 
 # ─────────────────────────────────────────────────────────────────
 # NAV  — 100% HTML using <a href="?nav=...">
 # No Streamlit buttons, no CSS tricks, always visible
 # ─────────────────────────────────────────────────────────────────
 p    = st.session_state.page
 wlc  = len(st.session_state.watchlist)
 badge = f'<span class="nav-badge">{wlc}</span>' if wlc else ''
 
 def nc(page_id):   # nav link class
     return "nav-link active" if p == page_id else "nav-link"
 
 st.markdown(f"""
 <div class="nav">
   <div class="nav-inner">
-    <a class="nav-logo" href="?nav=logo">CineMatch</a>
+    <a class="nav-logo" href="?nav=logo" target="_self">CineMatch</a>
     <div class="nav-sep"></div>
     <nav class="nav-links">
-      <a class="{nc('home')}"      href="{nav_href('home')}">Browse</a>
-      <a class="{nc('recs')}"      href="{nav_href('recs')}">For You</a>
-      <a class="{nc('watchlist')}" href="{nav_href('watchlist')}">Watchlist{badge}</a>
+      <a class="{nc('home')}"      href="{nav_href('home')}" target="_self">Browse</a>
+      <a class="{nc('recs')}"      href="{nav_href('recs')}" target="_self">For You</a>
+      <a class="{nc('watchlist')}" href="{nav_href('watchlist')}" target="_self">Watchlist{badge}</a>
     </nav>
   </div>
 </div>
 """, unsafe_allow_html=True)
 
 # ─────────────────────────────────────────────────────────────────
 # DATA GUARD
 # ─────────────────────────────────────────────────────────────────
 if ratings_df is None:
     st.error("Data files not found. Add data/ratings.csv and data/movies.csv.")
     st.stop()
 
 # ─────────────────────────────────────────────────────────────────
 # CARD GRID RENDERER
 # Pure HTML <a> tags — zero Streamlit buttons anywhere
 # ─────────────────────────────────────────────────────────────────
 def render_grid(items, prev_page):
     """items = list of (title, genre1)"""
     cols = st.columns(8, gap="medium")
     for i, (title, genre1) in enumerate(items):
         purl  = poster_url(title)
         icon  = GENRE_ICON.get(genre1, '🎬')
         short = title[:17] + '…' if len(title) > 17 else title
+        safe_short = safe_html_text(short)
+        safe_genre = safe_html_text(genre1)
         href  = card_href(title, prev_page)
-        img   = (f'<img class="mcard-img" src="{purl}" loading="lazy" alt="{short}" '
+        img   = (f'<img class="mcard-img" src="{purl}" loading="lazy" alt="{safe_html_attr(short)}" '
                  f'onerror="this.style.display=\'none\';this.nextSibling.style.display=\'flex\'"/>'
                  f'<div class="mcard-ph" style="display:none;">{icon}</div>'
                  if purl else f'<div class="mcard-ph">{icon}</div>')
         with cols[i % 8]:
             st.markdown(f"""
-<a class="mc" href="{href}">
+<a class="mc" href="{href}" target="_self">
   <div class="mcard">
     {img}
     <div class="mcard-ov">
-      <div class="mcard-ov-t">{short}</div>
-      <div class="mcard-ov-s">{genre1}</div>
+      <div class="mcard-ov-t">{safe_short}</div>
+      <div class="mcard-ov-s">{safe_genre}</div>
     </div>
     <div class="mcard-body">
-      <div class="mcard-title">{short}</div>
-      <div class="mcard-genre">{genre1}</div>
+      <div class="mcard-title">{safe_short}</div>
+      <div class="mcard-genre">{safe_genre}</div>
     </div>
   </div>
 </a>""", unsafe_allow_html=True)
 
 # ─────────────────────────────────────────────────────────────────
 # DETAIL PAGE
 # ─────────────────────────────────────────────────────────────────
 def show_detail(title):
     # Back button (only Streamlit button needed here)
     st.markdown('<div class="det-top">', unsafe_allow_html=True)
     st.markdown('<div class="btn-ghost">', unsafe_allow_html=True)
     if st.button("← Back", key="back_btn"):
         st.session_state.movie = None
         st.session_state.page  = st.session_state.prev
         st.rerun()
     st.markdown('</div></div>', unsafe_allow_html=True)
 
     det = tmdb_details(title)
     if not det or 'title' not in det:
         st.warning("Could not load details from TMDB.")
         return
 
     # Plain string values — used directly inside HTML text nodes (no escaping needed;
     # Streamlit's unsafe_allow_html renders them literally as text content).
     ptitle   = str(det.get('title', title) or title)
     overview = str(det.get('overview', '') or '')
     tagline  = str(det.get('tagline', '') or '')
+    safe_title = safe_html_text(ptitle)
+    safe_overview = safe_html_text(overview)
+    safe_tagline = safe_html_text(tagline)
     year     = str(det.get('release_date', '')[:4])
     rating   = round(det.get('vote_average', 0), 1)
     rt       = det.get('runtime') or 0
     runtime  = f"{rt//60}h {rt%60}m" if rt else ''
     genres   = [str(g['name']) for g in det.get('genres', [])]
     poster   = f"{IMG_BASE}/w400{det['poster_path']}"    if det.get('poster_path')   else None
     backdrop = f"{IMG_BASE}/w1280{det['backdrop_path']}" if det.get('backdrop_path') else None
 
     if backdrop:
         st.markdown(f'<div class="det-backdrop"><img src="{backdrop}" alt=""/>'
                     f'<div class="det-fade"></div></div>', unsafe_allow_html=True)
 
-    # For HTML attribute (alt=""), use a safe version without quotes
-    alt_title = ptitle.replace('"', '')
+    # For HTML attributes and rendered content, escape text so TMDB copy never appears as raw HTML.
+    alt_title = safe_html_attr(ptitle)
     pimg = (f'<img class="det-poster" src="{poster}" alt="{alt_title}" '
             f'onerror="this.style.display=\'none\';this.nextSibling.style.display=\'flex\'"/>'
             f'<div class="det-poster-ph" style="display:none;">🎬</div>'
             if poster else '<div class="det-poster-ph">🎬</div>')
-    gpills = ''.join(f'<span class="det-pill">{g}</span>' for g in genres)
+    gpills = ''.join(f'<span class="det-pill">{safe_html_text(g)}</span>' for g in genres)
 
-    tagline_html = f'<div class="det-tagline">"{tagline}"</div>' if tagline else ''
+    tagline_html = f'<div class="det-tagline">"{safe_tagline}"</div>' if tagline else ''
 
     st.markdown(f"""
 <div class="det-body">
   <div class="det-flex">
     {pimg}
     <div class="det-info">
-      <div class="det-title">{ptitle}</div>
+      <div class="det-title">{safe_title}</div>
       <div class="det-meta">
         <span class="det-year">{year}</span>
         <span class="det-rating">&#9733; {rating} / 10</span>
         <span class="det-rt">{runtime}</span>
       </div>
       {tagline_html}
-      <div class="det-ov">{overview}</div>
+      <div class="det-ov">{safe_overview}</div>
       <div>{gpills}</div>
     </div>
   </div>
 </div>""", unsafe_allow_html=True)
 
     # Watchlist toggle
     in_wl = title in st.session_state.watchlist
     st.markdown('<div style="max-width:1200px;margin:0 auto;padding:8px 80px 16px;'
                 'display:flex;gap:12px;">', unsafe_allow_html=True)
     if in_wl:
         st.markdown('<div class="btn-danger">', unsafe_allow_html=True)
         if st.button("✓ In Watchlist — Remove", key="wl_tog"):
             st.session_state.watchlist.remove(title); st.rerun()
         st.markdown('</div>', unsafe_allow_html=True)
     else:
         if st.button("＋ Add to Watchlist", key="wl_tog"):
             st.session_state.watchlist.append(title); st.rerun()
     st.markdown('</div>', unsafe_allow_html=True)
 
     # Trailer
     videos  = det.get('videos', {}).get('results', [])
     trailer = next((v for v in videos
                     if v.get('type') == 'Trailer' and v.get('site') == 'YouTube'), None)
     if trailer:
         st.markdown(f'<div class="sub-sec" style="border-top:none;padding-top:0;padding-bottom:16px;">'
                     f'<a class="trailer-a" href="https://www.youtube.com/watch?v={trailer["key"]}"'
                     f' target="_blank">▶ Watch Trailer</a></div>', unsafe_allow_html=True)
 
     # Where to Watch
     pd_data = det.get('watch/providers', {}).get('results', {})
     region  = pd_data.get('IN', pd_data.get('US', {}))
     jtw     = region.get('link', '')
     seen, combp = set(), []
     for lbl, lst in [('Stream', region.get('flatrate', [])),
                      ('Rent',   region.get('rent',     [])),
                      ('Buy',    region.get('buy',      []))]:
         for p2 in lst:
             if p2['provider_name'] not in seen:
                 seen.add(p2['provider_name']); combp.append((p2, lbl))
 
     st.markdown('<div class="sub-sec"><div class="sub-h">Where to Watch</div>', unsafe_allow_html=True)
     if combp:
         cards = ''
         for p2, lbl in combp[:10]:
             logo = f'{IMG_BASE}/w92{p2["logo_path"]}' if p2.get('logo_path') else None
             img  = f'<img src="{logo}" alt="{p2["provider_name"]}"/>' if logo \
                    else '<div style="width:40px;height:40px;background:var(--surf2);border-radius:7px;"></div>'
             href = prov_href(p2['provider_name'], ptitle, jtw)
             cards += (f'<a class="prov" href="{href}" target="_blank" rel="noopener">'
-                      f'{img}<div class="prov-n">{p2["provider_name"]}</div>'
+                      f'{img}<div class="prov-n">{safe_html_text(p2["provider_name"])}</div>'
                       f'<div class="prov-t">{lbl}</div></a>')
         st.markdown(f'<div class="providers">{cards}</div>', unsafe_allow_html=True)
     else:
         st.markdown('<p style="font-size:.8rem;color:var(--muted);">No streaming data for your region.</p>',
                     unsafe_allow_html=True)
     st.markdown('</div>', unsafe_allow_html=True)
 
     # Cast
     cast = det.get('credits', {}).get('cast', [])[:14]
     if cast:
         ch = ''
         for c in cast:
-            cname = str(c.get('name', '') or '')
-            cchar = str(c.get('character', '') or '')[:22]
+            cname = safe_html_text(c.get('name', '') or '')
+            cchar = safe_html_text(str(c.get('character', '') or '')[:22])
             img = f'{IMG_BASE}/w185{c["profile_path"]}' if c.get('profile_path') else None
             ph  = f'<img class="cast-img" src="{img}" alt=""/>' if img \
                   else '<div class="cast-ph">👤</div>'
             ch += (f'<div class="cast-card">{ph}'
                    f'<div class="cast-name">{cname}</div>'
                    f'<div class="cast-char">{cchar}</div></div>')
         st.markdown(f'<div class="sub-sec"><div class="sub-h">Cast</div>'
                     f'<div class="cast-grid">{ch}</div></div>', unsafe_allow_html=True)
 
 # ─────────────────────────────────────────────────────────────────
 # ROUTER
 # ─────────────────────────────────────────────────────────────────
 if st.session_state.movie:
     show_detail(st.session_state.movie)
     st.stop()
 
 # ─────────────────────────────────────────────────────────────────
 # HOME
 # ─────────────────────────────────────────────────────────────────
 if st.session_state.page == 'home':
 
     st.markdown("""
 <div class="hero">
   <div class="hero-body">
     <div class="hero-eye">AI-Powered Discovery</div>
     <div class="hero-h">Your next favourite<br><em>film</em> awaits.</div>
     <div class="hero-p">Browse thousands of movies or let our engine recommend
     films tailored to your taste — no account needed.</div>
   </div>
 </div>""", unsafe_allow_html=True)
 
     # CTA row
-    st.markdown('<div style="max-width:1200px;margin:0 auto;padding:28px 80px 0;">', unsafe_allow_html=True)
-    ca, cb, _ = st.columns([1.6, 1.8, 8])
-    with ca:
-        if st.button("Get Recommendations →", key="hero_cta"):
-            st.session_state.page = 'recs'; st.rerun()
-    with cb:
+    st.markdown('<div class="toolbar-card">', unsafe_allow_html=True)
+    cta1, cta2, _ = st.columns([1.5, 1.7, 5], gap="small")
+    with cta1:
+        render_link_button("Get Recommendations →", nav_href('recs'))
+    with cta2:
         if st.session_state.watchlist:
-            st.markdown('<div style="border:1px solid rgba(201,169,110,0.28);border-radius:6px;display:inline-block;">',
-                        unsafe_allow_html=True)
-            if st.button(f"🎯  My Watchlist  ({wlc})", key="wl_hero"):
-                st.session_state.page = 'watchlist'; st.rerun()
-            st.markdown('</div>', unsafe_allow_html=True)
+            render_link_button(f"My Watchlist ({wlc})", nav_href('watchlist'), variant='ghost')
     st.markdown('</div>', unsafe_allow_html=True)
 
     st.markdown('<div class="divider" style="margin-top:28px;"></div>', unsafe_allow_html=True)
 
     # Browse section
-    st.markdown('<div style="max-width:1200px;margin:0 auto;padding:32px 80px 0;">', unsafe_allow_html=True)
+    st.markdown('<div class="content-section">', unsafe_allow_html=True)
     st.markdown('<div class="sec-eye">Explore</div>', unsafe_allow_html=True)
     st.markdown('<div class="sec-h">Browse Movies</div>', unsafe_allow_html=True)
 
     f1, f2 = st.columns([3, 1])
     with f1:
         search = st.text_input("", placeholder="Search by title…",
                                label_visibility="collapsed", key="search_home")
     with f2:
         gpick  = st.selectbox("", ['All'] + all_genres,
                               label_visibility="collapsed", key="gpick")
 
     if search:
         pool = movies_df[movies_df['title'].str.contains(search, case=False, na=False)]
     elif gpick != 'All':
         pool = movies_df[movies_df['genres'].str.contains(gpick, case=False, na=False)]
     else:
         pool = movies_df.sample(min(40, len(movies_df)), random_state=42)
     filtered = pool.head(40)
 
     st.markdown(f'<div class="count">{len(filtered)} TITLES</div>', unsafe_allow_html=True)
     st.markdown('</div>', unsafe_allow_html=True)
 
-    st.markdown('<div style="max-width:1200px;margin:0 auto;padding:0 80px;">', unsafe_allow_html=True)
+    st.markdown('<div class="content-section flush">', unsafe_allow_html=True)
     items = [(row['title'], row['genres'].split('|')[0].strip() if row['genres'] else '')
              for _, row in filtered.iterrows()]
     render_grid(items, 'home')
     st.markdown('</div>', unsafe_allow_html=True)
 
-    # Bottom banner — "Find My Movies" button is inline right of the text
-    st.markdown('<div class="btm"><div class="btm-inner">', unsafe_allow_html=True)
-    st.markdown("""
-  <div class="btm-text">
-    <div class="btm-title">Not sure what to watch?</div>
-    <div class="btm-sub">Tell us your favourite genres or describe what you're in the mood for.</div>
-  </div>""", unsafe_allow_html=True)
-    if st.button("Find My Movies →", key="banner_cta"):
-        st.session_state.page = 'recs'; st.rerun()
-    st.markdown('</div></div><div style="height:48px;"></div>', unsafe_allow_html=True)
+    # Bottom banner
+    st.markdown('<div class="banner-card">', unsafe_allow_html=True)
+    bn1, bn2 = st.columns([3.8, 1.2], gap="medium")
+    with bn1:
+        st.markdown('<div class="banner-title">Not sure what to watch?</div>', unsafe_allow_html=True)
+        st.markdown('<div class="banner-subtitle">Tell us your favourite genres or describe what you\'re in the mood for, and CineMatch will surface polished, relevant picks.</div>', unsafe_allow_html=True)
+    with bn2:
+        render_link_button("Find My Movies →", nav_href('recs'))
+    st.markdown('</div><div style="height:48px;"></div>', unsafe_allow_html=True)
 
 # ─────────────────────────────────────────────────────────────────
 # WATCHLIST
 # ─────────────────────────────────────────────────────────────────
 elif st.session_state.page == 'watchlist':
 
     st.markdown("""
 <div class="hero">
   <div class="hero-body">
     <div class="hero-eye">Your Collection</div>
     <div class="hero-h">My <em>Watchlist</em></div>
     <div class="hero-p">Films you've saved to watch later.</div>
   </div>
 </div>""", unsafe_allow_html=True)
 
     wl = st.session_state.watchlist
     if not wl:
         st.markdown("""<div class="wl-empty">
   <div class="wl-empty-icon">🎬</div>
   <div class="wl-empty-h">Your watchlist is empty</div>
   <p style="font-size:.83rem;">Open any movie and tap "Add to Watchlist".</p>
 </div>""", unsafe_allow_html=True)
     else:
-        st.markdown(f'<div style="max-width:1200px;margin:0 auto;padding:28px 80px 0;">'
-                    f'<div class="count">{len(wl)} SAVED</div></div>', unsafe_allow_html=True)
-        st.markdown('<div style="max-width:1200px;margin:0 auto;padding:0 80px;">', unsafe_allow_html=True)
+        st.markdown(f'<div class="content-section tight"><div class="count">{len(wl)} SAVED</div></div>', unsafe_allow_html=True)
+        st.markdown('<div class="content-section flush">', unsafe_allow_html=True)
         wl_items = []
         for t in wl:
             gr     = movies_df[movies_df['title'] == t]
             genre1 = gr['genres'].values[0].split('|')[0].strip() if len(gr) else ''
             wl_items.append((t, genre1))
         render_grid(wl_items, 'watchlist')
         st.markdown('</div>', unsafe_allow_html=True)
-        st.markdown('<div style="max-width:1200px;margin:0 auto;padding:20px 80px 48px;">', unsafe_allow_html=True)
+        st.markdown('<div class="content-section tight" style="padding-bottom:48px;">', unsafe_allow_html=True)
         st.markdown('<div class="btn-ghost">', unsafe_allow_html=True)
         if st.button("Clear Watchlist", key="clear_wl"):
             st.session_state.watchlist = []; st.rerun()
         st.markdown('</div></div>', unsafe_allow_html=True)
 
 # ─────────────────────────────────────────────────────────────────
 # FOR YOU
 # ─────────────────────────────────────────────────────────────────
 elif st.session_state.page == 'recs':
 
     st.markdown("""
 <div class="hero">
   <div class="hero-body">
     <div class="hero-eye">Personalised</div>
     <div class="hero-h">Made <em>For You</em></div>
     <div class="hero-p">Pick genres, or describe what you're in the mood for.</div>
   </div>
 </div>""", unsafe_allow_html=True)
 
-    st.markdown('<div style="max-width:1200px;margin:0 auto;padding:40px 80px 0;">', unsafe_allow_html=True)
+    st.markdown('<div class="content-section">', unsafe_allow_html=True)
 
     # Mode toggle
     st.markdown('<div class="sec-eye" style="margin-bottom:16px;">Recommendation Mode</div>',
                 unsafe_allow_html=True)
     mc1, mc2, _sp = st.columns([1.4, 1.6, 9])
     with mc1:
         if st.button("🎭  By Genre", key="mode_genre"):
             st.session_state.rec_mode = 'genre'; st.session_state.recs = None
     with mc2:
         st.markdown('<div class="btn-ghost">', unsafe_allow_html=True)
         if st.button("🔍  By Search", key="mode_search"):
             st.session_state.rec_mode = 'search'; st.session_state.recs = None
         st.markdown('</div>', unsafe_allow_html=True)
 
     mode = st.session_state.rec_mode
     st.markdown(f'<p style="font-size:.61rem;color:var(--gold);margin:12px 0 36px;'
                 f'letter-spacing:2px;font-weight:600;">'
                 f'{"GENRE-BASED" if mode=="genre" else "SEARCH-BASED"}</p>',
                 unsafe_allow_html=True)
 
     # ── Genre mode ──────────────────────────────────────────────
     if mode == 'genre':
         st.markdown('<div class="sec-eye" style="margin-bottom:16px;">Select genres you love</div>',
                     unsafe_allow_html=True)
         gcols  = st.columns(9)
@@ -934,72 +1028,73 @@ elif st.session_state.page == 'recs':
                 st.session_state.page = 'home'; st.rerun()
             st.markdown('</div>', unsafe_allow_html=True)
 
         if srch_btn:
             if not sq.strip():
                 st.warning("Please enter a description.")
             else:
                 with st.spinner("Searching…"):
                     st.session_state.recs = search_recs(sq, movies_df, top_n)
 
     # ── Results ─────────────────────────────────────────────────
     if st.session_state.recs is not None:
         recs = st.session_state.recs
         if recs.empty:
             st.warning("No results found. Try something different.")
         else:
             st.markdown('<div class="divider" style="margin:32px 0;"></div>', unsafe_allow_html=True)
             st.markdown('<div class="sec-eye">Results</div>', unsafe_allow_html=True)
             st.markdown(f'<div class="sec-h">Top {len(recs)} Picks For You</div>',
                         unsafe_allow_html=True)
 
             lc, rc = st.columns([1.2, 1])
             with lc:
                 for i, row in recs.iterrows():
                     bw    = int(row['score'] * 100)
-                    pills = ''.join(f'<span class="rec-pill">{g.strip()}</span>'
+                    pills = ''.join(f'<span class="rec-pill">{safe_html_text(g.strip())}</span>'
                                     for g in row['genres'].split('|') if g.strip())
                     purl  = poster_url(row['title'])
                     g0    = row['genres'].split('|')[0].strip()
                     ph    = (f'<img class="rec-poster" src="{purl}" alt="poster" '
                              f'onerror="this.style.display=\'none\';'
                              f'this.nextSibling.style.display=\'flex\'"/>'
                              f'<div class="rec-poster-ph" style="display:none;">'
                              f'{GENRE_ICON.get(g0,"🎬")}</div>'
                              if purl else
                              f'<div class="rec-poster-ph">{GENRE_ICON.get(g0,"🎬")}</div>')
                     num  = f"0{i+1}" if i+1 < 10 else str(i+1)
                     href = card_href(row['title'], 'recs')
+                    safe_row_title = safe_html_text(row['title'])
 
                     # Entire rec card wrapped in <a> — click anywhere opens detail
                     st.markdown(f"""
-<a class="rc" href="{href}">
+<a class="rc" href="{href}" target="_self">
   <div class="rec-card">
     <div class="rec-num">{num}</div>
     {ph}
     <div class="rec-body">
-      <div class="rec-title">{row['title']}</div>
+      <div class="rec-title">{safe_row_title}</div>
       <div class="rec-pills">{pills}</div>
       <div class="rec-bar-bg"><div class="rec-bar" style="width:{bw}%;"></div></div>
       <div class="rec-score">Match: {row['score']:.2f}</div>
       <div class="rec-hint">Click to view full details →</div>
     </div>
   </div>
 </a>""", unsafe_allow_html=True)
 
             with rc:
                 st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
                 fig, ax = plt.subplots(figsize=(5.5, max(4, len(recs) * 0.52)))
                 fig.patch.set_facecolor('#0e0f18'); ax.set_facecolor('#0e0f18')
                 tch = [r['title'][:26] + '…' if len(r['title']) > 26 else r['title']
                        for _, r in recs.iterrows()]
                 sc  = recs['score'].values
                 pal = (['#c9a96e','#cdb07c','#d1b78a','#d5be98','#d9c5a6',
                         '#ddcab0','#e1d0ba','#e5d5c4','#e9dace','#eddfd8'] * 2)[:len(sc)]
                 ax.barh(tch[::-1], sc[::-1], color=pal[::-1], height=0.5, edgecolor='none')
                 for b, s in zip(ax.patches, sc[::-1]):
                     ax.text(b.get_width() + 0.014, b.get_y() + b.get_height() / 2,
                             f'{s:.2f}', va='center', color='#5a5a72', fontsize=7.5)
                 ax.set_xlim(0, 1.22)
                 ax.set_xlabel('Match Score', color='#3a3a52', fontsize=8)
                 ax.set_title('Match Chart', color='#c9a96e', fontsize=9,
                              fontweight='bold', pad=10)
