import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════
#  CONFIG PAGE
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="Career Recommender",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════
#  CSS CUSTOM
# ══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Inter:wght@300;400;500&display=swap');

* { font-family: 'Inter', sans-serif; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f;
    color: #e8e8f0;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0a0f 0%, #0f0f1a 50%, #0a0a0f 100%);
}

/* Header */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #6ee7f7, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -1px;
}
.hero p {
    color: #888;
    font-size: 1.05rem;
    margin-top: 0.5rem;
    font-weight: 300;
}

/* Cards de formulaire */
.form-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.form-card h3 {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #a78bfa;
    margin: 0 0 1rem 0;
}

/* Résultat */
.result-box {
    background: linear-gradient(135deg, rgba(110,231,247,0.08), rgba(167,139,250,0.08));
    border: 1px solid rgba(110,231,247,0.25);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    margin-top: 1rem;
}
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: #6ee7f7;
    margin-bottom: 0.5rem;
}
.result-career {
    font-family: 'Syne', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #6ee7f7, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0.5rem 0;
}
.result-emoji { font-size: 3rem; margin-bottom: 0.5rem; }

/* Confidence bars */
.conf-row {
    display: flex;
    align-items: center;
    margin: 0.4rem 0;
    gap: 0.8rem;
}
.conf-label {
    width: 180px;
    font-size: 0.82rem;
    color: #bbb;
    text-align: right;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.conf-bar-bg {
    flex: 1;
    height: 8px;
    background: rgba(255,255,255,0.07);
    border-radius: 10px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg, #6ee7f7, #a78bfa);
}
.conf-pct {
    font-size: 0.82rem;
    color: #6ee7f7;
    width: 38px;
    text-align: right;
}

/* Inputs */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e8e8f0 !important;
}
.stSlider > div > div > div { color: #6ee7f7 !important; }
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] { color: #666 !important; }

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #6ee7f7, #a78bfa) !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    letter-spacing: 1px !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Metric */
.metric-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #6ee7f7;
}
.metric-lbl { font-size: 0.78rem; color: #666; text-transform: uppercase; letter-spacing: 1px; }

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  CHARGEMENT & MODÈLE (cache)
# ══════════════════════════════════════════════
@st.cache_resource
def load_model():
    df = pd.read_csv('career_cleaned_ml.csv')

    def group_job(title):
        t = str(title).lower()
        if any(x in t for x in ['software','developer','programmer','web','devops','front','back']):
            return 'Software / Dev'
        elif any(x in t for x in ['data','analyst','scientist']):
            return 'Data / Analytics'
        elif any(x in t for x in ['mechanical','civil','structural','production','quality','site']):
            return 'Mechanical / Civil'
        elif any(x in t for x in ['computer software','system','network','cyber','it','instrumentation']):
            return 'IT / Systems'
        elif any(x in t for x in ['business','consultant','project manager']):
            return 'Business / Consulting'
        elif any(x in t for x in ['teacher','professor','research']):
            return 'Education / Research'
        elif any(x in t for x in ['sales','marketing','hr','executive','tele','financial']):
            return 'Sales / HR / Finance'
        else:
            return 'Other'

    df['job_category'] = df['job_title'].apply(group_job)

    def group_spec(s):
        s = str(s).lower()
        if any(x in s for x in ['computer','software','it','data','engineering','electronic','information']):
            return 0
        elif any(x in s for x in ['commerce','finance','business','accounting','economics','management']):
            return 1
        elif any(x in s for x in ['psychology','arts','literature','history','sociology']):
            return 2
        elif any(x in s for x in ['maths','mathematics','physics','chemistry','science']):
            return 3
        else:
            return 4

    df_enc = df.copy()
    encoders = {}
    for col in ['gender', 'ug_degree', 'has_certification', 'is_working']:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le

    df_enc['spec_group']   = df_enc['ug_specialization'].apply(group_spec)
    df_enc['skills_count'] = df_enc['skills'].astype(str).str.split(r'[;,]').apply(len)
    df_enc['has_masters']  = (df_enc['masters_field'].astype(str).str.lower() != 'none').astype(int)

    feature_cols = ['gender','ug_degree','spec_group','cgpa_normalized',
                    'has_certification','is_working','skills_count','has_masters']

    X = df_enc[feature_cols]
    y = df_enc['job_category']

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    # Best K
    scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k), X_sc, y, cv=5).mean()
              for k in range(1, 21)]
    best_k = range(1, 21)[scores.index(max(scores))]
    best_cv = max(scores)

    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_sc, y)

    return knn, scaler, encoders, group_spec, X, y, best_k, best_cv

knn, scaler, encoders, group_spec, X_all, y_all, best_k, best_cv = load_model()

# Icônes par catégorie
ICONS = {
    'Software / Dev'       : '💻',
    'Data / Analytics'     : '📊',
    'Mechanical / Civil'   : '⚙️',
    'IT / Systems'         : '🖧',
    'Business / Consulting': '💼',
    'Education / Research' : '🎓',
    'Sales / HR / Finance' : '📈',
    'Other'                : '🔍',
}

# ══════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <h1>Career Recommender</h1>
    <p>Intelligence artificielle · KNN · Recommandation de carrière personnalisée</p>
</div>
""", unsafe_allow_html=True)

# Stats rapides
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-box"><div class="metric-val">{best_k}</div><div class="metric-lbl">Meilleur K</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-box"><div class="metric-val">{best_cv*100:.0f}%</div><div class="metric-lbl">CV Accuracy</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-box"><div class="metric-val">{len(X_all)}</div><div class="metric-lbl">Profils entraînés</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-box"><div class="metric-val">8</div><div class="metric-lbl">Catégories</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  FORMULAIRE  |  RÉSULTAT
# ══════════════════════════════════════════════
col_form, col_result = st.columns([1, 1], gap="large")

with col_form:
    # ── Infos personnelles ──
    st.markdown('<div class="form-card"><h3>👤 Profil personnel</h3>', unsafe_allow_html=True)
    gender = st.selectbox("Genre", ["Male", "Female"])
    is_working = st.selectbox("Travaillez-vous actuellement ?", ["No", "Yes"])
    has_masters_bool = st.selectbox("Avez-vous fait un Master ?", ["Non", "Oui"])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Formation ──
    st.markdown('<div class="form-card"><h3>🎓 Formation</h3>', unsafe_allow_html=True)
    ug_degree = st.selectbox("Diplôme de licence", ["B.E", "B.SC", "BA", "B.TECH", "MBA", "B.COM", "BCA", "B.ARCH"])
    ug_specialization = st.text_input("Spécialisation", placeholder="ex: Computer Science Engineering")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Compétences ──
    st.markdown('<div class="form-card"><h3>🔧 Compétences & Notes</h3>', unsafe_allow_html=True)
    cgpa = st.slider("CGPA / Moyenne (%)", min_value=40, max_value=100, value=75, step=1)
    skills = st.text_area("Compétences (séparées par virgule)",
                          placeholder="ex: Python, SQL, Data Analysis, Machine Learning",
                          height=80)
    has_cert = st.selectbox("Avez-vous des certifications ?", ["No", "Yes"])
    st.markdown('</div>', unsafe_allow_html=True)

    predict_btn = st.button("🎯  Recommander une carrière")

# ── Résultat ──
with col_result:
    st.markdown("<br><br>", unsafe_allow_html=True)

    if predict_btn:
        if not ug_specialization.strip():
            st.warning("⚠️ Veuillez entrer votre spécialisation.")
        else:
            # Encoder
            g_enc = encoders['gender'].transform([gender.strip().title()])[0]
            deg   = ug_degree.strip().upper()
            d_enc = encoders['ug_degree'].transform([deg])[0] if deg in encoders['ug_degree'].classes_ else 0
            c_enc = 1 if has_cert == "Yes" else 0
            w_enc = 1 if is_working == "Yes" else 0
            s_enc = group_spec(ug_specialization)
            sk_cnt = len([s for s in skills.split(',') if s.strip()]) if skills else 1
            m_enc = 1 if has_masters_bool == "Oui" else 0

            X_new = np.array([[g_enc, d_enc, s_enc, float(cgpa),
                                c_enc, w_enc, sk_cnt, m_enc]])
            X_new_sc = scaler.transform(X_new)

            pred = knn.predict(X_new_sc)[0]
            distances, indices = knn.kneighbors(X_new_sc)
            neighbor_labels = y_all.iloc[indices[0]].values
            unique, counts = np.unique(neighbor_labels, return_counts=True)
            confidence = dict(zip(unique, (counts / best_k * 100).round(1)))
            confidence_sorted = dict(sorted(confidence.items(), key=lambda x: x[1], reverse=True))

            icon = ICONS.get(pred, '🎯')

            # Box résultat
            st.markdown(f"""
            <div class="result-box">
                <div class="result-emoji">{icon}</div>
                <div class="result-title">Carrière recommandée</div>
                <div class="result-career">{pred}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Barres de confiance
            st.markdown("**Confiance des voisins les plus proches**")
            for label, pct in confidence_sorted.items():
                bar_icon = ICONS.get(label, '•')
                is_best = "style='color:#6ee7f7;font-weight:600'" if label == pred else ""
                st.markdown(f"""
                <div class="conf-row">
                    <div class="conf-label" {is_best}>{bar_icon} {label}</div>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill" style="width:{pct}%"></div>
                    </div>
                    <div class="conf-pct">{pct}%</div>
                </div>
                """, unsafe_allow_html=True)

            # Détails du profil soumis
            with st.expander("📋 Résumé du profil soumis"):
                st.markdown(f"""
                | Champ | Valeur |
                |---|---|
                | Genre | {gender} |
                | Diplôme | {ug_degree} |
                | Spécialisation | {ug_specialization} |
                | CGPA | {cgpa}% |
                | En emploi | {is_working} |
                | Certifications | {has_cert} |
                | Master | {has_masters_bool} |
                | Nb compétences | {sk_cnt} |
                """)
    else:
        st.markdown("""
        <div style="text-align:center; padding:4rem 2rem; color:#444; border:1px dashed #222; border-radius:16px;">
            <div style="font-size:3rem">🎯</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.1rem; margin-top:1rem; color:#555;">
                Remplissez le formulaire<br>et cliquez sur <b style="color:#6ee7f7">Recommander</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#333; font-size:0.78rem; padding:1rem 0; border-top:1px solid #1a1a2e;">
    Career Recommender · KNN · Career Recommendation Dataset · Kaggle
</div>
""", unsafe_allow_html=True)
