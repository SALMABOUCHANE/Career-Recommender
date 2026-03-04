import streamlit as st
import numpy as np
import time

st.set_page_config(
    page_title="CareerIQ — Find Your Path",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=Space+Mono&display=swap');

:root {
    --bg:      #0a0a0f;
    --surface: #111118;
    --border:  #1e1e2e;
    --accent1: #e8c97e;
    --accent2: #6c63ff;
    --accent3: #ff6b6b;
    --text:    #e8e6f0;
    --muted:   #6b6882;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stAppViewContainer"] > .main { background-color: var(--bg); }
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, header { visibility: hidden; }

.hero {
    position: relative;
    text-align: center;
    padding: 70px 20px 50px;
    margin-bottom: 40px;
}
.hero::before {
    content: "";
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 50% at 50% 0%, rgba(108,99,255,.25) 0%, transparent 70%),
        radial-gradient(ellipse 40% 40% at 80% 80%, rgba(232,201,126,.10) 0%, transparent 60%);
    pointer-events: none;
}
.hero-tag {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent1);
    background: rgba(232,201,126,.08);
    border: 1px solid rgba(232,201,126,.25);
    padding: 6px 16px;
    border-radius: 20px;
    margin-bottom: 20px;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(42px, 7vw, 78px);
    line-height: 1.05;
    color: var(--text);
    margin: 0 0 16px;
    font-style: italic;
}
.hero-title span { color: var(--accent1); font-style: normal; }
.hero-sub {
    font-size: 16px;
    color: var(--muted);
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.6;
    font-weight: 300;
}

.divider { border: none; border-top: 1px solid var(--border); margin: 0 0 40px; }

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 20px;
}

div[data-baseweb="select"] > div {
    background-color: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
}
div[data-baseweb="select"] span { color: var(--text) !important; }
[data-testid="stCheckbox"] label { font-size: 14px !important; color: var(--text) !important; }
[data-testid="stSlider"] label { color: var(--muted) !important; font-size: 13px !important; }

[data-testid="stForm"] button[kind="primaryFormSubmit"] {
    background: linear-gradient(135deg, var(--accent2), #8b5cf6) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    border: none !important;
    border-radius: 12px !important;
    width: 100% !important;
    box-shadow: 0 4px 24px rgba(108,99,255,.35) !important;
}

.result-header {
    font-family: 'DM Serif Display', serif;
    font-size: 28px;
    font-style: italic;
    color: var(--text);
    margin-bottom: 6px;
}
.result-sub { font-size: 13px; color: var(--muted); margin-bottom: 28px; }

.job-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 22px 26px 20px 30px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}
.job-card::before {
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
}
.job-card.rank-1::before { background: var(--accent1); }
.job-card.rank-2::before { background: var(--accent2); }
.job-card.rank-3::before { background: var(--accent3); }

.job-title { font-weight: 600; font-size: 17px; color: var(--text); margin-bottom: 6px; }
.rank-badge {
    float: right;
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    color: var(--muted);
    background: rgba(255,255,255,.04);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 10px;
}
.job-num { font-family:'Space Mono',monospace; font-size:10px; color:var(--muted); margin-bottom:4px; }
.job-desc { font-size:13px; color:var(--muted); margin:4px 0 10px; line-height:1.5; }
.job-tag {
    display: inline-block;
    font-size: 11px;
    font-family: 'Space Mono', monospace;
    color: var(--muted);
    background: rgba(255,255,255,.04);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 3px 9px;
    margin: 3px 4px 3px 0;
}
.match-wrap { margin-top: 12px; display: flex; align-items: center; gap: 10px; }
.match-bg { flex: 1; height: 4px; background: var(--border); border-radius: 4px; overflow: hidden; }
.match-fill { height: 100%; border-radius: 4px; background: linear-gradient(90deg, var(--accent2), var(--accent1)); }
.match-pct { font-family:'Space Mono',monospace; font-size:11px; color:var(--accent1); min-width:36px; text-align:right; }

.footer-quote {
    text-align: center;
    padding: 40px 20px 30px;
    font-family: 'DM Serif Display', serif;
    font-style: italic;
    font-size: 18px;
    color: var(--muted);
}
</style>
""", unsafe_allow_html=True)

# ── Job Database ──
JOB_DB = {
    "Software Engineer":         {"tags":["Backend","Systems","APIs"],        "skills":["python","java","c_cpp","sql","web_dev"],                     "desc":"Design, build, and maintain scalable software systems."},
    "Data Scientist":            {"tags":["Analytics","ML","Research"],       "skills":["python","sql","machine_learning","data_analysis"],            "desc":"Extract insights from complex datasets to drive decisions."},
    "Machine Learning Engineer": {"tags":["AI","Deep Learning","MLOps"],     "skills":["python","machine_learning","data_analysis","cloud_computing"], "desc":"Build and deploy production-grade ML models at scale."},
    "Cloud Architect":           {"tags":["AWS/GCP","Infrastructure","SRE"], "skills":["cloud_computing","devops","cybersecurity"],                   "desc":"Design resilient and cost-efficient cloud solutions."},
    "DevOps Engineer":           {"tags":["CI/CD","Automation","SRE"],       "skills":["devops","cloud_computing","python"],                          "desc":"Bridge dev and ops for fast, reliable software delivery."},
    "Frontend Developer":        {"tags":["UI/UX","React","Performance"],    "skills":["web_dev","python","data_analysis"],                           "desc":"Craft pixel-perfect, accessible, performant interfaces."},
    "Cybersecurity Analyst":     {"tags":["Threat Intel","SIEM","Pen Test"], "skills":["cybersecurity","python","sql"],                              "desc":"Protect systems and data from evolving digital threats."},
    "AI Researcher":             {"tags":["NLP","Vision","Papers"],          "skills":["python","machine_learning","data_analysis","c_cpp"],           "desc":"Push the boundaries of intelligence through research."},
    "Data Engineer":             {"tags":["Pipelines","ETL","Warehousing"],  "skills":["sql","python","cloud_computing","data_analysis"],              "desc":"Build the infrastructure powering data-driven orgs."},
    "Product Manager (Tech)":    {"tags":["Strategy","Roadmap","Agile"],     "skills":["communication","leadership","problem_solving"],               "desc":"Lead cross-functional teams to ship products users love."},
}

def score_jobs(feat):
    active = [k for k, v in feat.items() if v is True]
    results = []
    for title, info in JOB_DB.items():
        overlap = len(set(active) & set(info["skills"]))
        raw = overlap / max(len(info["skills"]), 1)
        bonus = 0.0
        if feat.get("gpa", 0) >= 3.5: bonus += 0.05
        if feat.get("years_experience", 0) >= 3: bonus += 0.07
        if feat.get("degree_level") in ["Master","PhD"]: bonus += 0.06
        if title == "Product Manager (Tech)" and feat.get("leadership") and feat.get("communication"): bonus += 0.15
        pct = min(round((raw + bonus + np.random.uniform(0, 0.06)) * 100), 99)
        results.append((title, pct, info))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:3]

# ── Hero ──
st.markdown("""
<div class="hero">
    <div class="hero-tag">✦ AI-Powered Career Intelligence</div>
    <h1 class="hero-title">Discover Your<br><span>Ideal Career Path</span></h1>
    <p class="hero-sub">Tell us about your background and skills — we'll match you to the roles where you'll truly thrive.</p>
</div>
<hr class="divider">
""", unsafe_allow_html=True)

# ── Form ──
with st.form("career_form"):
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown('<div class="section-label">① Personal Profile</div>', unsafe_allow_html=True)
        age              = st.slider("Age", 18, 65, 24)
        gender           = st.selectbox("Gender", ["Prefer not to say","Male","Female","Non-binary / Other"])
        degree_level     = st.selectbox("Degree Level", ["Bachelor","Master","PhD","Associate / Diploma","Self-taught"])
        field_of_study   = st.selectbox("Field of Study", ["Computer Science","Data Science","Information Technology","Software Engineering","Mathematics / Statistics","Electrical Engineering","Other"])
        gpa              = st.slider("GPA (0 – 4.0)", 0.0, 4.0, 3.3, 0.1)
        years_experience = st.slider("Years of Experience", 0, 25, 1)

    with col_r:
        st.markdown('<div class="section-label">② Technical Skills</div>', unsafe_allow_html=True)
        t1, t2 = st.columns(2)
        with t1:
            python           = st.checkbox("Python")
            java             = st.checkbox("Java")
            c_cpp            = st.checkbox("C / C++")
            sql              = st.checkbox("SQL")
            web_dev          = st.checkbox("Web Development")
        with t2:
            machine_learning = st.checkbox("Machine Learning")
            data_analysis    = st.checkbox("Data Analysis")
            cloud_computing  = st.checkbox("Cloud Computing")
            cybersecurity    = st.checkbox("Cybersecurity")
            devops           = st.checkbox("DevOps")

        st.markdown('<div class="section-label" style="margin-top:20px">③ Soft Skills</div>', unsafe_allow_html=True)
        s1, s2 = st.columns(2)
        with s1:
            communication   = st.checkbox("Communication")
            leadership      = st.checkbox("Leadership")
            problem_solving = st.checkbox("Problem Solving")
        with s2:
            teamwork     = st.checkbox("Teamwork")
            adaptability = st.checkbox("Adaptability")
            creativity   = st.checkbox("Creativity")

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("✦  Reveal My Career Matches")

# ── Results ──
if submitted:
    feat = dict(
        age=age, gender=gender, degree_level=degree_level,
        field_of_study=field_of_study, gpa=gpa, years_experience=years_experience,
        python=python, java=java, c_cpp=c_cpp, sql=sql,
        machine_learning=machine_learning, data_analysis=data_analysis,
        cloud_computing=cloud_computing, cybersecurity=cybersecurity,
        web_dev=web_dev, devops=devops,
        communication=communication, leadership=leadership,
        problem_solving=problem_solving, teamwork=teamwork,
        adaptability=adaptability, creativity=creativity
    )

    with st.spinner("Analyzing your profile…"):
        time.sleep(1.0)

    top = score_jobs(feat)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<div class="result-header">Your Top Matches</div>', unsafe_allow_html=True)
    st.markdown('<div class="result-sub">Ranked by compatibility with your skills and background.</div>', unsafe_allow_html=True)

    labels = ["Best Match", "Strong Fit", "Worth Exploring"]
    for i, (title, pct, info) in enumerate(top):
        tags_html = "".join(f'<span class="job-tag">{t}</span>' for t in info["tags"])
        st.markdown(f"""
        <div class="job-card rank-{i+1}">
            <span class="rank-badge">{labels[i]}</span>
            <div class="job-num">0{i+1}</div>
            <div class="job-title">{title}</div>
            <div class="job-desc">{info['desc']}</div>
            <div>{tags_html}</div>
            <div class="match-wrap">
                <div class="match-bg"><div class="match-fill" style="width:{pct}%"></div></div>
                <div class="match-pct">{pct}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    tech_count = sum(1 for k in ["python","java","c_cpp","sql","web_dev",
                                  "machine_learning","data_analysis",
                                  "cloud_computing","cybersecurity","devops"]
                     if feat.get(k))
    c1, c2, c3 = st.columns(3)
    c1.metric("Technical Skills", f"{tech_count} / 10")
    c2.metric("Work Experience", f"{years_experience} yr{'s' if years_experience != 1 else ''}")
    c3.metric("GPA", f"{gpa:.1f}", "Strong" if gpa >= 3.5 else "")

# ── Footer ──
st.markdown("""
<div class="footer-quote">
    "Every skill you master is a door that no one can close behind you."
</div>
""", unsafe_allow_html=True)