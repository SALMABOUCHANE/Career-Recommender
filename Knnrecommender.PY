import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════
#  ÉTAPE 1 — CHARGEMENT & PRÉPARATION
# ══════════════════════════════════════════════
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

# ══════════════════════════════════════════════
#  ÉTAPE 2 — ENCODAGE
# ══════════════════════════════════════════════
df_enc = df.copy()

encoders = {}
for col in ['gender', 'ug_degree', 'has_certification', 'is_working']:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    encoders[col] = le

def group_spec(s):
    s = str(s).lower()
    if any(x in s for x in ['computer','software','it','data','engineering','electronic','information']):
        return 0   # tech
    elif any(x in s for x in ['commerce','finance','business','accounting','economics','management']):
        return 1   # business
    elif any(x in s for x in ['psychology','arts','literature','history','sociology']):
        return 2   # humanities
    elif any(x in s for x in ['maths','mathematics','physics','chemistry','science']):
        return 3   # science
    else:
        return 4   # other

df_enc['spec_group']   = df_enc['ug_specialization'].apply(group_spec)
df_enc['skills_count'] = df_enc['skills'].astype(str).str.split(r'[;,]').apply(len)
df_enc['has_masters']  = (df_enc['masters_field'].astype(str).str.lower() != 'none').astype(int)

feature_cols = ['gender', 'ug_degree', 'spec_group', 'cgpa_normalized',
                'has_certification', 'is_working', 'skills_count', 'has_masters']

X = df_enc[feature_cols]
y = df_enc['job_category']

# ══════════════════════════════════════════════
#  ÉTAPE 3 — TROUVER LE MEILLEUR K
# ══════════════════════════════════════════════
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

print("Recherche du meilleur K...")
cv_scores = []
for k in range(1, 21):
    score = cross_val_score(KNeighborsClassifier(n_neighbors=k), X_sc, y, cv=5).mean()
    cv_scores.append(score)

best_k = range(1, 21)[cv_scores.index(max(cv_scores))]
print(f"✅ Meilleur K = {best_k}  (CV Accuracy = {max(cv_scores)*100:.2f}%)")

# ══════════════════════════════════════════════
#  ÉTAPE 4 — ENTRAÎNEMENT FINAL
# ══════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X_sc, y, test_size=0.2, random_state=42, stratify=y)

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

acc = accuracy_score(y_test, knn.predict(X_test))
print(f"✅ Accuracy Test = {acc*100:.2f}%")
print()
print(classification_report(y_test, knn.predict(X_test)))

# ══════════════════════════════════════════════
#  ÉTAPE 5 — SAUVEGARDER LE MODÈLE
# ══════════════════════════════════════════════
# On entraîne sur TOUT le dataset pour la version finale
knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_sc, y)

with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn_final, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

print("✅ Modèle sauvegardé : knn_model.pkl")

# ══════════════════════════════════════════════
#  ÉTAPE 6 — FONCTION DE RECOMMANDATION
# ══════════════════════════════════════════════
def recommander_carriere(gender, ug_degree, ug_specialization,
                          cgpa, has_certification, is_working,
                          skills, has_masters):
    """
    Recommande une catégorie de carrière pour un profil donné.

    Paramètres :
    ─────────────
    gender           : 'Male' ou 'Female'
    ug_degree        : 'B.E', 'B.SC', 'BA', 'B.TECH', 'MBA'...
    ug_specialization: ex. 'Computer Science Engineering'
    cgpa             : note entre 0 et 100
    has_certification: 'Yes' ou 'No'
    is_working       : 'Yes' ou 'No'
    skills           : liste de compétences séparées par virgule
    has_masters      : True ou False
    """

    # Encoder les valeurs
    gender_enc = encoders['gender'].transform([gender.strip().title()])[0]

    deg = ug_degree.strip().upper()
    if deg in encoders['ug_degree'].classes_:
        deg_enc = encoders['ug_degree'].transform([deg])[0]
    else:
        deg_enc = 0   # valeur par défaut si diplôme inconnu

    cert_enc    = 1 if str(has_certification).strip().title() == 'Yes' else 0
    working_enc = 1 if str(is_working).strip().title() == 'Yes' else 0
    spec_enc    = group_spec(ug_specialization)
    skills_cnt  = len([s for s in str(skills).split(',') if s.strip()])
    masters_enc = 1 if has_masters else 0

    # Normaliser le CGPA
    cgpa_norm = cgpa * 10 if cgpa <= 10 else cgpa

    # Construire le vecteur
    X_new = np.array([[gender_enc, deg_enc, spec_enc, cgpa_norm,
                        cert_enc, working_enc, skills_cnt, masters_enc]])

    X_new_sc = scaler.transform(X_new)

    # Prédiction + probabilités des voisins
    prediction   = knn_final.predict(X_new_sc)[0]
    distances, indices = knn_final.kneighbors(X_new_sc)
    neighbor_labels = y.iloc[indices[0]].values
    unique, counts  = np.unique(neighbor_labels, return_counts=True)
    confidence = dict(zip(unique, (counts / best_k * 100).round(1)))

    return prediction, confidence


# ══════════════════════════════════════════════
#  ÉTAPE 7 — TESTS DE DÉMONSTRATION
# ══════════════════════════════════════════════
print("\n" + "=" * 55)
print("  DÉMONSTRATION — RECOMMANDATIONS")
print("=" * 55)

profils_test = [
    {
        "nom"              : "Ahmed",
        "gender"           : "Male",
        "ug_degree"        : "B.E",
        "ug_specialization": "Computer Science Engineering",
        "cgpa"             : 78,
        "has_certification": "Yes",
        "is_working"       : "Yes",
        "skills"           : "Python, SQL, Machine Learning, Data Analysis",
        "has_masters"      : False,
    },
    {
        "nom"              : "Sara",
        "gender"           : "Female",
        "ug_degree"        : "BA",
        "ug_specialization": "Psychology",
        "cgpa"             : 72,
        "has_certification": "No",
        "is_working"       : "No",
        "skills"           : "Communication, People Management, Leadership",
        "has_masters"      : True,
    },
    {
        "nom"              : "Rami",
        "gender"           : "Male",
        "ug_degree"        : "B.TECH",
        "ug_specialization": "Mechanical Engineering",
        "cgpa"             : 65,
        "has_certification": "Yes",
        "is_working"       : "No",
        "skills"           : "AutoCAD, SolidWorks, Problem Solving",
        "has_masters"      : False,
    },
]

for p in profils_test:
    pred, conf = recommander_carriere(
        p['gender'], p['ug_degree'], p['ug_specialization'],
        p['cgpa'], p['has_certification'], p['is_working'],
        p['skills'], p['has_masters']
    )
    print(f"\n  Profil : {p['nom']}")
    print(f"  Formation   : {p['ug_degree']} — {p['ug_specialization']}")
    print(f"  CGPA        : {p['cgpa']}  |  Skills : {p['skills']}")
    print(f"  ➤ Carrière recommandée : ★ {pred} ★")
    print(f"  Confiance des voisins  : {conf}")
    print("  " + "-"*50)