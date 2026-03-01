import pandas as pd

df = pd.read_csv('career_recommender.csv')

# ── 1. Rename columns ──────────────────────────────────────────────────────────
df.columns = [
    'name', 'gender', 'ug_degree', 'ug_specialization',
    'interests', 'skills', 'cgpa', 'has_certification',
    'certification_details', 'is_working', 'job_title', 'masters_field'
]

# ── 2. Drop name (not useful for ML) ──────────────────────────────────────────
df.drop(columns=['name'], inplace=True)

# ── 3. Standardize gender ──────────────────────────────────────────────────────
df['gender'] = df['gender'].str.strip().str.title()

# ── 4. Standardize yes/no columns ─────────────────────────────────────────────
for col in ['has_certification', 'is_working']:
    df[col] = df[col].str.strip().str.title().map({'Yes': 1, 'No': 0})

# ── 5. Clean CGPA (convert to float, normalize scale) ─────────────────────────
# If value <= 10, it's a CGPA (out of 10) → multiply by 10 to match percentage scale
df['cgpa'] = pd.to_numeric(df['cgpa'], errors='coerce')
df['cgpa_normalized'] = df['cgpa'].apply(lambda x: x * 10 if pd.notna(x) and x <= 10 else x)

# ── 6. Clean job_title (target variable) ──────────────────────────────────────
df['job_title'] = df['job_title'].fillna('Unemployed/Student')
df['job_title'] = df['job_title'].str.strip()
df['job_title'] = df['job_title'].replace({
    'NA': 'Unemployed/Student',
    'Student (Unemployed)': 'Unemployed/Student'
})
df['job_title'] = df['job_title'].str.title()

# Merge near-duplicate job titles
df['job_title'] = df['job_title'].replace({
    'Software developer': 'Software Developer',
    'Software Engineer Trainee': 'Software Engineer',
    'Senior Software Engineer': 'Software Engineer',
    'Teaching': 'Teacher',
})

# ── 7. Clean masters_field ────────────────────────────────────────────────────
df['masters_field'] = df['masters_field'].fillna('None')
df['masters_field'] = df['masters_field'].str.strip()

# ── 8. Clean skills & interests (strip extra whitespace/newlines) ──────────────
df['skills'] = df['skills'].fillna('Unknown').str.replace(r'\s+', ' ', regex=True).str.strip()
df['interests'] = df['interests'].str.replace(r'\s+', ' ', regex=True).str.strip()

# ── 9. Clean ug_degree & specialization ───────────────────────────────────────
df['ug_degree'] = df['ug_degree'].str.strip().str.upper()
df['ug_specialization'] = df['ug_specialization'].str.strip().str.title()

# ── 10. Drop rows where cgpa is still null ────────────────────────────────────
df.dropna(subset=['cgpa'], inplace=True)

# ── 11. Keep only rows with a valid job title for ML dataset ──────────────────
df_ml = df[df['job_title'] != 'Unemployed/Student'].copy()

# ── 12. Remove job titles with very few samples (< 3) ─────────────────────────
title_counts = df_ml['job_title'].value_counts()
valid_titles = title_counts[title_counts >= 3].index
df_ml = df_ml[df_ml['job_title'].isin(valid_titles)]

# ── Save both versions ────────────────────────────────────────────────────────
df.to_csv('career_cleaned_full.csv', index=False)
df_ml.to_csv('career_cleaned_ml.csv', index=False)

print("✅ Full cleaned dataset:", df.shape)
print("✅ ML-ready dataset (with valid job titles):", df_ml.shape)
print("\nFinal columns:", df_ml.columns.tolist())
print("\nTop job titles (target):")
print(df_ml['job_title'].value_counts().head(15))
print("\nMissing values in ML dataset:")
print(df_ml.isnull().sum())
