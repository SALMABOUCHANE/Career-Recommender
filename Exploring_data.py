import pandas as pd

df = pd.read_csv('career_recommender.csv')

print('Shape:', df.shape)
print('\nColumns:', df.columns.tolist())
print('\nMissing values:\n', df.isnull().sum())
print('\nDtypes:\n', df.dtypes)
print('\nTarget column (job title) unique values sample:')
print(df.iloc[:, 10].value_counts().head(20))

# Check CGPA column issues
print('\nCGPA unique problematic values:')
cgpa = df['What was the average CGPA or Percentage obtained in under graduation?']
non_numeric = cgpa[pd.to_numeric(cgpa, errors='coerce').isna()]
print(non_numeric.value_counts())

# Check job title NA values
print('\nJob title NA values:')
job_col = df.columns[10]
print(df[job_col].isna().sum(), 'NaN')
print(df[df[job_col].str.strip().str.upper() == 'NA'][job_col].count(), 'string NA')
