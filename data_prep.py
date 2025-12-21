import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi

username = 'divyabuni'
repo_id = f'{username}/tourism-package-dataset'

df = pd.read_csv(f'https://huggingface.co/datasets/{repo_id}/resolve/main/tourism.csv')
df_cleaned = df.drop(['Unnamed: 0', 'CustomerID'], axis=1, errors='ignore')
df_cleaned['Gender'] = df_cleaned['Gender'].replace('Fe Male', 'Female')
df_cleaned['MaritalStatus'] = df_cleaned['MaritalStatus'].replace('Unmarried', 'Single')

le = LabelEncoder()
for col in df_cleaned.select_dtypes(include=['object']).columns:
    df_cleaned[col] = le.fit_transform(df_cleaned[col])

train_df, test_df = train_test_split(df_cleaned, test_size=0.2, random_state=42, stratify=df_cleaned['ProdTaken'])
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

api = HfApi()
api.upload_file(path_or_fileobj='train.csv', path_in_repo='train.csv', repo_id=repo_id, repo_type='dataset')
api.upload_file(path_or_fileobj='test.csv', path_in_repo='test.csv', repo_id=repo_id, repo_type='dataset')
print('Data Prep Done.')