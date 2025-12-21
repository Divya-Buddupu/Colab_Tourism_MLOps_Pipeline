import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from huggingface_hub import HfApi

username = 'divyabuni'
dataset_repo = f'{username}/tourism-package-dataset'
model_repo = f'{username}/tourism-package-model'

train_df = pd.read_csv(f'https://huggingface.co/datasets/{dataset_repo}/resolve/main/train.csv')
X_train = train_df.drop('ProdTaken', axis=1)
y_train = train_df['ProdTaken']

# Save feature names for Step 8 (app.py)
feature_names = X_train.columns.tolist()
joblib.dump(feature_names, 'features.pkl')

model = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, 'model.pkl')

api = HfApi()
api.create_repo(repo_id=model_repo, repo_type='model', exist_ok=True)
api.upload_file(path_or_fileobj='model.pkl', path_in_repo='model.pkl', repo_id=model_repo, repo_type='model')
api.upload_file(path_or_fileobj='features.pkl', path_in_repo='features.pkl', repo_id=model_repo, repo_type='model')
print('Training Done with Feature Metadata.')