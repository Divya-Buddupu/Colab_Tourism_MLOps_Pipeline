from huggingface_hub import HfApi
username = 'divyabuni'
space_id = f'{username}/tourism-package-predictor'
api = HfApi()
api.create_repo(repo_id=space_id, repo_type='space', space_sdk='docker', exist_ok=True)
for file in ['app.py', 'Dockerfile', 'requirements.txt']:
    api.upload_file(path_or_fileobj=file, path_in_repo=file, repo_id=space_id, repo_type='space')
print('Deployment Done.')