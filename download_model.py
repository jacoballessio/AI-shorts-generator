from huggingface_hub import hf_hub_download
import os

os.makedirs('Z:/Models', exist_ok=True)
print('Downloading Llama 3.1 8B Q4_K_M (~4.5GB)...')
print('This may take several minutes depending on your connection...')

hf_hub_download(
    repo_id='bartowski/Meta-Llama-3.1-8B-Instruct-GGUF',
    filename='Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf',
    local_dir='Z:/Models',
    local_dir_use_symlinks=False
)

print('\n✅ Download complete! Model saved to Z:/Models/')
print('\n📝 Add this to your .env file:')
print('LOCAL_LLM_PATH=Z:/Models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf')
