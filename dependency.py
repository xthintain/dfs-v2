import subprocess
import os
from huggingface_hub import snapshot_download
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_TOKEN'] = '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'
def install():
    subprocess.call(['pip', 'install', '-r', 'requirements.txt'])
    subprocess.call(['pip', 'install','torch', 'torchvision', 'torchaudio','--index-url https://download.pytorch.org/whl/cu124'])# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

def mkdirs():
    os.makedirs('./models/stablediffusion/MeinaMix_V11',exist_ok=True)
    os.makedirs('./models/stablediffusion/diffusion_pytorch',exist_ok=True)
    os.makedirs('./models/GPTSovits',exist_ok=True)
    snapshot_download('Meina/MeinaMix_V11', local_dir='./models/stablediffusion/MeinaMix_V11')
    snapshot_download('lllyasviel/control_v11p_sd15_canny', local_dir='./models/stablediffusion/diffusion_pytorch')


if __name__ == '__main__':
    install()
    mkdirs()