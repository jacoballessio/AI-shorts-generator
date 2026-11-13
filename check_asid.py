import torch
print('torch', torch.__version__)
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
print('transformers ok')
import qwen_omni_utils
print('qwen_omni_utils ok')
