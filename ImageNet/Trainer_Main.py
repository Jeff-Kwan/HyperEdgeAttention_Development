'''
Use this file with torchrun.
Single Node (Machine) Command:
torchrun --standalone --nproc_per_node=gpu Training/Trainer_Main.py

torchrun --standalone --nproc_per_node=1 Training/Trainer_Main.py

Consider using Slurm for multi-node training.
'''
import torch
from torch.distributed import init_process_group, destroy_process_group
from datasets import load_dataset
from datetime import datetime
import random
import numpy as np


import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Distributed Trainer Class
from Functions.Trainer import Trainer, distributed_dataloader

# Developed Available Models
from Model.LangCAT import LangCAT, LangCAT_Config
from Model.VisionCAT import VisionCAT, VisionCAT_Config

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    init_process_group(backend="nccl")

    # Retrieve the local rank from environment variables set by torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # Configuration
    if args['modality'] == 'language' or args['modality'] == 'lang':
        config = LangCAT_Config()
    elif args['modality'] == 'vision' or args['modality'] == 'vis':
        config = VisionCAT_Config()
    else:
        raise ValueError(f"Modality {args['modality']} not recognized.")
    
    if args['config']:
        config.load(args['config'])
    else:
        raise ValueError("No configuration file provided.")
    
    # Model
    if args['modality'] == 'language' or args['modality'] == 'lang':
        model = LangCAT(config)
    elif args['modality'] == 'vision' or args['modality'] == 'vis':
        model = VisionCAT(config)
    else:
        raise ValueError(f"Modality {args['modality']} not recognized.")

    # Directories
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    now = datetime.now()
    timestamp = now.strftime("%H-%M")
    date = now.strftime("%Y-%m-%d")
    config.timestamp = timestamp
    config.date = date
    config.output_dir = os.path.join(root_dir, 'Output', date, f'{timestamp}-{config.name}-{config.data}')
    os.makedirs(config.output_dir, exist_ok=True)

    # Manage Batch Sizes
    config.batch = config.batch // torch.cuda.device_count()
    config.minibatch = config.minibatch // torch.cuda.device_count() if config.minibatch else None

    # Manual Seed
    if config.manual_seed:
        torch.cuda.manual_seed_all(config.manual_seed)

    # Precision
    torch.set_float32_matmul_precision(config.matmul_precision)    

    # Need to generalize code to get dataset class
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from Streaming.Training_Streaming_Language import TextDatasetStreaming
    torch.autograd.set_detect_anomaly(True)
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
    dataset = load_dataset('Skylion007/openwebtext', split='train', streaming=False, trust_remote_code=True)
    train_dataset = TextDatasetStreaming(dataset.to_iterable_dataset(), tokenizer, config.seq_len)


    # Distributed dataloader
    train_loader = distributed_dataloader(train_dataset, config.batch)

    trainer = Trainer(config, model, train_loader)
    trainer.train()

    destroy_process_group()





if __name__ == '__main__':
    args = {
        'modality': 'language',
        'config': 'Training/Example-LCAT-OpenWebText-config.json',
    }
    main(args)