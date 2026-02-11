sweep_config = {
    'pretrained_path': '/Dataset4/hmh_data/PR_data/checkpoint/multi_checkpoint/uni/epoch=35-step=40176.ckpt', 
    'retake_training': False,
    'batch_size': 128,
    'lr': 1e-4,
    'warmup': 5000,
    'max_epochs': 1226446, 
    'extract_layers': [11],
    'function_layers': 'mean',
    'pool': None,
    'dim_output': 512,
    'temperature': 1.0,
    'without_context': True,
    'margin': 1,
    'p': 2,
    'eps': 1e-6,
    'enable_val': False,
    }

spot_config = {
    'pretrained_path': "/Dataset4/hmh_data/PR_data/checkpoint/bert_checkpoint/w_spatial/epoch=1-step=3868.ckpt",
    'dim_feedforward': 1024,
    'nheads': 16,
    'nlayers': 12,
    'dropout': 0.0,
    'dim_model': 512,
    'batch_first': True,
    'n_tokens': 20340, 
    'context_length': 1500,
    'autoregressive': False,
    'pool': None,
    'learnable_pe': True,
    'spatial_aware': False
    }

visual_config = {
    'pretrained_path': "/home/hmh/weights/uni/pytorch_model.bin",
    'model_name': 'uni', 
    }