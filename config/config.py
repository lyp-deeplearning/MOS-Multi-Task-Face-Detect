cfg_mos_s = {
    'name': 'mos_s',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 260,
    'decay1': 22,
    'decay2': 35,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage2': 1, 'stage3': 2, 'stage4': 3},
    'in_channel': 32,
    'out_channel': 64
}
## mobilenetv2 16
cfg_mos_m = {
    'name': 'mos_m',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'pretrain': True,
    'return_layers': {'feature_2': 1, 'feature_4': 2, 'feature_6': 3},
    'in_channel': 32,
    'out_channel': 64
}
