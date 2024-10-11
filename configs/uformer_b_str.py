total_iter=200*172
val_interval=1000
log_interval=100
start_save_iter=total_iter-val_interval*10
random_seed=12345
val=True
snapshot_dir='ckpt'
data_root = 'data/'

img_size=[512,512]

model_cfg = dict(
    model_name='UformerSTR',
    img_size=img_size,
    embed_dim=32,
    win_size=8,
    token_projection='linear',
    token_mlp='leff',
    depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],
    # depths=[2, 2, 2, 2, 2, 2, 2, 2, 2],
    modulator=True,
    use_checkpoint=True,
)

train_data_cfg = dict(
    data_type='str',
    data_class='mask',
    data_dir=data_root+'text_rmv/SCUT-EnsText/train',
    img_size=img_size,
    augment=True,
    crop=True,
    batch_size=16,
    num_workers=2,
)

val_data_cfg = dict(
    data_type='str',
    data_class='standard',
    data_dir=data_root+'text_rmv/SCUT-EnsText/test',
    img_size=img_size,
    augment=False,
    batch_size=8,
    num_workers=1,
)

test_data_cfg = dict(
    data_type='str',
    data_class='standard', # mask
    data_dir=data_root+'text_rmv/SCUT-EnsText/test',
    img_size=None,
    augment=False,
)

optim_cfg = dict(
    optim='adamw',
    optim_kwargs=dict(
        lr=2e-5,
        betas=(0.9,0.999),
        weight_decay=0.001,
    ),
    scheduler=None,
)