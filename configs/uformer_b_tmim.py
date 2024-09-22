total_iter=10*13884
val_interval=13884
log_interval=100
start_save_iter=total_iter-val_interval*10
random_seed=12345
val=True
snapshot_dir='ckpt'
data_root = 'data/'

img_size=[512, 512]

model_cfg = dict(
    model_name='UformerTMIM',
    img_size=img_size,
    embed_dim=32,
    win_size=8,
    token_projection='linear',
    token_mlp='leff',
    depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],
    # depths=[2, 2, 2, 2, 2, 2, 2, 2, 2],
    modulator=True,
    use_checkpoint=True,
    tmim=True,
)

train_data_cfg = dict(
    data_type='tmim',
    data_class=[
        dict(
            data_class='2mask',
            data_dir=data_root+'text_det/textocr/train_images',
            anno_dir=data_root+'text_det/textocr/TextOCR_0.1_train.json',
            variant='textocr',
            img_size=img_size,
            augment=True,
            crop=False,
        ),
        dict(
            data_class='2mask',
            data_dir=data_root+'text_det/textocr/train_images',
            anno_dir=data_root+'text_det/textocr/TextOCR_0.1_val.json',
            variant='textocr',
            img_size=img_size,
            augment=True,
            crop=False,
        ),
        dict(
            data_class='2mask',
            data_dir=data_root+'text_det/totaltext/train_images',
            anno_dir=data_root+'text_det/totaltext/annotation.json',
            variant='totaltext',
            img_size=img_size,
            augment=True,
            crop=False,
        ),
        dict(
            data_class='2mask',
            data_dir=data_root+'text_det/ic15/train_images',
            anno_dir=data_root+'text_det/ic15/annotation.json',
            variant='ic15',
            img_size=img_size,
            augment=True,
            crop=False,
        ),
        dict(
            data_class='2mask',
            data_dir=data_root+'text_det/cocotext/train2014',
            anno_dir=data_root+'text_det/cocotext/cocotext.v2.json',
            variant='cocotext',
            img_size=img_size,
            augment=True,
            crop=False,
        ),
        dict(
            data_class='2mask',
            data_dir=data_root+'text_det/mlt19/train_images',
            anno_dir=data_root+'text_det/mlt19/annotation.json',
            variant='mlt',
            img_size=img_size,
            augment=True,
            crop=False,
        ),
        dict(
            data_class='2mask',
            data_dir=data_root+'text_det/art/train_images',
            anno_dir=data_root+'text_det/art/annotation.json',
            variant='art',
            img_size=img_size,
            augment=True,
            crop=False,
        ),
        dict(
            data_class='2mask',
            data_dir=data_root+'text_det/lsvt/train_images',
            anno_dir=data_root+'text_det/lsvt/annotation.json',
            variant='lsvt',
            img_size=img_size,
            augment=True,
            crop=False,
        ),
        dict(
            data_class='2mask',
            data_dir=data_root+'text_det/rects/img',
            anno_dir=data_root+'text_det/rects/annotation.json',
            variant='rects',
            img_size=img_size,
            augment=True,
            crop=False,
        ),
    ],
    batch_size=8,
    num_workers=1,
)

val_data_cfg = dict(
    data_type='str',
    data_class='standard',
    data_dir=data_root+'text_rmv/SCUT-EnsText/test',
    img_size=img_size,
    augment=False,
    batch_size=8,
    num_workers=4,
)

test_data_cfg = dict(
    data_type='str',
    data_class='mask',
    data_dir=data_root+'text_rmv/SCUT-EnsText/test',
    img_size=None,
    augment=False,
)

optim_cfg = dict(
    optim='adamw',
    optim_kwargs=dict(
        lr=2e-4,
        betas=(0.9,0.999),
        weight_decay=0.02,
    ),
    scheduler='OneCycleLR',
    scheduler_kwargs=dict(
        max_lr=2e-4,
        total_steps=total_iter,
        pct_start=0.075,
        cycle_momentum=False,
    ),
)
