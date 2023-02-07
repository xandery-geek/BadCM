import os
import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import TensorBoardLogger


def get_save_name(cfg):
    percentage = cfg['percentage']

    attack_method = 'Nomal' if percentage == 0 else cfg['attack']
    if attack_method == 'BadCM' and cfg['badcm'] is not None:
        attack_method = attack_method + cfg['badcm']

    save_name = '{}_{}_{}_p={}_t={}'.format(cfg['module_name'], cfg['dataset'], attack_method, percentage, cfg['trial_tag'])
    return save_name


def run_cmr(module, cfg):

    percentage = cfg['percentage']
    save_name = cfg['save_name']
    
    print("save_name: {}".format(save_name))

    checkpoint_dir = 'checkpoints/' + save_name
    checkpoint_callback = callbacks.ModelCheckpoint(
        monitor='val_map', 
        dirpath=checkpoint_dir,
        save_last=True,
        mode='max')

    tb_logger = TensorBoardLogger('log/tensorboard', save_name) if cfg["enable_tb"] else False
    trainer = pl.Trainer(
        devices=len(cfg['device']),
        accelerator='gpu',
        max_epochs=cfg['epochs'],
        check_val_every_n_epoch=cfg["valid_interval"],
        callbacks=[checkpoint_callback],
        logger=tb_logger
    )
    
    
    train_loader = module.poi_train_loader if percentage > 0 else module.train_loader
    test_loader = module.test_loader

    if cfg['phase'] == 'train':
        module.flogger.log("=> Training on poisoned data with p={} and target={}".format(percentage, cfg['target']))
        trainer.fit(
            model=module, 
            ckpt_path=cfg["checkpoint"], 
            train_dataloaders=train_loader, 
            val_dataloaders=test_loader
        )

    ckpt = (cfg["checkpoint"] or os.path.join(checkpoint_dir, 'last.ckpt')) if cfg['phase'] == 'test' else 'best'

    if percentage > 0:
        module.flogger.log("=> Testing on poisoned data with p={} and target={}".format(percentage, cfg['target']))
        trainer.test(model=module, dataloaders=module.poi_test_loader, ckpt_path=ckpt)

    module.flogger.log("=> Testing on clean data ...")
    trainer.test(model=module, dataloaders=test_loader, ckpt_path=ckpt)
