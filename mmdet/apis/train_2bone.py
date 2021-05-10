import random
import warnings

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner)
from mmcv.utils import build_from_cfg

from mmdet.core import DistEvalHook, EvalHook
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_root_logger


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_detector_2bone(model, dataset_vis, dataset_ir, cfg, distributed=False, validate=False, timestamp=None, meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    if isinstance(dataset_vis, (list, tuple)):
        dataset_vis = dataset_vis
    else:
        dataset_vis = [dataset_vis] # [<mmdet.datasets.dataset_wrappers.RepeatDataset object at 0x7f307240fba8>]
    # prepare data loaders
    if isinstance(dataset_ir, (list, tuple)):
        dataset_ir = dataset_ir
    else:
        dataset_ir = [dataset_ir] # [<mmdet.datasets.dataset_wrappers.RepeatDataset object at 0x7f307240fba8>]
    
    if 'imgs_per_gpu' in cfg.data_vis:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data_vis:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data_vis.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data_vis.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data_vis.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data_vis.imgs_per_gpu} in this experiments')
        cfg.data_vis.samples_per_gpu = cfg.data_vis.imgs_per_gpu

    # cfg.gpus will be ignored if distributed
    data_loaders_vis = [build_dataloader(ds, cfg.data_vis.samples_per_gpu, cfg.data_vis.workers_per_gpu, len(cfg.gpu_ids), dist=False, shuffle=False, seed=cfg.seed) for ds in dataset_vis]
    data_loaders_ir = [build_dataloader(ds, cfg.data_ir.samples_per_gpu, cfg.data_ir.workers_per_gpu, len(cfg.gpu_ids), dist=False, shuffle=False, seed=cfg.seed) for ds in dataset_ir]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        print("cfg.gpu_ids[0] = ",cfg.gpu_ids[0])
        print("cfg.gpu_ids = ",cfg.gpu_ids)
        model = MMDataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # here enter
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data_vis.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data_vis.val.pipeline = replace_ImageToTensor(
                cfg.data_vis.val.pipeline)
        val_dataset_vis = build_dataset(cfg.data_vis.val, dict(test_mode=True))
        val_dataset_ir = build_dataset(cfg.data_ir.val, dict(test_mode=True))
        val_dataloader_vis = build_dataloader(
            val_dataset_vis,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data_vis.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        val_dataloader_ir = build_dataloader(
            val_dataset_ir,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data_vis.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader_vis, **eval_cfg))
        runner.register_hook(eval_hook(val_dataloader_ir, **eval_cfg))
        runner.val_dataloader_vis = val_dataloader_vis
        runner.val_dataloader_ir = val_dataloader_ir

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    runner.run(data_loaders_vis, data_loaders_ir, cfg.workflow)
