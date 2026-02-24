import torch


def make_optimizer_1stage(cfg, model):
    params = []
    keys = []
    for key, value in model.named_parameters():
        if "prompt_learner" in key:
            lr = cfg.SOLVER.STAGE1.BASE_LR
            weight_decay = cfg.SOLVER.STAGE1.WEIGHT_DECAY
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            keys += [key]
    if cfg.SOLVER.STAGE1.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE1.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.STAGE1.MOMENTUM)
    elif cfg.SOLVER.STAGE1.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.STAGE1.BASE_LR, weight_decay=cfg.SOLVER.STAGE1.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE1.OPTIMIZER_NAME)(params)
    return optimizer

def make_optimizer_2stage(cfg, model, center_criterion):
    """Stage2 optimizer (paper): lr(image encoder)=5e-6, lr(other modules)=5e-5.

    - Freeze text_encoder and prompt_learner
    - Fine-tune image encoder with a smaller LR
    - Train newly added modules (e.g., TGI/decoder, projections, classifiers) with a larger LR
    """
    params = []

    lr_img = cfg.SOLVER.STAGE2.BASE_LR            # expected 5e-6
    lr_other = cfg.SOLVER.STAGE2.BASE_LR * 10.0   # expected 5e-5

    for key, value in model.named_parameters():
        # Freeze text/prompt in stage2 (paper statement)
        if "text_encoder" in key or "prompt_learner" in key:
            value.requires_grad_(False)
            continue

        if not value.requires_grad:
            continue

        # LR split rule:
        # - image_encoder.*  -> lr_img
        # - BUT MSCSA modules are treated as "other" (newly added) -> lr_other
        if key.startswith("image_encoder.") and ("mscsa" not in key):
            lr = lr_img
        else:
            lr = lr_other

        weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY
        params.append({"params": [value], "lr": lr, "weight_decay": weight_decay})

    if cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(
            params, momentum=cfg.SOLVER.STAGE2.MOMENTUM
        )
    elif cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(
            params, lr=lr_img, weight_decay=cfg.SOLVER.STAGE2.WEIGHT_DECAY
        )
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(params)

    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.STAGE2.CENTER_LR)
    return optimizer, optimizer_center

