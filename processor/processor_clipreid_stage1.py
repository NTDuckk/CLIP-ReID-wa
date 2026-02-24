import logging
import os
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from torch.cuda import amp
import torch.distributed as dist
import collections
from torch.nn import functional as F

def do_train_stage1(cfg,
             model,
             train_loader_stage1,
             optimizer,
             scheduler,
             local_rank):
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD 

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  

    loss_meter = AverageMeter()
    scaler = amp.GradScaler()
    # Stage1 loss (paper Eq.1-3): Li2t + Lt2i
    eps = 1e-12
    
    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))
    image_features = []
    labels = []
    with torch.no_grad():
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage1):
            img = img.to(device)
            target = vid.to(device)
            with amp.autocast(enabled=True):
                image_feature = model(img, target, get_image = True)
                for i, img_feat in zip(target, image_feature):
                    labels.append(i)
                    image_features.append(img_feat.cpu())
        labels_list = torch.stack(labels, dim=0).cuda() #N
        image_features_list = torch.stack(image_features, dim=0).cuda()

        batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH
        num_image = labels_list.shape[0]
        i_ter = num_image // batch
    del labels, image_features

    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        scheduler.step(epoch)
        model.train()

        iter_list = torch.randperm(num_image).to(device)
        for i in range(i_ter+1):
            optimizer.zero_grad()
            if i != i_ter:
                b_list = iter_list[i*batch:(i+1)* batch]
            else:
                b_list = iter_list[i*batch:num_image]
            
            target = labels_list[b_list]
            image_features = image_features_list[b_list]
            with amp.autocast(enabled=True):
                text_features = model(label=target, get_text=True)

            # Normalize features (cosine similarity) and use CLIP's logit_scale
            if isinstance(model, nn.DataParallel):
                logit_scale = model.module.logit_scale.exp().float()
            else:
                logit_scale = model.logit_scale.exp().float()

            v = F.normalize(image_features.float(), dim=-1)
            t = F.normalize(text_features.float(), dim=-1)

            # Similarity matrix s(V_i, T_j)
            sim = logit_scale * (v @ t.t())  # (B,B)

            # Eq.(1) Li2t: standard CE with diagonal positives
            targets_i2t = torch.arange(sim.size(0), device=sim.device)
            loss_i2t = F.cross_entropy(sim, targets_i2t)

            # Eq.(2) Lt2i: sum of positives mass for each text column
            exp_sim = torch.exp(sim)
            labels = target
            pos_mask = (labels.view(1, -1) == labels.view(-1, 1)).float()  # (B,B)  (row=image, col=text)
            numer = (exp_sim * pos_mask).sum(dim=0)  # (B,)
            denom = exp_sim.sum(dim=0)               # (B,)
            loss_t2i = (-torch.log((numer + eps) / (denom + eps))).mean()

            loss = loss_i2t + loss_t2i

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), target.size(0))

            torch.cuda.synchronize()
            if (i + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (i + 1), len(train_loader_stage1),
                                    loss_meter.avg, scheduler._get_lr(epoch)[0]))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Stage1 running time: {}".format(total_time))
