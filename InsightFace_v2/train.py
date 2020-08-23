import os
from shutil import copyfile

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from config import device, grad_clip, print_freq
from data_gen import ArcFaceDataset, AdverserialFaceDataset
from focal_loss import FocalLoss
from lfw_eval import lfw_test
from models import resnet18, resnet34, resnet50, resnet101, resnet152, ArcMarginModel
from optimizer import InsightFaceOptimizer
from utils import parse_args, save_checkpoint, AverageMeter, accuracy, get_logger


def full_log(epoch):
    full_log_dir = 'data/full_log'
    if not os.path.isdir(full_log_dir):
        os.mkdir(full_log_dir)
    filename = 'angles_{}.txt'.format(epoch)
    dst_file = os.path.join(full_log_dir, filename)
    src_file = 'data/angles.txt'
    copyfile(src_file, dst_file)


def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_acc = float('-inf')
    writer = SummaryWriter()
    epochs_since_improvement = 0
    is_adverserial = args.adverserial

    # Initialize / load checkpoint
    if checkpoint is None:
        if args.network == 'r18':
            model = resnet18(args)
        elif args.network == 'r34':
            model = resnet34(args)
        elif args.network == 'r50':
            model = resnet50(args)
        elif args.network == 'r101':
            model = resnet101(args)
        elif args.network == 'r152':
            model = resnet152(args)
        else:
            raise TypeError('network {} is not supported.'.format(args.network))

        # print(model)
        model = nn.DataParallel(model)
        metric_fc = ArcMarginModel(args)
        metric_fc = nn.DataParallel(metric_fc)

        if args.optimizer == 'sgd':
            optimizer = InsightFaceOptimizer(
                torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay))
        else:
            optimizer = InsightFaceOptimizer(
                torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                 lr=args.lr, weight_decay=args.weight_decay))

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        try:
            model = model.module
        except:
            pass
        model = nn.DataParallel(model)
        metric_fc = checkpoint['metric_fc']
        try:
            metric_fc = metric_fc.module
        except:
            pass
        metric_fc = nn.DataParallel(metric_fc)
        optimizer = checkpoint['optimizer']

    logger = get_logger()

    # Move to GPU, if available
    model = model.to(device)
    metric_fc = metric_fc.to(device)

    # Loss function
    if args.focal_loss:
        criterion = FocalLoss(gamma=args.gamma).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    train_dataset = ArcFaceDataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    if is_adverserial:
        adv_dataset = AdverserialFaceDataset('train')
        adv_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=args.batch_size//2, shuffle=True, num_workers=4)
        adv_criterion = nn.BCEWithLogitsLoss().to(device)
    max_train_rounds = 1000 if is_adverserial else -1

    # Epochs
    for epoch in range(start_epoch, args.end_epoch):
        # One epoch's training

        # Standard epoch
        train_loss, train_acc = train(train_loader=train_loader,
                                      model=model,
                                      metric_fc=metric_fc,
                                      criterion=criterion,
                                      optimizer=optimizer,
                                      epoch=epoch,
                                      logger=logger,
                                      max_rounds=max_train_rounds)

        writer.add_scalar('model/train_loss', train_loss, epoch)
        writer.add_scalar('model/train_acc', train_acc, epoch)

        logger.info('Learning rate={}, step number={}\n'.format(optimizer.lr, optimizer.step_num))

        # One epoch's validation
        lfw_acc, threshold = lfw_test(model)
        # lfw_acc, threshold = 0, 75

        writer.add_scalar('model/valid_acc', lfw_acc, epoch)
        writer.add_scalar('model/valid_thres', threshold, epoch)

        # Check if there was an improvement
        is_best = lfw_acc > best_acc
        best_acc = max(lfw_acc, best_acc)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        if is_adverserial:
            # Adverserial epoch
            train_loss, train_acc = train_adv(train_loader=adv_loader,
                                          model=model,
                                          threshold=threshold,
                                          criterion=adv_criterion,
                                          optimizer=optimizer,
                                          epoch=epoch,
                                          logger=logger)

            writer.add_scalar('model/adv_train_loss', train_loss, epoch)
            writer.add_scalar('model/adv_train_acc', train_acc, epoch)

            logger.info('Learning rate={}, step number={}\n'.format(optimizer.lr, optimizer.step_num))

            # One epoch's validation
            lfw_acc, threshold = lfw_test(model)
            writer.add_scalar('model/adv_valid_acc', lfw_acc, epoch)
            writer.add_scalar('model/adv_valid_thres', threshold, epoch)

            # Check if there was an improvement
            is_best = lfw_acc > best_acc
            best_acc = max(lfw_acc, best_acc)
            if not is_best:
                epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
            else:
                epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, metric_fc, optimizer, best_acc, is_best)


def train(train_loader, model, metric_fc, criterion, optimizer, epoch, logger, max_rounds=-1):
    model.train()  # train mode (dropout and batchnorm is used)
    metric_fc.train()

    losses = AverageMeter()
    top5_accs = AverageMeter()

    # Batches
    for i, (img, label) in enumerate(train_loader):
        # Move to GPU, if available
        img = img.to(device)
        label = label.to(device)  # [N, 1]

        # Forward prop.
        feature = model(img)  # embedding => [N, 512]
        output = metric_fc(feature, label)  # class_id_out => [N, 93431]

        # Calculate loss
        loss = criterion(output, label)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        optimizer.clip_gradient(grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())
        top5_accuracy = accuracy(output, label, 5)
        top5_accs.update(top5_accuracy)

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Top5 Accuracy {top5_accs.val:.3f} ({top5_accs.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                                         loss=losses,
                                                                                         top5_accs=top5_accs))
        # NOAM: Trying to overfit the adverserial training, make this part shorter
        if 0 < max_rounds <= i:
            break

    return losses.avg, top5_accs.avg


def train_adv(train_loader, model, threshold, criterion, optimizer, epoch, logger):
    model.train()  # train mode (dropout and batchnorm is used)
    #metric_fc.train()

    losses = AverageMeter()
    top5_accs = AverageMeter()

    # Batches
    for i, (img1, img2, label) in enumerate(train_loader):
        # Move to GPU, if available
        bs = img1.shape[0]
        img1 = img1.to(device)
        img2 = img2.to(device)
        label = label.to(device).type_as(img1)  # [N, 1]

        #img1 = img1[:1,:,:,:]
        imgs_concatted = torch.cat((img1, img2), 0)
        features_concat = model(imgs_concatted)

        f1 = features_concat[:bs]
        f2 = features_concat[bs:]
        assert f1.shape == f2.shape
        # Forward prop.
        #f1 = model(img1)  # embedding => [N, 512]
        #f2 = model(img1)  # embedding => [N, 512]

        # https://github.com/pytorch/pytorch/issues/18027 No batch dot product
        dot_product = (f1 * f2).sum(-1)
        normalized = (f1.norm(dim=1) * f2.norm(dim=1) + 1e-5)
        cosdistance = dot_product / normalized
        # Change from -1 (opposite) -> 1 (same) range to 0 (same) - 1 (different)
        features_diff = (torch.ones_like(cosdistance) - cosdistance) / 2
        features_diff = features_diff.unsqueeze(1)


        #output = metric_fc(feature, label)  # class_id_out => [N, 93431]
        #threshold_decision = features_diff.sum(dim=1) - threshold
        threshold_decision = dot_product - threshold

        # Calculate loss
        loss = criterion(threshold_decision, label)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        optimizer.clip_gradient(grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())
        #top5_accuracy = accuracy(threshold_decision, label, 5)
        #top5_accs.update(top5_accuracy)

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Adv Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Adv Top5 Accuracy {top5_accs.val:.3f} ({top5_accs.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                                         loss=losses,
                                                                                         top5_accs=top5_accs))

    return losses.avg, top5_accs.avg


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
