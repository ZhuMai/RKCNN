import argparse
import shutil
import datetime
import time
import random
import os


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import pytorch_models.RKCNN as RKCNN


parser = argparse.ArgumentParser(description='RKCNN CIFAR Training')

parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')

parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run (default: 300)')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate (default: 0.1)')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')

parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')

parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

parser.add_argument('--attention', dest='attention', action='store_true', help='use attentional transition')

########################################################################
parser.add_argument('--k1', default=120, type=int,
                    metavar='Growth Rate 1', help='the number of filters per layer in period 1 (default: 120)')

parser.add_argument('--k2', default=120, type=int,
                    metavar='Growth Rate 2', help='the number of filters per layer in period 2 (default: 120)')

parser.add_argument('--k3', default=120, type=int,
                    metavar='Growth Rate 3', help='the number of filters per layer in period 3 (default: 120)')

parser.add_argument('--s1', default=5, type=int,
                    metavar='Stage 1', help='the stage of RK method in period 1 (default: 5)')

parser.add_argument('--s2', default=5, type=int,
                    metavar='Stage 2', help='the stage of RK method in period 2 (default: 5)')

parser.add_argument('--s3', default=5, type=int,
                    metavar='Stage 3', help='the stage of RK method in period 3 (default: 5)')

parser.add_argument('--keep_prob', default=0.8, type=float, metavar='K',
                    help='1 - dropout_rate (default: 0.8)')

parser.add_argument('--out_features', default=10, type=int,
                    metavar='O', help='the number of out features (default: 10)')

parser.add_argument('--bottleneck', dest='bottleneck', action='store_true', help='use bottleneck')

parser.add_argument('--save', default=None, type=str, metavar='PATH',
                    help='path to save checkpoint')

parser.add_argument('--update1', default=1, type=int,
                    metavar='L1', help='the update num in period 1 (default: 1)')

parser.add_argument('--update2', default=1, type=int,
                    metavar='L2', help='the update num in period 2 (default: 1)')

parser.add_argument('--update3', default=1, type=int,
                    metavar='L3', help='the update num in period 3 (default: 1)')

parser.add_argument('--r1', default=1, type=int, 
                    help='the number of steps in period 1 (default: 1)')

parser.add_argument('--r2', default=1, type=int, 
                    help='the number of steps in period 2 (default: 1)')

parser.add_argument('--r3', default=1, type=int, 
                    help='the number of steps in period 3 (default: 1)')

parser.add_argument('--neck', default=1, type=int, metavar='w',
                    help='bottleneck outputs wk channels (default: 1)')

parser.add_argument('--no_multiscale', dest='no_multiscale', action='store_true',
                    help='not to use multiscale feature strategy')

parser.add_argument('--data_augmentation', dest='data_augmentation', action='store_true',
                    help='use data augmentation')

parser.add_argument('--lrD1', default=0.5, type=float,
                    help='milestone to lr decay')

parser.add_argument('--lrD2', default=0.75, type=float,
                    help='milestone to lr decay again')

parser.add_argument('--gamma', default=0.1, type=float,
                    help='Multiplicative factor of learning rate decay')

parser.add_argument('--cos', dest='cos', action='store_true',
                    help='whether to use Cosine Annealing LR')

parser.add_argument('--replace', dest='replace', action='store_true',
                    help='replace in the alternate update.')
########################################################################

best_prec1 = 0


def main():

    global args, best_prec1
    args = parser.parse_args()
    if args.attention:
        print('Attentional transition is used.')

    if args.bottleneck:
        print('Bottleneck is used.')
    else:
        print("Only bottelneck mode is supportted now.")
        return

    if args.no_multiscale:
        print('Multiscale is not used.')
    else:
        print("Multiscale is used.")

    if 0 == args.update1 and 0 == args.update2 and 0 == args.update3:
        print('This model is an RKCNN-E.')
    elif args.replace:
        print('This model is an RKCNN-I.')
    else:
        print('This model is an RKCNN-R.')

    # create model
    model = RKCNN.build_RKCNN(growth_rates=[args.k1, args.k2, args.k3],
            stages=[args.s1, args.s2, args.s3], if_att=args.attention, update_nums=[args.update1,args.update2,args.update3],
            keep_prob=args.keep_prob, out_features=args.out_features, if_b=args.bottleneck,
            steps=[args.r1, args.r2, args.r3], neck=args.neck, multiscale=not args.no_multiscale, replace=args.replace)
    model = nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    if args.cos:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.lrD1 * args.epochs, args.lrD2 * args.epochs],
                                                     gamma=args.gamma)

   # # optionally resume from a checkpoint
    if args.resume:
        filename = args.resume + '/RKCNN.pth.tar'
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(filename))

        if args.save is None:
            args.save = args.resume
    cudnn.benchmark = True


    # Data loading code
    # The output of torchvision datasets are PILImage images of range [0, 255].
    # We transform them to Tensors of normalized range [-1, 1].

    normalize_10 = transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
    normalize_100 = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])

    if args.data_augmentation:
        if 10 == args.out_features:
            trainset = datasets.CIFAR10(root='./data', train=True, download=True,
                                        transform=transforms.Compose(
                                            [transforms.RandomHorizontalFlip(),
                                             transforms.RandomCrop(32, padding=4),
                                             transforms.ToTensor(),
                                             normalize_10]))
        elif 100 == args.out_features:
            trainset = datasets.CIFAR100(root='./data', train=True, download=True, 
                                        transform=transforms.Compose(
                                            [transforms.RandomHorizontalFlip(),
                                             transforms.RandomCrop(32, padding=4),
                                             transforms.ToTensor(),
                                             normalize_100]))
    else:
        if 10 == args.out_features:
            trainset = datasets.CIFAR10(root='./data', train=True, download=True,
                                        transform=transforms.Compose(
                                            [transforms.ToTensor(),
                                             normalize_10]))
        elif 100 == args.out_features:
            trainset = datasets.CIFAR100(root='./data', train=True, download=True, 
                                        transform=transforms.Compose(
                                            [transforms.ToTensor(),
                                             normalize_100]))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.workers, pin_memory=True)

    if 10 == args.out_features:
        testset = datasets.CIFAR10(root='./data', train=False, download=True, 
                                    transform=transforms.Compose(
                                        [transforms.ToTensor(),
                                         normalize_10]))
    elif 100 == args.out_features:
        testset = datasets.CIFAR100(root='./data', train=False, download=True, 
                                    transform=transforms.Compose(
                                        [transforms.ToTensor(),
                                         normalize_100]))
    val_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.workers, pin_memory=True)


    if args.evaluate:
        filename = args.save + '/best_RKCNN.pth.tar'
        if os.path.isfile(filename):
            print("=> loading best checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            model.load_state_dict(checkpoint['state_dict'])
            validate(val_loader, model, criterion)
        else:
            print("=> no best checkpoint found at '{}'".format(filename))

        return

    count = get_number_of_param(model)

    if args.save:
        isExists=os.path.exists(args.save)
        if False == isExists:
            os.mkdir(args.save)
        filename = args.save + '/RKCNN.pth.tar'
        best_filename = args.save + '/best_RKCNN.pth.tar'
    else:
        args.save = '.'
        filename = './RKCNN.pth.tar'
        best_filename = './best_RKCNN.pth.tar'

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        scheduler.step()

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filename, best_filename)

        print('Best top 1 err till now: %.3f%%'%(100-best_prec1))

    with open("results.txt", "a") as text_file:
        text_file.write(args.save + ': Params: %d. Best top 1 err: %.3f%%.\n' % (count, 100-best_prec1))


def get_number_of_param(model):
    """get the number of param for every element"""
    count = 0
    for param in model.parameters():
        param_size = param.size()
        count_of_one_param = 1
        for dis in param_size:
            count_of_one_param *= dis

        count += count_of_one_param

    print('total number of parameters is %d'%count)
    return count


def train(train_loader, model, criterion, optimizer, epoch):
    """train model"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    # last_datetime = datetime.datetime.now()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = input.cuda()
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

    print(time.ctime())


def validate(val_loader, model, criterion):
    """validate model"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\t'
                      'Total time {batch_time.sum:.3f} s'
          .format(top1=top1, top5=top5, batch_time=batch_time))

    return top1.avg


def save_checkpoint(state, is_best, filename, best_filename):
    """Save the trained model"""
    torch.save(state, filename)

    if is_best:
        shutil.copyfile(filename, best_filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
