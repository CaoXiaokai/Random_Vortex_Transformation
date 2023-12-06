import torch
from tqdm import tqdm
import os
import sys
import time
from dataloader import *
from utils import *
import argparse

parser = argparse.ArgumentParser(description='Vortex Encryption Training')

# hardware
parser.add_argument('--num_workers', type=int, default=8,
                    help='number of workers loading data')

# dataset
parser.add_argument('--dataset', type=str, default='cifar10', required=True, choices=['cifar10', 'cifar10_vor', 'cifar10_random', 'mnist', 'mnist_vor', 'mnist_random', 'fashion', 'fashion_vor', 'fashion_random'],
                    help='dataset for training')
parser.add_argument('--dataroot', type=str, default='dataset/',
                    help='path to dataset')

# model
parser.add_argument('--model', '-m', type=str, default='ResNet18', required=True, choices=['ResNet18', 'ViT'],
                    help='model name')
                  

# hyperparameters
parser.add_argument('--epoch', '-e', type=int, default=100,
                    help='number of epochs: (default: 100 for training cifar_vor)')
parser.add_argument('--batch', type=int, default=128,
                    help='batchsize')
parser.add_argument('--lr', type=float, default=0.1,
                    help='default learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum')
parser.add_argument('--wd', type=float, default=1e-4,
                    help='weight decay: (default: 0.0001)')

# data augmentation
parser.add_argument('--crop', type=int, default=32,
                    help='crop size')
parser.add_argument('--beta_of_ricap', type=float, default=0.3,
                    help='beta of ricap augmentation')

# save and resume
parser.add_argument('--loadpath', type=str, default=None,
                    help='the path of the checkpoints loaded')
parser.add_argument('--resume', '-r', type=int, default=0,
                    help='epoch at which resume from checkpoint. -1 for latest')
parser.add_argument('--savefreq', type=int, default=10,
                    help='frequency to save model and to mark it the latest')
parser.add_argument('--dev', type=int, default=0,
                    help="device rank for training")
parser.add_argument('--postfix', '-pf', type=str, default='',
                    help='discription')
args = parser.parse_args()

#tensorboard
comment = '_{model}_{dataset}_bs{batch}_epoch{epoch}_imgsize{crop}_wd{wd}_{ricap}_{postfix}'.format(
    model=args.model,
    dataset=args.dataset,
    batch=args.batch,
    epoch=args.epoch,
    crop=args.crop,
    wd=args.wd,
    ricap=args.beta_of_ricap if args.beta_of_ricap != None else "",
    postfix=args.postfix if args.postfix != '' else ''
)

#seed
seed = 3407
torch.manual_seed(seed)
# device
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.dev)    

dataset_name = str(args.dataset)
lr = args.lr
EPOCH = args.epoch
BATCH_SIZE = args.batch
T_max = EPOCH
weight_decay = args.wd
momentum = args.momentum
img_size = args.crop
NUM_WORKER = args.num_workers
INPUT_CHANNEL = 3 if 'cifar10' in dataset_name else 1
CLASSIFY_NUM = 10

test_best_acc = 0.0

# dataset
train_loader, test_loader = dataloader(name=dataset_name, batch_size=BATCH_SIZE, img_size=img_size, num_workers=NUM_WORKER)

model = get_model(model_name=args.model, input_channel=INPUT_CHANNEL, crop_size=args.crop, classify_num=CLASSIFY_NUM)

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)

model.cuda()

criterion =torch.nn.CrossEntropyLoss().cuda()

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, verbose=True)


def train(model, train_loader, optimizer, epoch):
    model.train()
    sum_loss = 0.0
    correct = 0
    total = 0
    cnt = 0
    print(f"EPOCH: {epoch}")
    for inputs, labels in tqdm(train_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()

        if args.beta_of_ricap == None:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        else:
            inputs, (c_, W_) = ricap(inputs, labels, beta=args.beta_of_ricap)
            outputs = model(inputs)
            loss = ricap_criterion(outputs, c_, W_)

        optimizer.zero_grad()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
    print('Train Loss: {:.4f}, Acc: {:.4f}'.format(sum_loss / len(train_loader), correct / total), end=' ')


def eval(model, test_loader, epoch):
    global test_best_acc
    model.eval()
    test_loss = 0.0
    correct = 0.0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        test_best_acc = max(test_best_acc, correct / total)
        print("EPOCH {:d}, Test Loss : {:.4f}, acc : {:.2%}, Best acc:{:.2%}".format(
            epoch, test_loss / len(test_loader), correct / total, test_best_acc))

    return correct / total


def main():
    checkpoint_path = 'checkpoint/'+comment
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    start_epoch = 0
    if args.loadpath is not None:
        load_checkpoint_path = args.loadpath
        checkpoint = torch.load(load_checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print(f"Having loaded the checkpoint of {start_epoch} epoch")
        test_acc = eval(model, test_loader, 0)

    for epoch in range(start_epoch, start_epoch+EPOCH+1):
        start = time.time()
        train(model, train_loader, optimizer, epoch)
        end = time.time()
        print(f'Train time: {int((end-start)//60)}min{int((end-start)%60)}s')
        test_acc = eval(model, test_loader, epoch)
        scheduler.step()

        if epoch % args.savefreq == 0:
            save_checkpoint(checkpoint_path, model, optimizer, epoch)

if __name__ == "__main__":
    main()

