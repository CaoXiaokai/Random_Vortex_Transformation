import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import sys
import time
import os

def save_checkpoint(save_checkpoint_path, model, optimizer, epoch):
    print('==>saving checkpoints...')
    state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
    if not os.path.exists(save_checkpoint_path):
        os.makedirs(save_checkpoint_path)
    torch.save(state, save_checkpoint_path+'/'+ str(int(epoch)))

def display():
    global train_loss_list  
    global test_loss_list 
    global train_err_list
    global test_err_list

    # assert len(train_err_list) == len(test_err_list):

    x1 = range(len(train_err_list))
    x2 = range(len(train_loss_list))

    fig, ax1 = plt.subplots()
    plt.ylabel("error(%)")
    plt.xlabel("epoch")
    ax1.plot(x1, train_err_list, label="train_err", linestyle=(0, (3, 1, 1, 1)), linewidth=1.0)
    # ax1.plot(x2,
    #          train_loss_list,
    #          label="train_loss",
    #          linestyle=(0, (3, 1, 1, 1)),
    #          linewidth=1.0)
    ax1.plot(x1, test_err_list, label="val_err")
    # ax1.plot(x2,
    #          test_loss_list,
    #          label="val_loss",
    #          linestyle=(0, (3, 1, 1, 1)),
    #          linewidth=1.0)
    plt.ylim(ymin=0, ymax=20)
    plt.legend()
    ax1.set_yticks([0, 5, 10, 15, 20])
    labels = ax1.get_yticklabels()
    labels[3].set_visible(False)
    # ax1.grid(axis="y", color='black', linestyle=(0, (5, 10)), linewidth=0.5)
    ax1.axhline(y=5, color='black', linestyle=(0, (5, 10)), linewidth=0.5)
    ax1.axhline(y=10, color='black', linestyle=(0, (5, 10)), linewidth=0.5)
    ax1.axhline(y=20, color='black', linestyle=(0, (5, 10)), linewidth=0.5)
    plt.savefig("result_1.png")

def record_info(test_best_acc, train_best_acc):
    print("Best test acc: {:.4f}".format(test_best_acc))
    # with open("./record.txt",'a+') as f:
    #     f.write("20230823_resent152_5_30, Best train accuracy: " + str(train_best_acc))
    #     f.write("\n")

    #     f.write("Best test accuracy: "+ str(test_best_acc))
    #     f.write("\n")
    
def record_info(log_path, content):
    with open(log_path, 'a+') as f:
        f.write(str(content))

class ManifoldMixupModel(nn.Module):
    def __init__(self, model, num_classes = 10, alpha = 1):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.lam = None
        self.num_classes = num_classes
        self.bce_loss = nn.BCELoss()
        self.softmax = nn.Softmax(dim=1)
        ##选择需要操作的层，在ResNet中各block的层名为layer1,layer2...所以可以写成如下。其他网络请自行修改
        self. module_list = []
        for n,m in self.model.named_modules():
            #if 'conv' in n:
            if n[:-1]=='layer':
                self.module_list.append(m)

    def forward(self, x, target=None):

        def to_one_hot(inp, num_classes):
            y_onehot = torch.FloatTensor(inp.size(0), num_classes).to(inp.device)
            y_onehot.zero_()
            inp = inp.long()
            y_onehot.scatter_(1, inp.unsqueeze(1).data, 1)
            return y_onehot
            
        if target==None:
            out = self.model(x)
            return out
        else:
            if self.alpha <= 0:
                self.lam = 1
            else:
                self.lam = np.random.beta(self.alpha, self.alpha)
            k = np.random.randint(-1, len(self.module_list))
            self.indices = torch.randperm(target.size(0)).cuda()
            target_onehot = to_one_hot(target, self.num_classes)
            target_shuffled_onehot = target_onehot[self.indices]
            if k == -1:
                x = x * self.lam + x[self.indices] * (1 - self.lam)
                out = self.model(x)
            else:
                modifier_hook = self.module_list[k].register_forward_hook(self.hook_modify)
                out = self.model(x)
                modifier_hook.remove()
            target_reweighted = target_onehot* self.lam + target_shuffled_onehot * (1 - self.lam)
            
            loss = self.bce_loss(self.softmax(out), target_reweighted)
            return out, loss
        
    def hook_modify(self, module, input, output):
        output = self.lam * output + (1 - self.lam) * output[self.indices]
        return output

def ricap(images, targets, beta):

    beta = beta  # hyperparameter

    # size of image
    I_x, I_y = images.size()[2:]

    # generate boundary position (w, h)
    w = int(np.round(I_x * np.random.beta(beta, beta)))
    h = int(np.round(I_y * np.random.beta(beta, beta)))
    w_ = [w, I_x - w, w, I_x - w]
    h_ = [h, h, I_y - h, I_y - h]

    # select four images
    cropped_images = {}
    c_ = {}
    W_ = {}
    for k in range(4):
        # print(images.size())
        index = torch.randperm(targets.size(0))
        x_k = np.random.randint(0, I_x - w_[k] + 1)
        y_k = np.random.randint(0, I_y - h_[k] + 1)
        cropped_images[k] = images[index][:, :, x_k:x_k + w_[k], y_k:y_k + h_[k]]
        c_[k] = targets[index]
        W_[k] = (w_[k] * h_[k]) / (I_x * I_y)

    # patch cropped images
    patched_images = torch.cat(
        (torch.cat((cropped_images[0], cropped_images[1]), 2),
            torch.cat((cropped_images[2], cropped_images[3]), 2)),
        3)

    targets = (c_, W_)
    return patched_images, targets

def ricap_criterion(outputs, c_, W_):
    loss = sum([W_[k] * F.cross_entropy(outputs, c_[k]) for k in range(4)])
    return loss

class LabelSmoothingLoss(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x.float(), dim=-1)
        # print(target.unsqueeze(1).shape)
        nll_loss = -logprobs.gather(dim=-1, index=target.long().unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_model(model_name='ResNet18', input_channel=3, crop_size=32, classify_num=10):
    model = None
    if model_name == 'ResNet18':
        model = torchvision.models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_ftrs, classify_num))
        if crop_size == 32 or crop_size == 28:
            model.maxpool = nn.MaxPool2d(1, 1, 0)
            model.conv1 = nn.Conv2d(input_channel, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        elif crop_size == 224:
            model.conv1 = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    elif model_name == 'ViT':
        model = torchvision.models.vit_b_16(pretrained=True)
        model.conv_proj = nn.Conv2d(input_channel, 768, kernel_size=(16, 16), stride=(16, 16))
    
    else:
        raise Exception("can not find the model")

    return model

def weight_init(model):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_normal_(layer.weight.data, gain=1)
            if layer.bias is not None:
                nn.init.constant_(layer.bias.data, 0.3)
            nn.init.kaiming_normal_(layer.weight.data)
        if isinstance(layer, nn.BatchNorm2d):
            # nn.init.normal_(layer.weight.data, mean=0.0, std=0.01)
            nn.init.kaiming_normal_(layer.weight.data)
            nn.init.constant_(layer.bias.data, 0.3)
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight.data, mean=0.0, std=0.01)
            if layer.bias is not None:
                nn.init.zeros(layer.bias.data)

if __name__ == "__main__":
    modle = get_model('Resnet18')