# coding=utf-8
from prototypical_batch_sampler import PrototypicalBatchSampler
from prototypical_loss import prototypical_loss as loss_fn
from evaluate import prototypical_eval
from fundus_dataset import FundusDataset
from protonet import Model
from parser_util import get_parser
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import torch
import os

import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(6)

def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)


def init_dataset(opt, mode):
    trsm = transforms.Compose([
          transforms.Resize(342),
          transforms.CenterCrop(299),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    dataset = FundusDataset(mode=mode, root=opt.dataset_root, transform = trsm)
    n_classes = len(np.unique(dataset.y))
    if n_classes < opt.classes_per_it_tr or n_classes < opt.classes_per_it_val:
        raise(Exception('There are not enough classes in the dataset in order ' +
                        'to satisfy the chosen classes_per_it. Decrease the ' +
                        'classes_per_it_{tr/val} option and try again.'))
    return dataset


def init_sampler(opt, labels, mode):
    if 'train' in mode:
        classes_per_it = opt.classes_per_it_tr
        num_samples = opt.num_support_tr + opt.num_query_tr
    else:
        classes_per_it = opt.classes_per_it_val
        num_samples = opt.num_support_val + opt.num_query_val

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=opt.iterations)


def init_dataloader(opt, mode):
    dataset = init_dataset(opt, mode)
    sampler = init_sampler(opt, dataset.y, mode)
    dataloader1 = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    dataloader2 = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
    return dataloader1, dataloader2


def init_protonet(opt, device):
    '''
    Initialize the ProtoNet
    '''
    #device = 'cuda:6' if torch.cuda.is_available() and opt.cuda else 'cpu'
    pretrained_weights_path = '../../inceptionv3_NEWFUNDUS_1006.pt'
    #model = Model(pretrained_weights_path, device)
    
    # define model
    model = models.inception_v3(pretrained=True) 
    # modify auxlogit layer
    num_aux_ftrs = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(num_aux_ftrs, 6)
    # modify fc layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 6)
    # load 6-classes weights
    model.load_state_dict(torch.load(pretrained_weights_path, map_location=device))
    # add my fc layer
    #model.AuxLogits.fc = nn.Sequential(nn.Linear(num_aux_ftrs, len(class_names)), nn.LogSoftmax(dim=1))
    #model.fc = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    #model.fc = nn.Tanh()#forward函数里面用到了，所以没法直接删掉，所以换成了激活函数
    #del model._modules['AuxLogits']
    #del model._modules['fc']
    model.aux_logits = False
    model = nn.DataParallel(model, device_ids=[2,4])
    model = model.to(device)
    return model


def init_optim(opt, model):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(),
                            lr=opt.learning_rate)


def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=opt.lr_scheduler_gamma,
                                           step_size=opt.lr_scheduler_step)


def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)

def cal_centroid(output, target):
    target_cpu = target.to('cpu')
    output_cpu = output.to('cpu')
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    #print('看看有多少类：', target_cpu.size(), n_classes)
    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero().squeeze(1)
    support_idxs = list(map(supp_idxs, classes))
    # idx_list是某一类的support样本，取均值然后叠加，得到的prototypes就是每类的聚类中心的叠加
    prototypes = torch.stack([output_cpu[idx_list].mean(0) for idx_list in support_idxs])
    return prototypes
            
def train(opt, tr_dataloader, model, optim, lr_scheduler, device, val_dataloader=None, test_dataloader=None, mytrain_dataloader=None, mytest_dataloader=None):
    '''
    Train the model with the prototypical learning algorithm
    '''
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print('0:', meminfo.used)
    #device = 'cuda:6' if torch.cuda.is_available() and opt.cuda else 'cpu'

    if val_dataloader is None:
        best_state = None
    
    avg_train_loss = []
    avg_train_acc = []
    avg_val_loss = []
    avg_val_acc = []
    avg_test_acc = []
    #放在epoch外面，把每个epoch的每个batch的结果都保存了，没次计算只取最近的一个batch
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    test_loss = []
    test_acc = []
    best_acc = 0

    best_model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(opt.experiment_root, 'last_model.pth')

    for epoch in range(opt.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        #---------------train stage-------------------------#
        tr_iter = iter(tr_dataloader)
        model.train()
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)
            with torch.set_grad_enabled(True):
                model_output = model(x)
                loss, acc = loss_fn(model_output, target=y,
                                    n_support=opt.num_support_tr)
                loss.backward()
                optim.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())
        avg_loss = np.mean(train_loss[-opt.iterations:])
        avg_acc = np.mean(train_acc[-opt.iterations:])
        avg_train_loss.append(avg_loss)
        avg_train_acc.append(avg_acc)
        print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        lr_scheduler.step()
        
        #---------------eval stage-------------------------#
        if val_dataloader is None:
            continue
        val_iter = iter(val_dataloader)
        model.train()
        for batch in val_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            with torch.set_grad_enabled(False):
                model_output = model(x)
                loss, acc = loss_fn(model_output, target=y,
                                    n_support=opt.num_support_val) 
            val_loss.append(loss.item())
            val_acc.append(acc.item())
        avg_loss = np.mean(val_loss[-opt.iterations:])
        avg_acc = np.mean(val_acc[-opt.iterations:])
        avg_val_loss.append(avg_loss)
        avg_val_acc.append(avg_acc)
        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
            best_acc)
        print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
            avg_loss, avg_acc, postfix))
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()
        
        #---------------inference stage-------------------------#        
        train_iter = iter(mytrain_dataloader)
        model.eval()
        i = 0
        for batch in train_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            with torch.set_grad_enabled(False):
                model_output = model(x)
                if i==0:
                    train_outputs = model_output
                    train_targets = y
                else:
                    train_outputs = torch.cat((train_outputs, model_output), 0)
                    train_targets = torch.cat((train_targets, y), 0)
                i+=1                
        #---------------test stage-------------------------#
        if test_dataloader is None:
            continue
        prototypes = cal_centroid(train_outputs, train_targets)
        test_iter = iter(mytest_dataloader)
        model.eval()
        for batch in test_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            with torch.set_grad_enabled(False):
                model_output = model(x)
                acc = prototypical_eval(model_output, target=y, prototypes=prototypes) 
            test_acc.append(acc.item())
        avg_acc = np.mean(test_acc)
        avg_test_acc.append(avg_acc)
        print('Avg Test Acc: {}'.format(avg_acc))
        
    torch.save(model.state_dict(), last_model_path)

    for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        save_list_to_file(os.path.join(opt.experiment_root,
                                       name + '.txt'), locals()[name])

    return avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc, avg_test_acc

def main():
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()
    device = 'cuda:2' if torch.cuda.is_available() and options.cuda else 'cpu'    
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)
    tr_dataloader, mytrain_dataloader = init_dataloader(options, 'train')
    val_dataloader, _ = init_dataloader(options, 'val')
    # trainval_dataloader = init_dataloader(options, 'trainval')
    test_dataloader, mytest_dataloader = init_dataloader(options, 'val')   
    model = init_protonet(options, device=device)
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)
    res = train(opt=options,
                tr_dataloader=tr_dataloader,
                val_dataloader=val_dataloader,
                test_dataloader=test_dataloader,
                mytrain_dataloader = mytrain_dataloader,
                mytest_dataloader = mytest_dataloader,
                model=model,
                optim=optim,
                lr_scheduler=lr_scheduler,
                device = device)
    avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc, avg_test_acc = res
    print('draw training loss and accuracy curve!!!')
    plt.subplot(121)
    plt.plot(range(options.epochs), avg_train_loss, 'g', label='train loss')
    plt.plot(range(options.epochs), avg_val_loss, 'r', label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.subplot(122)
    plt.plot(range(options.epochs), avg_train_acc, 'g', label='train acc')
    plt.plot(range(options.epochs), avg_val_acc, 'r', label='val acc')
    plt.plot(range(options.epochs), avg_test_acc, 'b', label='test acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.savefig('result.jpg')
    plt.show()


if __name__ == '__main__':
    main()
