# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

"""# Import Libraries"""

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from utils.model import select_model
from utils.options import parse_args_function
from utils.dataset import Dataset
from utils.metric import *
from tqdm import tqdm
from models.hourglass import HeatmapLoss

args = parse_args_function()

"""# Load Dataset"""

root = args.input_file

#mean = np.array([120.46480086, 107.89070987, 103.00262132])
#std = np.array([5.9113948 , 5.22646725, 5.47829601])

transform = transforms.Compose([transforms.Resize((256, 256)),
                                transforms.ToTensor()])

if args.train or args.pre_train:
    trainset = Dataset(root=root, load_set='train', transform=transform, with_object=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16, drop_last=True)
    
    print('Train files loaded')

if args.val:
    valset = Dataset(root=root, load_set='val', transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=True)
    
    print('Validation files loaded')

if args.test:
    testset = Dataset(root=root, load_set='test', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=True)
    
    print('Test files loaded')

"""# Model"""

use_cuda = False
if args.gpu:
    use_cuda = True

model = select_model(args.model_def)

if use_cuda and torch.cuda.is_available():
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=args.gpu_number)

"""# Load Snapshot"""

if args.pretrained_model != '':
    model.load_state_dict(torch.load(args.pretrained_model))
    losses = np.load(args.pretrained_model[:-4] + '-losses.npy').tolist()
    start = len(losses)
else:
    losses = []
    start = 0

"""# Optimizer"""

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_step_gamma)
scheduler.last_epoch = start
lambda_1 = 0.01
lambda_2 = 1
lambda_hm = 1
lambda_pre_2d = 0.001

def calc_loss(combined_hm_preds, heatmaps):
    heatmapLoss = HeatmapLoss()
    combined_loss = []
    for i in range(2):
        # combined_loss.append(heatmapLoss(combined_hm_preds[0][:,i], heatmaps))
        combined_loss.append(heatmapLoss(combined_hm_preds[i], heatmaps))
    combined_loss = torch.stack(combined_loss, dim=1)
    return combined_loss 

"""# Train"""

if args.train:
    print('Begin training the network...')
    best_val_error = 9999999
    best_train_error = 9999999
    
    for epoch in range(start, args.num_iterations):  # loop over the dataset multiple times
    
        with tqdm(total=trainloader.__len__(), 
                bar_format='{n_fmt}/{total_fmt} |{bar}| {rate_fmt} {postfix}', ncols=120) as process_bar: 
            running_loss = 0.0
            train_loss = 0.0
            e_dis3d_train = []
            for i, tr_data in enumerate(trainloader):
                # get the inputs
                inputs, labels2d, labels3d, scale_label = tr_data
                scale_label = np.array(scale_label)
        
                # wrap them in Variable
                inputs = Variable(inputs)
                labels2d = Variable(labels2d)
                labels3d = Variable(labels3d)
                
                if use_cuda and torch.cuda.is_available():
                    inputs = inputs.float().cuda(device=args.gpu_number[0])
                    labels2d = labels2d.float().cuda(device=args.gpu_number[0])
                    labels3d = labels3d.float().cuda(device=args.gpu_number[0])
        
                # zero the parameter gradients
                optimizer.zero_grad()
        
                # forward + backward + optimize
                outputs2d_init, outputs2d, outputs3d = model(inputs)
                loss2d_init = criterion(outputs2d_init, labels2d)
                loss2d = criterion(outputs2d, labels2d)
                loss3d = criterion(outputs3d, labels3d)
                loss = (lambda_1)*loss2d_init + (lambda_1)*loss2d + (lambda_2)*loss3d
                loss.backward()
                optimizer.step()

                # calculate the distance between label3d and output3d
                b_labels3d = labels3d.reshape(-1, 21, 3) #[B, 21, 3]
                b_out3d = outputs3d.reshape(-1, 21, 3)
                b_dis3d = torch.norm(labels3d - outputs3d, dim = -1)
                b_dis3d = b_dis3d.cpu().detach().numpy() #* 1000 [B, 21]
                b_dis3d = b_dis3d / np.repeat(scale_label.reshape(-1, 1), 21, axis=-1)
                e_dis3d_train.append(b_dis3d)
                
                # print statistics
                train_loss += loss.data
                # running_loss += loss.data
                # if (i+1) % args.log_batch == 0:    # print every log_iter mini-batches
                #     print('[%d, %5d] loss: %.5f' % (epoch + 1, i + 1, running_loss / args.log_batch))
                #     running_loss = 0.0

                process_bar.set_postfix_str('loss=%.5f, 2d_ini=%.5f, 2d=%.5f, 3d=%.5f'\
                                           % (loss.data, loss2d_init.data, loss2d.data, loss3d.data))
                process_bar.update()
            e_dis3d_train = np.r_[e_dis3d_train]
            auc_train = calc_auc(e_dis3d_train.reshape(-1), 20, 50) #for freihand
            m_train_error = train_loss / (i+1)
            if best_train_error > m_train_error:
                best_train_error = m_train_error
                torch.save(model.state_dict(), args.output_file+'best_train.pkl')
            print('%d epoch training done, train loss=%.5f, AUC_20_50=%.5f' % (epoch+1, m_train_error, auc_train))
            
            if args.val and (epoch+1) % args.val_epoch == 0:
                val_loss = 0.0
                val_2d_ini = 0.0
                val_2d = 0.0
                val_3d = 0.0
                e_dis3d = []
                for v, val_data in enumerate(valloader):
                    # get the inputs
                    inputs, labels2d, labels3d, scale_label_val = val_data
                    scale_label_val = np.array(scale_label_val)
                    
                    # wrap them in Variable
                    inputs = Variable(inputs)
                    labels2d = Variable(labels2d)
                    labels3d = Variable(labels3d)
            
                    if use_cuda and torch.cuda.is_available():
                        inputs = inputs.float().cuda(device=args.gpu_number[0])
                        labels2d = labels2d.float().cuda(device=args.gpu_number[0])
                        labels3d = labels3d.float().cuda(device=args.gpu_number[0])
            
                    outputs2d_init, outputs2d, outputs3d = model(inputs)
                    loss2d = criterion(outputs2d, labels2d)
                    loss3d = criterion(outputs3d, labels3d)
                    loss2d_init = criterion(outputs2d_init, labels2d)
                    loss = (lambda_1)*loss2d_init + (lambda_1)*loss2d + (lambda_2)*loss3d
                    val_loss += loss.data
                    val_2d_ini += loss2d_init.data
                    val_2d += loss2d.data
                    val_3d += loss3d.data

                    # calculate the distance between label3d and output3d
                    b_labels3d = labels3d.reshape(-1, 21, 3) #[B, 21, 3]
                    b_out3d = outputs3d.reshape(-1, 21, 3)
                    b_dis3d = torch.norm(labels3d - outputs3d, dim = -1)
                    b_dis3d = b_dis3d.cpu().detach().numpy() #* 1000 [B, 21]
                    b_dis3d = b_dis3d / np.repeat(scale_label_val.reshape(-1, 1), 21, axis=-1)

                    e_dis3d.append(b_dis3d)
                e_dis3d = np.r_[e_dis3d]
                auc_val = calc_auc(e_dis3d.reshape(-1), 20, 50) #for freihand
                m_val_error = val_loss / (v+1)
                if best_val_error > m_val_error:
                    best_val_error = m_val_error
                    torch.save(model.state_dict(), args.output_file+'best_val.pkl')
                print('val error: %.5f; 2d_ini: %.5f; 2d: %.5f; 3d: %.5f; val AUC_20_50: %.5f' % \
                     (m_val_error, val_2d_ini / (v+1), val_2d / (v+1), val_3d / (v+1), auc_val))

            losses.append((train_loss / (i+1)).cpu().numpy())
            
            if (epoch+1) % args.snapshot_epoch == 0:
                torch.save(model.state_dict(), args.output_file+str(epoch+1)+'.pkl')
                np.save(args.output_file+str(epoch+1)+'-losses.npy', np.array(losses))

            # Decay Learning Rate
            scheduler.step()
        
    print('Finished Training')

"""# Test"""

if args.test:
    print('Begin testing the network...')
    
    running_loss = 0.0
    e_dis3d = []
    for i, ts_data in enumerate(testloader):
        # get the inputs
        inputs, labels2d, labels3d, scale_label = ts_data
        scale_label = np.array(scale_label)
        
        # wrap them in Variable
        inputs = Variable(inputs)
        labels2d = Variable(labels2d)
        labels3d = Variable(labels3d)
        # scale_label = Variable(scale_label)

        if use_cuda and torch.cuda.is_available():
            inputs = inputs.float().cuda(device=args.gpu_number[0])
            labels2d = labels2d.float().cuda(device=args.gpu_number[0])
            labels3d = labels3d.float().cuda(device=args.gpu_number[0])
            # scale_label = scale_label.float().cuda(device=args.gpu_number[0])

        outputs2d_init, outputs2d, outputs3d = model(inputs)

        # calculate the distance between label3d and output3d
        b_labels3d = labels3d.reshape(-1, 21, 3) #[B, 21, 3]
        b_out3d = outputs3d.reshape(-1, 21, 3)
        b_dis3d = torch.norm(labels3d - outputs3d, dim = -1)
        b_dis3d = b_dis3d.cpu().detach().numpy() #* 1000 [B, 21]
        b_dis3d = b_dis3d / np.repeat(scale_label.reshape(-1, 1), 21, axis=-1)

        e_dis3d.append(b_dis3d)
        
        loss = criterion(outputs3d, labels3d)
        running_loss += loss.data
    e_dis3d = np.r_[e_dis3d] 
    print(e_dis3d.shape)
    # e_dis3d = e_dis3d.reshape(-1, 21)
    auc_test = calc_auc(e_dis3d.reshape(-1), 20, 50)
    print('test auc: %f' % (auc_test))
    print('test 3d loss: %.5f' % (running_loss / (i+1)))
