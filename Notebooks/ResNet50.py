#ResNet50 model

import torch
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchsummary import summary
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import os


#To compite average of a list
def update_average(self, val: float, n: int = 1) -> None:
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count
    
#To compute triplet loss
def batch_all_triplet_loss(labels, embeddings, margin, squared=False, epoch=0):
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)
    anchor_positive_dist = pairwise_dist.unsqueeze(2)
    anchor_negative_dist = pairwise_dist.unsqueeze(1)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
    mask = _get_triplet_mask(labels)
    triplet_loss = mask.float() * triplet_loss
    triplet_loss[triplet_loss < margin] = 0
    valid_triplets = triplet_loss[triplet_loss > 1e-16]
    num_positive_triplets = valid_triplets.size(0)
    num_valid_triplets = mask.sum()
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets.float() + 1e-16)
    triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)
    # print(triplet_loss, fraction_positive_triplets)

    return triplet_loss, fraction_positive_triplets

#To compute final result
def result(model,dataloader, device):
    dist = []
    with torch.no_grad():
        model.eval()
        for _, eval_data in enumerate(tqdm(dataloader)):
            eval_image = eval_data['image'].to(device)
            eval_out = model(eval_image)
            eval_pair = eval_data['pair_image'].to(device)
            eval_pait_out = model(eval_pair)
            distance = torch.norm(eval_out - eval_pait_out, dim=1)
            dist.append(list(distance.cpu().numpy()))

    new_dist = []
    for i in range(len(dist)):
        for j in range(len(dist[i])):
            new_dist.append(dist[i][j])
    dist = np.asarray(new_dist)

    return dist

#To evalute the model to compute thresholds
def evalulate(model, eval_loader1, eval_loader2, device):
    # same target pairs
    dist1 = result(model,eval_loader1, device)
    # diff target pairs
    dist2 = result(model,eval_loader2, device)

    with open('test.npy', 'wb') as f:
        np.save(f, dist1)
        np.save(f, dist2)

    same_hist = plt.hist(dist1, 100, range=[np.floor(np.min([dist1.min(), dist2.min()])),np.ceil(np.max([dist1.max(), dist2.max()]))], alpha=0.5, label='same')
    diff_hist = plt.hist(dist2, 100, range=[np.floor(np.min([dist1.min(), dist2.min()])),np.ceil(np.max([dist1.max(), dist2.max()]))], alpha=0.5, label='diff')
    difference = same_hist[0] - diff_hist[0]
    difference[:same_hist[0].argmax()] = np.Inf
    difference[diff_hist[0].argmax():] = np.Inf
    return (same_hist[1][np.where(difference <= 0)[0].min()] + same_hist[1][np.where(difference <= 0)[0].min() - 1])/2
        
#To train the model
def train(model,train_loader,valid_loader,valid_loader1,valid_loader2,optimizer,scheduler,num_epochs,eval_every,margin,device,name):
    epoch_loss_list = {'train':[], 'valid':[]}
    errorMetric1_list = []
    global_step = 0
    trainLoss = update_average()
    valid_loss = update_average()
    best_errorMetric1 = 1
    total_step = len(train_loader)*num_epochs
    count = 0
    print(f'total steps: {total_step}')
    for epoch in range(num_epochs):
        print(f'epoch {epoch+1}')
        for _, data in enumerate(tqdm(train_loader)):
            if count > 0:
                break
            count += 1
            model.train()
            inputs = data['image'].to(device)
            target = data['target'].to(device) 
            embeddings = model(inputs)
            loss, _ = batch_all_triplet_loss(target, embeddings, margin=margin, epoch=epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            trainLoss.update(loss.item())
            global_step += 1
            current_lr = optimizer.param_groups[0]['lr']

            if global_step % eval_every == 0:
                model.eval()
                for _, data in enumerate(tqdm(valid_loader)):
                    inputs = data['image'].to(device)
                    target = data['target'].to(device)
                    embeddings = model(inputs)
                    loss, _ = batch_all_triplet_loss(target, embeddings, margin=margin, epoch=epoch)
                    valid_loss.update(loss.item())

                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}, lr: {:.4f}'
                          .format(epoch+1, num_epochs, global_step, total_step, trainLoss.avg, valid_loss.avg ,current_lr))

        dist1 = result(model,valid_loader1,device)
        dist2 = result(model,valid_loader2,device)
        same_hist = plt.hist(dist1, 100, range=[np.floor(np.min([dist1.min(), dist2.min()])),np.ceil(np.max([dist1.max(), dist2.max()]))], alpha=0.5, label='same')
        diff_hist = plt.hist(dist2, 100, range=[np.floor(np.min([dist1.min(), dist2.min()])),np.ceil(np.max([dist1.max(), dist2.max()]))], alpha=0.5, label='diff')
        plt.legend(loc='upper right')
        difference = same_hist[0] - diff_hist[0]
        difference[:same_hist[0].argmax()] = np.Inf
        difference[diff_hist[0].argmax():] = np.Inf
        dist_threshold = (same_hist[1][np.where(difference <= 0)[0].min()] + same_hist[1][np.where(difference <= 0)[0].min() - 1])/2
        overlap = np.sum(dist1>=dist_threshold) + np.sum(dist2<=dist_threshold)
        errorMetric1 = overlap / (dist1.shape[0] * 2 - overlap)
        print('dist_threshold:',dist_threshold,'overlap:',overlap,'errorMetric1:',errorMetric1)
        plt.clf()

        epoch_loss_list['train'].append(trainLoss.avg)
        epoch_loss_list['valid'].append(valid_loss.avg)
        errorMetric1_list.append(errorMetric1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,9))
        ax1.plot(range(len(epoch_loss_list['train'])), epoch_loss_list['train'], label=('trainLoss'))
        ax1.plot(range(len(epoch_loss_list['valid'])), epoch_loss_list['valid'], label=('valid_loss'))
        ax2.plot(range(len(errorMetric1_list)), errorMetric1_list, label=('errorMetric1'))
        ax1.legend(prop={'size': 15})
        ax2.legend(prop={'size': 15})
        plt.savefig('../Result/loss.png')
        plt.clf()

        if errorMetric1 < best_errorMetric1:
            best_errorMetric1 = errorMetric1
            checkpoint = {'model': model,'optimizer': optimizer,'scheduler': scheduler,]
            save_path = '../Model/' + save_path
            torch.save(checkpoint, save_path)

        trainLoss.reset()
        valid_loss.reset()

        scheduler.step()
    print('Finished Training')
    
#To test the model by printing evaluation metrics
def test(model, test_loader, dist_threshold, device):
    label = []
    pred = []
    with torch.no_grad():
        model.eval()
        for _, test_data in enumerate(tqdm(test_loader)):
            test_image = test_data['image'].to(device)
            test_target = test_data['target']
            test_out = model(test_image)
            test_pair = test_data['pair_image'].to(device)
            test_pair_target = test_data['pair_target']
            test_pait_out = model(test_pair)
            distance = torch.norm(test_out - test_pait_out, dim=1)
            label.append(list((test_target == test_pair_target).cpu().numpy()))
            pred.append(list((distance <= dist_threshold).cpu().numpy()))

    new_label = []
    new_pred = []
    for i in range(len(label)):
        for j in range(len(label[i])):
            new_label.append(label[i][j])
            new_pred.append(pred[i][j])

    new_pred = [0 if i == False else 1 for i in new_pred]
    new_label = [0 if i == False else 1 for i in new_label]
    new_pred = np.array(new_pred)
    new_label = np.array(new_label)
    num_true = np.sum(new_pred==new_label)
    acc = num_true/len(new_label)
    print('Accuracy:', acc)
    
# TO load saved models from path
def load(name, model, optimizer):
    checkpoint = torch.load('../Model/' + name)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda() 

               
    
if __name__ == "__main__":
    # config
    BATCH_SIZE=128
    NUM_WORKERS = 2
    embedding_size = 512
    sampler = None
    weight_decay = 1e-3
    lr = 0.0001
    dropout = 0.3
    # resnet, vg16 or None(IncepetionResNet)
    resnet = "IncepetionResNet"
    pretrain = True
    pool= None
    # sgd or None(adam) or rmsprop
    optimizer_type = None
    num_epochs = 30
    eval_every = 1000
    name = 'facenet_best.pth'
    load_local_model = False

    # read scv
    train_dataset = pd.read_csv('../Data/train2.csv')
    valid_dataset = pd.read_csv('../Data/valid.csv')
    eval_dataset1 = pd.read_csv('../Data/eval_same.csv')
    eval_dataset2 = pd.read_csv('../Data/eval_diff.csv')
    test_dataset = pd.read_csv('../Data/test.csv')

 

    # model, optimizer, scheduler
    facenet = FaceNet(resnet, pool=pool, embedding_size=embedding_size, dropout=dropout, pretrain=pretrain, device=device).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    if load_local_model:
        load(name, facenet, optimizer)
        temp = name.split('.')
        name = temp[0] + '_new.' + temp[1]
    scheduler = get_Scheduler(optimizer, lr, scheduler_name) # scheduler

    # train
    train(facenet,train_loader,valid_loader,eval_loader1,eval_loader2,optimizer,scheduler,num_epochs,eval_every,margin,device,name)
    dist_threshold = evalulate(facenet, eval_loader1, eval_loader2, device)
    test(facenet,test_loader,dist_threshold,device)
