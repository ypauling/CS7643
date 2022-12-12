import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import numpy as np

from params import get_parser
from recipe_model_vgg import JointEmbeddingModel
from data_loader import ImagerLoader

parser = get_parser()
params = parser.parse_args()

if torch.cuda.is_available():
    print('Using GPU...')
    device = torch.device('cuda')
else:
    print('Using CPU...')
    device = torch.device('cpu')


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0.
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_metric(img_emb, txt_emb, sample_ids):

    emb_type = params.emb_type
    np.random.seed(params.seed)

    ids_sorted = np.sort(sample_ids)
    n_samples = params.medr
    n_iter = 10

    final_ranks = []
    final_recalls = {1: 0., 5: 0., 10: 0.}

    for _ in range(n_iter):

        idx_sample = np.random.choice(
            len(ids_sorted), n_samples, replace=False)
        img_emb_sample = img_emb[idx_sample]
        txt_emb_sample = txt_emb[idx_sample]

        if emb_type == 'image':
            sim_matrix = np.dot(img_emb_sample, txt_emb_sample.T)
        else:
            sim_matrix = np.dot(txt_emb_sample, img_emb_sample.T)

        med_ranks = []
        recalls = {1: 0., 5: 0., 10: 0.}
        for j in range(n_samples):
            sim_j = sim_matrix[j, :]
            idx_j_sorted = np.argsort(sim_j)[::-1]
            pair_rank = np.where(idx_j_sorted == j)[0][0]

            if (pair_rank == 0):
                recalls[1] += 1
            if (pair_rank < 5):
                recalls[5] += 1
            if (pair_rank < 10):
                recalls[10] += 1

            med_ranks.append(pair_rank + 1)

        for k, v in recalls.items():
            recalls[k] = v / n_samples

        for k, v in final_recalls.items():
            final_recalls[k] = v + recalls[k]
        final_ranks.append(np.median(med_ranks))

    for k in final_recalls.keys():
        final_recalls[k] /= n_iter

    return np.mean(final_ranks), final_recalls


def save_checkpoint(state):
    torch.save(state, os.path.join(
        params.checkpoint_path,
        'model_epoch_{:03d}_val_{:.3f}.pth.tar'.format(
            state['epoch'], state['best_val']
        )))


def train(train_loader, model, criterion, optimizer, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    cosine_losses = AverageMeter()
    if params.sem_reg:
        img_losses = AverageMeter()
        txt_losses = AverageMeter()

    model.train()

    start = time.time()
    for i, (inp, target) in enumerate(train_loader):
        if i >= params.early_stop:
            break

        data_time.update(time.time() - start)
        if (i+1) % 20 == 0:
            print('Epoch {:03d}, iteration {:7d}, '
                  'data loading time: {:.4f}'.format(
                      epoch, i, data_time.val))

        inp_device = []
        for j in range(len(inp)):
            inp_device.append(inp[j].to(device))

        tar_device = []
        for j in range(len(target)):
            tar_device.append(target[j].to(device))

        output = model(*inp_device)

        if params.sem_reg:
            loss_cosine = criterion[0](
                output[0], output[1], tar_device[0].float())
            loss_img = criterion[1](output[2], tar_device[1])
            loss_txt = criterion[1](output[3], tar_device[2])

            loss = params.cos_weight * loss_cosine +\
                params.reg_weight * loss_img +\
                params.reg_weight * loss_txt

            n_samples = inp[0].size(0)
            cosine_losses.update(loss_cosine, n_samples)
            img_losses.update(loss_img, n_samples)
            txt_losses.update(loss_txt, n_samples)
        else:
            loss_cosine = criterion(
                output[0], output[1], tar_device[0].float())

            loss = loss_cosine

            n_samples = inp[0].size(0)
            cosine_losses.update(loss_cosine, n_samples)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - start)
        if (i+1) % 20 == 0:
            print('Epoch {:03d}, iteration {:7d}, '
                  'batch time: {:.4f}'.format(
                      epoch, i, batch_time.val))
        start = time.time()

    if params.sem_reg:
        print(
            'Epoch {}:\n'
            '\tAverage Cosine Loss: {:.4f}\n'
            '\tAverage Image Regularization Loss: {:.4f}\n'
            '\tAverage Text Regularization Loss: {:.4f}\n'
            '\tImage Learning Rate: {:.4f}\n'
            '\tText Learning Rate: {:.4f}\n'.format(
                epoch, cosine_losses.avg,
                img_losses.avg, txt_losses.avg,
                optimizer.param_groups[1]['lr'],
                optimizer.param_groups[0]['lr']
            )
        )
    else:
        print(
            'Epoch {}:\n'
            '\tAverage Cosine Loss: {:.4f}\n'
            '\tImage Learning Rate: {:.4f}\n'
            '\tText Learning Rate: {:.4f}\n'.format(
                epoch, cosine_losses.avg,
                optimizer.param_groups[1]['lr'],
                optimizer.param_groups[0]['lr']
            )
        )

    return cosine_losses.avg


def validate(val_loader, model, criterion):

    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.eval()

    start = time.time()
    for i, (inp, target) in enumerate(val_loader):
        if i >= params.early_stop:
            break

        data_time.update(time.time() - start)

        inp_device = []
        for j in range(len(inp)):
            inp_device.append(inp[j].to(device))

        output = model(*inp_device)

        if i == 0:
            img_embs = output[0].data.cpu().numpy()
            txt_embs = output[1].data.cpu().numpy()
            sample_ids = target[-2]
        else:
            img_embs = np.concatenate(
                (img_embs, output[0].data.cpu().numpy()), axis=0)
            txt_embs = np.concatenate(
                (txt_embs, output[1].data.cpu().numpy()), axis=0)
            sample_ids = np.concatenate((sample_ids, target[-2]), axis=0)

        batch_time.update(time.time() - start)
        start = time.time()

    med_rank, recalls = calculate_metric(img_embs, txt_embs, sample_ids)

    print(
        'Validation Set: \n'
        '\tMedian Rank: {:.4f}\n'
        '\tRecalls: {}'.format(
            med_rank,
            recalls
        )
    )

    return med_rank, recalls


def adjust_learning_rate(optimizer, epoch, params):
    optimizer.param_groups[0]['lr'] = params.lr * params.txt_train
    optimizer.param_groups[1]['lr'] = params.lr * params.img_train
    params.patience = 3


def main():
    model = JointEmbeddingModel()
    model.cnn = nn.DataParallel(model.cnn)
    model.to(device)

    cosine_criterion = nn.CosineEmbeddingLoss(0.1).to(device)
    if params.sem_reg:
        weights = torch.ones(params.n_classes)
        weights[0] = 0
        regularization_criterion = nn.CrossEntropyLoss(weight=weights)
        regularization_criterion = regularization_criterion.to(device)
        criterion = [cosine_criterion, regularization_criterion]
    else:
        criterion = cosine_criterion

    img_params = list(map(id, model.cnn.parameters()))
    txt_params = filter(lambda p: id(p) not in img_params, model.parameters())

    optimizer = optim.Adam([
        {'params': txt_params},
        {'params': model.cnn.parameters(), 'lr': params.lr * params.img_train}
    ], lr=params.lr * params.txt_train)

    if params.resume_path:
        if os.path.isfile(params.resume_path):
            print('Loading checkpoint {}'.format(params.resume_path))
            checkpoint = torch.load(params.resume_path)
            params.start_epoch = checkpoint['epoch']
            best_val = checkpoint['best_val']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('Finish loading...')
        else:
            print('Starting from 0 epoch...')
            best_val = float('inf')
    else:
        print('Starting from 0 epoch...')
        best_val = float('inf')

    valtrack = 0

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    cudnn.benchmark = True

    train_loader = torch.utils.data.DataLoader(
        ImagerLoader(
            params.img_path,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]),
            data_path=params.data_path,
            partition='train',
            sem_reg=params.sem_reg),
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.n_workers,
        pin_memory=True
    )
    print('Training loader prepared.')

    val_loader = torch.utils.data.DataLoader(
        ImagerLoader(
            params.img_path,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]),
            data_path=params.data_path,
            sem_reg=params.sem_reg,
            partition='val'),
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.n_workers,
        pin_memory=True
    )
    print('Validation loader prepared.')

    for epoch in range(params.start_epoch, params.epochs):

        train_loss = train(train_loader, model, criterion, optimizer, epoch)
        with open(params.train_loss_output, 'a') as f:
            f.write('{}\n'.format(train_loss))

        if (epoch+1) % params.valfreq == 0 and epoch != 0:
            val_loss, recalls = validate(val_loader, model, criterion)
            with open(params.val_loss_output, 'a') as f:
                f.write('{}\n'.format(val_loss))
            with open(params.val_recall_output, 'a') as f:
                f.write('{},{},{}\n'.format(
                    recalls[1], recalls[5], recalls[10]))

            if val_loss >= best_val:
                valtrack += 1
            else:
                valtrack = 0
            if valtrack >= params.patience:
                params.img_train = params.txt_train
                params.txt_train = not(params.img_train)
                adjust_learning_rate(optimizer, epoch, params)
                valtrack = 0

            is_best = val_loss < best_val
            best_val = min(val_loss, best_val)
            if is_best:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_val': best_val,
                    'optimizer': optimizer.state_dict(),
                    'valtrack': valtrack,
                    'freeVision': params.img_train,
                    'curr_val': val_loss,
                })

            print('** Validation: %f (best) - %d (valtrack)' %
                  (best_val, valtrack))


if __name__ == '__main__':
    main()
