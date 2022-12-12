import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from params import get_parser
from recipe_model import JointEmbeddingModel
from data_loader import ImagerLoader

parser = get_parser()
params = parser.parse_args()

if torch.cuda.is_available():
    print('Using GPU...')
    device = torch.device('cuda')
else:
    print('Using CPU...')
    device = torch.device('cpu')


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


def test(test_loader, model, metrics):

    model.eval()

    for i, (inp, target) in enumerate(test_loader):
        if i >= params.early_stop:
            break

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

    med_rank, recalls = metrics(img_embs, txt_embs, sample_ids)
    return med_rank, recalls


def main():

    model = JointEmbeddingModel()
    model.cnn = nn.DataParallel(model.cnn)
    model.to(device)

    if os.path.isfile(params.resume_path):
        print('Loading checkpoint {}'.format(params.resume_path))
        checkpoint = torch.load(params.resume_path)
        model.load_state_dict(checkpoint['state_dict'])
        print('Finish loading...')
    else:
        print('No model found...')
        sys.exit()

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    cudnn.benchmark = True

    test_loader = torch.utils.data.DataLoader(
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
            partition='test',
            sem_reg=params.sem_reg),
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.n_workers,
        pin_memory=True
    )
    print('Test loader prepared.')

    med_rank, recalls = test(test_loader, model, calculate_metric)

    print(med_rank, recalls)
