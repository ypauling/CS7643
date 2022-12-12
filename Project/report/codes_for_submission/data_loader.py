from PIL import Image
import os
import pickle
import lmdb
import torch
import sys
import torch.utils.data as data
import numpy as np

MAX_NUM_IMGS = 5
NUM_PATH_COMP = 4


def img_loader(path):

    try:
        img = Image.open(path).convert('RGB')
        return img
    except Exception:
        print('...', sys.stderr)
        return Image.new('RGB', (224, 224), 'white')


class ImagerLoader(data.Dataset):

    def __init__(self, imgs_path, transform=None, target_transform=None,
                 loader=img_loader, square=False, data_path=None,
                 partition=None, sem_reg=None):

        if data_path is None:
            raise Exception('No data path specified')

        if partition is None or partition not in ['train', 'val', 'test']:
            raise Exception('Unknown partition type: {}'.format(partition))
        else:
            self.partition = partition

        self.env = lmdb.open(os.path.join(
            data_path, '{}_lmdb'.format(partition)),
            max_readers=1, readonly=True, lock=False, readahead=False,
            meminit=False)

        with open(os.path.join(
                data_path, '{}_ids.pkl'.format(partition)), 'rb') as f:
            self.ids = pickle.load(f)

        self.transform = transform
        self.target_transform = target_transform
        self.square = square
        self.imgs_path = imgs_path
        self.loader = loader
        self.nids = len(self.ids)

        self.p_mismatch = 0.8
        self.maxlen = 20

        if sem_reg is not None:
            self.sem_reg = sem_reg
        else:
            self.sem_reg = False

    def __getitem__(self, index):

        if self.partition == 'train':
            match_flag = np.random.uniform() > self.p_mismatch
        else:
            match_flag = True

        target = match_flag and 1 or -1

        with self.env.begin(write=False) as txn:
            pickled_sample = txn.get(self.ids[index].encode('latin1'))
        sample = pickle.loads(pickled_sample, encoding='latin1')
        imgs_infos = sample['images']

        if match_flag is True:
            img_idx = 0
            if self.partition == 'train':
                img_idx = np.random.choice(
                    range(min(MAX_NUM_IMGS, len(imgs_infos))))

            img_path = [imgs_infos[img_idx]['id'][i]
                        for i in range(NUM_PATH_COMP)]
            img_path = os.path.join(*img_path)
            img_path = os.path.join(
                self.imgs_path, self.partition,
                img_path, imgs_infos[img_idx]['id'])

        else:
            rand_idx = np.random.choice(range(self.nids))
            while rand_idx == index:
                rand_idx = np.random.choice(range(self.nids))

            with self.env.begin(write=False) as txn:
                rand_pickled_sample = txn.get(
                    self.ids[rand_idx].encode('latin1'))
            rand_sample = pickle.loads(rand_pickled_sample, encoding='latin1')
            rand_imgs_infos = rand_sample['images']

            rand_img_idx = 0
            if self.partition == 'train':
                rand_img_idx = np.random.choice(
                    range(min(MAX_NUM_IMGS, len(rand_imgs_infos))))

            img_idx = rand_img_idx
            img_path = [rand_imgs_infos[img_idx]['id'][i]
                        for i in range(NUM_PATH_COMP)]
            img_path = os.path.join(*img_path)
            img_path = os.path.join(
                self.imgs_path, self.partition,
                img_path, rand_imgs_infos[img_idx]['id'])

        instrs = sample['instructions']
        n_instrs = len(instrs)
        instrs_tensor = np.zeros(
            (self.maxlen, np.shape(instrs)[1]), dtype=np.float32)
        instrs_tensor[:n_instrs] = instrs
        instrs = torch.FloatTensor(instrs_tensor)

        ingrs = sample['ingredients'].astype(int)
        n_ingrs = max(np.nonzero(ingrs)[0]) + 1
        ingrs = torch.LongTensor(ingrs)

        img = self.loader(img_path)
        if self.square:
            img = img.resize(self.square)
        if self.transform is not None:
            img = self.transform(img)

        sample_class = sample['class'] - 1
        sample_id = self.ids[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        if match_flag is True:
            img_class = sample_class
            img_id = sample_id
        else:
            img_class = rand_sample['class'] - 1
            img_id = self.ids[rand_idx]

        if self.partition == 'train':
            if self.sem_reg:
                return [img, instrs, n_instrs, ingrs, n_ingrs], \
                    [target, sample_class, img_class]
            else:
                return [img, instrs, n_instrs, ingrs, n_ingrs], [target]
        else:
            if self.sem_reg:
                return [img, instrs, n_instrs, ingrs, n_ingrs], \
                    [target, sample_class, img_class, sample_id, img_id]
            else:
                return [img, instrs, n_instrs, ingrs, n_ingrs], \
                    [target, sample_id, img_id]

    def __len__(self):
        return len(self.ids)
