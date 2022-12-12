import pickle
import tqdm
import torchfile
import time
import shutil
import lmdb
from proc import detect_ingrs
import utils
import os
import numpy as np
from params import get_parser

MAX_NUM_IMGS = 5


def load_st_vectors(file):
    vecfile = torchfile.load(file)

    img_ids = []
    for idrow in vecfile[b'ids']:
        img_ids.append(''.join(chr(c) for c in idrow))

    ret_dict = {}
    ret_dict['encs'] = vecfile[b'encs']
    ret_dict['rbps'] = vecfile[b'rbps']
    ret_dict['rlens'] = vecfile[b'rlens']
    ret_dict['ids'] = img_ids

    print(np.shape(ret_dict['encs']), len(ret_dict['rlens']), len(
        ret_dict['rbps']), len(ret_dict['ids']))
    return ret_dict


parser = get_parser()
params = parser.parse_args()

DATASET = params.dataset
output_dir = params.bigram_folder
st_dir = params.bigram_folder

with open(os.path.join(output_dir, 'remove1M.txt'), 'r') as f:
    remove_ids = {w.rstrip(): i for i, w in enumerate(f)}

start = time.time()
print('Start loading skip-thought vectors...')

partition = params.lmdb_partition

# st_vecs_train = load_st_vectors(os.path.join(st_dir, 'encs_train_1024.t7'))
# st_vecs_val = load_st_vectors(os.path.join(st_dir, 'encs_val_1024.t7'))
# st_vecs_test = load_st_vectors(os.path.join(st_dir, 'encs_test_1024.t7'))
st_vecs = load_st_vectors(os.path.join(
    st_dir, 'encs_{}_1024.t7'.format(partition)))

# st_vecs = {'train': st_vecs_train, 'val': st_vecs_val, 'test': st_vecs_test}

# stid2eid = {'train': {}, 'val': {}, 'test': {}}
# for partition in ['train', 'val', 'test']:
#     for i, eid in enumerate(st_vecs[partition]['ids']):
#         stid2eid[partition][eid] = i
stid2eid = {}
for i, eid in enumerate(st_vecs['ids']):
    stid2eid[eid] = i

print('Finish loading skip-thought vectors...')
print('Use {} seconds...'.format(time.time() - start))

print('Loading dataset...')
dataset = utils.Layer.merge(
    [utils.Layer.L1, utils.Layer.L2, utils.Layer.INGRS], DATASET)
print('Finish loading dataset...')

print('Loading vocab...')
with open(params.vocab, 'r') as f:
    ingr_vocab = {w.rstrip(): i+2 for i, w in enumerate(f)}
    ingr_vocab['</i>'] = 1
print('Finish loading vocab...')

print('Loading classes1M...')
with open(os.path.join(output_dir, 'classes1M.pkl'), 'rb') as f:
    class_dict = pickle.load(f)
    ind2class = pickle.load(f)
print('Finish loading classes1M...')


# if os.path.isdir(os.path.join(output_dir, 'train_lmdb')):
#     shutil.rmtree(os.path.join(output_dir, 'train_lmdb'))
# if os.path.isdir(os.path.join(output_dir, 'val_lmdb')):
#     shutil.rmtree(os.path.join(output_dir, 'val_lmdb'))
# if os.path.isdir(os.path.join(output_dir, 'test_lmdb')):
#     shutil.rmtree(os.path.join(output_dir, 'test_lmdb'))

if os.path.isdir(os.path.join(output_dir, '{}_lmdb'.format(partition))):
    shutil.rmtree(os.path.join(output_dir, '{}_lmdb'.format(partition)))

# env = {'train': [], 'val': [], 'test': []}
# env['train'] = lmdb.open(os.path.join(
#     output_dir, 'train_lmdb'), map_size=int(1e11))
# env['val'] = lmdb.open(os.path.join(
#     output_dir, 'val_lmdb'), map_size=int(1e11))
# env['test'] = lmdb.open(os.path.join(
#     output_dir, 'test_lmdb'), map_size=int(1e11))

env = lmdb.open(os.path.join(
    output_dir, '{}_lmdb'.format(partition)), map_size=int(5e10))

print('Creating dataset...')
# partition_ids = {'train': [], 'val': [], 'test': []}
partition_ids = []

for i, entry in tqdm.tqdm(enumerate(dataset)):

    if entry['partition'] != partition:
        continue

    ninstrs = len(entry['instructions'])
    ingrs_detected = detect_ingrs(entry, ingr_vocab)
    ningrs = len(ingrs_detected)
    img_infos = entry.get('images')

    if ninstrs >= params.maxlen or ningrs >= params.maxlen or ningrs == 0 or \
            not img_infos or remove_ids.get(entry['id']):
        continue

    ingr_vec = np.zeros((params.maxlen), dtype='uint16')
    ingr_vec[:ningrs] = ingrs_detected

    # partition = entry['partition']
    stidx = stid2eid[entry['id']]
    begidx = st_vecs['rbps'][stidx] - 1
    endidx = begidx + st_vecs['rlens'][stidx]

    dumped_rep = pickle.dumps({
        'ingredients': ingr_vec,
        'instructions': st_vecs['encs'][begidx:endidx],
        'class': class_dict[entry['id']] + 1,
        'images': img_infos[:MAX_NUM_IMGS]
    })

    with env.begin(write=True) as txn:
        txn.put('{}'.format(entry['id']).encode('latin1'), dumped_rep)
    partition_ids.append(entry['id'])

with open(os.path.join(output_dir, '{}_ids.pkl'.format(partition)), 'wb') as f:
    pickle.dump(partition_ids, f)

print('{} samples: {}'.format(partition, len(partition_ids)))
print('Finish creating dataset...')
