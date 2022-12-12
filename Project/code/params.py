import argparse

# Copied from
# https://github.com/torralba-lab/im2recipe-Pytorch/blob/master/scripts/params.py


def get_parser():

    parser = argparse.ArgumentParser(description='model parameters')

    parser.add_argument('-partition', dest='partition', default='test')
    parser.add_argument('-nlosscurves', dest='nlosscurves',
                        default=3, type=int)
    parser.add_argument('-epochs', dest='epochs', default=720, type=int)
    parser.add_argument('-start_epoch', dest='start_epoch',
                        default=0, type=int)
    parser.add_argument('-embedding', dest='embedding', default='images')
    parser.add_argument('-medr', dest='medr', default=1000, type=int)
    parser.add_argument('-tsamples', dest='tsamples', default=20, type=int)
    parser.add_argument('-maxlen', dest='maxlen', default=20, type=int)
    parser.add_argument('-maxims', dest='maxims', default=5, type=int)
    parser.add_argument('-imsize', dest='imsize', default=256, type=int)
    parser.add_argument('-batch_size', dest='batch_size',
                        default=160, type=int)
    parser.add_argument('-n_workers', dest='n_workers', default=30, type=int)
    parser.add_argument('-dispfreq', dest='dispfreq', default=1000, type=int)
    parser.add_argument('-valfreq', dest='valfreq', default=10, type=int)
    parser.add_argument('-patience', dest='patience', default=1, type=int)
    parser.add_argument('-test_feats', dest='test_feats',
                        default='../results/')
    parser.add_argument('-cos_weight', dest='cos_weight',
                        default=0.98, type=float)
    parser.add_argument('-reg_weight', dest='reg_weight',
                        default=0.01, type=float)
    parser.add_argument('-img_train', dest='img_train',
                        default=False, type=bool)
    parser.add_argument('-txt_train', dest='txt_train',
                        default=True, type=bool)
    parser.add_argument('-lr', dest='lr', default=0.0001, type=float)
    parser.add_argument('-resume_path', dest='resume_path', default='')
    parser.add_argument('-data_path', dest='data_path', default='../data/')
    parser.add_argument('-img_path', dest='img_path', default='../data/images')
    parser.add_argument('-early_stop', dest='early_stop',
                        default=1000, type=int)

    parser.add_argument('-seed', dest='seed', default=42, type=int)
    parser.add_argument('-f101_cats', dest='f101_cats',
                        default='../data/food101_classes_renamed.txt')
    parser.add_argument('-vocab', dest='vocab',
                        default='../data/text/vocab.txt')
    parser.add_argument('-stvecs', dest='stvecs', default='../data/text/')
    parser.add_argument('-dataset', dest='dataset', default='../data/recipe1M')
    parser.add_argument('-suffix', dest='suffix', default='1M')
    parser.add_argument('-logfile', dest='logfile', default='')
    parser.add_argument(
        '--nocrtbgrs', dest='create_bigrams', action='store_false')
    parser.add_argument('--crtbgrs', dest='create_bigrams',
                        action='store_true')
    parser.set_defaults(create_bigrams=False)
    parser.add_argument(
        '-bigram_folder', dest='bigram_folder', default='../data/')
    parser.add_argument('-lmdb_partition',
                        dest='lmdb_partition', default='train')

    parser.add_argument('-w2v_dim', dest='w2v_dim', default=300, type=int)
    parser.add_argument('-irnn_hdim', dest='irnn_hdim', default=300, type=int)
    parser.add_argument('-w2v_bin', dest='w2v_bin',
                        default='../data/text/vocab.bin')

    parser.add_argument('-stv_dim', dest='stv_dim', default=1024, type=int)
    parser.add_argument('-srnn_hdim', dest='srnn_hdim', default=1024, type=int)

    parser.add_argument('-img_feature', dest='img_feature',
                        default=2048, type=int)
    parser.add_argument('-emb_dim', dest='emb_dim', default=1024, type=int)
    parser.add_argument('-sem_reg', dest='sem_reg', default=True, type=bool)
    parser.add_argument('-n_classes', dest='n_classes', default=1048, type=int)

    parser.add_argument('-emb_type', dest='emb_type', default='image')
    parser.add_argument('-checkpoint_path',
                        dest='checkpoint_path', default='./snapshot/')
    parser.add_argument('-train_loss_output', dest='train_loss_output',
                        default='./snapshot/train_loss.txt')
    parser.add_argument('-val_loss_output', dest='val_loss_output',
                        default='./snapshot/val_loss.txt')
    parser.add_argument(
        '-val_recall_output', dest='val_recall_output',
        default='./snapshot/val_recall.txt')
    parser.add_argument('-force_save', dest='force_save', default=20, type=int)

    return parser
