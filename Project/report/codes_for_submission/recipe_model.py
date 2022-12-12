import torch
import gensim
import torch.nn as nn
import torch.optim
import torchvision.models as models

from params import get_parser

parser = get_parser()
params = parser.parse_args()


class ingredientRNN(nn.Module):

    def __init__(self):
        super(ingredientRNN, self).__init__()
        self.input_size = params.w2v_dim
        self.hidden_size = params.irnn_hdim

        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
            params.w2v_bin, binary=True)
        w2v_vecs = torch.FloatTensor(w2v_model.vectors)
        vocab_size = w2v_vecs.size(0)

        self.embedding = nn.Embedding(
            vocab_size, self.input_size, padding_idx=0)
        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            bidirectional=True,
            batch_first=True)
        self.embedding.from_pretrained(w2v_vecs)

    def forward(self, x, seq_lengths):
        x = self.embedding(x)

        lengths_sorted, idx_sorted = seq_lengths.sort(dim=0, descending=True)
        x_sorted = x.gather(0, idx_sorted.view(-1, 1, 1).expand_as(x).long())
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(
            x_sorted, lengths_sorted.cpu().data.numpy(), batch_first=True)
        _, (hidden, cstate) = self.rnn(x_packed)

        _, idx_orig = idx_sorted.sort(dim=0, descending=False)
        output = hidden.gather(1, idx_orig.view(
            1, -1, 1).expand_as(hidden).long())
        output = output.transpose(0, 1).contiguous()
        output = output.view(output.size(0), output.size(1) * output.size(2))

        return output


class instructionRNN(nn.Module):

    def __init__(self):
        super(instructionRNN, self).__init__()
        self.input_size = params.stv_dim
        self.hidden_size = params.srnn_hdim

        self.lstm = nn.LSTM(
            self.input_size, self.hidden_size, batch_first=True)

    def forward(self, x, seq_lengths):
        lengths_sorted, idx_sorted = seq_lengths.sort(dim=0, descending=True)
        x_sorted = x.gather(0, idx_sorted.view(-1, 1, 1).expand_as(x).long())
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(
            x_sorted, lengths_sorted.cpu().data.numpy(), batch_first=True)
        output, _ = self.lstm(x_packed)

        output_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True)
        _, idx_orig = idx_sorted.sort(dim=0, descending=False)
        idx_unsorted = idx_orig.view(-1, 1, 1).expand_as(output_unpacked)
        idx_last = (seq_lengths - 1).view(-1, 1).expand(
            output_unpacked.size(0), output_unpacked.size(2)).unsqueeze(1)
        output = output_unpacked.gather(0, idx_unsorted).gather(1, idx_last)
        output = output.reshape(output.size(0), -1)

        return output


class JointEmbeddingModel(nn.Module):

    def __init__(self):
        super(JointEmbeddingModel, self).__init__()

        img_model = models.resnet50(True)
        img_modules = list(img_model.children())[:-1]
        self.cnn = nn.Sequential(*img_modules)

        self.img_embed = nn.Sequential(
            nn.Linear(params.img_feature, params.emb_dim),
            nn.Tanh()
        )
        self.txt_embed = nn.Sequential(
            nn.Linear(2*params.irnn_hdim + params.srnn_hdim, params.emb_dim),
            nn.Tanh()
        )

        self.ingr_rnn = ingredientRNN()
        self.inst_rnn = instructionRNN()

        if params.sem_reg:
            self.sem_reg = nn.Linear(params.emb_dim, params.n_classes)

    def forward(self, img, inst, n_inst, ingr, n_ingr):

        img_features = self.cnn(img)
        img_emb = self.img_embed(img_features.view(img_features.size(0), -1))
        norm_factor = torch.norm(img_emb, p=2, dim=1, keepdim=True)
        norm_factor = torch.clamp(norm_factor, min=1e-10).expand_as(img_emb)
        img_emb = img_emb / norm_factor

        ingr_emb = self.ingr_rnn(ingr, n_ingr)
        inst_emb = self.inst_rnn(inst, n_inst)
        txt_features = torch.cat([ingr_emb, inst_emb], dim=1)
        txt_emb = self.txt_embed(txt_features)
        norm_factor = torch.norm(txt_emb, p=2, dim=1, keepdim=True)
        norm_factor = torch.clamp(norm_factor, min=1e-10).expand_as(txt_emb)
        txt_emb = txt_emb / norm_factor

        if params.sem_reg:
            img_sem = self.sem_reg(img_emb)
            txt_sem = self.sem_reg(txt_emb)
            output = [img_emb, txt_emb, img_sem, txt_sem]
        else:
            output = [img_emb, txt_emb]

        return output
