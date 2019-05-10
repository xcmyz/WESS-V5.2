import torch
from torch.utils.data import Dataset, DataLoader
from text import text_to_sequence, symbols
from layers import get_bert_embedding
import hparams
from WESS import WESS_Encoder, WESS_Decoder, WESS

import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from multiprocessing import cpu_count
import os

# import visualize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device("cpu")

# def get_separator(text):
#     total_len = len(text)
#     sep_list = list([0])
#     for i in range(total_len):
#         if ((not text[i].isalpha()) and (not text[i].isdigit()) and (i != 0)):
#             sep_list.append(i)

#     if sep_list[len(sep_list)-1] != total_len - 1:
#         sep_list.append(total_len-1)
#     return sep_list


def gen_text_sep(token_list):
    index = 0
    out_sep = list([0])
    out_list = list()

    for str_piece in token_list:
        temp_seq = text_to_sequence(str_piece, hparams.text_cleaners)
        if not temp_seq:
            # print("#########")
            temp_seq = [len(symbols)]
        out_list = out_list + temp_seq
        index = index + len(temp_seq)
        out_sep.append(index)

    # print(out_list)
    # print(out_sep)
    return out_list, out_sep


def get_separator(text, tokenized):
    # print(text)
    # print(tokenized)
    total_len = len(text)
    tokenized = tokenized[1:len(tokenized)-1]
    # start = 0
    output_sep = list()
    # output_sep.append(start)
    cnt = 0
    for i in range(len(text)):
        # print(tokenized[cnt])
        word = text[i:i+len(tokenized[cnt])]
        word = word.lower()
        # print(word)
        if word == tokenized[cnt]:
            output_sep.append(i)
            cnt = cnt + 1
            # print(cnt)
            if cnt >= len(tokenized):
                break
    # for word in tokenized:
    #     output_sep.append(start+len(word)+1)
    #     start = start + len(word)

    # print(output_sep)
    output_sep.append(total_len)
    # print(output_sep)

    if output_sep[0] != 0:
        output_sep = [0] + output_sep

    return output_sep


def cut_text(text, sep_list):
    for ind in range(len(sep_list) - 1):
        print(text[sep_list[ind]:sep_list[ind+1]])


class WESSDataLoader(Dataset):
    """LJSpeech"""

    def __init__(self, tokenizer, model_bert, dataset_path=hparams.dataset_path):
        self.dataset_path = dataset_path
        self.text_path = os.path.join(self.dataset_path, "train.txt")
        self.text = process_text(self.text_path)
        self.model_bert = model_bert
        self.tokenizer = tokenizer
        # with open(textPath, "r", encoding='utf-8') as f:
        #     training_text = len(f.read())
        # self.trainingText = training_text

    def __len__(self):
        # print(self.text_path)
        # print(len(self.text))
        return len(self.text)

    def gen_str(self, token_list):
        out_str = str()
        for str_piece in token_list:
            out_str = out_str + str_piece

        return out_str

    def gen_text_sep(self, token_list):
        index = 0
        out_sep = list([0])
        out_list = list()

        for str_piece in token_list:
            temp_seq = text_to_sequence(str_piece, hparams.text_cleaners)
            if not temp_seq:
                # print("#########")
                temp_seq = [len(symbols)]
            out_list = out_list + temp_seq
            index = index + len(temp_seq)
            out_sep.append(index)

        # print(out_list)
        # print(out_sep)
        return out_list, out_sep

    def __getitem__(self, idx):
        index = idx + 1

        mel_name = os.path.join(
            self.dataset_path, "ljspeech-mel-%05d.npy" % index)
        mel_np = np.load(mel_name)

        character = self.text[idx]
        # print(character)

        embeddings, tokens = get_bert_embedding(
            character, self.model_bert, self.tokenizer, return_token=True)
        embeddings = embeddings[1:(embeddings.size(0)-1)]
        # print(embeddings.size())
        # print(tokens)
        tokens = tokens[1:(len(tokens)-1)]
        # print(tokens)
        # character = self.gen_str(tokens)
        character, sep_list = self.gen_text_sep(tokens)
        # print(character)
        # sep_list = get_separator(character, tokens)

        # print()
        # print(embeddings.size())
        # print(character)
        # # print(len(character))
        # # print(tokens)
        # print(sep_list)
        # print(len(sep_list))
        # print()

        # character = text_to_sequence(character, hparams.text_cleaners)
        character = np.array(character)

        return {"text": character, "mel": mel_np, "embeddings": embeddings, "sep": sep_list}


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        inx = 0
        txt = []
        for line in f.readlines():
            # print("*******")
            cnt = 0
            for index, ele in enumerate(line):
                if ele == '|':
                    cnt = cnt + 1
                    if cnt == 2:
                        inx = index
                        end = len(line)
                        # print(line)
                        txt.append(line[inx+1:end-1])
                        break
        return txt


def collate_fn(batch):
    texts = [d['text'] for d in batch]
    mels = [d['mel'] for d in batch]
    embeddings = [d['embeddings'] for d in batch]
    sep_lists = [d['sep'] for d in batch]
    # print(d['embeddings'].size())

    # print()
    # for d in batch:
    #     # print(np.shape(d['text']))
    #     print(d['sep'][len(d['sep'])-1])

    texts = pad_seq_text(texts)
    # print(np.shape(texts))

    text_max_length = np.shape(texts)[1]
    for ind, sep_list in enumerate(sep_lists):
        if sep_list[len(sep_list)-1] < text_max_length:
            sep_lists[ind].append(text_max_length)

    # for d in batch:
    #     # print(np.shape(d['text']))
    #     print(d['sep'][len(d['sep'])-1])

    # print()
    mels, gate_target = pad_seq_spec(mels)

    return {"text": texts, "mel": mels, "embeddings": embeddings, 'sep': sep_lists, "gate": gate_target}


def pad_seq_text(inputs):
    def pad_data(x, length):
        pad = 0
        return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=pad)

    max_len = max((len(x) for x in inputs))
    return np.stack([pad_data(x, max_len) for x in inputs])


def pad_seq_spec(inputs):

    def pad(x, max_len):
        # print(type(x))
        if np.shape(x)[0] > max_len:
            # print("ERROR!")
            raise ValueError("not max_len")
        s = np.shape(x)[1]
        # print(s)
        x = np.pad(x, (0, max_len - np.shape(x)
                       [0]), mode='constant', constant_values=0)
        return x[:, :s]

    max_len = max(np.shape(x)[0] for x in inputs)

    def gen_gate(batchlen, maxlen):
        list_A = [0 for i in range(batch_len - 1)]
        list_B = [1 for i in range(max_len - batch_len + 1)]

        output = list_A + list_B
        output = np.array(output)

        return output

    gate_target = list()
    for batch in inputs:
        batch_len = np.shape(batch)[0]
        gate_target.append(gen_gate(batch_len, max_len))

    gate_target = np.stack(gate_target)

    # print(max_len)
    # for x in inputs:
    #     x = pad(x, max_len)
    #     # print(np.shape(x))
    #     # print(x)
    # print(np.stack([pad(x,max_len) for x in inputs]))
    # a  = np.stack([pad(x,max_len) for x in inputs])
    # print(np.shape(a))
    # print(type(a))
    mel_output = np.stack([pad(x, max_len) for x in inputs])

    # print(np.shape(mel_output))
    # print(np.shape(gate_target))

    return mel_output, gate_target


if __name__ == "__main__":
    # Test text
    print(text_to_sequence(":;\"-'AaXx", hparams.text_cleaners))
    print(len(symbols))

    # seq = np.ndarray([])
    # seq = np.append(seq, np.array([1, 2, 3]))
    # seq = np.append(seq, np.array([1]))
    # seq = np.append(seq, np.array([2, 3]))
    # seq = np.append(seq, [1])
    # seq = np.append(seq, [2, 3])
    # seq = [[[1, 2, 3], [1, 2, 3], [1, 2, 3]]]
    # # seq = torch.Tensor(seq)
    # seq = np.array(seq)
    # print(seq)
    # print(pad_sequence(seq))

    # Test
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    text = "I am doing my job."
    embeddings, tokenized_text = get_bert_embedding(
        text, model, tokenizer, return_token=True)
    print(tokenized_text)
    seq_sep = get_separator(text, tokenized_text)
    print(seq_sep)

    # print(len(text))
    # tokenized_text = tokenizer.tokenize(text)
    # print(tokenized_text)

    # seq_sep = get_separator(text, tokenized_text)
    # print(seq_sep)

    dataset = WESSDataLoader(tokenizer, model)
    training_loader = DataLoader(dataset, batch_size=2, shuffle=True,
                                 collate_fn=collate_fn, drop_last=True, num_workers=cpu_count())

    # Test Encoder
    WESS_Encoder = WESS_Encoder().to(device)
    print(WESS_Encoder)

    # Test WESS
    test_WESS = WESS().to(device)
    # test_WESS = test_WESS.train()
    test_WESS = test_WESS.eval()
    print(test_WESS)

    # visual_x = 0

    for i, data_of_batch in enumerate(training_loader):
        # print(data)
        a = 0
        # print(data)
        # print(i)

        texts = data_of_batch["text"]
        mels = data_of_batch["mel"]
        embeddings = data_of_batch["embeddings"]

        for index, batch in enumerate(embeddings):
            embeddings[index] = embeddings[index].to(device)

        # print("embedding size 0:", len(embeddings))
        # print("embedding size 1:", len(embeddings[0]))
        # print("embedding size 2:", len(embeddings[1]))
        sep_lists = data_of_batch["sep"]
        # print(data_of_batch["gate"])

        # Test Encoder
        texts = torch.from_numpy(texts).long().to(device)

        # output = WESS_Encoder(texts, embeddings, sep_lists)

        # print()
        # print("1:", output[0].size())
        # print("2:", output[1].size())

        # Test WESS
        mels = torch.from_numpy(mels).float().to(device)

        # output = test_WESS(texts, embeddings, sep_lists, mels)
        output = test_WESS(texts, embeddings, sep_lists, mels)

        print("mels:", mels.size())
        print("output:", output[0][1].size())
        # print("gate predict:", output[1].size())

        # print("test bert embeddings")
        # for embedding in embeddings:
        #     print(embedding.size())
        # print("test sep")
        # for sep_list in sep_lists:
        #     print(len(sep_list))

        # print(np.shape(mels))

        # visual_x = output[0][1]
        # break

    # visualize.make_dot(visual_x).view()

    # text = "I love you, ,,,12,too."
    # # print(get_separator("I love you, too."))
    # # sep_list = get_separator("I love you, too.")
    # # cut_text(text, sep_list)
    # emb, tokenized_text = get_bert_embedding(text, model, tokenizer)
    # print(emb.size())
    # print(tokenized_text)
    # sep = get_separator(text, tokenized_text)
    # cut_text(text, sep)

    # # Other test
    # test_list = [np.array([1, 2, 3]), np.array([1])]
    # # print(text_to_sequence("I like", [hparams.cleaners]))
    # # print(pad_sequence(test_list))

    # # test_list = [np.array([[1, 2, 3], [1, 2, 3]]),
    # #              np.array([[1, 2, 3], [1, 3]])]
    # # print(pad_sequence(test_list))
    # a = np.ndarray((2, 2))
    # print(a)
    # print(pad(a, 3))
