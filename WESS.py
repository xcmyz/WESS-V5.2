import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
# import random

from TransformerBlock import TransformerDecoderBlock as DecoderBlock
from TransformerBlock import get_sinusoid_encoding_table
# from bert_embedding import get_bert_embedding
from layers import BERT, LinearProjection, LinearNet_TwoLayer, FFN, PostNet, LinearProjection
from text.symbols import symbols
import hparams as hp
# print(len(symbols))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device("cpu")

if_parallel = False


class WESS_Encoder(nn.Module):
    """
    Encoder
    (pre-transformer replaced by GRU)
    """

    def __init__(self,
                 vocab_max_size=2000,
                 embedding_size=256,
                 GRU_hidden_size=768,
                 GRU_num_layers=1,
                 GRU_batch_first=True,
                 GRU_bidirectional=True,
                 bert_prenet_hidden=1024,
                 bert_prenet_output=256,
                 bert_hidden=256,
                 bert_n_layers=2,
                 bert_attn_heads=4,
                 #  embedding_postnet_hidden=1024,
                 #  embedding_postnet_output=256,
                 dropout=0.1):
        """
        :param encoder_hparams
        """

        super(WESS_Encoder, self).__init__()
        self.vocab_max_size = vocab_max_size
        self.embedding_size = embedding_size
        self.GRU_hidden = GRU_hidden_size
        self.GRU_num_layers = GRU_num_layers
        self.GRU_batch_first = GRU_batch_first
        self.GRU_bidirectional = GRU_bidirectional
        self.bert_prenet_hidden = bert_prenet_hidden
        self.bert_prenet_output = bert_prenet_output
        self.bert_hidden = bert_hidden
        self.bert_n_layers = bert_n_layers
        self.bert_attn_heads = bert_attn_heads
        # self.embedding_postnet_hidden = embedding_postnet_hidden
        # self.embedding_postnet_output = embedding_postnet_output
        self.dropout = dropout

        # Embeddings
        self.pre_embedding = nn.Embedding(len(symbols)+1, self.embedding_size)
        self.position_embedding = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self.vocab_max_size, self.embedding_size), freeze=True)

        # self.pre_GRU = nn.GRU(input_size=self.embedding_size,
        #                       hidden_size=self.GRU_hidden,
        #                       num_layers=self.GRU_num_layers,
        #                       batch_first=self.GRU_batch_first,
        #                       dropout=self.dropout,
        #                       bidirectional=self.GRU_bidirectional)

        self.pre_GRU = nn.GRU(input_size=self.embedding_size,
                              hidden_size=self.GRU_hidden,
                              num_layers=self.GRU_num_layers,
                              batch_first=self.GRU_batch_first,
                              bidirectional=self.GRU_bidirectional)

        self.bert_pre_net = LinearNet_TwoLayer(
            self.GRU_hidden, self.bert_prenet_hidden, self.bert_prenet_output)

        self.bi_Transformer = BERT(hidden=self.bert_hidden,
                                   n_layers=self.bert_n_layers,
                                   attn_heads=self.bert_attn_heads,
                                   dropout=self.dropout)

        # self.EmbeddingNet = LinearNet_TwoLayer(
        #     self.embedding_size, self.embedding_postnet_hidden, self.embedding_postnet_output)

    def init_GRU_hidden(self, batch_size, num_layers, hidden_size):
        if self.GRU_bidirectional:
            return torch.zeros(num_layers*2, batch_size,  hidden_size).to(device)
        else:
            return torch.zeros(num_layers*1, batch_size,  hidden_size).to(device)

    def get_GRU_embedding(self, GRU_output):
        # print(GRU_output.size())
        out_1 = GRU_output[:, 0:1, :]
        out_2 = GRU_output[:, GRU_output.size(1)-1:, :]

        out = out_1 + out_2
        out = out[:, :, 0:out.size(2)//2] + out[:, :, out.size(2)//2:]

        # print(out.size())
        return out

    def cal_P_GRU(self, batch, gate_for_words_batch):
        list_input = list()
        list_output = list()

        for ind in range(len(gate_for_words_batch)-1):
            list_input.append(
                batch[gate_for_words_batch[ind]:gate_for_words_batch[ind+1]])

        # print(len(list_input))
        for one_word in list_input:
            one_word = torch.stack([one_word])

            # pos_input = torch.Tensor(
            #     [i for i in range(one_word.size(1))]).long().to(device)
            # position_embedding = self.position_embedding(pos_input)
            # position_embedding = position_embedding.unsqueeze(0)

            # one_word = one_word + position_embedding
            # output_one_word = self.P_transformer_block(one_word)
            # print(output_one_word.size())

            # self.pre_GRU.flatten_parameters()
            output_one_word = self.pre_GRU(one_word)[0]
            output_one_word = self.get_GRU_embedding(output_one_word)
            word = output_one_word.squeeze(0)
            # word = output_one_word[output_one_word.size()[0]-1]
            list_output.append(word)

        output = torch.stack(list_output)
        output = output.squeeze(1)
        # print(output.size())

        return output

    # def pad_by_word(self, words_batch):
    #     len_arr = np.array(list())
    #     for ele in words_batch:
    #         len_arr = np.append(len_arr, ele.size(0))
    #     max_size = int(len_arr.max())
    #     # print(max_size)

    #     def pad(tensor, target_length):
    #         embedding_size = tensor.size(1)
    #         pad_tensor = torch.zeros(1, embedding_size).to(device)

    #         for i in range(target_length-tensor.size(0)):
    #             tensor = torch.cat((tensor, pad_tensor))

    #         return tensor

    #     padded = list()
    #     for one_batch in words_batch:
    #         one_batch = pad(one_batch, max_size)
    #         padded.append(one_batch)
    #     padded = torch.stack(padded)

    #     return padded

    # def pad_all(self, word_batch, embeddings):
    #     # print(word_batch.size())
    #     # print(embeddings.size())
    #     if word_batch.size(1) == embeddings.size(1):
    #         return word_batch, embeddings

    #     if word_batch.size(1) > embeddings.size(1):
    #         pad_len = word_batch.size(1) - embeddings.size(1)
    #         pad_vec = torch.zeros(word_batch.size(
    #             0), pad_len, embeddings.size(2)).float().to(device)
    #         embeddings = torch.cat((embeddings, pad_vec), 1)
    #         return word_batch, embeddings

    #     if word_batch.size(1) < embeddings.size(1):
    #         pad_len = embeddings.size(1) - word_batch.size(1)
    #         pad_vec = torch.zeros(word_batch.size(
    #             0), pad_len, embeddings.size(2)).float().to(device)
    #         word_batch = torch.cat((word_batch, pad_vec), 1)
    #         return word_batch, embeddings

    def pad_bert_embedding_and_GRU_embedding(self, bert_embedding, GRU_embedding):
        """One batch"""

        len_bert = len(bert_embedding)
        len_GRU = GRU_embedding.size(0)
        max_len = max(len_bert, len_GRU)

        # pad_embedding = torch.zeros(1, GRU_embedding.size(1)).to(device)

        # for i in range(max_len - len_GRU):
        #     GRU_embedding = torch.cat((GRU_embedding, pad_embedding), 0)

        # for i in range(max_len - len_bert):
        #     bert_embedding = torch.cat((bert_embedding, pad_embedding), 0)

        # print(GRU_embedding.size())
        # print(bert_embedding.size())

        GRU_embedding = torch.cat((GRU_embedding, torch.zeros(
            max_len - len_GRU, GRU_embedding.size(1)).to(device)), 0)
        bert_embedding = torch.cat((bert_embedding, torch.zeros(
            max_len - len_bert, GRU_embedding.size(1)).to(device)), 0)

        # output = bert_embedding + GRU_embedding

        return bert_embedding, GRU_embedding

    def pad_all(self, bert_transformer_input):
        len_list = list()
        for batch in bert_transformer_input:
            len_list.append(batch.size(0))

        max_len = max(len_list)

        # pad_embedding = torch.zeros(
        #     1, bert_transformer_input[0].size(1)).to(device)

        for index, batch in enumerate(bert_transformer_input):
            bert_transformer_input[index] = torch.cat((bert_transformer_input[index], torch.zeros(
                max_len - bert_transformer_input[index].size(0), bert_transformer_input[index].size(1)).to(device)), 0)

        bert_transformer_input = torch.stack(bert_transformer_input)

        return bert_transformer_input

    def forward(self, x, bert_embeddings, gate_for_words, indexs_list):
        """
        :param: x: (batch, length)
        :param: bert_embeddings: (batch, length, 768)
        :param: gate_for_words: (batch, indexs)
        """

        # Embedding
        x = self.pre_embedding(x)
        # print("x:", x.size())

        # P_GRU
        words_batch = list()
        for index, batch in enumerate(x):
            words_batch.append(self.cal_P_GRU(batch, gate_for_words[index]))

        # words_batch = self.pad_by_word(words_batch)
        # bert_embeddings = self.pad_by_word(bert_embeddings)
        # words_batch, bert_embeddings = self.pad_all(
        #     words_batch, bert_embeddings)
        # # print(words_batch.size())
        # # print(bert_embeddings.size())
        # bert_input = words_batch + bert_embeddings

        # # Add Position Embedding
        # pos_input = torch.stack([torch.Tensor([i for i in range(
        #     bert_input.size(1))]).long() for i in range(bert_input.size(0))]).to(device)
        # pos_embedding = self.position_embedding(pos_input)
        # bert_input = bert_input + pos_embedding

        # encoder_output = self.bert_encoder(bert_input)

        # New

        # If you need data parallel, you will need some extra processing.
        if self.training and if_parallel:
            # print(indexs_list)
            bert_embeddings = [bert_embeddings[i] for i in indexs_list]

        if not (len(words_batch) == len(bert_embeddings)):
            raise ValueError(
                "the length of bert embeddings is not equal to the length of GRU embeddings.")

        bert_transformer_input = list()

        for batch_index in range(len(words_batch)):

            # print("bert_embedding:", bert_embeddings[batch_index])
            # print(words_batch[batch_index])

            bert_embedding, GRU_embedding = self.pad_bert_embedding_and_GRU_embedding(
                bert_embeddings[batch_index], words_batch[batch_index])
            add_bert_GRU = bert_embedding + GRU_embedding
            bert_transformer_input.append(add_bert_GRU)

        bert_transformer_input = self.pad_all(bert_transformer_input)

        pos_input_one_batch = torch.Tensor(
            [i for i in range(bert_transformer_input.size(1))]).long()
        pos_input = torch.stack([pos_input_one_batch for _ in range(
            bert_transformer_input.size(0))]).to(device)
        position_embedding = self.position_embedding(pos_input)

        #  print("position_embedding:", position_embedding.size())
        # print("position embedding:", position_embedding)

        # bert_transformer_input = bert_transformer_input + position_embedding

        # print("bert_transformer_input:", bert_transformer_input.size())
        bert_transformer_input = self.bert_pre_net(bert_transformer_input)
        # print("bert_transformer_input:", bert_transformer_input.size())

        bert_transformer_input = bert_transformer_input + position_embedding

        encoder_output_word = self.bi_Transformer(bert_transformer_input)

        # encoder_output_alpha = self.EmbeddingNet(x)
        encoder_output_alpha = x

        return encoder_output_word, encoder_output_alpha


class WESS_Decoder(nn.Module):
    """
    Decoder
    """

    def __init__(self,
                 vocab_max_size=2000,
                 embedding_size=256,
                 decoder_input_hidden=256,
                 decoder_prenet_word_hidden=384,
                 decoder_prenet_word_output=256,
                 decoder_prenet_alpha_hidden=384,
                 decoder_prenet_alpha_output=256,
                 decoder_word_attn_heads=4,
                 decoder_word_feed_forward_hidden=4*256,
                 decoder_alpha_attn_heads=4,
                 decoder_alpha_feed_forward_hidden=4*256,
                 decoder_n_layer_word=1,
                 decoder_n_layer_alpha=1,
                 max_decode_length=1000,
                 #  decoder_postnet_hidden=384,
                 decoder_postnet_output=80,
                 #  linear_projection_hidden=128,
                 linear_projection_output=1,
                 #  teacher_forced=0.9,
                 decode_per_step=3,
                 gate_threshold=0.5,
                 dropout=0.1):
        """
        :param decoder_hparams
        """

        super(WESS_Decoder, self).__init__()
        self.vocab_max_size = vocab_max_size
        self.embedding_size = embedding_size
        self.decoder_input_hidden = decoder_input_hidden
        self.decoder_prenet_word_hidden = decoder_prenet_word_hidden
        self.decoder_prenet_word_output = decoder_prenet_word_output
        self.decoder_prenet_alpha_hidden = decoder_prenet_alpha_hidden
        self.decoder_prenet_alpha_output = decoder_prenet_alpha_output
        self.decoder_word_attn_heads = decoder_word_attn_heads
        self.decoder_word_feed_forward_hidden = decoder_word_feed_forward_hidden
        self.decoder_alpha_attn_heads = decoder_alpha_attn_heads
        self.decoder_alpha_feed_forward_hidden = decoder_alpha_feed_forward_hidden
        self.decoder_n_layer_word = decoder_n_layer_word
        self.decoder_n_layer_alpha = decoder_n_layer_alpha
        self.max_decode_length = max_decode_length
        # self.decoder_postnet_hidden = decoder_postnet_hidden
        self.decoder_postnet_output = decoder_postnet_output
        # self.linear_projection_hidden = linear_projection_hidden
        self.linear_projection_output = linear_projection_output
        # self.teacher_forced = teacher_forced
        self.decode_per_step = decode_per_step
        self.gate_threshold = gate_threshold
        self.dropout = dropout

        # Position Embedding
        self.position_embedding = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self.vocab_max_size, self.embedding_size), freeze=True)

        # PreNet
        self.PreNet_word = LinearNet_TwoLayer(
            self.decoder_postnet_output, self.decoder_prenet_word_hidden, self.decoder_prenet_word_output)
        self.PreNet_alpha = LinearNet_TwoLayer(
            self.decoder_postnet_output, self.decoder_prenet_alpha_hidden, self.decoder_prenet_alpha_output)

        # Decoder Block
        self.decoder_block_word = DecoderBlock(self.decoder_input_hidden,
                                               self.decoder_word_attn_heads,
                                               self.decoder_word_feed_forward_hidden,
                                               self.dropout)
        self.decoder_block_alpha = DecoderBlock(self.decoder_input_hidden,
                                                self.decoder_alpha_attn_heads,
                                                self.decoder_alpha_feed_forward_hidden,
                                                self.dropout)

        self.decoder_layer_stack_word = nn.ModuleList(
            [self.decoder_block_word for _ in range(self.decoder_n_layer_word)])
        self.decoder_layer_stack_alpha = nn.ModuleList(
            [self.decoder_block_alpha for _ in range(self.decoder_n_layer_alpha)])

        # PostNet
        # self.decoder_postnet = FFN(
        #     self.decoder_input_hidden, self.decoder_postnet_hidden, self.decoder_postnet_output)
        self.decoder_postnet = FFN(
            self.decoder_input_hidden, self.decoder_postnet_output)

        self.postnet = PostNet()

        # Linear Projection
        # self.linear_projection = LinearProjection(
        #     self.decoder_postnet_output, self.linear_projection_hidden, self.linear_projection_output)
        self.linear_projection = LinearProjection(
            self.decoder_postnet_output, self.linear_projection_output)

    def forward(self, encoder_output_word, encoder_output_alpha, teacher_forced=0.9, mel_target=None):
        """Decoder"""

        # Init
        position_input_word = torch.stack([torch.Tensor([i for i in range(encoder_output_word.size(
            1))]).to(device) for _ in range(encoder_output_word.size(0))]).long()
        position_embedding_word = self.position_embedding(position_input_word)
        encoder_output_word = encoder_output_word + position_embedding_word

        position_input_alpha = torch.stack([torch.Tensor([i for i in range(encoder_output_alpha.size(
            1))]).to(device) for _ in range(encoder_output_alpha.size(0))]).long()
        position_embedding_alpha = self.position_embedding(
            position_input_alpha)
        encoder_output_alpha = encoder_output_alpha + position_embedding_alpha

        if self.training:

            # Init
            mel_output = list()
            mel_first_input_w = torch.zeros(mel_target.size(
                0), self.decode_per_step, mel_target.size(2)).to(device)
            mel_first_input_a = torch.zeros(mel_target.size(
                0), self.decode_per_step, mel_target.size(2)).to(device)

            # print(mel_first_input.size())

            mel_first_input_word = self.PreNet_word(mel_first_input_w)
            mel_first_input_alpha = self.PreNet_alpha(mel_first_input_a)

            pos_input = torch.stack([torch.Tensor([i for i in range(self.decode_per_step)]).to(
                device) for _ in range(mel_target.size(0))]).long()
            pos_embedding = self.position_embedding(pos_input)

            # print("pos_emb:", pos_embedding.size())

            mel_first_input_word = mel_first_input_word + pos_embedding
            mel_first_input_alpha = mel_first_input_alpha + pos_embedding

            # Word
            for decoder_layer in self.decoder_layer_stack_word:
                mel_first_input_word = decoder_layer(
                    mel_first_input_word, encoder_output_word)

            decoder_first_output_word = mel_first_input_word

            # Alpha
            for decoder_layer in self.decoder_layer_stack_alpha:
                mel_first_input_alpha = decoder_layer(
                    mel_first_input_alpha, encoder_output_alpha)

            decoder_first_output_alpha = mel_first_input_alpha

            decoder_first_output = decoder_first_output_word + decoder_first_output_alpha
            decoder_first_output = self.decoder_postnet(decoder_first_output)

            mel_output.append(decoder_first_output)

            for cnt_pos in range(mel_target.size(1) // self.decode_per_step):
                # Position Embedding
                # position = cnt_pos + 1
                # pos_input = torch.stack(
                #     [torch.Tensor([position]).long() for i in range(mel_target.size(0))]).to(device)
                # pos_emb = self.position_embedding(pos_input)

                range_start = cnt_pos * self.decode_per_step
                range_end = range_start + self.decode_per_step

                pos_input = torch.stack([torch.Tensor([i for i in range(range_start, range_end)]).to(
                    device) for _ in range(mel_target.size(0))]).long()
                pos_embedding = self.position_embedding(pos_input)

                # Teacher Forced
                # print(teacher_forced)
                if np.random.uniform(0.0, 1.0, 1)[0] < teacher_forced:
                    # Word
                    model_input_word = mel_target[:, range_start:range_end, :]
                    model_input_word = self.PreNet_word(model_input_word)
                    model_input_word = model_input_word + pos_embedding

                    # Alpha
                    model_input_alpha = mel_target[:, range_start:range_end, :]
                    model_input_alpha = self.PreNet_alpha(model_input_alpha)
                    model_input_alpha = model_input_alpha + pos_embedding

                else:
                    # Word
                    # model_input_word = mel_target[:, range_start:range_end, :]
                    model_input_word = mel_output[len(mel_output)-1]
                    # print("model_input_word:", model_input_word.size())
                    model_input_word = self.PreNet_word(model_input_word)
                    model_input_word = model_input_word + pos_embedding

                    # Alpha
                    # model_input_alpha = mel_target[:, range_start:range_end, :]
                    model_input_alpha = mel_output[len(mel_output)-1]
                    # print("model_input_alpha:", model_input_alpha.size())
                    model_input_alpha = self.PreNet_alpha(model_input_alpha)
                    model_input_alpha = model_input_alpha + pos_embedding

                # print("model_input_word:", model_input_word.size())

                for decoder_layer in self.decoder_layer_stack_word:
                    model_input_word = decoder_layer(
                        model_input_word, encoder_output_word)

                for decoder_layer in self.decoder_layer_stack_alpha:
                    model_input_alpha = decoder_layer(
                        model_input_alpha, encoder_output_alpha)

                model_output = model_input_word + model_input_alpha
                model_output = self.decoder_postnet(model_output)

                # print("model_output:", model_output.size())

                mel_output.append(model_output)

            mel_output = torch.cat(mel_output, 1)[:, 0:mel_target.size(1), :]
            # print("mel_output:", mel_output.size())
            mel_output_postnet = self.postnet(mel_output) + mel_output

            # print("mel_output:", mel_output.size())

            gate_predicted = self.linear_projection(mel_output)
            # print(gate_predicted)
            gate_predicted = gate_predicted.squeeze(2)

            return mel_output, mel_output_postnet, gate_predicted

        else:

            # Test One Text Once
            # Init
            mel_output = list()

            mel_first_input_w = torch.zeros(
                1, self.decode_per_step, 80).to(device)
            mel_first_input_a = torch.zeros(
                1, self.decode_per_step, 80).to(device)

            mel_first_input_word = self.PreNet_word(mel_first_input_w)
            mel_first_input_alpha = self.PreNet_alpha(mel_first_input_a)

            pos_input = torch.stack([torch.Tensor(
                [i for i in range(self.decode_per_step)]).to(device) for _ in range(1)]).long()

            # print(pos_input.size())

            pos_embedding = self.position_embedding(pos_input)

            # print(pos_embedding.size())

            mel_first_input_word = mel_first_input_word + pos_embedding
            mel_first_input_alpha = mel_first_input_alpha + pos_embedding

            # Word
            for decoder_layer in self.decoder_layer_stack_word:
                mel_first_input_word = decoder_layer(
                    mel_first_input_word, encoder_output_word)

            decoder_first_output_word = mel_first_input_word

            # Alpha
            for decoder_layer in self.decoder_layer_stack_alpha:
                mel_first_input_alpha = decoder_layer(
                    mel_first_input_alpha, encoder_output_alpha)

            decoder_first_output_alpha = mel_first_input_alpha

            decoder_first_output = decoder_first_output_word + decoder_first_output_alpha
            decoder_first_output = self.decoder_postnet(decoder_first_output)

            mel_output.append(decoder_first_output)

            for cnt_pos in range(self.max_decode_length // self.decode_per_step):

                if (cnt_pos + 1) == self.max_decode_length // self.decode_per_step:
                    print("Warning! Reached max decoder steps.")

                # Position Embedding
                range_start = cnt_pos * self.decode_per_step
                range_end = range_start + self.decode_per_step

                pos_input = torch.stack([torch.Tensor([i for i in range(
                    range_start, range_end)]).to(device) for _ in range(1)]).long()
                pos_embedding = self.position_embedding(pos_input)

                # Default Teacher Forced

                # Word
                model_input_word = mel_output[len(mel_output)-1]
                model_input_word = self.PreNet_word(model_input_word)
                model_input_word = model_input_word + pos_embedding

                # print("model_input_word:", model_input_word.size())

                for decoder_layer in self.decoder_layer_stack_word:
                    model_input_word = decoder_layer(
                        model_input_word, encoder_output_word)

                # Alpha
                model_input_alpha = mel_output[len(mel_output)-1]
                model_input_alpha = self.PreNet_alpha(model_input_alpha)
                model_input_alpha = model_input_alpha + pos_embedding

                for decoder_layer in self.decoder_layer_stack_alpha:
                    model_input_alpha = decoder_layer(
                        model_input_alpha, encoder_output_alpha)

                model_output = model_input_word + model_input_alpha
                model_output = self.decoder_postnet(model_output)

                mel_output.append(model_output)

                # print(model_output.size())

                gate_predicted = self.linear_projection(model_output)
                # print(gate_predicted)
                gate_predicted = torch.sigmoid(gate_predicted)
                # print(gate_predicted)

                # print(gate_predicted.size())
                # print(gate_predicted)
                # print(max(gate_predicted[0]).data)

                if max(gate_predicted[0]).data > self.gate_threshold:
                    # print(max(gate_predicted[0]).data)
                    break

            mel_output = torch.cat(mel_output, 1)
            mel_output_postnet = self.postnet(mel_output) + mel_output

            # print(mel_output.size())

            return mel_output, mel_output_postnet


class WESS(nn.Module):
    """
    WESS: Word Embedding Speech Synthesizer
    """

    def __init__(self):
        super(WESS, self).__init__()
        self.encoder = WESS_Encoder()
        self.decoder = WESS_Decoder()

    def forward(self, x, bert_embeddings, gate_for_words, teacher_forced=0.9, indexs_list=None, mel_target=None, gate_target=None):

        # If you need data parallel, you will need some extra processing for bert embeddings

        encoder_output_word, encoder_output_alpha = self.encoder(
            x, bert_embeddings,  gate_for_words, indexs_list)

        # print("encoder_output_word:", encoder_output_word)
        # print("encoder_output_alpha:", encoder_output_alpha)

        # print("encoder output word:", encoder_output_word.size())
        # print("encoder output alpha:", encoder_output_alpha.size())

        output = self.decoder(encoder_output_word,
                              encoder_output_alpha, teacher_forced, mel_target)

        return output, mel_target, gate_target


if __name__ == "__main__":
    # Test
    # test_encoder = WESS_Encoder()
    # print(test_encoder)
    # # x = torch.randn(2, 180, 768)
    # # print("x: ", x.size())

    # # out = test_GRU(x)

    # # test input
    # test_x = torch.stack(
    #     [torch.Tensor([11, 12, 34, 21, 33]), torch.Tensor([23, 32, 56, 0, 0])])
    # test_bert_embedding = [torch.randn(2, 768), torch.randn(3, 768)]

    test_encoder = WESS_Encoder().to(device)
    print(test_encoder)

    test_decoder = WESS_Decoder().to(device)
    print(test_decoder)

    test_WESS = WESS().to(device)
    print(test_WESS)
