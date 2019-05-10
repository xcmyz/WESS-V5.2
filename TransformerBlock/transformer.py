import torch
import torch.nn as nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from TransformerBlock.attention import MultiHeadedAttention
from TransformerBlock.utils import SublayerConnection, PositionwiseFeedForward, get_sinusoid_encoding_table
from TransformerBlock.visualize import make_dot


class TransformerEncoderBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    Input = Input + Position_Embedding
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        # position_embeddings = torch.stack([get_sinusoid_encoding_table(
        #     x.size()[1], self.hidden) for i in range(x.size()[0])])
        # x = x + position_embeddings
        x = self.input_sublayer(
            x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class TransformerDecoderBlock(nn.Module):
    """
    Transformer Decoder
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super(TransformerDecoderBlock, self).__init__()
        self.hidden = hidden
        self.self_attention = MultiHeadedAttention(
            h=attn_heads, d_model=hidden)
        self.encoder_attention = MultiHeadedAttention(
            h=attn_heads, d_model=hidden)
        self.pos_ffn = PositionwiseFeedForward(
            d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    # def forward(self, dec_input, enc_output_1, enc_output_2):
    #     # position_embeddings = torch.stack([get_sinusoid_encoding_table(
    #     #     dec_input.size()[1], self.hidden) for i in range(dec_input.size()[0])])
    #     # print(position_embeddings.size())
    #     # dec_input = position_embeddings + dec_input
    #     dec_output = self.self_attention(dec_input, dec_input, dec_input)

    #     # print(dec_output.size())
    #     dec_output = self.encoder_attention(
    #         dec_output, enc_output_1, enc_output_2)
    #     # print(dec_output.size())
    #     dec_output = self.pos_ffn(dec_output)

    #     return dec_output

    def forward(self, dec_input, enc_output):
        # position_embeddings = torch.stack([get_sinusoid_encoding_table(
        #     dec_input.size()[1], self.hidden) for i in range(dec_input.size()[0])])
        # print(position_embeddings.size())
        # dec_input = position_embeddings + dec_input
        dec_output = self.self_attention(dec_input, dec_input, dec_input)

        dec_output = self.encoder_attention(dec_output, enc_output, enc_output)
        dec_output = self.pos_ffn(dec_output)

        return dec_output


if __name__ == "__main__":
    # Test
    # test_transformer = TransformerBlock(784, 1, 4*784, 0.1)
    # print(test_transformer)
    # # torch.save({'model': test_transformer.state_dict()}, "test_save.pth.tar")
    # test_input = Variable(torch.randn(2, 12, 784))
    # # position_embeddings = get_sinusoid_encoding_table(12, 784)
    # # position_embeddings = torch.stack([position_embeddings for i in range(2)])
    # # print(position_embeddings.size())
    # # test_input = position_embeddings + test_input
    # output = test_transformer(test_input)
    # print(output.size())

    # test_input = Variable(torch.randn(2, 12, 784))
    # with SummaryWriter(comment='TransformerBlock') as w:
    #     w.add_graph(test_transformer, (test_input,))

    # g = make_dot(output)
    # g.view()

    test_decoder = TransformerDecoderBlock(784, 1, 4*784, 0.1)
    print(test_decoder)
