import os
import math
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from utils import config
from utils.metric import rouge, moses_multi_bleu, _prec_recall_f1_score, compute_prf, compute_exact_match
from tensorflow.keras import layers
import matplotlib.pyplot as plt

'''
Define the EncoderLayer Class for Transformer, which contains Layer Normalization Layer -> Multi-Head Attention Layer ->
Dropout Layer -> Layer Normalization Layer -> Positionwise Feed Forward Layer -> Dropout Layer
'''
class EncoderLayer(layers.Layer):
    """
    Represents one Encoder layer of the Transformer Encoder
    Refer Fig. 1 in https://arxiv.org/pdf/1706.03762.pdf
    NOTE: The layer normalization step has been moved to the input as per latest version of Transformer
    """
    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 bias_mask=None, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        """
        Parameters:
            hidden_size: Hidden size
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            layer_dropout: Dropout for this layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """
        super(EncoderLayer, self).__init__()
        
        self.multi_head_attention = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth, 
                                                       hidden_size, num_heads, bias_mask, attention_dropout)
        
        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, filter_size, hidden_size,
                                                                 layer_config='cc', padding = 'both', 
                                                                 dropout=relu_dropout)
        self.dropout = layers.Dropout(layer_dropout)
        self.layer_norm_mha = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_ffn = layers.LayerNormalization(epsilon=1e-6)

    def __call__(self, inputs, mask=None, training=True):
        x = inputs
        
        # Layer Normalization
        x_norm = self.layer_norm_mha(x)
        # Multi-head attention
        y, _ = self.multi_head_attention(x_norm, x_norm, x_norm, mask)
        # Dropout and residual
        x = self.dropout(x + y, training=training)
        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)
        # Positionwise Feedforward
        y = self.positionwise_feed_forward(x_norm)
        # Dropout and residual
        y = self.dropout(x + y, training=training)
        # y = self.layer_norm_end(y)
        return y

'''
Define the DecoderLayer Class for Transformer, which contains Layer Normalization Layer -> Multi-Head Attention Layer ->
Dropout Layer -> Layer Normalization Layer -> Multi-Head Encoder-Decoder Attention Layer -> Dropout Layer
-> Layer Normalization Layer -> Positionwise Feed Forward Layer -> Dropout Layer
'''
class DecoderLayer(layers.Layer):
    """
    Represents one Decoder layer of the Transformer Decoder
    Refer Fig. 1 in https://arxiv.org/pdf/1706.03762.pdf
    NOTE: The layer normalization step has been moved to the input as per latest version of T2T
    """
    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 bias_mask, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        """
        Parameters:
            hidden_size: Hidden size
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            layer_dropout: Dropout for this layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """ 
        super(DecoderLayer, self).__init__()
        self.multi_head_attention_dec = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth, 
                                                       hidden_size, num_heads, bias_mask, attention_dropout)

        self.multi_head_attention_enc_dec = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth, 
                                                       hidden_size, num_heads, None, attention_dropout)
        
        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, filter_size, hidden_size,
                                                                 layer_config='cc', padding = 'left', 
                                                                 dropout=relu_dropout)
        self.dropout = layers.Dropout(layer_dropout)
        self.layer_norm_mha_dec = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_mha_enc = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_ffn = layers.LayerNormalization(epsilon=1e-6)
        # self.layer_norm_end = layers.LayerNormalization(epsilon=1e-6)
    
    def __call__(self, inputs, training=True):
        """
        NOTE: Inputs is a tuple consisting of decoder inputs and encoder output
        """

        x, encoder_outputs, attention_weight, mask = inputs
        mask_src, dec_mask = mask
        
        # Layer Normalization before decoder self attention
        x_norm = self.layer_norm_mha_dec(x)
        # Masked Multi-head attention
        y, _ = self.multi_head_attention_dec(x_norm, x_norm, x_norm, dec_mask)
        # Dropout and residual after self-attention
        x = self.dropout(x + y, training=training)
        # Layer Normalization before encoder-decoder attention
        x_norm = self.layer_norm_mha_enc(x)
        # Multi-head encoder-decoder attention
        y, attention_weight = self.multi_head_attention_enc_dec(x_norm, encoder_outputs, encoder_outputs, mask_src)
        # Dropout and residual after encoder-decoder attention
        x = self.dropout(x + y, training=training)
        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)
        # Positionwise Feedforward
        y = self.positionwise_feed_forward(x_norm)
        # Dropout and residual after positionwise feed forward layer
        y = self.dropout(x + y, training=training)
        # y = self.layer_norm_end(y)
        # Return encoder outputs as well to work with tf.keras.models.Sequential
        return y, encoder_outputs, attention_weight, mask

'''
MultiExpertMultiHeadAttention is similar as MultiHeadAttention, while it can be used for multi-decoder, indicates weights sharing
'''
class MultiExpertMultiHeadAttention(layers.Layer):
    def __init__(self, num_experts, input_depth, total_key_depth, total_value_depth, output_depth, 
                 num_heads, bias_mask=None, dropout=0.0):
        """
        Parameters:
            expert_num: Number of experts
            input_depth: Size of last dimension of input
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(MultiExpertMultiHeadAttention, self).__init__()
        # Checks borrowed from 
        # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
        if total_key_depth % num_heads != 0:
            print("Key depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (total_key_depth, num_heads))
            total_key_depth = total_key_depth - (total_key_depth % num_heads)
        if total_value_depth % num_heads != 0:
            print("Value depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (total_value_depth, num_heads))
            total_value_depth = total_value_depth - (total_value_depth % num_heads)
        self.num_experts = num_experts
        self.num_heads = num_heads
        self.query_scale = (total_key_depth//num_heads)**-0.5 ## sqrt
        self.bias_mask = bias_mask
        
        # Key and query depth will be same
        self.query_linear = layers.Dense(total_key_depth*num_experts, use_bias=False)
        self.key_linear = layers.Dense(total_key_depth*num_experts, use_bias=False)
        self.value_linear = layers.Dense(total_value_depth*num_experts, use_bias=False)
        self.output_linear = layers.Dense(output_depth*num_experts, use_bias=False)
        
        self.dropout = layers.Dropout(dropout)
    
    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_experts ,num_heads, seq_length, depth/(num_heads*num_experts)]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return tf.transpose(tf.reshape(x, [shape[0], shape[1], self.num_experts, self.num_heads, 
                               shape[2]//(self.num_heads*self.num_experts)]), perm=[0, 2, 3, 1, 4])
    
    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_experts ,num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, num_experts, depth/num_experts]
        """
        if len(x.shape) != 5:
            raise ValueError("x must have rank 5")
        shape = x.shape
        return tf.reshape(tf.transpose(x, perm=[0, 3, 1, 2, 4]), [shape[0], shape[3], self.num_experts, shape[4]*self.num_heads])
    
    def __call__(self, queries, keys, values, mask):
        
        # Do a linear for each component
        queries = self.query_linear(queries)
        keys = self.key_linear(keys)
        values = self.value_linear(values)
        # Split into multiple heads
        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)
        # Scale queries
        queries *= self.query_scale
        # Combine queries and keys
        logits = tf.matmul(queries, keys, transpose_b=True)
        
        if mask is not None:
            #logits += (mask * -1e18)
            mask = tf.expand_dims(tf.expand_dims(mask, 1), 1)
            logits = logits*(1 - tf.cast(mask, tf.float32))+ (-1e18) * tf.cast(mask, tf.float32) 

        ## attention weights 
        # attetion_weights = logits.sum(dim=1)/self.num_heads
        # Convert to probabilites
        weights = tf.nn.softmax(logits, axis=-1)
        # Dropout
        weights = self.dropout(weights)
        # Combine with values to get context
        contexts = tf.matmul(weights, values)
        # Merge heads
        contexts = self._merge_heads(contexts)
        #contexts = torch.tanh(contexts)
        # Linear to get output
        outputs = self.output_linear(contexts)
        return outputs

'''
The implementation of vanilla MultiHeadAttention, which refers to the implementation of vanilla Transformer in torch version, 
see reference "The Annotated Transformer" at http://nlp.seas.harvard.edu/2018/04/03/attention.html
'''
class MultiHeadAttention(layers.Layer):
    """
    Multi-head attention as per https://arxiv.org/pdf/1706.03762.pdf
    Refer Figure 2
    """
    def __init__(self, input_depth, total_key_depth, total_value_depth, output_depth, 
                 num_heads, bias_mask=None, dropout=0.0):
        """
        Parameters:
            input_depth: Size of last dimension of input
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(MultiHeadAttention, self).__init__()
        # Checks borrowed from 
        # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
        # if total_key_depth % num_heads != 0:
        #     raise ValueError("Key depth (%d) must be divisible by the number of "
        #                      "attention heads (%d)." % (total_key_depth, num_heads))
        # if total_value_depth % num_heads != 0:
        #     raise ValueError("Value depth (%d) must be divisible by the number of "
        #                      "attention heads (%d)." % (total_value_depth, num_heads))
            
        if total_key_depth % num_heads != 0:
            print("Key depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (total_key_depth, num_heads))
            total_key_depth = total_key_depth - (total_key_depth % num_heads)
        if total_value_depth % num_heads != 0:
            print("Value depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (total_value_depth, num_heads))
            total_value_depth = total_value_depth - (total_value_depth % num_heads)

        self.num_heads = num_heads
        self.query_scale = (total_key_depth//num_heads)**-0.5 ## sqrt
        self.bias_mask = bias_mask
        
        # Key and query depth will be same
        self.query_linear = layers.Dense(total_key_depth, use_bias=False)
        self.key_linear = layers.Dense(total_key_depth, use_bias=False)
        self.value_linear = layers.Dense(total_value_depth, use_bias=False)
        self.output_linear = layers.Dense(output_depth, use_bias=False)
        self.dropout = layers.Dropout(dropout)
    
    #devide the input into mutiple heads
    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return tf.transpose(tf.reshape(x, [shape[0], shape[1], self.num_heads, shape[2]//self.num_heads]), perm=[0, 2, 1, 3])
    
    # merge the results after attention calculation, such that the transformer model can attend to different part of an input sequence
    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return tf.reshape(tf.transpose(x, perm=[0, 2, 1, 3]), [shape[0], shape[2], shape[3]*self.num_heads])
    
    def __call__(self, queries, keys, values, mask):
        
        # Do a linear for each component
        queries = self.query_linear(queries)
        keys = self.key_linear(keys)
        values = self.value_linear(values)
        # Split into multiple heads
        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)
        # Scale queries
        queries *= self.query_scale
        # Combine queries and keys
        logits = tf.matmul(queries, keys, transpose_b=True)
        
        if mask is not None:
            #logits += (mask * -1e18)
            mask = tf.expand_dims(mask, 1)
            logits = logits*(1 - tf.cast(mask, tf.float32))+ (-1e18) * tf.cast(mask, tf.float32)

        ## attention weights 
        attetion_weights = tf.reduce_sum(logits, axis=1)/self.num_heads
        # Convert to probabilites
        weights = tf.nn.softmax(logits, axis=-1)
        # Dropout
        weights = self.dropout(weights)
        # Combine with values to get context
        contexts = tf.matmul(weights, values)
        # Merge heads
        contexts = self._merge_heads(contexts)
        #contexts = torch.tanh(contexts)
        # Linear to get output
        outputs = self.output_linear(contexts)       
        return outputs, attetion_weights

# 1d Cnovolution to improve the representation of input sequence
class Conv(layers.Layer):
    """
    Convenience class that does padding and convolution for inputs in the format
    [batch_size, sequence length, hidden size]
    """
    def __init__(self, output_size, kernel_size, pad_type):
        """
        Parameters:
            input_size: Input feature size
            output_size: Output feature size
            kernel_size: Kernel width
            pad_type: left -> pad on the left side (to mask future data), 
                      both -> pad on both sides
        """
        super(Conv, self).__init__()
        padding = [kernel_size - 1, 0] if pad_type == 'left' else [kernel_size//2, (kernel_size - 1)//2]
        self.padding = [[0, 0], [0, 0], padding]
        self.output_size = output_size
        self.kernel_size = kernel_size

    def __call__(self, inputs, training=True):
        inputs = tf.pad(tf.transpose(inputs, perm=[0, 2, 1]), self.padding, "CONSTANT") 
        outputs = layers.Conv1D(self.output_size, self.kernel_size, padding='valid')(tf.transpose(inputs, [0, 2, 1]), training=training)
        return outputs

'''
The implementation of PositionwiseFeedForward Layer, compared to the vanilla Transformer which contains only two Dense Layers
are added, additional 1-d convolution can be implemented as well to improve the sequence representation and cross layer
information sharing if the input layer_config contains the character 'c'
'''
class PositionwiseFeedForward(layers.Layer):
    """
    Does a Linear + RELU + Linear on each of the timesteps
    """
    def __init__(self, input_depth, filter_size, output_depth, layer_config='ll', padding='left', dropout=0.0):
        """
        Parameters:
            input_depth: Size of last dimension of input
            filter_size: Hidden size of the middle layer
            output_depth: Size last dimension of the final output
            layer_config: ll -> linear + ReLU + linear
                          cc -> conv + ReLU + conv etc.
            padding: left -> pad on the left side (to mask future data), 
                     both -> pad on both sides
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(PositionwiseFeedForward, self).__init__()
        
        self.layer_config = layer_config
        self.linear_layer1 = layers.Dense(filter_size)
        self.linear_layer2 = layers.Dense(output_depth)
        self.conv_layer1 = Conv(filter_size, kernel_size=3, pad_type=padding)
        self.conv_layer2 = Conv(output_depth, kernel_size=3, pad_type=padding)
        self.relu_layer = layers.ReLU()
        self.dropout_layer = layers.Dropout(dropout)
        
    def __call__(self, inputs):
        x = inputs
        for i, cur_layer in enumerate(self.layer_config):
            if i < len(self.layer_config)-1:
                if cur_layer == 'l':
                    x = self.linear_layer1(x)
                elif cur_layer == 'c':
                    x = self.conv_layer1(x)
                else:
                    raise ValueError("Unknown layer type {}".format(cur_layer))
            else:
                if cur_layer == 'l':
                    x = self.linear_layer2(x)
                elif cur_layer == 'c':
                    x = self.conv_layer2(x)
                else:
                    raise ValueError("Unknown layer type {}".format(cur_layer))
            x = self.relu_layer(x)
            x = self.dropout_layer(x)
        return x

def _gen_bias_mask(max_length):
    """
    Generates bias values (-Inf) to mask future timesteps during attention
    """
    np_mask = np.triu(np.full([max_length, max_length], -np.inf), 1)
    tf_mask = tf.cast(np_mask, dtype=tf.float32)
    return tf.expand_dims(tf.expand_dims(tf_mask, 0), 1)

def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """
    Generates a [1, length, channels] timing signal consisting of sinusoids
    If Universal Transformer is used, time signal would be calculated in addition to position encoding
    Adapted from:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = ( math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * np.exp(np.arange(num_timescales).astype(np.float) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]], 'constant', constant_values=[0.0, 0.0])
    signal = tf.reshape(signal, [1,length, channels])
    #signal =  signal.reshape([1, length, channels])
    return tf.cast(signal, dtype=tf.float32)

def _get_attn_subsequent_mask(size):
    """
    Get an attention mask to avoid using the subsequent info.
    Args:
        size: int
    Returns:
        (`LongTensor`):
        * subsequent_mask `[1 x size x size]`
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = tf.cast(subsequent_mask, dtype=tf.uint8)
    return subsequent_mask

class OutputLayer(layers.Layer):
    """
    Abstract base class for output layer. 
    Handles projection to output labels
    """
    def __init__(self, hidden_size, output_size):
        super(OutputLayer, self).__init__()
        self.output_size = output_size
        self.output_projection = layers.Dense(output_size)

    def loss(self, hidden, labels):
        raise NotImplementedError('Must implement {}.loss'.format(self.__class__.__name__))

class SoftmaxOutputLayer(OutputLayer):
    """
    Implements a softmax based output layer to transform the logits into probability distribution
    """
    def forward(self, hidden):
        logits = self.output_projection(hidden)
        probs = tf.nn.softmax(logits, axis=-1)
        predictions = tf.math.argmax(probs, axis=-1)
        return predictions

    def loss(self, hidden, labels):
        logits = self.output_projection(hidden)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        return -tf.math.reduce_sum(tf.one_hot(labels, depth=self.output_size)*tf.nn.log_softmax(log_probs, -1))/len(labels)

'''
Since the self-attention implementation will destory the position information of input sequence, external position information
should be added to indicates the position of each token, as position relationship is always crucial in Natural Language
Processing tasks
'''
def position_encoding(sentence_size, embedding_dim):
    encoding = np.ones((embedding_dim, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_dim + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (embedding_dim+1)/2) * (j - (sentence_size+1)/2)
    encoding = 1 + 4 * encoding / embedding_dim / sentence_size
    # Make position encoding of time words identity to avoid modifying them
    # encoding[:, -1] = 1.0
    return np.transpose(encoding)

def gen_embeddings(vocab):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """
    embeddings = np.random.randn(vocab.n_words, config.emb_dim) * 0.01 
    print('Embeddings: %d x %d' % (vocab.n_words, config.emb_dim))
    if config.emb_file is not None:
        print('Loading embedding file: %s' % config.emb_file)
        pre_trained = 0
        for line in open(config.emb_file).readlines():
            sp = line.split()
            if(len(sp) == config.emb_dim + 1):
                if sp[0] in vocab.word2index:
                    pre_trained += 1
                    embeddings[vocab.word2index[sp[0]]] = [float(x) for x in sp[1:]]
            else:
                print(sp[0])
        print('Pre-trained: %d (%.2f%%)' % (pre_trained, pre_trained * 100.0 / vocab.n_words))
    return embeddings

# class TF_Embedding(layers.Layer):
#     def __init__(self, vocab, d_model, padding_idx=0, pretrain=True, **kwargs):
#         super(TF_Embedding, self).__init__(**kwargs)
#         self.input_dim = vocab.n_words
#         self.output_dim = d_model
#         self.padding_idx = padding_idx
#         if pretrain:
#             pre_embedding = gen_embeddings(vocab)
#             self.embeddings =  layers.Embedding(self.input_dim, self.output_dim, weights=[pre_embedding])
#         else:
#             self.embeddings = self.add_weight(
#                 shape=(self.input_dim, self.output_dim),
#                 initializer='random_normal',
#                 dtype='float32')

#     def call(self, inputs): 
#         def compute_mask():
#             return tf.not_equal(inputs, self.padding_idx)
        
#         out = tf.nn.embedding_lookup(self.embeddings, inputs)
#         masking = compute_mask() # [B, T], bool
#         masking = tf.cast(tf.tile(masking[:,:, tf.newaxis], [1,1,self.output_dim]), 
#                           dtype=tf.float32) # [B, T, D]
#         return tf.multiply(out, masking)

# class Embeddings(layers.Layer):
#     def __init__(self, vocab, d_model, padding_idx=None, pretrain=True):
#         super(Embeddings, self).__init__()
#         self.lut = TF_Embedding(vocab, d_model, padding_idx=padding_idx, pretrain=pretrain)
#         self.d_model = d_model

#     def forward(self, x):
#         return self.lut(x) * math.sqrt(self.d_model)

# def share_embedding(vocab, pretrain=True):
#     embedding = Embeddings(vocab, config.emb_dim, padding_idx=config.PAD_idx, pretrain=pretrain)
#     #embedding.lut.weight.data.copy_(torch.FloatTensor(pre_embedding))
#     #embedding.lut.weight.data.requires_grad = True
#     return embedding

#Initialize the embedding of input tokens
class Embeddinglayer(layers.Layer):
    def __init__(self, vocab_size, d_model):
        # model hyper parameter variables
        super(Embeddinglayer, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = layers.Embedding(vocab_size, d_model)

    def __call__(self, sequences):
        output = self.embedding(sequences) * tf.sqrt(tf.cast(self.d_model, dtype=tf.float32))
        return output

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence such that they will not influence the gradient calculation. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.shape[1]
    padding_mask = tf.math.equal(seq_k, config.PAD_idx)
    padding_mask = tf.broadcast_to(tf.expand_dims(padding_mask, 1), [-1, len_q, -1]) # b x lq x lk
    return padding_mask

'''
The input batch was stored in a tuple, this function was used to extract the input_batch index sequence, input_batch padding
mask, input batch length, enc_batch_extend_vocab from the tuple
'''
def get_input_from_batch(batch):
    enc_batch = batch[0] #enc_batch = batch["input_batch"]
    enc_lens = batch[1] # enc_lens = batch["input_lengths"]
    batch_size, max_enc_len = enc_batch.shape
    assert len(enc_lens) == batch_size

    enc_padding_mask = tf.cast(sequence_mask(enc_lens, max_len=max_enc_len), dtype=tf.float32)

    extra_zeros = None
    enc_batch_extend_vocab = None

    c_t_1 = tf.zeros((batch_size, 2 * config.hidden_dim))

    coverage = None
    if config.is_coverage:
        coverage = tf.zeros(enc_batch.shape)

    return enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage

'''
The input batch was stored in a tuple, this function was used to extract the target_batch index sequence, decoder batch padding
mask, max decoding length from the tuple
'''
def get_output_from_batch(batch):
    #dec_batch = batch["target_batch"]
    dec_batch = batch[3]
    target_batch = dec_batch       
    #dec_lens_var = batch["target_lengths"]
    dec_lens_var = batch[4]
    #print('*******DEC BATCH*******', dec_lens_var) ----- Tensor("input_x_4:0", shape=(32, 1), dtype=int64)
    #max_dec_len = max(dec_lens_var)
    #max_dec_len = tf.math.reduce_max(dec_lens_var)
    max_dec_len = target_batch.shape[1]
    dec_padding_mask = tf.cast(sequence_mask(dec_lens_var, max_len=max_dec_len), dtype=tf.float32)
    
    return dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch

'''
Decoder can't see the future information. That is, for a sequence, at time step t, the decoding output should only depend on the
output before t, not after t. Therefore, we need to find a way to hide the information after t.
'''
def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = tf.math.reduce_max(sequence_length)
    batch_size = sequence_length.shape[0]
    seq_range = tf.range(0, max_len, dtype=tf.int32)
    seq_range_expand = tf.broadcast_to(tf.expand_dims(seq_range, 0), [batch_size, max_len])
    seq_range_expand = seq_range_expand
    #seq_length_expand = tf.broadcast_to(tf.expand_dims(sequence_length, 1), seq_range_expand.shape)
    seq_length_expand = tf.broadcast_to(sequence_length, seq_range_expand.shape)
    return seq_range_expand < seq_length_expand

# Update the config file
def write_config():
    if(not config.test):
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path)
        with open(config.save_path+'config.txt', 'w') as the_file:
            for k, v in config.arg.__dict__.items():
                if("False" in str(v)):
                    pass
                elif("True" in str(v)):
                    the_file.write("--{} ".format(k))
                else:
                    the_file.write("--{} {} ".format(k,v))

def print_custum(emotion,dial,ref,hyp_g,hyp_b):
    print("emotion:{}".format(emotion))
    print("Context:{}".format(dial))
    #print("Topk:{}".format(hyp_t))
    print("Beam: {}".format(hyp_b))
    print("Greedy:{}".format(hyp_g))
    print("Ref:{}".format(ref))
    print("----------------------------------------------------------------------")
    print("----------------------------------------------------------------------")

def count_parameters(model):
    return sum(np.prod(p.get_shape()) for p in model.trainable_weights)

def make_infinite(dataloader):
    while True:
        for x in dataloader:
            yield x

def top_k_top_p_filtering(logits, top_k=0, top_p=0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (..., vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < tf.math.top_k(logits, top_k)[0][:, -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits = tf.sort(logits, direction='DESCENDING')
        sorted_indices = tf.argsort(logits, direction='DESCENDING')
        cumulative_probs = tf.math.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[:, 1:] = tf.identity(sorted_indices_to_remove[:, :-1])
        sorted_indices_to_remove[:, 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits
