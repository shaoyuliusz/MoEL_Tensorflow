### Architecture based on
#https://github.com/kolloldas/torchnlp and https://github.com/HLTCHKUST/MoEL
#https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

import tensorflow as tf

import numpy as np
import math
from model.common_layer import EncoderLayer, DecoderLayer, MultiHeadAttention, Conv, _gen_bias_mask ,_gen_timing_signal, Embeddinglayer, _get_attn_subsequent_mask,  get_input_from_batch, get_output_from_batch, top_k_top_p_filtering
from utils import config
import random
# from numpy import random
import os
import pprint
from tqdm import tqdm
pp = pprint.PrettyPrinter(indent=1)
import os
import time
from copy import deepcopy
from sklearn.metrics import accuracy_score

tf.random.set_seed(0)
np.random.seed(0)

class Encoder(tf.keras.layers.Layer):
    """
    A Transformer Encoder module. 
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    Refer https://github.com/HLTCHKUST/MoEL
    """
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.0, layer_dropout=0.0, 
                 attention_dropout=0.0, relu_dropout=0.0, use_mask=False, universal=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN(Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """
        
        super(Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        
        if(self.universal):  
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params =(hidden_size, 
                 total_key_depth or hidden_size,
                 total_value_depth or hidden_size,
                 filter_size, 
                 num_heads, 
                 _gen_bias_mask(max_length) if use_mask else None,
                 layer_dropout, 
                 attention_dropout, 
                 relu_dropout)
        
        #the embedding projection layer
        self.embedding_proj = tf.keras.layers.Dense(units = hidden_size, activation=None, use_bias = False) 
        if(self.universal):
            self.enc = EncoderLayer(*params)
        else:
            self.enc = [EncoderLayer(*params) for _ in range(num_layers)]
        
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.input_dropout = tf.keras.layers.Dropout(input_dropout)
        
        if(config.act):
            self.act_fn = ACT_basic(hidden_size)
            self.remainders = None
            self.n_updates = None
    
    def __call__(self, inputs, mask, training=True):
        #Add input dropout
        x = self.input_dropout(inputs)
        
        # Project to hidden size
        x = self.embedding_proj(x)
        
        if(self.universal):
            if(config.act):
                x, (self.remainders, self.n_updates) = self.act_fn(x, inputs, self.enc, self.timing_signal, self.position_signal, self.num_layers)
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += tf.cast(self.timing_signal[:, :inputs.shape[1], :], inputs.dtype)
                    x += tf.cast(tf.tile(tf.expand_dims(self.position_signal[:, l, :], 1), [1,inputs.shape[1],1]), inputs.dtype)
                    x += self.position_signal[:, l, :]
                    x = self.enc(x, mask=mask, training=training)
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += tf.cast(self.timing_signal[:, :inputs.shape[1], :], inputs.dtype)
            for i in range(self.num_layers):
                x = self.enc[i](x, mask, training=training)
        
            y = self.layer_norm(x)
        return y

class Decoder(tf.keras.layers.Layer):
    """
    A Transformer Decoder module. 
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.0, layer_dropout=0.0, 
                 attention_dropout=0.0, relu_dropout=0.0, universal=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """
        
        super(Decoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        
        if(self.universal):  
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.mask = _get_attn_subsequent_mask(max_length)

        params =(hidden_size, 
                 total_key_depth or hidden_size,
                 total_value_depth or hidden_size,
                 filter_size, 
                 num_heads, 
                 _gen_bias_mask(max_length), # mandatory
                 layer_dropout, 
                 attention_dropout, 
                 relu_dropout)
        
        if(self.universal):
            self.dec = DecoderLayer(*params)
        else:
            #stack a number of decoder layers
            self.dec = [DecoderLayer(*params) for l in range(num_layers)]
            
        self.embedding_proj = tf.keras.layers.Dense(units = hidden_size, activation=None, use_bias = False) 
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.input_dropout = tf.keras.layers.Dropout(input_dropout)

    def __call__(self, inputs, encoder_output, mask, training=True):#ADHOC = TRUE
        mask_src, mask_trg = mask
        dec_mask = tf.math.greater(tf.cast(mask_trg, tf.uint8) + self.mask[:, :mask_trg.shape[-1], :mask_trg.shape[-1]], 0)

        #Add input dropout
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)
            
        if(self.universal):
            if(config.act):
                x, attn_dist, (self.remainders,self.n_updates) = self.act_fn(x, inputs, self.dec, self.timing_signal, self.position_signal, self.num_layers, encoder_output, decoding=True)
                y = self.layer_norm(x)

            else:
                x += tf.cast(self.timing_signal[:, :inputs.shape[1], :], inputs.dtype)
                for l in range(self.num_layers):
                    x += tf.cast(tf.tile(tf.expand_dims(self.position_signal[:, l, :], 1), [1,inputs.shape[1],1]), inputs.dtype)
                    x, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src,dec_mask)))
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += tf.cast(self.timing_signal[:, :inputs.shape[1], :], inputs.dtype)
            
            # Run decoder
            for i in range(self.num_layers):
                x, _, attn_dist, _ = self.dec[i]((x, encoder_output, [], (mask_src,dec_mask)),training=training)

            # Final layer normalization
            y = self.layer_norm(x)
        return y, attn_dist


class Generator(tf.keras.layers.Layer):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = tf.keras.layers.Dense(vocab)
        self.p_gen_linear = tf.keras.layers.Dense(1)

    def __call__(self, x, attn_dist=None, enc_batch_extend_vocab=None, extra_zeros=None, temp=1, beam_search=False, attn_dist_db=None):

        if config.pointer_gen:
            p_gen = self.p_gen_linear(x)
            alpha = tf.math.sigmoid(p_gen)

        logit = self.proj(x)

        if(config.pointer_gen):
            vocab_dist = tf.nn.softmax(vocab_dist, axis = 2)
            vocab_dist_ = alpha * vocab_dist

            attn_dist = tf.nn.softmax(attn_dist/temp, axis = -1)
            attn_dist_ = (1 - alpha) * attn_dist     
            
            
            enc_batch_extend_vocab_ = tf.concat([tf.expand_dims(enc_batch_extend_vocab, axis = 1)]*x.shape[1],axis=1) ## extend for all seq
            if(beam_search):
                enc_batch_extend_vocab_ = tf.concat([tf.expand_dims(enc_batch_extend_vocab_[0], axis = 0)]*x.shape[0],axis=0)
            
            logit = tf.math.log(vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_))
            return logit
        else:
            return tf.nn.log_softmax(logit, axis = -1)

class Transformer(tf.keras.Model):

    def __init__(self, vocab, decoder_number,  model_file_path=None, is_eval=False, load_optim=False):
        super(Transformer, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words
        self.embedding = Embeddinglayer(self.vocab.n_words, config.emb_dim)
        self.encoder = Encoder(config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads, 
                                total_key_depth=config.depth, total_value_depth=config.depth,
                                filter_size=config.filter,universal=config.universal)
        
        ## multiple decoders
        self.decoder = Decoder(config.emb_dim, hidden_size = config.hidden_dim,  num_layers=config.hop, num_heads=config.heads, 
                                    total_key_depth=config.depth,total_value_depth=config.depth,
                                    filter_size=config.filter)
        
        self.decoder_key = tf.keras.layers.Dense(decoder_number, use_bias=False)
        self.generator = Generator(config.hidden_dim, self.vocab_size)

        if config.weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.generator.proj.weight = self.embedding.lut.weight

        self.criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')
        if (config.label_smoothing):
            ###### ignore label smoothing
            #self.criterion = LabelSmoothing(size=self.vocab_size, padding_idx=config.PAD_idx, smoothing=0.1)
            #self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)
            self.criterion_ppl = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = config.lr)

    def __call__(self, batch, training=True):
        enc_batch, _, _, enc_batch_extend_vocab, extra_zeros, _, _ = get_input_from_batch(batch)
        dec_batch, _, _, _, _ = get_output_from_batch(batch)
        
#         if(config.noam):
#             self.optimizer.optimizer.zero_grad()
#         else:
#             self.optimizer.zero_grad()
        
        ## Encode
        
        mask_src = tf.expand_dims(tf.math.equal(enc_batch, config.PAD_idx), axis = 1)

        emb_mask = self.embedding(batch[2]) #"mask_input"
        encoder_outputs = self.encoder(self.embedding(enc_batch)+emb_mask, mask_src, training=training)

        # Decode 
        sos_token = tf.expand_dims(tf.cast([config.SOS_idx] * enc_batch.shape[0], dtype = tf.int32), axis = 1)
        dec_batch_shift = tf.concat([sos_token,dec_batch[:, :-1]],1)

        mask_trg = tf.expand_dims(tf.math.equal(dec_batch_shift, config.PAD_idx), axis = 1)
        pre_logit, attn_dist = self.decoder(self.embedding(dec_batch_shift),encoder_outputs, (mask_src,mask_trg))
        
        ## compute output distribution using generator
        logit = self.generator(pre_logit,attn_dist,enc_batch_extend_vocab if config.pointer_gen else None, extra_zeros, attn_dist_db=None)
        #logit = F.log_softmax(logit,dim=-1) #fix the name later
        ## loss: NNL if ptr else Cross entropy
        #loss = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1))
        
        #loss = self.criterion(tf.reshape(logit, [-1, logit.shape[-1]]), tf.reshape(dec_batch, -1))
        
        #multi-task
        q_h = encoder_outputs[:,0] #moved out of config.multitask
        logit_prob = self.decoder_key(q_h) #moved out of config.multitask
        if config.multitask:                
            pred_program = np.argmax(logit_prob.detach().numpy(), axis=1)
            program_acc = accuracy_score(batch["program_label"], pred_program)
        
        return logit, logit_prob

    def compute_act_loss(self,module):    
        R_t = module.remainders
        N_t = module.n_updates
        p_t = R_t + N_t
        avg_p_t = tf.math.reduce_sum(tf.math.reduce_sum(p_t, axis = 1)/p_t.shape[1])/p_t.shape[1]
        loss = config.act_loss_weight * avg_p_t.item()
        return loss

    def decoder_greedy(self, batch, max_dec_step=30):
        enc_batch, _, _, enc_batch_extend_vocab, extra_zeros, _, _ = get_input_from_batch(batch)
        mask_src = tf.expand_dims(tf.math.equal(enc_batch, config.PAD_idx), 1)
        emb_mask = self.embedding(batch[2]) #"mask_input"
        encoder_outputs = self.encoder(self.embedding(enc_batch)+emb_mask,mask_src, training=training)

        ys = tf.cast(tf.fill([1,1], config.SOS_idx), dtype = tf.int32)

        mask_trg = tf.expand_dims(tf.math.equal(ys.data, config.PAD_idx), axis = 1)
        decoded_words = []
        for i in range(max_dec_step+1):
            if(config.project):
                out, attn_dist = self.decoder(self.embedding_proj_in(self.embedding(ys)),self.embedding_proj_in(encoder_outputs), (mask_src,mask_trg))
            else:
                out, attn_dist = self.decoder(self.embedding(ys),encoder_outputs, (mask_src,mask_trg))
            
            prob = self.generator(out,attn_dist,enc_batch_extend_vocab, extra_zeros, attn_dist_db=None)
            next_word = tf.math.argmax(prob[:,-1], axis = 1)
            decoded_words.append(['<EOS>' if ni.item() == config.EOS_idx else self.vocab.index2word[ni.item()] for ni in next_word.view(-1)])
            next_word = next_word.data[0]

            if config.USE_CUDA:
                ys = tf.concat([ys, tf.ones([1,1], dtype=tf.dtypes.int64).fill(next_word).cuda()], 1)
                ys = ys
            else:
                ys = tf.concat([ys, tf.ones([1,1], dtype=tf.dtypes.int64).fill(next_word)], 1)
            mask_trg = tf.expand_dims(tf.math.equal(ys.data, config.PAD_idx), axis = 1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == '<EOS>': break
                else: st+= e + ' '
            sent.append(st)
        return sent
    
    def decoder_topk(self, batch, max_dec_step=30, training = False):
        enc_batch, _, _, enc_batch_extend_vocab, extra_zeros, _, _ = get_input_from_batch(batch)
        mask_src = tf.expand_dims(tf.math.equal(enc_batch.data,1),1)
        
        emb_mask = self.embedding(batch[2]) #"mask_input"
        encoder_outputs = self.encoder(self.embedding(enc_batch)+emb_mask,mask_src)

        ys = tf.cast(tf.fill([1,1], config.SOS_idx), dtype = tf.int32)

        mask_trg = tf.expand_dims(tf.math.equal(ys.data, config.PAD_idx), axis = 1)
        decoded_words = []
        for i in range(max_dec_step+1):
            if(config.project):
                out, attn_dist = self.decoder(self.embedding_proj_in(self.embedding(ys)),self.embedding_proj_in(encoder_outputs), (mask_src,mask_trg))
            else:
                out, attn_dist = self.decoder(self.embedding(ys),encoder_outputs, (mask_src,mask_trg))
            
            logit = self.generator(out,attn_dist,enc_batch_extend_vocab, extra_zeros, attn_dist_db=None)
            filtered_logit = top_k_top_p_filtering(logit[:, -1], top_k=3, top_p=0, filter_value=-float('Inf'))
            # Sample from the filtered distribution
            next_word = tf.squeeze(tf.random.categorical(tf.nn.softmax(filtered_logit, axis=-1), 1))
            decoded_words.append(['<EOS>' if ni.item() == config.EOS_idx else self.vocab.index2word[ni.item()] for ni in next_word.view(-1)])
            next_word = next_word.data[0]

#             if config.USE_CUDA:
#                 #ys = torch.cat([ys, torch.ones(1, 1).long().fill_(next_word).cuda()], dim=1)
#                 ys = tf.concat([ys, tf.cast(tf.fill([1,1], next_word), dtype = tf.int64).cuda()], 1)
#                 ys = ys.cuda()

            ys = tf.concat([ys, tf.cast(tf.fill([1,1], next_word), dtype = tf.int64)], 1)
            mask_trg = tf.expand_dims(tf.math.equal(ys.data, config.PAD_idx), axis = 1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == '<EOS>': break
                else: st+= e + ' '
            sent.append(st)
        return sent

### CONVERTED FROM https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/universal_transformer_util.py#L1062
class ACT_basic(tf.keras.layers.Layer):
    def __init__(self,hidden_size):
        super(ACT_basic, self).__init__()
        self.sigma = tf.keras.layers.Activation(activation='sigmoid')
        self.p = tf.keras.layers.Dense(1, bias_initializer='ones')
        self.threshold = 1 - 0.1

    def call(self, state, inputs, fn, time_enc, pos_enc, max_hop, encoder_output=None, decoding=False):
        """
        Without any annotations, TensorFlow automatically decides whether to use the GPU or 
        CPU for an operation???copying the tensor between CPU and GPU memory, if necessary. 
        """
        # init_hdd
        ## [B, S]
        halting_probability = tf.zeros(inputs.shape[0],inputs.shape[1])
        ## [B, S]
        
        remainders = tf.zeros(shape = [inputs.shape[0],inputs.shape[1]])
        ## [B, S]
        n_updates = tf.zeros(shape = [inputs.shape[0],inputs.shape[1]])
        ## [B, S, HDD]
        previous_state = tf.zeros_like(inputs)
        

        step = 0
        # for l in range(self.num_layers):
        while tf.reduce_any(tf.cast((halting_probability<self.threshold) & (n_updates < max_hop), tf.bool)):
            # Add timing signal
            state = state + tf.cast(time_enc[:, :inputs.shape[1], :], inputs.data.dtype) 
            #**follows from https://blog.csdn.net/g11d111/article/details/103756562**
            state = tf.cast(tf.tile(tf.expand_dims(state + pos_enc[:, step, :], 1), [1,inputs.shape[1],1]), inputs.data.dtype)

            p = tf.expand_dims(self.sigma(self.p(state)), axis = -1)
            
            # Mask for inputs which have not halted yet
            still_running = tf.cast((halting_probability < 1.0), tf.float32)

            # Mask of inputs which halted at this step
            new_halted = tf.cast((halting_probability + p * still_running > self.threshold), tf.float32) * still_running

            # Mask of inputs which haven't halted, and didn't halt this step
            still_running = tf.cast((halting_probability + p * still_running <= self.threshold), tf.float32) * still_running

            # Add the halting probability for this step to the halting
            # probabilities for those input which haven't halted yet
            halting_probability = halting_probability + p * still_running

            # Compute remainders for the inputs which halted at this step
            remainders = remainders + new_halted * (1 - halting_probability)

            # Add the remainders to those inputs which halted at this step
            halting_probability = halting_probability + new_halted * remainders

            # Increment n_updates for all inputs which are still running
            n_updates = n_updates + still_running + new_halted

            # Compute the weight to be applied to the new state and output
            # 0 when the input has already halted
            # p when the input hasn't halted yet
            # the remainders when it halted this step
            update_weights = p * still_running + new_halted * remainders

            if(decoding):
                state, _, attention_weight = fn((state,encoder_output,[]))
            else:
                # apply transformation on the state
                state = fn(state)

            # update running part in the weighted state and keep the rest
            previous_state = ((state * tf.expand_dims(update_weights, axis = -1) + (previous_state * (1 - tf.expand_dims(update_weights, axis = -1)))))
                              
            if(decoding):
                if(step==0):  previous_att_weight = tf.zeros_like(attention_weight)

                previous_att_weight = ((attention_weight * tf.expand_dims(update_weights, axis = -1)) + (previous_att_weight * (1- tf.expand_dims(update_weights, axis = -1))))
            ## previous_state is actually the new_state at end of hte loop 
            ## to save a line I assigned to previous_state so in the next 
            ## iteration is correct. Notice that indeed we return previous_state
            step+=1

        if(decoding):
            return previous_state, previous_att_weight, (remainders,n_updates)
        else:
            return previous_state, (remainders,n_updates)
