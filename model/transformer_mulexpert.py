import os
import math
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from model.common_layer import EncoderLayer, DecoderLayer, MultiHeadAttention, Conv, PositionwiseFeed, _gen_bias_mask ,_gen_timing_signal, share_embedding, NoamOpt, _get_attn_subsequent_mask,  get_input_from_batch, get_output_from_batch, top_k_top_p_filtering
from utils import config
from utils.metric import rouge, moses_multi_bleu, _prec_recall_f1_score, compute_prf, compute_exact_match
from copy import deepcopy
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers

class Encoder(layers.Layer):
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
    filter_size, max_length=1000, input_dropout=0.0, layer_dropout=0.0, attention_dropout=0.0, 
    relu_dropout=0.0, use_mask=False, universal=False):

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
        
        self.embedding_proj = layers.Dense(hidden_size, use_bias = False)
        if(self.universal):
            self.enc = EncoderLayer(*params)
        else:
            self.enc = [EncoderLayer(*params) for _ in range(num_layers)]
        
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.input_dropout = layers.Dropout(input_dropout)
        
        if(config.act):
            self.act_fn = ACT_basic(hidden_size)
            self.remainders = None
            self.n_updates = None
    
    def __call__(self, inputs, mask, training=True): #ADHOC = TRUE
        #Add input dropout
        
        inputs = tf.cast(inputs, dtype = tf.float32) #inputs dim (32, 38)
        x = self.input_dropout(inputs)
        
        # Project to hidden size
        x = self.embedding_proj(x) #x dim [32, 100]
        
        if(self.universal):
            if(config.act):
                x, (self.remainders, self.n_updates) = self.act_fn(x, inputs, self.enc, self.timing_signal, self.position_signal, self.num_layers)
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += tf.cast(self.timing_signal[:, :inputs.shape[1], :], inputs.dtype)
                    x += tf.cast(tf.tile(tf.expand_dims(self.position_signal[:, l, :], 1), [1,inputs.shape[1],1]), inputs.dtype)
                    x = self.enc(x, mask=mask, training=training)
                y = self.layer_norm(x)
        else:
            # Add timing signal
#             MASK  Tensor("transformer_experts/Less_2:0", shape=(32, 38), dtype=bool)
#             x (32, 100)
#             inputs (32, 38)
#             TIMING SIGNAL (1, 1000, 100)
#             TIMING SIGNAL2 (1, 38, 100)
            print('x', x.shape)
            print('inputs', inputs.shape)
            print('TIMING SIGNAL', self.timing_signal.shape)
            print('TIMING SIGNAL2', self.timing_signal[:, :inputs.shape[1], :].shape)
            x += tf.cast(self.timing_signal[:, :inputs.shape[1], :], inputs.dtype)   
            for i in range(self.num_layers):
                x = self.enc[i](x, mask, training=training)
            y = self.layer_norm(x)
        return y

class Decoder(layers.Layer):
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.0, layer_dropout=0.0, 
                 attention_dropout=0.0, relu_dropout=0.0, universal=False):
        super(Decoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        
        if(self.universal):  
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
            self.dec = [DecoderLayer(*params) for l in range(num_layers)]
        
        self.embedding_proj = layers.Dense(hidden_size, use_bias = False)
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.input_dropout = layers.Dropout(input_dropout)


    def __call__(self, inputs, encoder_output, mask, training=True):#ADHOC = TRUE
        mask_src, mask_trg = mask
        dec_mask = tf.math.greater(mask_trg + self.mask[:, :mask_trg.shape[-1], :mask_trg.shape[-1]], 0)
        #Add input dropout
        x = self.input_dropout(inputs)
        if (not config.project):
            x = self.embedding_proj(x)
            
        if(self.universal):
            if(config.act):
                x, attn_dist, (self.remainders,self.n_updates) = self.act_fn(x, inputs, self.dec, self.timing_signal, self.position_signal, self.num_layers, encoder_output, decoding=True)
                y = self.layer_norm(x)
            else:
                x += tf.cast(self.timing_signal[:, :inputs.shape[1], :], inputs.dtype)
                for l in range(self.num_layers):
                    x += tf.cast(tf.tile(tf.expand_dims(self.position_signal[:, l, :], 1), [1,inputs.shape[1],1]), inputs.dtype)
                    x, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src,dec_mask)),training=training)
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += tf.cast(self.timing_signal[:, :inputs.shape[1], :], inputs.dtype) 
            # Run decoder
            for i in range(self.num_layers):
                x, _, attn_dist, _ = self.dec[i]((x, encoder_output, [], (mask_src,dec_mask)),training=training)
            y = self.layer_norm(x)
        return y, attn_dist

class MulDecoder(layers.Layer):
    def __init__(self, expert_num,  embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.0, layer_dropout=0.0, 
                 attention_dropout=0.0, relu_dropout=0.0):
        super(MulDecoder, self).__init__()
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
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
        if config.basic_learner:
            self.basic = DecoderLayer(*params)
        self.experts = [DecoderLayer(*params) for e in range(expert_num)]
        self.dec = [DecoderLayer(*params) for l in range(num_layers)]
        
        self.embedding_proj = layers.Dense(hidden_size, use_bias = False)
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.input_dropout = layers.Dropout(input_dropout)
    
    def __call__(self, inputs, encoder_output, mask, attention_epxert, training=True):#ADHOC = TRUE
        mask_src, mask_trg = mask
        dec_mask = tf.math.greater(tf.cast(mask_trg, tf.uint8) + self.mask[:, :mask_trg.shape[-1], :mask_trg.shape[-1]], 0)
        #Add input dropout
        x = self.input_dropout(inputs)
        if (not config.project):
            x = self.embedding_proj(x)
        # Add timing signal
        x += tf.cast(self.timing_signal[:, :inputs.shape[1], :], inputs.dtype)
        expert_outputs = []
        if config.basic_learner:
            basic_out , _, attn_dist, _ = self.basic((x, encoder_output, [], (mask_src,dec_mask)), training=training)

        #compute experts
        #TODO forward all experts in parrallel
        if (attention_epxert.shape[0]==1 and config.topk>0):
            for i, expert in enumerate(self.experts):
                if attention_epxert[0, i]>0.0001: #speed up inference
                    expert_out , _, attn_dist, _ = expert((x, encoder_output, [], (mask_src,dec_mask)), training=training)
                    expert_outputs.append(attention_epxert[0, i]*expert_out)
            x = tf.stack(expert_outputs, axis=1)
            x = x.sum(dim=1)
                    
        else:
            for i, expert in enumerate(self.experts):
                expert_out , _, attn_dist, _ = expert((x, encoder_output, [], (mask_src,dec_mask)), training=training)
                expert_outputs.append(expert_out)
            x = tf.stack(expert_outputs, axis=1) #(batch_size, expert_number, len, hidden_size)
            x = attention_epxert * x
            x = tf.reduce_sum(x, axis=1)#(batch_size, len, hidden_size)
        if config.basic_learner:
            x+=basic_out
        # Run decoder
        for i in range(self.num_layers):
            x, _, attn_dist, _ = self.dec[i]((x, encoder_output, [], (mask_src,dec_mask)), training=training)
        # Final layer normalization
        y = self.layer_norm(x)
        return y, attn_dist

class Generator(layers.Layer):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = layers.Dense(vocab)
        self.p_gen_linear = layers.Dense(1)

    def __call__(self, x, attn_dist=None, enc_batch_extend_vocab=None, extra_zeros=None, temp=1, beam_search=False, attn_dist_db=None):
        if config.pointer_gen:
            p_gen = self.p_gen_linear(x)
            alpha = tf.math.sigmoid(p_gen)
        logit = self.proj(x)

        if(config.pointer_gen):
            vocab_dist = tf.nn.softmax(logit/temp, axis=2)
            vocab_dist_ = alpha * vocab_dist

            attn_dist = tf.nn.softmax(attn_dist/temp, axis=-1)
            attn_dist_ = (1 - alpha) * attn_dist     
            tf.expand_dims(enc_batch_extend_vocab, axis = 1)     
            enc_batch_extend_vocab_ = tf.concat([tf.expand_dims(enc_batch_extend_vocab, axis=1)]*x.shape[1], axis=1) ## extend for all seq
            if(beam_search):
                enc_batch_extend_vocab_ = tf.concat([tf.expand_dims(enc_batch_extend_vocab_[0], axis=0)]*x.shape[0], axis=0) ## extend for all seq
            logit = tf.math.log(vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_))#scatter_add后面记得要手动实现一下，根据输入维度
            return logit
        else:
            return tf.nn.log_softmax(logit, axis = -1)

class Transformer_experts(layers.Layer):
    def __init__(self, vocab, decoder_number, model_file_path=None, is_eval=False, load_optim=False):
        super(Transformer_experts, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words

        self.embedding = Embeddinglayer(self.vocab.n_words, config.emb_dim)
        self.encoder = Encoder(config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads, 
                                total_key_depth=config.depth, total_value_depth=config.depth,
                                filter_size=config.filter,universal=config.universal)
        self.decoder_number = decoder_number
        ## multiple decoders
        self.decoder = MulDecoder(decoder_number, config.emb_dim, config.hidden_dim,  num_layers=config.hop, num_heads=config.heads, 
                                    total_key_depth=config.depth,total_value_depth=config.depth,
                                    filter_size=config.filter)
 
        self.decoder_key = layers.Dense(decoder_number, use_bias=False)

        self.generator = Generator(config.hidden_dim, self.vocab_size)
        self.emoji_embedding = layers.Dense(config.emb_dim, use_bias=False)

        if config.weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.generator.proj.weight = self.embedding.lut.weight

        self.criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')
        if (config.label_smoothing):
            self.criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none', label_smoothing=0.1)
        
        if config.softmax:
            self.attention_activation =  layers.Softmax(axis = 1)
        else:
            self.attention_activation =  layers.Activation('sigmoid') #nn.Softmax()

        self.optimizer = tf.keras.optimizers.Adam(lr=config.lr)
        if(config.noam):
            self.optimizer = NoamOpt(config.hidden_dim, 1, 8000, tf.keras.optimizers.Adam(lr=0, beta_1=0.9, beta_2=0.98, epsilon=1e-9))

        #if model_file_path is not None:
            #print("loading weights")
            #model = tf.keras.models.load_model(model_file_path)

        #self.model_dir = config.save_path
        #if not os.path.exists(self.model_dir):
            #os.makedirs(self.model_dir)
        #self.best_path = ""
    
    def __call__(self, batch, training=True): #ADHC training=True
        enc_batch, _, _, enc_batch_extend_vocab, extra_zeros, _, _ = get_input_from_batch(batch)
        dec_batch, _, _, _, _ = get_output_from_batch(batch)
        ## Encode
        mask_src = tf.expand_dims(tf.math.equal(enc_batch, config.PAD_idx), axis = 1)
        if config.dataset=="empathetic":
            emb_mask = self.embedding(batch[2]) #"mask_input"
            encoder_outputs = self.encoder(self.embedding(enc_batch)+emb_mask, mask_src, training=training)
        else:
            encoder_outputs = self.encoder(self.embedding(enc_batch), mask_src, training=training)

        ## Attention over decoder
        q_h = tf.math.reduce_mean(encoder_outputs, axis=1) if config.mean_query else encoder_outputs[:,0]
        #q_h = encoder_outputs[:,0]
        logit_prob = self.decoder_key(q_h) #(bsz, num_experts)
        if(config.topk>0):
            k_max_value, k_max_index = tf.math.top_k(logit_prob, config.topk)
            a = np.empty([logit_prob.shape[0], self.decoder_number])
            a.fill(float('-inf'))
            mask = tf.cast(tf.constant(a), dtype=tf.float32)
            logit_prob_ = mask.scatter_(1,tf.cast(k_max_index, dtype=tf.int64),k_max_value)
            attention_parameters = self.attention_activation(logit_prob_)
        else:
            attention_parameters = self.attention_activation(logit_prob)
        if(config.oracle):
            attention_parameters = self.attention_activation(tf.cast(batch[5], dtype=tf.float32)*1000) #'target_program'
        attention_parameters = tf.expand_dims(tf.expand_dims(attention_parameters, axis=-1), axis=-1) # (batch_size, expert_num, 1, 1)
        # Decode 
        sos_token = tf.expand_dims(tf.cast([config.SOS_idx] * enc_batch.shape[0], dtype=tf.int32), axis=1)
        dec_batch_shift = tf.concat((sos_token,dec_batch[:, :-1]), axis=1)
        mask_trg = tf.expand_dims(tf.math.equal(dec_batch_shift, config.PAD_idx), axis=1)
       
        pre_logit, attn_dist = self.decoder(self.embedding(dec_batch_shift), encoder_outputs, (mask_src,mask_trg), attention_parameters, training=training)
        ## compute output dist
        logit = self.generator(pre_logit, attn_dist, enc_batch_extend_vocab if config.pointer_gen else None, extra_zeros, attn_dist_db=None)
        return logit, logit_prob
    
    def decoder_greedy(self, batch, max_dec_step=30, training=False):
        enc_batch, _, _, enc_batch_extend_vocab, extra_zeros, _, _ = get_input_from_batch(batch)
        mask_src = tf.expand_dims(tf.math.equal(enc_batch, config.PAD_idx), axis = 1)
        emb_mask = self.embedding(batch[2]) #"mask_input"
        encoder_outputs = self.encoder(self.embedding(enc_batch)+emb_mask, mask_src, training=training)
        ## Attention over decoder
        q_h = tf.math.reduce_mean(encoder_outputs, axis=1) if config.mean_query else encoder_outputs[:,0]
        #q_h = encoder_outputs[:,0]
        logit_prob = self.decoder_key(q_h)
        
        if(config.topk>0): 
            k_max_value, k_max_index = tf.math.top_k(logit_prob, config.topk)
            a = np.empty([logit_prob.shape[0], self.decoder_number])
            a.fill(float('-inf'))
            mask = tf.cast(tf.constant(a), dtype=tf.float32)
            logit_prob = mask.scatter_(1,tf.cast(k_max_index, dtype=tf.int64),k_max_value)

        attention_parameters = self.attention_activation(logit_prob)
        
        if(config.oracle):
            attention_parameters = self.attention_activation(tf.cast(batch[5], dtype=tf.float32)*1000) #'target_program'
        attention_parameters = tf.expand_dims(tf.expand_dims(attention_parameters, axis=-1), axis=-1) # (batch_size, expert_num, 1, 1)

        ys = tf.cast(tf.fill([1, 1], config.SOS_idx), dtype=tf.int64)
        mask_trg = tf.expand_dims(tf.math.equal(ys, config.PAD_idx), axis=1)
        decoded_words = []
        for i in range(max_dec_step+1):
            if(config.project):
                out, attn_dist = self.decoder(self.embedding_proj_in(self.embedding(ys)),self.embedding_proj_in(encoder_outputs), (mask_src,mask_trg), attention_parameters, training=training)
            else:
                out, attn_dist = self.decoder(self.embedding(ys),encoder_outputs, (mask_src,mask_trg), attention_parameters, training=training)
            logit = self.generator(out,attn_dist,enc_batch_extend_vocab, extra_zeros, attn_dist_db=None)
            next_word = tf.math.argmax(logit[:, -1], axis = 1)
            decoded_words.append(['<EOS>' if ni.item() == config.EOS_idx else self.vocab.index2word[ni] for ni in tf.reshape(next_word, -1).numpy()])
            next_word = next_word[0]
            ys = tf.concat([ys, tf.cast(tf.fill([1, 1], next_word), dtype=tf.int64)], axis=1)
            mask_trg = tf.expand_dims(tf.math.equal(ys, config.PAD_idx), axis=1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == '<EOS>': 
                    break
                else: 
                    st+= e + ' '
            sent.append(st)
        return sent
    
    def decoder_topk(self, batch, max_dec_step=30, training=False):
        enc_batch, _, _, enc_batch_extend_vocab, extra_zeros, _, _ = get_input_from_batch(batch)
        mask_src = tf.expand_dims(tf.math.equal(enc_batch, config.PAD_idx), axis = 1)
        emb_mask = self.embedding(batch[2]) #"mask_input"
        encoder_outputs = self.encoder(self.embedding(enc_batch)+emb_mask, mask_src, training=training)
        ## Attention over decoder
        q_h = tf.math.reduce_mean(encoder_outputs, axis=1) if config.mean_query else encoder_outputs[:,0]
        #q_h = encoder_outputs[:,0]
        logit_prob = self.decoder_key(q_h)
        
        if(config.topk>0): 
            k_max_value, k_max_index = tf.math.top_k(logit_prob, config.topk)
            a = np.empty([logit_prob.shape[0], self.decoder_number])
            a.fill(float('-inf'))
            mask = tf.cast(tf.constant(a), dtype=tf.float32)
            logit_prob = mask.scatter_(1,tf.cast(k_max_index, dtype=tf.int64),k_max_value)

        attention_parameters = self.attention_activation(logit_prob)
        
        if(config.oracle):
            attention_parameters = self.attention_activation(tf.cast(batch[5], dtype=tf.float32)*1000) #'target_program'
        attention_parameters = tf.expand_dims(tf.expand_dims(attention_parameters, axis=-1), axis=-1) # (batch_size, expert_num, 1, 1)

        ys = tf.cast(tf.fill([1, 1], config.SOS_idx), dtype=tf.int64)
        mask_trg = tf.expand_dims(tf.math.equal(ys, config.PAD_idx), axis=1)
        decoded_words = []
        for i in range(max_dec_step+1):
            if(config.project):
                out, attn_dist = self.decoder(self.embedding_proj_in(self.embedding(ys)),self.embedding_proj_in(encoder_outputs), (mask_src,mask_trg), attention_parameters, training=training)
            else: 
                out, attn_dist = self.decoder(self.embedding(ys),encoder_outputs, (mask_src,mask_trg), attention_parameters, training=training)
            logit = self.generator(out,attn_dist,enc_batch_extend_vocab, extra_zeros, attn_dist_db=None)
            filtered_logit = top_k_top_p_filtering(logit[:, -1], top_k=3, top_p=0, filter_value=-float('Inf'))
            # Sample from the filtered distribution
            next_word = tf.squeeze(tf.random.categorical(tf.nn.softmax(filtered_logit, axis=-1), 1))
            decoded_words.append(['<EOS>' if ni.item() == config.EOS_idx else self.vocab.index2word[ni] for ni in tf.reshape(next_word, -1).numpy()])
            next_word = next_word[0]

            ys = tf.concat([ys, tf.cast(tf.fill([1, 1], next_word), dtype=tf.int64)], axis=1)
            mask_trg = tf.expand_dims(tf.math.equal(ys, config.PAD_idx), axis=1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == '<EOS>':
                    break
                else:
                    st+= e + ' '
            sent.append(st)
        return sent

class ACT_basic(layers.Layer):
    def __init__(self,hidden_size):
        super(ACT_basic, self).__init__()
        self.sigma = layers.Activation('sigmoid')
        self.p = layers.Dense(1, bias_initializer='ones')
        self.threshold = 1 - 0.1
    
    def __call__(self, state, inputs, fn, time_enc, pos_enc, max_hop, encoder_output=None, decoding=False):
        # init_hdd
        ## [B, S]
        halting_probability = tf.zeros(inputs.shape[0],inputs.shape[1])
        ## [B, S
        remainders = tf.zeros(inputs.shape[0],inputs.shape[1])
        ## [B, S]
        n_updates = tf.zeros(inputs.shape[0],inputs.shape[1])
        ## [B, S, HDD]
        previous_state = tf.zeros(inputs.shape)
        step = 0
        # for l in range(self.num_layers):
        while( ((halting_probability<self.threshold) & (n_updates < max_hop)).byte().any()):
            # Add Timing Signal
            state = state + tf.cast(time_enc[:, :inputs.shape[1], :], inputs.dtype)
            state = state + tf.cast(tf.tile(tf.expand_dims(pos_enc[:, step, :], 1), [1,inputs.shape[1],1]), inputs.dtype)

            p = tf.squeeze(self.sigma(self.p(state)), axis=-1)
            # Mask for inputs which have not halted yet
            still_running = tf.cast((halting_probability < 1.0), dtype=tf.float32)
            # Mask of inputs which halted at this step
            new_halted = tf.cast((halting_probability + p * still_running > self.threshold), dtype=tf.float32) * still_running
            # Mask of inputs which haven't halted, and didn't halt this step
            still_running = tf.cast((halting_probability + p * still_running <= self.threshold), dtype=tf.float32) * still_running
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
            previous_state = ((state * tf.expand_dims(update_weights, axis=-1)) + (previous_state * (1 - tf.expand_dims(update_weights, axis=-1))))
            if(decoding):
                if(step==0): 
                    previous_att_weight = tf.zeros(attention_weight.shape)## [B, S, src_size]
            previous_att_weight = ((attention_weight * tf.expand_dims(update_weights, axis=-1)) + (previous_att_weight * (1 - tf.expand_dims(update_weights, axis=-1))))
            ## previous_state is actually the new_state at end of hte loop 
            ## to save a line I assigned to previous_state so in the next 
            ## iteration is correct. Notice that indeed we return previous_state
            step+=1
        if(decoding):
            return previous_state, previous_att_weight, (remainders,n_updates)
        else:
            return previous_state, (remainders,n_updates)