# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Sequence-to-Sequence Decoders
"""
from typing import Callable, List, NamedTuple, Tuple
from typing import Optional

import mxnet as mx

import sockeye.attention
import sockeye.constants as C
import sockeye.coverage
import sockeye.encoder
import sockeye.lexicon
import sockeye.rnn
import sockeye.utils


def get_decoder(num_embed: int,
                vocab_size: int,
                num_layers: int,
                rnn_num_hidden: int,
                attention: sockeye.attention.Attention,
                cell_type: str, residual: bool,
                forget_bias: float,
                dropout=0.,
                weight_tying: bool = False,
                lexicon: Optional[sockeye.lexicon.Lexicon] = None,
                context_gating: bool = False,
                scheduled_sampling_type: str = None,
                scheduled_sampling_decay_params: List[float] = None) -> 'Decoder':
    """
    Returns a StackedRNNDecoder with the following properties.

    :param num_embed: Target word embedding size.
    :param vocab_size: Target vocabulary size.
    :param num_layers: Number of RNN layers in the decoder.
    :param rnn_num_hidden: Number of hidden units per decoder RNN cell.
    :param attention: Attention model.
    :param cell_type: RNN cell type.
    :param residual: Whether to add residual connections to multi-layer RNNs.
    :param forget_bias: Initial value of the RNN forget bias.
    :param dropout: Dropout probability for decoder RNN.
    :param weight_tying: Whether to share embedding and prediction parameter matrices.
    :param lexicon: Optional Lexicon.
    :param context_gating: Whether to use context gating.
    :return: Decoder instance.
    """
    return StackedRNNDecoder(rnn_num_hidden,
                             attention,
                             vocab_size,
                             num_embed,
                             num_layers,
                             weight_tying=weight_tying,
                             dropout=dropout,
                             cell_type=cell_type,
                             residual=residual,
                             forget_bias=forget_bias,
                             lexicon=lexicon,
                             context_gating=context_gating,
                             scheduled_sampling_type=scheduled_sampling_type,
                             scheduled_sampling_decay_params=scheduled_sampling_decay_params)


class Decoder:
    """
    Generic decoder interface.
    """

    def get_num_hidden(self) -> int:
        """
        Returns the representation size of this decoder.

        :raises: NotImplementedError
        """
        raise NotImplementedError()

    def get_rnn_cells(self) -> List[mx.rnn.BaseRNNCell]:
        """
        Returns a list of RNNCells used by this decoder.

        :raises: NotImplementedError
        """
        raise NotImplementedError()


DecoderState = NamedTuple('DecoderState', [
    ('hidden', mx.sym.Symbol),
    ('layer_states', List[mx.sym.Symbol]),
])
"""
Decoder state.

:param hidden: Hidden state after attention mechanism. Shape: (batch_size, num_hidden).
:param layer_states: Hidden states for RNN layers of StackedRNNDecoder. Shape: List[(batch_size, rnn_num_hidden)]

"""


class StackedRNNDecoder(Decoder):
    """
    Class to generate the decoder part of the computation graph in sequence-to-sequence models.
    The architecture is based on Luong et al, 2015: Effective Approaches to Attention-based Neural Machine Translation

    :param num_hidden: Number of hidden units in decoder RNN.
    :param attention: Attention model.
    :param target_vocab_size: Size of target vocabulary.
    :param num_target_embed: Size of target word embedding.
    :param num_layers: Number of decoder RNN layers.
    :param prefix: Decoder symbol prefix.
    :param weight_tying: Whether to share embedding and prediction parameter matrices.
    :param dropout: Dropout probability for decoder RNN.
    :param cell_type: RNN cell type.
    :param residual: Whether to add residual connections to multi-layer RNNs.
    :param forget_bias: Initial value of the RNN forget bias.
    :param lexicon: Optional Lexicon.
    :param context_gating: Whether to use context gating.
    """

    def __init__(self,
                 num_hidden: int,
                 attention: sockeye.attention.Attention,
                 target_vocab_size: int,
                 num_target_embed: int,
                 num_layers=1,
                 prefix=C.DECODER_PREFIX,
                 weight_tying=False,
                 dropout=0.0,
                 cell_type: str = C.LSTM_TYPE,
                 residual: bool = False,
                 forget_bias: float = 0.0,
                 lexicon: Optional[sockeye.lexicon.Lexicon] = None,
                 context_gating: bool = False,
                 scheduled_sampling_type: str = None,
                 scheduled_sampling_decay_params: List[float] = None):
        # TODO: implement variant without input feeding
        self.num_layers = num_layers
        self.prefix = prefix
        self.dropout = dropout
        self.num_hidden = num_hidden
        self.attention = attention
        self.target_vocab_size = target_vocab_size
        self.num_target_embed = num_target_embed
        self.context_gating = context_gating
        if self.context_gating:
            self.gate_w = mx.sym.Variable("%sgate_weight" % prefix)
            self.gate_b = mx.sym.Variable("%sgate_bias" % prefix)
            self.mapped_rnn_output_w = mx.sym.Variable("%smapped_rnn_output_weight" % prefix)
            self.mapped_rnn_output_b = mx.sym.Variable("%smapped_rnn_output_bias" % prefix)
            self.mapped_context_w = mx.sym.Variable("%smapped_context_weight" % prefix)
            self.mapped_context_b = mx.sym.Variable("%smapped_context_bias" % prefix)

        self.scheduled_sampling_type = scheduled_sampling_type
        self.scheduled_sampling_decay_params = scheduled_sampling_decay_params

        # Decoder stacked RNN
        self.rnn = sockeye.rnn.get_stacked_rnn(cell_type, num_hidden, num_layers, dropout, prefix, residual,
                                               forget_bias)

        # Decoder parameters
        # RNN init state parameters
        self._create_layer_parameters()
        # Hidden state parameters
        self.hidden_w = mx.sym.Variable("%shidden_weight" % prefix)
        self.hidden_b = mx.sym.Variable("%shidden_bias" % prefix)
        # Embedding & output parameters
        self.embedding = sockeye.encoder.Embedding(self.num_target_embed, self.target_vocab_size,
                                                   prefix=C.TARGET_EMBEDDING_PREFIX, dropout=0.)  # TODO dropout?
        if weight_tying:
            assert self.num_hidden == self.num_target_embed, \
                "Weight tying requires target embedding size and rnn_num_hidden to be equal"
            self.cls_w = self.embedding.embed_weight
        else:
            self.cls_w = mx.sym.Variable("%scls_weight" % prefix)
        self.cls_b = mx.sym.Variable("%scls_bias" % prefix)

        self.lexicon = lexicon
        self._get_sampling_threshold = self._link_sampling_scheduler()

    def _link_sampling_scheduler(self):
        i = mx.sym.Variable('updates')

        def _check_params(params):
            if self.scheduled_sampling_type == 'inv-sigmoid-decay':
                assert len(params) == 1, \
                    ('The inverse sigmoid decaying option for scheduled sampling requires 1 parameter,'
                    ' but given {}').format(len(params))

                k = params[0]
                assert k >= 1, 'Offset should be greater than or equal to 1'

            elif self.scheduled_sampling_type == 'exponential-decay':
                assert len(params) == 1, \
                    ('The exponential decaying option for scheduled sampling requires 1 parameter,'
                    ' but given {}').format(len(params))
                k = params[0]
                assert k < 1, 'Offset for the exponential decay should be less than 1.'

            elif self.scheduled_sampling_type == 'linear-decay':
                assert len(params) == 3, \
                    ('The linear decaying option for scheduled sampling requires 3 parameter,'
                    ' but given {}').format(len(params))

                k, epsilon, slope = params

                assert k <= 1, 'Offset for the linear decay should be less than 1.'
                assert epsilon >= 0, 'Epsilonfor the linear decay should be greather than or equal to 0.'

        _check_params(self.scheduled_sampling_decay_params)

        if self.scheduled_sampling_type == 'inv-sigmoid-decay':
            k = self.scheduled_sampling_decay_params[0]
            sampling_scheduler = lambda: k /(k + mx.sym.exp(i / k))
        elif self.scheduled_sampling_type == 'exponential-decay':
            k = self.scheduled_sampling_decay_params[0]
            sampling_scheduler = lambda: pow(k, i)
        elif self.scheduled_sampling_type == 'linear-decay':
            k, epsilon, slope = self.scheduled_sampling_decay_params
            sampling_scheduler = lambda: mx.sym.clip(k - slope * i, epsilon, k)
        else:
            sampling_scheduler = lambda: 1

        return sampling_scheduler

    def get_num_hidden(self) -> int:
        """
        Returns the representation size of this decoder.

        :return: Number of hidden units.
        """
        return self.num_hidden

    def get_rnn_cells(self) -> List[mx.rnn.BaseRNNCell]:
        """
        Returns a list of RNNCells used by this decoder.
        """
        return [self.rnn]

    def _create_layer_parameters(self):
        """
        Creates parameters for encoder last state transformation into decoder layer initial states.
        """
        self.init_ws, self.init_bs = [], []
        for state_idx, (_, init_num_hidden) in enumerate(self.rnn.state_shape):
            self.init_ws.append(mx.sym.Variable("%senc2decinit_%d_weight" % (self.prefix, state_idx)))
            self.init_bs.append(mx.sym.Variable("%senc2decinit_%d_bias" % (self.prefix, state_idx)))

    def create_layer_input_variables(self, batch_size: int) \
            -> Tuple[List[mx.sym.Symbol], List[mx.io.DataDesc], List[str]]:
        """
        Creates RNN layer state variables. Used for inference.
        Returns nested list of layer_states variables, flat list of layer shapes (for module binding),
        and a flat list of layer names (for BucketingModule's data names)

        :param batch_size: Batch size.
        """
        layer_states, layer_shapes, layer_names = [], [], []
        for state_idx, (_, init_num_hidden) in enumerate(self.rnn.state_shape):
            name = "%senc2decinit_%d" % (self.prefix, state_idx)
            layer_states.append(mx.sym.Variable(name))
            layer_shapes.append(mx.io.DataDesc(name=name, shape=(batch_size, init_num_hidden), layout=C.BATCH_MAJOR))
            layer_names.append(name)
        return layer_states, layer_shapes, layer_names

    def compute_init_states(self,
                            source_encoded: mx.sym.Symbol,
                            source_length: mx.sym.Symbol) -> DecoderState:
        """
        Computes initial states of the decoder, hidden state, and one for each RNN layer.
        Init states for RNN layers are computed using 1 non-linear FC with the last state of the encoder as input.

        :param source_encoded: Concatenated encoder states. Shape: (source_seq_len, batch_size, encoder_num_hidden).
        :param source_length: Lengths of source sequences. Shape: (batch_size,).
        :return: Decoder state.
        """
        # initial decoder hidden state
        hidden = mx.sym.tile(data=mx.sym.expand_dims(data=source_length * 0, axis=1), reps=(1, self.num_hidden))
        # initial states for each layer
        layer_states = []
        for state_idx, (_, init_num_hidden) in enumerate(self.rnn.state_shape):
            init = mx.sym.FullyConnected(data=mx.sym.SequenceLast(data=source_encoded,
                                                                  sequence_length=source_length,
                                                                  use_sequence_length=True),
                                         num_hidden=init_num_hidden,
                                         weight=self.init_ws[state_idx],
                                         bias=self.init_bs[state_idx],
                                         name="%senc2decinit_%d" % (self.prefix, state_idx))
            init = mx.sym.Activation(data=init, act_type="tanh",
                                     name="%senc2dec_inittanh_%d" % (self.prefix, state_idx))
            layer_states.append(init)
        return DecoderState(hidden, layer_states)

    def _step(self,
              word_vec_prev: mx.sym.Symbol,
              state: DecoderState,
              attention_func: Callable,
              attention_state: sockeye.attention.AttentionState,
              seq_idx: int = 0) -> Tuple[DecoderState, sockeye.attention.AttentionState]:

        """
        Performs single-time step in the RNN, given previous word vector, previous hidden state, attention function,
        and RNN layer states.

        :param word_vec_prev: Embedding of previous target word. Shape: (batch_size, num_target_embed).
        :param state: Decoder state consisting of hidden and layer states.
        :param attention_func: Attention function to produce context vector.
        :param attention_state: Previous attention state.
        :param seq_idx: Decoder time step.
        :return: (new decoder state, updated attention state).
        """
        # (1) RNN step
        # concat previous word embedding and previous hidden state
        rnn_input = mx.sym.concat(word_vec_prev, state.hidden, dim=1,
                                  name="%sconcat_target_context_t%d" % (self.prefix, seq_idx))
        # rnn_output: (batch_size, rnn_num_hidden)
        # next_layer_states: num_layers * [batch_size, rnn_num_hidden]
        rnn_output, layer_states = self.rnn(rnn_input, state.layer_states)

        # (2) Attention step
        attention_input = self.attention.make_input(seq_idx, word_vec_prev, rnn_output)
        attention_state = attention_func(attention_input, attention_state)

        # (3) Combine context with hidden state
        if self.context_gating:
            # context: (batch_size, encoder_num_hidden)
            # gate: (batch_size, rnn_num_hidden)
            gate = mx.sym.FullyConnected(data=mx.sym.concat(word_vec_prev, rnn_output, attention_state.context, dim=1),
                                         num_hidden=self.num_hidden, weight=self.gate_w, bias=self.gate_b)
            gate = mx.sym.Activation(data=gate, act_type="sigmoid",
                                     name="%sgate_activation_t%d" % (self.prefix, seq_idx))

            # mapped_rnn_output: (batch_size, rnn_num_hidden)
            mapped_rnn_output = mx.sym.FullyConnected(data=rnn_output,
                                                      num_hidden=self.num_hidden,
                                                      weight=self.mapped_rnn_output_w,
                                                      bias=self.mapped_rnn_output_b,
                                                      name="%smapped_rnn_output_fc_t%d" % (self.prefix, seq_idx))
            # mapped_context: (batch_size, rnn_num_hidden)
            mapped_context = mx.sym.FullyConnected(data=attention_state.context,
                                                   num_hidden=self.num_hidden,
                                                   weight=self.mapped_context_w,
                                                   bias=self.mapped_context_b,
                                                   name="%smapped_context_fc_t%d" % (self.prefix, seq_idx))

            # hidden: (batch_size, rnn_num_hidden)
            hidden = mx.sym.Activation(data=gate * mapped_rnn_output + (1 - gate) * mapped_context,
                                       act_type="tanh",
                                       name="%snext_hidden_t%d" % (self.prefix, seq_idx))

        else:
            # hidden: (batch_size, rnn_num_hidden)
            hidden = mx.sym.FullyConnected(data=mx.sym.concat(rnn_output, attention_state.context, dim=1),
                                           # use same number of hidden states as RNN
                                           num_hidden=self.num_hidden,
                                           weight=self.hidden_w,
                                           bias=self.hidden_b)
            # hidden: (batch_size, rnn_num_hidden)
            hidden = mx.sym.Activation(data=hidden, act_type="tanh",
                                       name="%snext_hidden_t%d" % (self.prefix, seq_idx))

        return DecoderState(hidden, layer_states), attention_state

    def decode(self,
               source_encoded: mx.sym.Symbol,
               source_seq_len: int,
               source_length: mx.sym.Symbol,
               target: mx.sym.Symbol,
               target_label: mx.sym.Symbol,
               target_seq_len: int,
               source_lexicon: Optional[mx.sym.Symbol] = None) -> mx.sym.Symbol:
        """
        Returns decoder logits with batch size and target sequence length collapsed into a single dimension.

        :param source_encoded: Concatenated encoder states. Shape: (source_seq_len, batch_size, encoder_num_hidden).
        :param source_seq_len: Maximum source sequence length.
        :param source_length: Lengths of source sequences. Shape: (batch_size,).
        :param target: Target sequence. Shape: (batch_size, target_seq_len).
        :param target_seq_len: Maximum target sequence length.
        :param source_lexicon: Lexical biases for current sentence.
               Shape: (batch_size, target_vocab_size, source_seq_len)
        :return: Logits of next-word predictions for target sequence.
                 Shape: (batch_size * target_seq_len, target_vocab_size)
        """
        # process encoder states
        source_encoded_batch_major = mx.sym.swapaxes(source_encoded, dim1=0, dim2=1, name='source_encoded_batch_major')

        # embed and slice target words
        # target_embed: (batch_size, target_seq_len, num_target_embed)
        target_embed = self.embedding.encode(target, None, target_seq_len)
        # target_embed: target_seq_len * (batch_size, num_target_embed)
        target_embed = mx.sym.split(data=target_embed, num_outputs=target_seq_len, axis=1, squeeze_axis=True)

        # get recurrent attention function conditioned on source
        attention_func = self.attention.on(source_encoded_batch_major, source_length, source_seq_len)
        attention_state = self.attention.get_initial_state(source_length, source_seq_len)

        # initialize decoder states
        # hidden: (batch_size, rnn_num_hidden)
        # layer_states: List[(batch_size, state_num_hidden]
        state = self.compute_init_states(source_encoded, source_length)

        # pros_all: target_seq_len * (batch_size, 1, rnn_num_hidden)
        probs_all = []

        updates = mx.sym.Variable('updates')

        def inv_sigmoid_decay(_updates):
            assert self.offset >= 1, 'Offset should be greater than or equal to 1'
            return self.offset / (self.offset + mx.sym.exp(_updates / self.offset))

        self.rnn.reset()

        splitted_target = mx.sym.split(target_label, num_outputs=target_seq_len, axis=1, squeeze_axis=True)

        threshold = self._get_sampling_threshold()

        for seq_idx in range(target_seq_len):

            rnd_val = mx.sym.random_uniform(low=0, high=1.0, shape=1)[0]
            if seq_idx == 0:
                trg_emb = target_embed[seq_idx]
            else:
                trg_emb = mx.sym.where(rnd_val <= threshold,
                                       target_embed[seq_idx],
                                       self.embedding.encode(sampled_words, None, 1))

            state, attention_state = self._step(trg_emb,
                                                state,
                                                attention_func,
                                                attention_state,
                                                seq_idx)

            # logits: (batch_size, target_vocab_size)
            logits = mx.sym.FullyConnected(data=state.hidden, num_hidden=self.target_vocab_size,
                                        weight=self.cls_w, bias=self.cls_b, name=C.LOGITS_NAME)
            # probs: (batch_size, target_vocab_size)
            prob = mx.sym.SoftmaxOutput(data=logits,
                                        label=splitted_target[seq_idx],
                                        ignore_label=C.PAD_ID,
                                        use_ignore=True,
                                        # normalization=normalization,
                                        name=C.SOFTMAX_NAME + '_%d' % seq_idx)
            sampled_words = mx.sym.sample_multinomial(prob)
            probs_all.append(prob)

        # losses: (batch_size, target_seq_len, target_vocab_size)
        losses = mx.sym.concat(*probs_all , dim=1)
        # losses: (batch_size * target_seq_len, target_vocab_size)
        losses = mx.sym.reshape(losses, shape=(-1, self.target_vocab_size), name=C.SOFTMAX_NAME)

        return losses

    def predict(self,
                word_id_prev: mx.sym.Symbol,
                state_prev: DecoderState,
                attention_func: Callable,
                attention_state_prev: sockeye.attention.AttentionState,
                source_lexicon: Optional[mx.sym.Symbol] = None,
                softmax_temperature: Optional[float] = None) -> Tuple[mx.sym.Symbol,
                                                                      DecoderState,
                                                                      sockeye.attention.AttentionState]:
        """
        Given previous word id, attention function, previous hidden state and RNN layer states,
        returns Softmax predictions (not a loss symbol), next hidden state, and next layer
        states. Used for inference.

        :param word_id_prev: Previous target word id. Shape: (1,).
        :param state_prev: Previous decoder state consisting of hidden and layer states.
        :param attention_func: Attention function to produce context vector.
        :param attention_state_prev: Previous attention state.
        :param source_lexicon: Lexical biases for current sentence.
               Shape: (batch_size, target_vocab_size, source_seq_len).
        :param softmax_temperature: Optional parameter to control steepness of softmax distribution.
        :return: (predicted next-word distribution, decoder state, attention state).
        """
        # target side embedding
        word_vec_prev = self.embedding.encode(word_id_prev, None, 1)

        # state.hidden: (batch_size, rnn_num_hidden)
        # attention_state.dynamic_source: (batch_size, source_seq_len, coverage_num_hidden)
        # attention_state.probs: (batch_size, source_seq_len)
        state, attention_state = self._step(word_vec_prev,
                                            state_prev,
                                            attention_func,
                                            attention_state_prev)

        # logits: (batch_size, target_vocab_size)
        logits = mx.sym.FullyConnected(data=state.hidden, num_hidden=self.target_vocab_size,
                                       weight=self.cls_w, bias=self.cls_b, name=C.LOGITS_NAME)

        if source_lexicon is not None:
            assert self.lexicon is not None
            # lex_bias: (batch_size, 1, target_vocab_size)
            lex_bias = self.lexicon.calculate_lex_bias(source_lexicon, attention_state.probs)
            # lex_bias: (batch_size, target_vocab_size)
            lex_bias = mx.sym.reshape(data=lex_bias, shape=(-1, self.target_vocab_size))
            logits = mx.sym.broadcast_add(lhs=logits, rhs=lex_bias, name='%s_plus_lex_bias' % C.LOGITS_NAME)

        if softmax_temperature is not None:
            logits /= softmax_temperature

        softmax_out = mx.sym.softmax(data=logits, name=C.SOFTMAX_NAME)
        return softmax_out, state, attention_state
