import logging
from typing import Dict, Tuple, List

import numpy as np
from overrides import overrides
import torch
from torch import autograd
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.modules import Attention
from allennlp.modules.feedforward import FeedForward
from allennlp.nn.beam_search import BeamSearch
from allennlp.nn import util, InitializerApplicator

class SPNet(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder, # just Embedding layer
                 encoder1: Seq2SeqEncoder, # user encoder
                 encoder2: Seq2SeqEncoder, # system encoder
                 attention: Attention, # decoding attention
                 max_decoding_steps: int = 200, # max timesteps of decoder
                 beam_size: int = 3, # beam search parameter
                 target_namespace: str = "target_tokens", # two separate vocabulary
                 target_embedding_dim: int = None, # target word embedding dimension
                 scheduled_sampling_ratio: float = 0., # maybe unnecessary
                 projection_dim: int = None, #
                 use_coverage: bool = False, # coverage penalty, optional
                 coverage_loss_weight: float = None,
                 domain_lambda: float = 0.5, # the penalty weight in final loss function, need to be tuned
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:

        super(SPNet, self).__init__(vocab)

        # General variables
        # target_namespace: target_tokens; source_namespace: tokens;
        self._target_namespace = target_namespace
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        self._source_unk_index = self.vocab.get_token_index(DEFAULT_OOV_TOKEN)
        self._target_unk_index = self.vocab.get_token_index(DEFAULT_OOV_TOKEN, self._target_namespace)
        self._source_vocab_size = self.vocab.get_vocab_size()
        self._target_vocab_size = self.vocab.get_vocab_size(self._target_namespace)

        # Encoder setting
        self._source_embedder = source_embedder
        self._encoder1 = encoder1
        self._encoder2 = encoder2
        # We assume that the 2 encoders have the same hidden state size
        self._encoder_output_dim = self._encoder1.get_output_dim()

        # Decoder setting
        self._target_embedding_dim = target_embedding_dim or source_embedder.get_output_dim()
        self._num_classes = self.vocab.get_vocab_size(self._target_namespace)
        self._target_embedder = Embedding(self._num_classes, self._target_embedding_dim)
        self._decoder_input_dim = self._encoder_output_dim * 2 # default as the decoder_output_dim
        # input projection of decoder: [context_attn, target_emb] -> [decoder_input_dim]
        self._input_projection_layer = Linear(self._target_embedding_dim + self._encoder_output_dim * 2,
                                                self._decoder_input_dim)
        self._decoder_output_dim = self._encoder_output_dim * 2
        self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_output_dim)
        self._projection_dim = projection_dim or self._source_embedder.get_output_dim()
        self._output_projection_layer = Linear(self._decoder_output_dim, self._num_classes)
        self._p_gen_layer = Linear(self._encoder_output_dim * 2 +
                                   self._decoder_output_dim * 2 + self._decoder_input_dim, 1)
        self._attention = attention

        # coverage penalty setting
        self._use_coverage = use_coverage
        self._coverage_loss_weight = coverage_loss_weight
        self._eps = 1e-45

        # Decoding strategy setting
        self._scheduled_sampling_ratio = scheduled_sampling_ratio
        self._max_decoding_steps = max_decoding_steps
        self._beam_search = BeamSearch(self._end_index, max_steps=max_decoding_steps, beam_size=beam_size)

        # multitasking of domain classification
        self._domain_penalty = domain_lambda # penalty term = 0.5 as default
        self._classifier_params = Params({
            "input_dim": self._decoder_output_dim, "hidden_dims": [128, 7], "activations": ["relu", "linear"],
            "dropout": [0.2, 0.0], "num_layers": 2
        })
        self._domain_classifier = FeedForward.from_params(self._classifier_params)
        
        initializer(self)
            
    """
        Make forward pass with decoder logic for producing the entire target sequence.
    """
    @overrides
    def forward(self,
                user_tokens: Dict[str, torch.LongTensor],
                user_token_ids: torch.Tensor,
                sys_tokens: Dict[str, torch.LongTensor],
                sys_token_ids: torch.Tensor,
                user_to_target: torch.Tensor,
                sys_to_target: torch.Tensor,
                user_value_mask: torch.Tensor, # 0-1 mask of the values 
                sys_value_mask: torch.Tensor,
                domain_labels: torch.Tensor,
                target_tokens: Dict[str, torch.LongTensor] = None,
                target_token_ids: torch.Tensor = None,
                metadata=None) -> Dict[str, torch.Tensor]:
        
        state = self._encode(user_tokens, sys_tokens)
        state["user_tokens"] = user_tokens
        state["sys_tokens"] = sys_tokens
        state["user_token_ids"] = user_token_ids
        state["sys_token_ids"] = sys_token_ids
        state["user_to_target"] = user_to_target
        state["sys_to_target"] = sys_to_target
        state["user_value_mask"] = user_value_mask
        state["sys_value_mask"] = sys_value_mask
        state["domain_labels"] = domain_labels.float()
        # state["domain_labels"] = domain_labels.float() / torch.sum(domain_labels, dim=1).unsqueeze(1).float()
    
        if target_tokens:
            state["target_tokens"] = target_tokens["tokens"]
            state = self._init_decoder_state(state)
            output_dict = self._forward_loop(state, target_tokens, target_token_ids)
        else:
            output_dict = {}

        output_dict["metadata"] = metadata
        
        if target_tokens is None:
            state["user_tokens"] = user_tokens["tokens"]
            state["sys_tokens"] = sys_tokens["tokens"]
            state = self._init_decoder_state(state)
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)

        return output_dict



    """ 
        Encoder steps of both user & system. 
    """
    def _encode(self,
                user_tokens: Dict[str, torch.Tensor],
                sys_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input_user = self._source_embedder(user_tokens)
        embedded_input_sys = self._source_embedder(sys_tokens)

        # shape: (batch_size, max_input_sequence_length)
        source_mask_user = util.get_text_field_mask(user_tokens)
        source_mask_sys = util.get_text_field_mask(sys_tokens)

        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        # note: encoder_outputs has all hidden states(timesteps x hidden_dim)
        encoder_outputs_user = self._encoder1.forward(embedded_input_user, source_mask_user)
        encoder_outputs_sys = self._encoder2.forward(embedded_input_sys, source_mask_sys)

        return {
                "source_mask_usr": source_mask_user,
                "encoder_outputs_usr": encoder_outputs_user,
                "source_mask_sys": source_mask_sys,
                "encoder_outputs_sys": encoder_outputs_sys
        }

    """ 
        Initialize the decoder state from the encoder last hidden state. 
    """
    def _init_decoder_state(self,
                            state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        batch_size = state["source_mask_usr"].size(0)

        # shape: (batch_size, encoder_output_dim)
        user_final_encoder_output = util.get_final_encoder_states(
                state["encoder_outputs_usr"],
                state["source_mask_usr"],
                self._encoder1.is_bidirectional())
        # shape: (batch_size, encoder_output_dim)
        sys_final_encoder_output = util.get_final_encoder_states(
                state["encoder_outputs_sys"],
                state["source_mask_sys"],
                self._encoder2.is_bidirectional())

        # shape: (batch_size, decoder_output_dim)
        # decoder_output_dim = 2 * encoder_output_dim
        encoder_output_concat = torch.cat((user_final_encoder_output,
                                           sys_final_encoder_output), 1)
        state["decoder_hidden"] = encoder_output_concat
        
        # multitasking: domain classification
        logits = self._domain_classifier(encoder_output_concat)
        # domain_possibilities = F.softmax(logits)
        state["loss"] = F.binary_cross_entropy_with_logits(logits, state["domain_labels"]) * self._domain_penalty


        # Initialize decoder cell / coverage vector as zeros
        # shape: (batch_size, max_input_seq_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs_usr"]
        state["decoder_cell"] = encoder_outputs.new_zeros(batch_size, self._decoder_output_dim)
        # coverage vector: (batch_size, max_input_seq_length)
        if self._use_coverage:
            state["coverage"] = encoder_outputs.new_zeros(batch_size, encoder_outputs.size(1))

        return state

    """
        Decoder step: decoder_state + last_prediction -> decoder_output(this timestep)
    """
    def _decoder_step(self,
                      last_predictions: torch.Tensor,
                      state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        # 1. Get the tensor in need from state (encoder/decoding timestep t-1)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs_usr = state["encoder_outputs_usr"]
        encoder_outputs_sys = state["encoder_outputs_sys"]

        # shape: (batch_size, max_input_sequence_length)
        source_mask_usr = state["source_mask_usr"]
        source_mask_sys = state["source_mask_sys"]

        # shape: (batch_size, decoder_output_dim)
        decoder_hidden = state["decoder_hidden"]
        # shape: (batch_size, decoder_output_dim)
        decoder_cell = state["decoder_cell"]

        # 3. Compute attn_weights, attn_context, decoder_input and update coverage vector,
        #    passing them into the decoder cell and get h_t, c_t
        # shape: (batch_size, target_embedding_dim), mapping the index to word embedding
        embedded_input = self._target_embedder(last_predictions)

        # shape: (batch_size, max_sequence_length)
        attn_weights_usr = self._attention(
            decoder_hidden[:, :self._encoder_output_dim], encoder_outputs_usr, source_mask_usr
        )
        
        attn_weights_sys = self._attention(
            decoder_hidden[:, self._encoder_output_dim:], encoder_outputs_sys, source_mask_sys
        )

        # state["coverage"] maintains coverage vector, of which c^t = \sum_{t'=0}^{t-1} a^t'
        # shape: (batch_size, max_sequence_length)
        if self._use_coverage:
            coverage_usr = state["coverage_usr"]
            coverage_usr = coverage_usr + attn_weights_usr
            coverage_sys = state["coverage_sys"]
            coverage_sys = coverage_sys + attn_weights_sys
            state["coverage_usr"] = coverage_usr
            state["coverage_sys"] = coverage_sys

        # attn_context = a^Th, shape: (batch_size, encoder_output_dim)
        attn_context_usr = util.weighted_sum(encoder_outputs_usr, attn_weights_usr)
        attn_context_sys = util.weighted_sum(encoder_outputs_sys, attn_weights_sys)
        attn_context = torch.cat((attn_context_usr, attn_context_sys), -1)

        # mapping (attn_context, target_word_emb) to (decoder_input)
        decoder_input_source = torch.cat((attn_context, embedded_input), -1)
        decoder_input = self._input_projection_layer(decoder_input_source)

        # shape (decoder_hidden): (batch_size, decoder_output_dim)
        # shape (decoder_cell): (batch_size, decoder_output_dim)
        decoder_hidden, decoder_cell = self._decoder_cell(
            decoder_input,
            (decoder_hidden, decoder_cell))

        # Same as the imp of Pointer-Gen: P_{vocab} = softmax(V'(V[st, ht*] + b) + b')
        # but we have not apply softmax cuz it will be computed in get_final_dist
        # output_projections = self._output_projection_layer(self._hidden_projection_layer(decoder_hidden))
        output_projections = self._output_projection_layer(decoder_hidden) # shape: (batch_size, target_vocab_size)

        # 4. Update state in timestep t
        state["decoder_input"] = decoder_input
        state["decoder_hidden"] = decoder_hidden
        state["decoder_cell"] = decoder_cell
        state["attn_scores_usr"] = attn_weights_usr
        state["attn_scores_sys"] = attn_weights_sys
        state["attn_context"] = attn_context

        return output_projections, state

    """
        Compute the final distribution on vocabulary:
        P(w) = p_gen * P_vocab(w) + (1-p_gen) \sum a^t
    
        p_gen is the "generation probability" in timestep t
    """
    def _get_final_dist(self,
                        state: Dict[str, torch.Tensor],
                        output_projections):

        user_tokens = state["user_tokens"]
        sys_tokens = state["sys_tokens"]
        attn_context = state["attn_context"] # shape: (batch_size, 2 * encoder_output_dim)
        decoder_input = state["decoder_input"]
        decoder_hidden = state["decoder_hidden"]
        decoder_cell = state["decoder_cell"]
        user_to_target = state["user_to_target"] # shape: (batch_size, max_sequence_length)
        sys_to_target = state["sys_to_target"] # shape: (batch_size, max_sequence_length)

        # Note that decoder state = [c, h] shape: (batch_size, 2 * decoder_output_dim)
        decoder_state = torch.cat((decoder_cell, decoder_hidden), 1)

        # p_gen = sigmoid(wh* ht* + ws st + wx xt + b_ptr) -> scalar [0, 1]
        # _p_gen_layer = Linear(2 * encoder_output_dim + decoder_output_dim * 2 + decoder_input_dim, 1)
        p_gen = torch.sigmoid(self._p_gen_layer(torch.cat((attn_context, decoder_state, decoder_input), 1)))

        # vocab_dist: (batch_size, target_vocab_size), which is the generation scores
        vocab_dist = F.softmax(output_projections, dim=-1)

        # get the final distribution as weighted sum result
        vocab_dist = vocab_dist * p_gen

        # user and system may have different shape but they represents the same
        # distribution on source vocabulary # shape: (batch_size, max_sequence_length)
        attn_dist_usr = state["attn_scores_usr"]
        attn_dist_sys = state["attn_scores_sys"]
        attn_dist_usr = attn_dist_usr * (1.0 - p_gen)
        attn_dist_sys = attn_dist_sys * (1.0 - p_gen)
        
        # !!! Attention-based Lexicalization
        usr_lex_mask = state["user_value_mask"]
        sys_lex_mask = state["sys_value_mask"]
        
        attn_lex_score_usr = F.softmax(state["attn_scores_usr"] * usr_lex_mask) # lower the unrelated words' score to zero
        attn_lex_score_sys = F.softmax(state["attn_scores_sys"] * sys_lex_mask) # shape: (batch_size, max_sequence_length)
        attn_max_usr = attn_lex_score_usr.argmax(dim=-1) # shape:(batch_size, 1)
        attn_max_sys = attn_lex_score_sys.argmax(dim=-1)
        

        # consider the copy scores (if they are known words in target vocabulary)
        unk_user_in_target = torch.eq(user_to_target, self._target_unk_index).float()
        unk_sys_in_target = torch.eq(sys_to_target, self._target_unk_index).float()

        # we want to mask the attention weights on unknown
        attn_dist_usr = attn_dist_usr - unk_user_in_target * attn_dist_usr
        attn_dist_sys = attn_dist_sys - unk_sys_in_target * attn_dist_sys

        user_token_index = (user_to_target).long()
        sys_token_index = (sys_to_target).long()

        # only apply those known words in target_namespace from source to the final distribution
        # final_dist: shape=(batch_size, target_vocab_size)
        # NOTE that the FINAL distribution is also on target_vocabulary, not extended of oov
        final_dist = vocab_dist.scatter_add(1, user_token_index, attn_dist_usr)
        final_dist = final_dist.scatter_add(1, sys_token_index, attn_dist_sys)
        normalization_factor = final_dist.sum(1, keepdim=True)
        final_dist = final_dist / normalization_factor
        
        return final_dist, attn_lex_score_usr, attn_lex_score_sys


    """
        Compute loss in one forward loop(training), use scheduled sampling.
        (a trick in RNN aimed at abridging the gap in training and prediction, Bengio et al.)
    """
    def _forward_loop(self,
                      state: Dict[str, torch.Tensor],
                      target_tokens: Dict[str, torch.LongTensor],
                      target_token_ids: torch.Tensor) -> Dict[str, torch.Tensor]:

        batch_size, target_sequence_length = target_tokens["tokens"].size()

        # shape: (batch_size, max_input_sequence_length)
        source_mask_usr = state["source_mask_usr"]
        source_mask_sys = state["source_mask_sys"]

        # shape: (batch_size, max_target_sequence_length)
        targets = target_tokens["tokens"]

        # The last token in the target tokens do not need to be processed (<END>)
        num_decoding_steps = target_sequence_length - 1

        # Initialize target predictions with <start_index>
        # shape: (batch_size, )
        last_predictions = source_mask_usr.new_full((batch_size,), fill_value=self._start_index)

        if self._use_coverage:
            coverage_loss = None

        step_proba: List[torch.Tensor] = []
        step_predictions: List[torch.Tensor] = []
        step_value_index: List[torch.Tensor] = []
        # begin the iterations of decoding
        for timestep in range(num_decoding_steps):
            # apply scheduled sampling to abridge the gap between training and testing of RNNs
            if torch.rand(1).item() < self._scheduled_sampling_ratio:
                input_choices = last_predictions
            else:
                input_choices = targets[:, timestep]

            if self._use_coverage:
                coverage_usr = state["coverage_usr"]
                coverage_sys = state["coverage_sys"]
            
            output_projections, state = self._decoder_step(input_choices, state)
            final_dist, attn_max_index, _ = self._get_final_dist(state, output_projections)
            step_proba.append(final_dist)
            step_value_index.append(attn_max_index.unsqueeze(1))

            if self._use_coverage:
                step_coverage_loss_usr = torch.sum(torch.min(state["attn_scores"], coverage_usr), 1)
                step_coverage_loss_sys = torch.sum(torch.min(state["attn_scores"], coverage_sys), 1)
                if coverage_loss is None:
                    coverage_loss = step_coverage_loss_usr + step_coverage_loss_sys
                else:
                    coverage_loss = coverage_loss + step_coverage_loss_usr + step_coverage_loss_sys

            _, predicted_classes = torch.max(final_dist, 1)
            last_predictions = predicted_classes
            step_predictions.append(last_predictions.unsqueeze(1))

        # shape: (batch_size, num_decoding_steps)
        predictions = torch.cat(step_predictions, 1)
        attn_max_index_all = torch.cat(step_value_index, 1)

        output_dict = {"predictions": predictions, "attn_max": attn_max_index_all}

        # shape: (batch_size, num_decoding_steps, num_classes)
        num_classes = step_proba[0].size(1)
        proba = step_proba[0].new_zeros((batch_size, num_classes, len(step_proba)))
        for i, p in enumerate(step_proba):
            proba[:, :, i] = p

        # Compute loss
        loss = self._get_loss(proba, state["target_tokens"], self._eps)

        # coverage penalty term:
        if self._use_coverage:
            coverage_loss = torch.mean(coverage_loss / num_decoding_steps)
            loss = loss + self._coverage_loss_weight * coverage_loss

        output_dict["loss"] = loss + state["loss"]

        return output_dict


    """
        _get_loss() returns the loss between the ground truth and the logits.
        
        Note that Pointer-Gen use NLL loss of the target token, while seq2seq
        uses cross entropy loss.
    """
    @staticmethod
    def _get_loss(proba: torch.LongTensor,
                  targets: torch.LongTensor,
                  eps: float) -> torch.Tensor:
        targets = targets[:, 1:]
        proba = torch.log(proba + eps)
        loss = torch.nn.NLLLoss(ignore_index=0)(proba, targets)
        return loss


    """ 
        Forward step when making predictions in evaluation. 
    """
    def _forward_beam_search(self,
                             state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        batch_size = state["source_mask_usr"].size()[0]
        # shape: (batch_size, ), just <START> symbol's index in target namespace
        start_predictions = state["source_mask_usr"].new_full(size=(batch_size,), fill_value=self._start_index)
        # shape (all_top_k_predictions): (batch_size, beam_size, max_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        del state["loss"]
        
        all_top_k_predictions, log_probabilities = self._beam_search.search(
            start_predictions, state, self.take_step)
        attn_max_usr = state["attn_max_usr"]
        attn_max_sys = state["attn_max_sys"]
        
        # return the beam search result and leave other works in ``decode`` method
        output_dict = {
            "class_log_probabilities": log_probabilities,
            "predictions": all_top_k_predictions,
            "attn_max_index_usr": attn_max_usr,
            "attn_max_index_sys": attn_max_sys
        }
        return output_dict


    """ 
        Used as param in beam_search.search(), responsible for computing the next most
        likely tokens, given the current state and the predictions from last timestep.
    """
    def take_step(self,
                  last_predictions: torch.Tensor,
                  state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        # shape: (group_size, num_classes)
        output_projections, state = self._decoder_step(last_predictions, state)
        final_dist, attn_max_index_usr, attn_max_index_sys = self._get_final_dist(state, output_projections)
        
        log_probabilities = torch.log(final_dist + self._eps)
        if("attn_max_usr" not in state.keys() and attn_max_index_usr.shape[0] > 1):
            attn_max_index_usr = attn_max_index_usr.unsqueeze(1)
            state["attn_max_usr"] =  attn_max_index_usr
        elif(attn_max_index_usr.shape[0] > 1):
            attn_max_index_usr = attn_max_index_usr.unsqueeze(1)
            state["attn_max_usr"] = torch.cat((state["attn_max_usr"], attn_max_index_usr), dim=1)
        
        if("attn_max_sys" not in state.keys() and attn_max_index_sys.shape[0] > 1):
            attn_max_index_sys = attn_max_index_sys.unsqueeze(1)
            state["attn_max_sys"] =  attn_max_index_sys
        elif(attn_max_index_sys.shape[0] > 1):
            attn_max_index_sys = attn_max_index_sys.unsqueeze(1)
            state["attn_max_sys"] = torch.cat((state["attn_max_sys"], attn_max_index_sys), dim=1)
        
        return log_probabilities, state


    """
        Finalize predictions, get called after Model.forward().
        
        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the output_dict.
        
        Note that the ``decode`` is not just indexing from target vocabulary as the simpleSeq2seq
        does, but needs to process ``unk`` token. (so many details in copy mechanism!)
    """
    @overrides
    def decode(self,
               output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        predicted_indices = output_dict["predictions"] # shape: (batch_size, num_decoding_steps)
        attn_max_index_usr = output_dict["attn_max_index_usr"]
        attn_max_index_sys = output_dict["attn_max_index_sys"]
        
        if not isinstance(predicted_indices, np.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
            attn_max_index_usr = attn_max_index_usr.detach().cpu().numpy()
            attn_max_index_sys = attn_max_index_sys.detach().cpu().numpy()

        all_predicted_tokens = []
        for indices in predicted_indices:
            # Beam search gives us the top k results for each source sentence in the batch
            # but we just want the single best. -> choose the best from beam search size
            if len(indices.shape) > 1:
                indices = indices[0]
                attn_max_index_usr = attn_max_index_usr[0]
                attn_max_index_sys = attn_max_index_sys[0]
            indices = list(indices)
            
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[:indices.index(self._end_index)]
            
            predicted_tokens = []
            tokens_delex = []
            for idx, x in enumerate(indices):
                if x < self._target_vocab_size:
                    token = self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                    tokens_delex.append(token)
                    if(token[0] == "["):
                        attn_scores_usr = attn_max_index_usr[idx, :]
                        attn_scores_sorted_usr = np.argsort(-attn_scores_usr)
                        attn_scores_sys = attn_max_index_sys[idx, :]
                        attn_scores_sorted_sys = np.argsort(-attn_scores_sys)
                        attn_idx = 0
                        for attn_idx in range(attn_scores_sorted_usr.shape[0]):
                            if(output_dict["metadata"][0]["user_tokens"][attn_scores_sorted_usr[attn_idx]] == token):
                                token = output_dict["metadata"][0]["user_values_dict"][attn_scores_sorted_usr[attn_idx]]
                                break
                        if token[0] == "[":
                            for attn_idx in range(attn_scores_sorted_sys.shape[0]):
                                if(output_dict["metadata"][0]["sys_tokens"][attn_scores_sorted_sys[attn_idx]] == token):
                                    token = output_dict["metadata"][0]["sys_values_dict"][attn_scores_sorted_sys[attn_idx]]
                                    break

                else:
                    logger.info("No! All the indices should within the target vocabulary")
                predicted_tokens.append(token)
            all_predicted_tokens.append(predicted_tokens)
        
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict