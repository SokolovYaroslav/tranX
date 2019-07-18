# coding=utf-8
from __future__ import print_function

import copy
import math
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils

from asdl.hypothesis import GenTokenAction
from asdl.transition_system import ApplyRuleAction, ReduceAction
from common.registerable import Registrable
from common.utils import update_args, init_arg_parser
from components.action_info import ActionInfo
from components.dataset import Batch
from components.decode_hypothesis import DecodeHypothesis
from model import nn_utils
from model.nn_utils import LabelSmoothing
from model.pointer_net import PointerNet
from model.transformer_utils import (
    PositionalEncoding,
    Embeddings,
    Encoder,
    EncoderLayer,
    MultiHeadedAttention,
    PositionwiseFeedForward,
    Decoder,
    DecoderLayer,
    subsequent_mask,
    StrictMultiHeadedAttention,
)


@Registrable.register("transformer_parser")
class TransformerParser(nn.Module):
    """Implementation of a semantic parser

    The parser translates a natural language utterance into an AST defined under
    the ASDL specification, using the transition system described in https://arxiv.org/abs/1810.02720
    """

    def __init__(self, args, vocab, transition_system):
        super(TransformerParser, self).__init__()

        self.args = args
        self.vocab = vocab
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        self.transition_system = transition_system
        self.grammar = self.transition_system.grammar

        # Transformer parameters
        self.num_layers = args.num_layers
        self.d_model = args.hidden_size
        self.d_ff = args.ffn_size
        self.h = args.num_heads
        self.dropout = args.dropout_model
        self.position = PositionalEncoding(self.d_model, self.dropout)
        attn = MultiHeadedAttention(self.h, self.d_model)
        parent_attn = StrictMultiHeadedAttention(self.h, 1, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)

        # Embedding layers

        # source token embedding
        self.src_embed = nn.Sequential(Embeddings(self.d_model, len(vocab.source)), copy.deepcopy(self.position))

        # embedding table of ASDL production rules (constructors), one for each ApplyConstructor action,
        # the last entry is the embedding for Reduce action
        self.action_embed_size = args.action_embed_size
        self.field_embed_size = args.field_embed_size
        self.type_embed_size = args.type_embed_size

        assert self.d_model == (
            self.action_embed_size
            + self.action_embed_size * (not self.args.no_parent_production_embed)
            + self.field_embed_size * (not self.args.no_parent_field_embed)
            + self.type_embed_size * (not self.args.no_parent_field_type_embed)
        )

        self.production_embed = Embeddings(self.action_embed_size, len(transition_system.grammar) + 1)

        # embedding table for target primitive tokens
        self.primitive_embed = Embeddings(self.action_embed_size, len(vocab.primitive))

        # embedding table for ASDL fields in constructors
        self.field_embed = Embeddings(self.field_embed_size, len(transition_system.grammar.fields))

        # embedding table for ASDL types
        self.type_embed = Embeddings(self.type_embed_size, len(transition_system.grammar.types))

        assert args.lstm == "transformer"
        self.encoder = Encoder(
            EncoderLayer(self.d_model, copy.deepcopy(attn), copy.deepcopy(ff), self.dropout), self.num_layers
        ).to(self.device)
        self.decoder = Decoder(
            DecoderLayer(
                self.d_model, copy.deepcopy(parent_attn), copy.deepcopy(attn), copy.deepcopy(ff), self.dropout
            ),
            self.num_layers,
        ).to(self.device)

        if args.no_copy is False:
            # pointer net for copying tokens from source side
            self.src_pointer_net = PointerNet(query_vec_size=args.att_vec_size, src_encoding_size=args.hidden_size)

            # given the decoder's hidden state, predict whether to copy or generate a target primitive token
            # output: [p(gen(token)) | s_t, p(copy(token)) | s_t]

            self.primitive_predictor = nn.Linear(args.att_vec_size, 2)

        if args.primitive_token_label_smoothing:
            self.label_smoothing = LabelSmoothing(
                args.primitive_token_label_smoothing, len(self.vocab.primitive), ignore_indices=[0, 1, 2]
            )

        # initialize the decoder's state and cells with encoder hidden states
        self.decoder_cell_init = nn.Linear(args.hidden_size, args.hidden_size)

        # attention: dot product attention
        # project source encoding to decoder rnn's hidden space

        self.att_src_linear = nn.Linear(args.hidden_size, args.hidden_size, bias=False)

        # transformation of decoder hidden states and context vectors before reading out target words
        # this produces the `attentional vector` in (Luong et al., 2015)

        self.att_vec_linear = nn.Linear(args.hidden_size + args.hidden_size, args.att_vec_size, bias=False)

        # bias for predicting ApplyConstructor and GenToken actions
        self.production_readout_b = nn.Parameter(torch.zeros(len(transition_system.grammar) + 1, dtype=torch.float32))
        self.tgt_token_readout_b = nn.Parameter(torch.zeros(len(vocab.primitive), dtype=torch.float32))

        if args.no_query_vec_to_action_map:
            # if there is no additional linear layer between the attentional vector (i.e., the query vector)
            # and the final softmax layer over target actions, we use the attentional vector to compute action
            # probabilities

            assert args.att_vec_size == args.action_embed_size
            self.production_readout = lambda q: F.linear(
                q * math.sqrt(self.d_model), self.production_embed.lut.weight, self.production_readout_b
            )
            self.tgt_token_readout = lambda q: F.linear(
                q * math.sqrt(self.d_model), self.primitive_embed.lut.weight, self.tgt_token_readout_b
            )
        else:
            # by default, we feed the attentional vector (i.e., the query vector) into a linear layer without bias, and
            # compute action probabilities by dot-producting the resulting vector and (GenToken, ApplyConstructor) action embeddings
            # i.e., p(action) = query_vec^T \cdot W \cdot embedding

            self.query_vec_to_action_embed = nn.Linear(
                args.att_vec_size, args.action_embed_size, bias=args.readout == "non_linear"
            )
            if args.query_vec_to_action_diff_map:
                # use different linear transformations for GenToken and ApplyConstructor actions
                self.query_vec_to_primitive_embed = nn.Linear(
                    args.att_vec_size, args.action_embed_size, bias=args.readout == "non_linear"
                )
            else:
                self.query_vec_to_primitive_embed = self.query_vec_to_action_embed

            self.read_out_act = F.tanh if args.readout == "non_linear" else nn_utils.identity

            self.production_readout = lambda q: F.linear(
                self.read_out_act(self.query_vec_to_action_embed(q)) * math.sqrt(self.d_model),
                self.production_embed.lut.weight,
                self.production_readout_b,
            )
            self.tgt_token_readout = lambda q: F.linear(
                self.read_out_act(self.query_vec_to_primitive_embed(q)) * math.sqrt(self.d_model),
                self.primitive_embed.lut.weight,
                self.tgt_token_readout_b,
            )

        # dropout layer
        self.dropout = nn.Dropout(args.dropout)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def score(self, examples, return_encode_state=False):
        """Given a list of examples, compute the log-likelihood of generating the target AST

        Args:
            examples: a batch of examples
            return_encode_state: return encoding states of input utterances
        output: score for each training example: Variable(batch_size)
        """

        batch = Batch(examples, self.grammar, self.vocab, copy=self.args.no_copy is False, cuda=self.args.cuda)

        # src_encodings: (batch_size, src_sent_len, hidden_size)
        src_encodings = self.encode(batch.src_sents_var)

        # tgt vector: (batch_size, src_sent_len, hidden_size)
        tgt_vector = self.prepare_tgt(batch)

        parent_indxs = [[a_t.parent_t if a_t else 0 for a_t in e.tgt_actions] for e in batch.examples]
        parent_indxs_np = np.zeros((len(parent_indxs), max(len(ind) for ind in parent_indxs)), dtype=np.long)
        for i in range(len(parent_indxs_np)):
            parent_indxs_np[i, : len(parent_indxs[i])] = parent_indxs[i]
            parent_indxs_np[i, 0] = 0

        # query vectors are sufficient statistics used to compute action probabilities
        query_vectors = self.decoder(
            tgt_vector, src_encodings, [parent_indxs_np], batch.src_token_mask_usual.unsqueeze(-2), batch.tgt_mask
        )
        # query_vectors: (tgt_action_len, batch_size, hidden_size)
        query_vectors = query_vectors.transpose(0, 1)

        # ApplyRule (i.e., ApplyConstructor) action probabilities
        # (tgt_action_len, batch_size, grammar_size)
        apply_rule_prob = F.softmax(self.production_readout(query_vectors), dim=-1)

        # probabilities of target (gold-standard) ApplyRule actions
        # (tgt_action_len, batch_size)
        tgt_apply_rule_prob = torch.gather(
            apply_rule_prob, dim=2, index=batch.apply_rule_idx_matrix.unsqueeze(2)
        ).squeeze(2)

        # compute generation and copying probabilities #

        # (tgt_action_len, batch_size, primitive_vocab_size)
        gen_from_vocab_prob = F.softmax(self.tgt_token_readout(query_vectors), dim=-1)

        # (tgt_action_len, batch_size)
        tgt_primitive_gen_from_vocab_prob = torch.gather(
            gen_from_vocab_prob, dim=2, index=batch.primitive_idx_matrix.unsqueeze(2)
        ).squeeze(2)

        if self.args.no_copy:
            # mask positions in action_prob that are not used

            if self.training and self.args.primitive_token_label_smoothing:
                # (tgt_action_len, batch_size)
                # this is actually the negative KL divergence size we will flip the sign later
                # tgt_primitive_gen_from_vocab_log_prob = -self.label_smoothing(
                #     gen_from_vocab_prob.view(-1, gen_from_vocab_prob.size(-1)).log(),
                #     batch.primitive_idx_matrix.view(-1)).view(-1, len(batch))

                tgt_primitive_gen_from_vocab_log_prob = -self.label_smoothing(
                    gen_from_vocab_prob.log(), batch.primitive_idx_matrix
                )
            else:
                tgt_primitive_gen_from_vocab_log_prob = tgt_primitive_gen_from_vocab_prob.log()

            # (tgt_action_len, batch_size)
            action_prob = (
                tgt_apply_rule_prob.log() * batch.apply_rule_mask
                + tgt_primitive_gen_from_vocab_log_prob * batch.gen_token_mask
            )
        else:
            # binary gating probabilities between generating or copying a primitive token
            # (tgt_action_len, batch_size, 2)
            primitive_predictor = F.softmax(self.primitive_predictor(query_vectors), dim=-1)

            # pointer network copying scores over source tokens
            # (tgt_action_len, batch_size, src_sent_len)
            primitive_copy_prob = self.src_pointer_net(src_encodings, batch.src_token_mask, query_vectors)

            # marginalize over the copy probabilities of tokens that are same
            # (tgt_action_len, batch_size)
            tgt_primitive_copy_prob = torch.sum(primitive_copy_prob * batch.primitive_copy_token_idx_mask, dim=-1)

            # mask positions in action_prob that are not used
            # (tgt_action_len, batch_size)
            action_mask_pad = torch.eq(batch.apply_rule_mask + batch.gen_token_mask + batch.primitive_copy_mask, 0.0)
            action_mask = 1.0 - action_mask_pad.float()

            # (tgt_action_len, batch_size)
            action_prob = (
                tgt_apply_rule_prob * batch.apply_rule_mask
                + primitive_predictor[:, :, 0] * tgt_primitive_gen_from_vocab_prob * batch.gen_token_mask
                + primitive_predictor[:, :, 1] * tgt_primitive_copy_prob * batch.primitive_copy_mask
            )

            # avoid nan in log
            action_prob.data.masked_fill_(action_mask_pad.data, 1.0e-7)
            eps = 1.0e-18
            action_prob += eps

            action_prob = action_prob.log() * action_mask

        scores = torch.sum(action_prob, dim=0)

        returns = [scores]

        return returns

    def prepare_tgt(self, batch):
        tgt_vector = []

        actions_embed = torch.zeros(
            (len(batch), batch.max_action_num - 1, self.action_embed_size), dtype=torch.float32, device=self.device
        )
        for e_num, example in enumerate(batch.examples):
            for a_num, action in enumerate(example.tgt_actions[:-1]):
                if isinstance(action.action, ApplyRuleAction):
                    action_embed = self.production_embed(torch.tensor(self.grammar.prod2id[action.action.production]))
                elif isinstance(action.action, ReduceAction):
                    action_embed = self.production_embed(torch.tensor(len(self.grammar)))
                else:
                    action_embed = self.primitive_embed(torch.tensor(self.vocab.primitive[action.action.token]))
                actions_embed[e_num, a_num] = action_embed
        tgt_vector.append(actions_embed)

        if self.args.no_parent_production_embed is False:
            parent_productions_embed = self.production_embed(
                torch.stack([batch.get_frontier_prod_idx(t) for t in range(1, batch.max_action_num)], dim=1)
            )
            tgt_vector.append(parent_productions_embed)

        if self.args.no_parent_field_embed is False:
            parent_field_embed = self.field_embed(
                torch.stack([batch.get_frontier_field_idx(t) for t in range(1, batch.max_action_num)], dim=1)
            )
            tgt_vector.append(parent_field_embed)

        if self.args.no_parent_field_type_embed is False:
            parent_field_type_embed = self.type_embed(
                torch.stack([batch.get_frontier_field_type_idx(t) for t in range(1, batch.max_action_num)], dim=1)
            )
            tgt_vector.append(parent_field_type_embed)

        tgt_vector = torch.cat(tgt_vector, dim=-1)

        start_vector = torch.zeros((len(batch), 1, tgt_vector.shape[2]))

        # initialize using the root type embedding
        offset = self.action_embed_size  # prev_action
        offset += self.action_embed_size * (not self.args.no_parent_production_embed)
        offset += self.field_embed_size * (not self.args.no_parent_field_embed)

        start_vector[:, 0, offset:] = self.type_embed(
            torch.tensor(
                [self.grammar.type2id[self.grammar.root_type] for _ in batch.examples],
                dtype=torch.long,
                device=self.device,
            )
        )

        tgt_vector = torch.cat([start_vector, tgt_vector], dim=1)

        return tgt_vector

    def encode(self, src_sents_var):
        """Encode the input natural language utterance

        Args:
            src_sents_var: a variable of shape (src_sent_len, batch_size), representing word ids of the input

        Returns:
            src_encodings: source encodings of shape (batch_size, src_sent_len, hidden_size)
        """
        src_sents_var = src_sents_var.transpose(0, 1)
        src_token_mask = (src_sents_var != 0).unsqueeze(-2)

        src_token_embed = self.src_embed(src_sents_var)

        src_encodings = self.encoder(src_token_embed, src_token_mask)

        return src_encodings

    def parse(self, src_sent, context=None, beam_size=5, debug=False):
        """Perform beam search to infer the target AST given a source utterance

        Args:
            src_sent: list of source utterance tokens
            context: other context used for prediction
            beam_size: beam size

        Returns:
            A list of `DecodeHypothesis`, each representing an AST
        """

        with torch.no_grad():
            args = self.args
            primitive_vocab = self.vocab.primitive

            src_sent_var = nn_utils.to_input_variable([src_sent], self.vocab.source, cuda=args.cuda, training=False)

            # Variable(1, src_sent_len, hidden_size)
            src_encodings = self.encode(src_sent_var)

            zero_action_embed = torch.zeros(args.action_embed_size)

            hyp_scores = torch.tensor([0.0])

            # For computing copy probabilities, we marginalize over tokens with the same surface form
            # `aggregated_primitive_tokens` stores the position of occurrence of each source token
            aggregated_primitive_tokens = OrderedDict()
            for token_pos, token in enumerate(src_sent):
                aggregated_primitive_tokens.setdefault(token, []).append(token_pos)

            t = 0
            hypotheses = [DecodeHypothesis()]
            completed_hypotheses = []

            while len(completed_hypotheses) < beam_size and t < args.decode_max_time_step:
                hyp_num = len(hypotheses)

                # (hyp_num, src_sent_len, hidden_size)
                exp_src_encodings = src_encodings.expand(hyp_num, src_encodings.size(1), src_encodings.size(2))

                if t == 0:
                    x = torch.zeros(1, self.d_model)
                    parent_ids = np.array([[0]])
                    if args.no_parent_field_type_embed is False:
                        offset = self.args.action_embed_size  # prev_action
                        offset += self.args.action_embed_size * (not self.args.no_parent_production_embed)
                        offset += self.args.field_embed_size * (not self.args.no_parent_field_embed)

                        x[0, offset : offset + self.type_embed_size] = self.type_embed(
                            torch.tensor(self.grammar.type2id[self.grammar.root_type])
                        )
                        x = x.unsqueeze(-2)
                else:
                    actions_tm1 = [hyp.actions[-1] for hyp in hypotheses]

                    a_tm1_embeds = []
                    for a_tm1 in actions_tm1:
                        if a_tm1:
                            if isinstance(a_tm1, ApplyRuleAction):
                                a_tm1_embed = self.production_embed(
                                    torch.tensor(self.grammar.prod2id[a_tm1.production])
                                )
                            elif isinstance(a_tm1, ReduceAction):
                                a_tm1_embed = self.production_embed(torch.tensor(len(self.grammar)))
                            else:
                                a_tm1_embed = self.primitive_embed(torch.tensor(self.vocab.primitive[a_tm1.token]))

                            a_tm1_embeds.append(a_tm1_embed)
                        else:
                            a_tm1_embeds.append(zero_action_embed)
                    a_tm1_embeds = torch.stack(a_tm1_embeds)

                    inputs = [a_tm1_embeds]
                    if args.no_parent_production_embed is False:
                        # frontier production
                        frontier_prods = [hyp.frontier_node.production for hyp in hypotheses]
                        frontier_prod_embeds = self.production_embed(
                            torch.tensor([self.grammar.prod2id[prod] for prod in frontier_prods], dtype=torch.long)
                        )
                        inputs.append(frontier_prod_embeds)
                    if args.no_parent_field_embed is False:
                        # frontier field
                        frontier_fields = [hyp.frontier_field.field for hyp in hypotheses]
                        frontier_field_embeds = self.field_embed(
                            torch.tensor([self.grammar.field2id[field] for field in frontier_fields], dtype=torch.long)
                        )

                        inputs.append(frontier_field_embeds)
                    if args.no_parent_field_type_embed is False:
                        # frontier field type
                        frontier_field_types = [hyp.frontier_field.type for hyp in hypotheses]
                        frontier_field_type_embeds = self.type_embed(
                            torch.tensor(
                                [self.grammar.type2id[type] for type in frontier_field_types], dtype=torch.long
                            )
                        )
                        inputs.append(frontier_field_type_embeds)

                    x = torch.cat([x, torch.cat(inputs, dim=-1).unsqueeze(-2)], dim=1)
                    recent_parents = np.array(
                        [[hyp.frontier_node.created_time] if hyp.frontier_node else 0 for hyp in hypotheses]
                    )
                    parent_ids = np.hstack([parent_ids, recent_parents])

                src_mask = torch.ones(
                    exp_src_encodings.shape[:-1], dtype=torch.uint8, device=exp_src_encodings.device
                ).unsqueeze(-2)
                tgt_mask = subsequent_mask(x.shape[-2])

                att_t = self.decoder(x, exp_src_encodings, [parent_ids], src_mask, tgt_mask)[:, -1]

                # Variable(batch_size, grammar_size)
                # apply_rule_log_prob = torch.log(F.softmax(self.production_readout(att_t), dim=-1))
                apply_rule_log_prob = F.log_softmax(self.production_readout(att_t), dim=-1)

                # Variable(batch_size, primitive_vocab_size)
                gen_from_vocab_prob = F.softmax(self.tgt_token_readout(att_t), dim=-1)

                if args.no_copy:
                    primitive_prob = gen_from_vocab_prob
                else:
                    # Variable(batch_size, src_sent_len)
                    primitive_copy_prob = self.src_pointer_net(src_encodings, None, att_t.unsqueeze(0)).squeeze(0)

                    # Variable(batch_size, 2)
                    primitive_predictor_prob = F.softmax(self.primitive_predictor(att_t), dim=-1)

                    # Variable(batch_size, primitive_vocab_size)
                    primitive_prob = primitive_predictor_prob[:, 0].unsqueeze(1) * gen_from_vocab_prob

                    # if src_unk_pos_list:
                    #     primitive_prob[:, primitive_vocab.unk_id] = 1.e-10

                gentoken_prev_hyp_ids = []
                gentoken_new_hyp_unks = []
                applyrule_new_hyp_scores = []
                applyrule_new_hyp_prod_ids = []
                applyrule_prev_hyp_ids = []

                for hyp_id, hyp in enumerate(hypotheses):
                    # generate new continuations
                    action_types = self.transition_system.get_valid_continuation_types(hyp)

                    for action_type in action_types:
                        if action_type == ApplyRuleAction:
                            productions = self.transition_system.get_valid_continuating_productions(hyp)
                            for production in productions:
                                prod_id = self.grammar.prod2id[production]
                                prod_score = apply_rule_log_prob[hyp_id, prod_id].item()
                                new_hyp_score = hyp.score + prod_score

                                applyrule_new_hyp_scores.append(new_hyp_score)
                                applyrule_new_hyp_prod_ids.append(prod_id)
                                applyrule_prev_hyp_ids.append(hyp_id)
                        elif action_type == ReduceAction:
                            action_score = apply_rule_log_prob[hyp_id, len(self.grammar)].item()
                            new_hyp_score = hyp.score + action_score

                            applyrule_new_hyp_scores.append(new_hyp_score)
                            applyrule_new_hyp_prod_ids.append(len(self.grammar))
                            applyrule_prev_hyp_ids.append(hyp_id)
                        else:
                            # GenToken action
                            gentoken_prev_hyp_ids.append(hyp_id)
                            hyp_copy_info = dict()  # of (token_pos, copy_prob)
                            hyp_unk_copy_info = []

                            if args.no_copy is False:
                                for (token, token_pos_list) in aggregated_primitive_tokens.items():
                                    sum_copy_prob = torch.gather(
                                        primitive_copy_prob[hyp_id], 0, torch.tensor(token_pos_list, dtype=torch.long)
                                    ).sum()
                                    gated_copy_prob = primitive_predictor_prob[hyp_id, 1] * sum_copy_prob

                                    if token in primitive_vocab:
                                        token_id = primitive_vocab[token]
                                        primitive_prob[hyp_id, token_id] = (
                                            primitive_prob[hyp_id, token_id] + gated_copy_prob
                                        )

                                        hyp_copy_info[token] = (token_pos_list, gated_copy_prob.item())
                                    else:
                                        hyp_unk_copy_info.append(
                                            {
                                                "token": token,
                                                "token_pos_list": token_pos_list,
                                                "copy_prob": gated_copy_prob.item(),
                                            }
                                        )

                            if args.no_copy is False and len(hyp_unk_copy_info) > 0:
                                unk_i = np.array([x["copy_prob"] for x in hyp_unk_copy_info]).argmax()
                                token = hyp_unk_copy_info[unk_i]["token"]
                                primitive_prob[hyp_id, primitive_vocab.unk_id] = hyp_unk_copy_info[unk_i]["copy_prob"]
                                gentoken_new_hyp_unks.append(token)

                                hyp_copy_info[token] = (
                                    hyp_unk_copy_info[unk_i]["token_pos_list"],
                                    hyp_unk_copy_info[unk_i]["copy_prob"],
                                )

                new_hyp_scores = None
                if applyrule_new_hyp_scores:
                    new_hyp_scores = torch.tensor(applyrule_new_hyp_scores)
                if gentoken_prev_hyp_ids:
                    primitive_log_prob = torch.log(primitive_prob)
                    gen_token_new_hyp_scores = (
                        hyp_scores[gentoken_prev_hyp_ids].unsqueeze(1) + primitive_log_prob[gentoken_prev_hyp_ids, :]
                    ).view(-1)

                    if new_hyp_scores is None:
                        new_hyp_scores = gen_token_new_hyp_scores
                    else:
                        new_hyp_scores = torch.cat([new_hyp_scores, gen_token_new_hyp_scores])

                top_new_hyp_scores, top_new_hyp_pos = torch.topk(
                    new_hyp_scores, k=min(new_hyp_scores.size(0), beam_size - len(completed_hypotheses))
                )

                live_hyp_ids = []
                new_hypotheses = []
                for new_hyp_score, new_hyp_pos in zip(top_new_hyp_scores.data.cpu(), top_new_hyp_pos.data.cpu()):
                    action_info = ActionInfo()
                    if new_hyp_pos < len(applyrule_new_hyp_scores):
                        # it's an ApplyRule or Reduce action
                        prev_hyp_id = applyrule_prev_hyp_ids[new_hyp_pos]
                        prev_hyp = hypotheses[prev_hyp_id]

                        prod_id = applyrule_new_hyp_prod_ids[new_hyp_pos]
                        # ApplyRule action
                        if prod_id < len(self.grammar):
                            production = self.grammar.id2prod[prod_id]
                            action = ApplyRuleAction(production)
                        # Reduce action
                        else:
                            action = ReduceAction()
                    else:
                        # it's a GenToken action
                        token_id = int((new_hyp_pos - len(applyrule_new_hyp_scores)) % primitive_prob.size(1))

                        k = (new_hyp_pos - len(applyrule_new_hyp_scores)) // primitive_prob.size(1)
                        # try:
                        # copy_info = gentoken_copy_infos[k]
                        prev_hyp_id = gentoken_prev_hyp_ids[k]
                        prev_hyp = hypotheses[prev_hyp_id]
                        # except:
                        #     print('k=%d' % k, file=sys.stderr)
                        #     print('primitive_prob.size(1)=%d' % primitive_prob.size(1), file=sys.stderr)
                        #     print('len copy_info=%d' % len(gentoken_copy_infos), file=sys.stderr)
                        #     print('prev_hyp_id=%s' % ', '.join(str(i) for i in gentoken_prev_hyp_ids), file=sys.stderr)
                        #     print('len applyrule_new_hyp_scores=%d' % len(applyrule_new_hyp_scores), file=sys.stderr)
                        #     print('len gentoken_prev_hyp_ids=%d' % len(gentoken_prev_hyp_ids), file=sys.stderr)
                        #     print('top_new_hyp_pos=%s' % top_new_hyp_pos, file=sys.stderr)
                        #     print('applyrule_new_hyp_scores=%s' % applyrule_new_hyp_scores, file=sys.stderr)
                        #     print('new_hyp_scores=%s' % new_hyp_scores, file=sys.stderr)
                        #     print('top_new_hyp_scores=%s' % top_new_hyp_scores, file=sys.stderr)
                        #
                        #     torch.save((applyrule_new_hyp_scores, primitive_prob), 'data.bin')
                        #
                        #     # exit(-1)
                        #     raise ValueError()

                        if token_id == int(primitive_vocab.unk_id):
                            if gentoken_new_hyp_unks:
                                token = gentoken_new_hyp_unks[k]
                            else:
                                token = primitive_vocab.id2word[primitive_vocab.unk_id]
                        else:
                            token = primitive_vocab.id2word[token_id]

                        action = GenTokenAction(token)

                        if token in aggregated_primitive_tokens:
                            action_info.copy_from_src = True
                            action_info.src_token_position = aggregated_primitive_tokens[token]

                        if debug:
                            action_info.gen_copy_switch = (
                                "n/a"
                                if args.no_copy
                                else primitive_predictor_prob[prev_hyp_id, :].log().cpu().data.numpy()
                            )
                            action_info.in_vocab = token in primitive_vocab
                            action_info.gen_token_prob = (
                                gen_from_vocab_prob[prev_hyp_id, token_id].log().cpu().item()
                                if token in primitive_vocab
                                else "n/a"
                            )
                            action_info.copy_token_prob = (
                                torch.gather(
                                    primitive_copy_prob[prev_hyp_id],
                                    0,
                                    torch.tensor(action_info.src_token_position, dtype=torch.long, device=self.device),
                                )
                                .sum()
                                .log()
                                .cpu()
                                .item()
                                if args.no_copy is False and action_info.copy_from_src
                                else "n/a"
                            )

                    action_info.action = action
                    action_info.t = t
                    if t > 0:
                        action_info.parent_t = prev_hyp.frontier_node.created_time
                        action_info.frontier_prod = prev_hyp.frontier_node.production
                        action_info.frontier_field = prev_hyp.frontier_field.field

                    if debug:
                        action_info.action_prob = new_hyp_score - prev_hyp.score

                    new_hyp = prev_hyp.clone_and_apply_action_info(action_info)
                    new_hyp.score = new_hyp_score

                    if new_hyp.completed:
                        completed_hypotheses.append(new_hyp)
                    else:
                        new_hypotheses.append(new_hyp)
                        live_hyp_ids.append(prev_hyp_id)

                if live_hyp_ids:
                    x = x[live_hyp_ids]
                    parent_ids = parent_ids[live_hyp_ids]
                    hypotheses = new_hypotheses
                    hyp_scores = torch.tensor([hyp.score for hyp in hypotheses])
                    t += 1
                else:
                    break

            completed_hypotheses.sort(key=lambda hyp: -hyp.score)

            return completed_hypotheses

    def save(self, path):
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        params = {
            "args": self.args,
            "transition_system": self.transition_system,
            "vocab": self.vocab,
            "state_dict": self.state_dict(),
        }
        torch.save(params, path)

    @classmethod
    def load(cls, model_path, cuda=False):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        vocab = params["vocab"]
        transition_system = params["transition_system"]
        saved_args = params["args"]
        # update saved args
        update_args(saved_args, init_arg_parser())
        saved_state = params["state_dict"]
        saved_args.cuda = cuda

        parser = cls(saved_args, vocab, transition_system)

        parser.load_state_dict(saved_state)

        if cuda:
            parser = parser.cuda()
        parser.eval()

        return parser
