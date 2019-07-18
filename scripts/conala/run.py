"""Script for train running. Analog to train.sh in the same directory."""
import os
import sys
sys.path.append("/home/y/PycharmProjects/tranX")

from exp import main


seed = 0
vocab = "data/conala/vocab.var_str_sep.src_freq3.code_freq3.bin"
train_file = "data/conala/train.var_str_sep.bin"
dev_file = "data/conala/dev.var_str_sep.bin"
test_file = "data/conala/test.var_str_sep.bin"
batch_size = 100
max_epoch = 50
log_every = 1

dropout = 0.3
hidden_size = 256
embed_size = 128
action_embed_size = 128
field_embed_size = 128
type_embed_size = 128
ptrnet_hidden_dim = 32
lr = 0.001
lr_decay = 0.5
lr_decay_after_epoch = 15
lstm = "lstm"  # lstm
beam_size = 15

model_name = f"model.sup.conala.{lstm}.hidden{hidden_size}.embed{embed_size}.action{action_embed_size}." \
    f"field{field_embed_size}.type{type_embed_size}.dr{dropout}.lr{lr}.lr_de{lr_decay}.lr_da{lr_decay_after_epoch}." \
    f"beam{beam_size}.{os.path.basename(vocab)}.{os.path.basename(train_file)}.glorot.par_state.seed{seed}"

ORIGINAL_ARGS_TRAIN = [
    # '--cuda',
    '--seed', str(seed),
    '--mode', 'train',
    '--batch_size', str(batch_size),
    '--evaluator', 'conala_evaluator',
    '--asdl_file', 'asdl/lang/py3/py3_asdl.simplified.txt',
    '--transition_system', 'python3',
    '--train_file', train_file,
    '--dev_file', dev_file,
    '--vocab', vocab,
    '--lstm', lstm,
    '--no_parent_field_type_embed',
    '--no_parent_production_embed',
    '--hidden_size', str(hidden_size),
    '--embed_size', str(embed_size),
    '--action_embed_size', str(action_embed_size),
    '--field_embed_size', str(field_embed_size),
    '--type_embed_size', str(type_embed_size),
    '--dropout', str(dropout),
    '--patience', str(5),
    '--max_num_trial', str(5),
    '--glorot_init',
    '--lr', str(lr),
    '--lr_decay', str(lr_decay),
    '--lr_decay_after_epoch', str(lr_decay_after_epoch),
    '--max_epoch', str(max_epoch),
    '--beam_size', str(beam_size),
    '--log_every', str(log_every),
    '--save_to', 'saved_models/conala/' + model_name
]
ORIGINAL_ARGS_TEST = [
    # '--cuda',
    '--mode', 'test',
    '--load_model', 'saved_models/conala/' + model_name + '.bin',
    '--beam_size', str(beam_size),
    '--test_file', test_file,
    '--evaluator', 'conala_evaluator',
    '--save_decode_to', 'decodes/conala/' + os.path.basename(model_name) + '.test.decode',
    '--decode_max_time_step', str(100),
]


dropout = 0.3
hidden_size = 256
num_layers = 1
ffn_size = 256
num_heads = 8
dropout_model = 0.1
embed_size = 128
action_embed_size = 128
field_embed_size = 64
type_embed_size = 64
ptrnet_hidden_dim = 32
lr = 0.001
lr_decay = 0.5
lr_decay_after_epoch = 15
lstm = "transformer"  # lstm
beam_size = 15

transformer_model_name = f"model.sup.conala.{lstm}.H{hidden_size}.ffn{ffn_size}.h{num_heads}." \
    f"L{num_layers}.dout_m{dropout_model}.emb{embed_size}.act{action_embed_size}." \
    f"fie{field_embed_size}.typ{type_embed_size}.dr{dropout}.lr{lr}.lr_de{lr_decay}.lr_da{lr_decay_after_epoch}." \
    f"B{beam_size}.{os.path.basename(vocab)}.{os.path.basename(train_file)}.seed{seed}"

TRANSFORMER_ARGS_TRAIN = [
    # '--cuda',
    '--seed', str(seed),
    '--parser', 'transformer_parser',
    '--valid_every_epoch', str(1),
    # '--no_copy',
    '--mode', 'train',
    '--batch_size', str(batch_size),
    '--evaluator', 'conala_evaluator',
    '--asdl_file', 'asdl/lang/py3/py3_asdl.simplified.txt',
    '--transition_system', 'python3',
    '--train_file', train_file,
    '--dev_file', dev_file,
    '--vocab', vocab,
    '--lstm', lstm,

    '--hidden_size', str(hidden_size),
    '--num_layers', str(num_layers),
    '--ffn_size', str(ffn_size),
    '--num_heads', str(num_heads),

    '--no_parent_production_embed',

    '--embed_size', str(embed_size),
    '--action_embed_size', str(action_embed_size),
    '--field_embed_size', str(field_embed_size),
    '--type_embed_size', str(type_embed_size),
    '--dropout', str(dropout),

    '--patience', str(5),
    '--max_num_trial', str(5),
    # '--glorot_init',
    '--lr', str(lr),
    '--lr_decay', str(lr_decay),
    '--lr_decay_after_epoch', str(lr_decay_after_epoch),
    '--max_epoch', str(max_epoch),
    '--beam_size', str(beam_size),

    '--log_every', str(log_every),
    '--save_to', 'saved_models/conala/' + transformer_model_name
]
TRANSFORMER_ARGS_TEST = [
    # '--cuda',
    '--mode', 'test',
    '--parser', 'transformer_parser',
    '--load_model', 'saved_models/conala/' + transformer_model_name + '.bin',
    '--beam_size', str(beam_size),
    '--test_file', test_file,
    '--evaluator', 'conala_evaluator',
    '--save_decode_to', 'decodes/conala/' + os.path.basename(transformer_model_name) + '.test.decode',
    '--decode_max_time_step', str(100),
]

if __name__ == '__main__':
    if sys.argv[1] == "train":
        sys.stdout = open(os.path.join("logs", "conala", model_name + ".log"), "wt", buffering=1)
        main(ORIGINAL_ARGS_TRAIN)
    elif sys.argv[1] == "test":
        sys.stdout = open(os.path.join("logs", "conala", model_name + ".log"), "wt", buffering=1)
        main(ORIGINAL_ARGS_TEST)
    elif sys.argv[1] == "transformer_train":
        sys.stdout = open(os.path.join("logs", "conala", transformer_model_name + ".log"), "wt", buffering=1)
        main(TRANSFORMER_ARGS_TRAIN)
    elif sys.argv[1] == "transformer_test":
        sys.stdout = open(os.path.join("logs", "conala", transformer_model_name + ".log"), "wt", buffering=1)
        main(TRANSFORMER_ARGS_TEST)
    else:
        raise RuntimeError('unknown mode')
