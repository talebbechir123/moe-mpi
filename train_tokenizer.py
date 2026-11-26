import argparse
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--vocab", type=int, default=4096)
ap.add_argument("--out", default="tok.json")
a = ap.parse_args()

tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=a.vocab, special_tokens=["<|endoftext|>"],
                              initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
tok.train([a.data], trainer)
tok.save(a.out)
print("vocab:", tok.get_vocab_size(), "saved to", a.out)
