import word2vec
import os
import sys

word2vec_file = sys.argv[1]
model = word2vec.load(word2vec_file)

output_file = os.path.join(os.path.dirname(word2vec_file), 'vocab.txt')
output = open(output_file, 'w')
output.write('\n'.join(model.vocab))
output.close()
