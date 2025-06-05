import os
import pandas as pd
from nltk.probability import FreqDist

lengths = []

for prompt in range(1, 9):

    for split in ('train', 'validation', 'test'):

        df = pd.read_csv(os.path.join(str(prompt), split + '_scaled.csv'))
        # print(prompt, split, df.domain1_score.min(), df.domain1_score.max())
        df['len'] = df.full_text.apply(lambda row: len(str.split(row)))
        lengths = lengths + list(df.len)

        if split == 'validation':
            print(len(df))

fd = FreqDist(lengths)
lengths = list(fd.keys())
lengths.sort()

under = 0
over = 0
thresh = 1024
for length in lengths:

    if length <= thresh:
        under += fd[length]
    else:
        over += fd[length]

print(under, over)

