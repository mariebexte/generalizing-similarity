import pandas as pd
import os
from nltk.probability import FreqDist

lengths = []
for prompt in range(15):

    for path in ['train_800.csv', 'validation.csv', 'test.csv']:

        df = pd.read_csv(os.path.join(str(prompt), path))
        df['length'] = df.full_text.apply(lambda row: len(row.split(' ')))
        lengths = lengths + list(df.length)

        print(prompt, list(df['prompt_name'])[0], list(df['task'])[0])


fd = FreqDist(lengths)
over = 0
under = 0
for key, value in fd.most_common():


    if key > 1024:
        over += value
        print(key, value)
    else:
        under += value

print('This many fit in BERT', under)
print('This many do not', over)

df_lengths = pd.DataFrame.from_dict(fd, orient='index').reset_index()
df_lengths.columns = ['length', 'frequency']
df_lengths.to_csv('answer_lengths.csv')
print(df_lengths)
