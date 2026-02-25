import pandas as pd

# Load the dataset data/utt_{train,test,valid}.tsv
#    Columns: uttid, speakerid, text
df_train = pd.read_csv('data/utt_train.tsv', sep='\t', header=None)
df_test = pd.read_csv('data/utt_test.tsv', sep='\t', header=None)
df_valid = pd.read_csv('data/utt_valid.tsv', sep='\t', header=None)

# Check if text is empty or very small
print("Checking dataset for very short or empty texts len(text) < 5")
for split in ['train', 'test', 'valid']:
    if split == 'train':
        df = df_train
    elif split == 'test':
        df = df_test
    else:
        df = df_valid
    df.columns = ['uttid', 'speakerid', 'text']
    print(f"Checking {split} split:")
    print(df[df['text'].isnull() | (df['text'].str.len() < 5)])
    print(f"\n{'= '*10}")


# Print the number of unique speakers
print("Number of unique speakers")
for split in ['train', 'test', 'valid']:
    if split == 'train':
        df = df_train
    elif split == 'test':
        df = df_test
    else:
        df = df_valid
    df.columns = ['uttid', 'speakerid', 'text']
    print(f"Number of unique speakers in {split} split: {df['speakerid'].nunique()}")
    print(f"\n{'= '*10}")

# Print the unique utterances
print("Unique utterances by text")
for split in ['train', 'test', 'valid']:
    if split == 'train':
        df = df_train
    elif split == 'test':
        df = df_test
    else:
        df = df_valid
    df.columns = ['uttid', 'speakerid', 'text']
    print(f"Number of unique utterances by text in {split} split: {df['text'].nunique()}")
    print(f"\n{'= '*10}")


# Print the speaker wise utterances
print("Speaker wise utterances")
for split in ['train', 'test', 'valid']:
    if split == 'train':
        df = df_train
    elif split == 'test':
        df = df_test
    else:
        df = df_valid
    df.columns = ['uttid', 'speakerid', 'text']
    print(f"Speaker wise utterances in {split} split:")
    print(df.groupby('speakerid')['text'].count())
    print(f"\n{'= '*10}")


# print the word length distribution
#  Length of words     number of text
#      1                    10
for split in ['train', 'test', 'valid']:
    if split == 'train':
        df = df_train
    elif split == 'test':
        df = df_test
    else:
        df = df_valid
    df.columns = ['uttid', 'speakerid', 'text']
    print(f"Word length distribution in {split} split:")
    print(df['text'].str.split().str.len().value_counts())
    print(f"\n{'= '*10}")
