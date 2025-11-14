from datetime import datetime, timedelta
import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
import re
from collections import Counter
import original_config as config

args = config.args

MAX_STOCKS = 20

USE_CUSTOM_DATE_RANGE = False
CUSTOM_START_DATE = '2020-01-01'
CUSTOM_END_DATE = '2021-12-31'

RANDOM_SAMPLE_STOCKS = True

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    text = ' '.join(text.split())
    return text


def tokenize_simple(text):
    return text.split()


def build_vocab_from_texts(all_texts, min_freq=2, max_vocab_size=10000):
    print("Building vocabulary...")

    word_freq = Counter()
    for text in all_texts:
        words = tokenize_simple(clean_text(text))
        word_freq.update(words)

    common_words = [word for word, freq in word_freq.most_common(max_vocab_size)
                    if freq >= min_freq]

    word2idx = {'<PAD>': 0, '<UNK>': 1}
    for idx, word in enumerate(common_words, start=2):
        word2idx[word] = idx

    idx2word = {idx: word for word, idx in word2idx.items()}

    print(f"Vocabulary size: {len(word2idx)}")
    print(f"Most common words: {common_words[:10]}")

    return word2idx, idx2word


def text_to_indices(text, word2idx, max_length):
    words = tokenize_simple(clean_text(text))[:max_length]
    indices = [word2idx.get(word, word2idx['<UNK>']) for word in words]

    if len(indices) < max_length:
        indices += [word2idx['<PAD>']] * (max_length - len(indices))

    return indices

filenames = os.listdir(args.data_dir)
stock_name_price = set([filename.split('.')[0] for filename in filenames])
stock_name_news = set(os.listdir(args.news_dir))
stock_names = list(set.intersection(stock_name_news, stock_name_price))

print(f"\nTotal stocks available: {len(stock_names)}")

if MAX_STOCKS is not None and MAX_STOCKS < len(stock_names):
    if RANDOM_SAMPLE_STOCKS:
        import random

        random.seed(args.seed)
        stock_names = random.sample(stock_names, MAX_STOCKS)
        print(f"Randomly sampled {MAX_STOCKS} stocks")
    else:
        stock_names = stock_names[:MAX_STOCKS]
        print(f"Using first {MAX_STOCKS} stocks")
else:
    print(f"Using all {len(stock_names)} stocks")

print(f"Selected stocks: {stock_names}")

if USE_CUSTOM_DATE_RANGE:
    start = datetime.strptime(CUSTOM_START_DATE, '%Y-%m-%d')
    end = datetime.strptime(CUSTOM_END_DATE, '%Y-%m-%d')
    print(f"\nUsing custom date range: {CUSTOM_START_DATE} to {CUSTOM_END_DATE}")
else:
    start = datetime.strptime(args.train_start_date, '%Y-%m-%d')
    end = datetime.strptime(args.test_end_date, '%Y-%m-%d')
    print(f"\nUsing full date range: {args.train_start_date} to {args.test_end_date}")

date_list = [start + timedelta(days=i) for i in range((end - start).days + 1)]
print(f"Total days: {len(date_list)}")

y = pd.DataFrame(index=date_list, columns=list(stock_names))

print("\nLoading price data...")
for filename in filenames:
    stock_name = filename.split(".")[0]
    if stock_name not in stock_names:
        continue

    filepath = args.data_dir + filename
    df = pd.read_csv(filepath, header=None, index_col=0, parse_dates=True, sep='\t')
    for index, move_per in zip(df.index, df[1]):
        if index in y.index:
            y[stock_name][index] = move_per

y[(-0.005 <= y) & (y <= 0.0055)] = float('nan')
y[y > 0.0055] = 1
y[y < -0.005] = 0

all_texts = []
news_data = {}

for stock_name in stock_names:
    file_names = os.listdir(args.news_dir + stock_name)

    for file_name in file_names:
        file_path = args.news_dir + stock_name + '/' + file_name
        key = stock_name + ' + ' + file_name

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            jsons = [json.loads(line) for line in lines]

            texts = []
            for i in range(min(len(jsons), args.max_num_tweets_len)):
                text = ' '.join(jsons[i]['text']) if isinstance(jsons[i]['text'], list) else jsons[i]['text']
                texts.append(text)
                all_texts.append(text)
            news_data[key] = texts


word2idx, idx2word = build_vocab_from_texts(all_texts, min_freq=2, max_vocab_size=10000)
vocab_size = len(word2idx)

vocab_path = args.dataset_save_dir
Path(vocab_path).mkdir(parents=True, exist_ok=True)
with open(vocab_path + 'vocab.pkl', 'wb') as f:
    pickle.dump({'word2idx': word2idx, 'idx2word': idx2word}, f)
print(f"\nVocabulary saved to {vocab_path}vocab.pkl")

print("\nTokenizing news data...")
tokenized_news = {}
for key, texts in news_data.items():
    tokenized_texts = []
    for text in texts:
        indices = text_to_indices(text, word2idx, args.max_num_tokens_len)
        tokenized_texts.append(indices)

    while len(tokenized_texts) < args.max_num_tweets_len:
        tokenized_texts.append([word2idx['<PAD>']] * args.max_num_tokens_len)

    tokenized_news[key] = np.array(tokenized_texts)

train_x = []
train_y = []
dev_x = []
dev_y = []
test_x = []
test_y = []

train_start_date = datetime.strptime(args.train_start_date, '%Y-%m-%d')
train_end_date = datetime.strptime(args.train_end_date, '%Y-%m-%d')
dev_start_date = datetime.strptime(args.dev_start_date, '%Y-%m-%d')
dev_end_date = datetime.strptime(args.dev_end_date, '%Y-%m-%d')
test_start_date = datetime.strptime(args.test_start_date, '%Y-%m-%d')
test_end_date = datetime.strptime(args.test_end_date, '%Y-%m-%d')

num_filtered_samples = 0
num_samples = 0

for stock_name in stock_names:

    for target_date in date_list:
        if target_date not in y.index or y[stock_name][target_date] not in (0, 1):
            continue

        sample = np.zeros((args.days, args.max_num_tweets_len, args.max_num_tokens_len), dtype=np.int32)

        num_no_news_days = 0
        for lag in range(args.days + 1, 1, -1):
            news_date = target_date - timedelta(days=lag)
            key = stock_name + ' + ' + str(news_date.date())

            if key in tokenized_news:
                sample[args.days - lag, :, :] = tokenized_news[key]
            else:
                num_no_news_days += 1
                if num_no_news_days > 1:
                    break

        if num_no_news_days > 1:
            num_filtered_samples += 1
            continue

        label = int(y[stock_name][target_date])
        num_samples += 1

        if train_start_date <= target_date <= train_end_date:
            train_x.append(sample.flatten())
            train_y.append(label)
        elif dev_start_date <= target_date <= dev_end_date:
            dev_x.append(sample.flatten())
            dev_y.append(label)
        elif test_start_date <= target_date <= test_end_date:
            test_x.append(sample.flatten())
            test_y.append(label)

train_x = np.array(train_x)
train_y = np.array(train_y)
dev_x = np.array(dev_x)
dev_y = np.array(dev_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

print(f"\nDataset Statistics:")
print(f"  Total samples: {num_samples}")
print(f"  Filtered samples: {num_filtered_samples}")
print(f"  Train: {len(train_y)} samples")
print(f"  Dev: {len(dev_y)} samples")
print(f"  Test: {len(test_y)} samples")
print(f"\nLabel distribution:")
print(f"  Train - Up: {np.sum(train_y == 1)}, Down: {np.sum(train_y == 0)}")
print(f"  Dev - Up: {np.sum(dev_y == 1)}, Down: {np.sum(dev_y == 0)}")
print(f"  Test - Up: {np.sum(test_y == 1)}, Down: {np.sum(test_y == 0)}")

print(f"\nSaving datasets to {vocab_path}...")
np.savetxt(vocab_path + 'train_x.csv', train_x, delimiter=',', fmt='%d')
np.savetxt(vocab_path + 'train_y.csv', train_y, delimiter=',', fmt='%d')
np.savetxt(vocab_path + 'dev_x.csv', dev_x, delimiter=',', fmt='%d')
np.savetxt(vocab_path + 'dev_y.csv', dev_y, delimiter=',', fmt='%d')
np.savetxt(vocab_path + 'test_x.csv', test_x, delimiter=',', fmt='%d')
np.savetxt(vocab_path + 'test_y.csv', test_y, delimiter=',', fmt='%d')

metadata = {
    'vocab_size': vocab_size,
    'num_stocks': len(stock_names),
    'stocks': stock_names,
    'days': args.days,
    'max_sentences': args.max_num_tweets_len,
    'max_words': args.max_num_tokens_len,
    'train_samples': len(train_y),
    'dev_samples': len(dev_y),
    'test_samples': len(test_y),
}

with open(vocab_path + 'metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE!")
print("=" * 60)
print(f"\nFiles created in {vocab_path}:")
print("  - train_x.csv, train_y.csv")
print("  - dev_x.csv, dev_y.csv")
print("  - test_x.csv, test_y.csv")
print("  - vocab.pkl (vocabulary)")
print("  - metadata.pkl (dataset info)")
print(f"\nVocabulary size: {vocab_size}")
print(f"Ready for training with original HAN model!")