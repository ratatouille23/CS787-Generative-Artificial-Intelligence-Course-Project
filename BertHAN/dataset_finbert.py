from datetime import date, datetime, timedelta
import json
import numpy as np
import os
import pandas as pd
from transformers import AutoTokenizer
from pathlib import Path
import config

args = config.args

MAX_STOCKS = None
RANDOM_SAMPLE = True

USE_CUSTOM_DATES = False
CUSTOM_START = '2021-01-01'
CUSTOM_END = '2021-12-31'

SPECIFIC_STOCKS = None

print("=" * 70)
print("FINBERT-HAN DATASET PREPROCESSING")
print("=" * 70)

filenames = os.listdir(args.data_dir)
stock_name_price = set([filename.split('.')[0] for filename in filenames])
stock_name_news = set(os.listdir(args.news_dir))
all_stocks = set.intersection(stock_name_news, stock_name_price)

print(f"\nTotal stocks available: {len(all_stocks)}")

if SPECIFIC_STOCKS is not None:
    stock_names = [s for s in SPECIFIC_STOCKS if s in all_stocks]
    print(f"Using {len(stock_names)} specified stocks: {stock_names}")
elif MAX_STOCKS is not None:
    stock_list = list(all_stocks)
    if RANDOM_SAMPLE:
        import random

        random.seed(args.seed)
        stock_names = random.sample(stock_list, min(MAX_STOCKS, len(stock_list)))
        print(f"Randomly sampled {len(stock_names)} stocks")
    else:
        stock_names = stock_list[:MAX_STOCKS]
        print(f"Using first {len(stock_names)} stocks")
    print(f"Selected stocks: {stock_names}")
else:
    stock_names = list(all_stocks)
    print(f"Using all {len(stock_names)} stocks")


if USE_CUSTOM_DATES:
    start = datetime.strptime(CUSTOM_START, '%Y-%m-%d')
    end = datetime.strptime(CUSTOM_END, '%Y-%m-%d')
    print(f"\nUsing custom date range: {CUSTOM_START} to {CUSTOM_END}")
else:
    start = datetime.strptime(args.train_start_date, '%Y-%m-%d')
    end = datetime.strptime(args.test_end_date, '%Y-%m-%d')
    print(f"\nUsing config date range: {args.train_start_date} to {args.test_end_date}")

date_list = [start + timedelta(days=i) for i in range((end - start).days + 1)]
print(f"Total days: {len(date_list)}")

print("\nLoading price data...")
y = pd.DataFrame(index=date_list, columns=list(stock_names))

for filename in filenames:
    stock_name = filename.split(".")[0]
    if stock_name not in stock_names:
        continue

    filepath = args.data_dir + filename
    df = pd.read_csv(filepath, header=None, index_col=0, parse_dates=True, sep='\t')
    for index, move_per in zip(df.index, df[1]):
        if index in y.index:
            y[stock_name][index] = move_per

print("Price data loaded")

print("Creating labels...")
y[(-0.005 <= y) & (y <= 0.0055)] = float('nan')
y[y > 0.0055] = 1
y[y < -0.005] = 0

total_labels = 0
up_count = 0
down_count = 0
for stock in stock_names:
    stock_labels = y[stock].dropna()
    total_labels += len(stock_labels)
    up_count += (stock_labels == 1).sum()
    down_count += (stock_labels == 0).sum()

print(f"Total labels: {total_labels}")
print(f"  Up (1): {up_count} ({100 * up_count / total_labels:.1f}%)")
print(f"  Down (0): {down_count} ({100 * down_count / total_labels:.1f}%)")

print(f"\nLoading FinBERT tokenizer from {args.modelpath}...")
BERT_tokenizer = AutoTokenizer.from_pretrained(args.modelpath)
print("Tokenizer loaded")

print("\nTokenizing news data...")
news_data = dict()

for idx, stock_name in enumerate(stock_names):
    print(f"  [{idx + 1}/{len(stock_names)}] Processing {stock_name}...")

    stock_news_dir = args.news_dir + stock_name
    if not os.path.exists(stock_news_dir):
        print(f"    Warning: No news directory for {stock_name}")
        continue

    file_names = os.listdir(stock_news_dir)

    for file_name in file_names:
        file_path = stock_news_dir + '/' + file_name
        key = stock_name + ' + ' + file_name

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                jsons = [json.loads(line) for line in lines]

                text_data = []
                for i in range(args.max_num_tweets_len):
                    if i < len(jsons):
                        text = ' '.join(jsons[i]['text']) if isinstance(jsons[i]['text'], list) else jsons[i]['text']
                        text_data.append(text)
                    else:
                        text_data.append('')

                tokens = BERT_tokenizer(
                    text_data,
                    max_length=args.max_num_tokens_len,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )

                news_data[key] = {
                    'input_ids': tokens['input_ids'].numpy(),
                    'token_type_ids': tokens['token_type_ids'].numpy(),
                    'attention_mask': tokens['attention_mask'].numpy()
                }
        except Exception as e:
            print(f" Error processing {file_path}: {e}")
            continue

print(f"Tokenized {len(news_data)} news files")

print("\nCreating train/dev/test splits...")
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
num_processed = 0

for stock_idx, stock_name in enumerate(stock_names):
    print(f"  [{stock_idx + 1}/{len(stock_names)}] Building samples for {stock_name}...")

    for target_date in date_list:
        if target_date not in y.index or y[stock_name][target_date] not in (0, 1):
            continue

        sample = np.zeros((args.days, args.max_num_tweets_len, 3, args.max_num_tokens_len), dtype=np.int32)

        num_no_news_days = 0
        for lag in range(args.days + 1, 1, -1):
            news_date = target_date - timedelta(days=lag)
            key = stock_name + ' + ' + str(news_date.date())

            if key in news_data:
                news_ids = news_data[key]
                sample[args.days - lag, :, 0, :] = news_ids['input_ids']
                sample[args.days - lag, :, 1, :] = news_ids['token_type_ids']
                sample[args.days - lag, :, 2, :] = news_ids['attention_mask']
            else:
                num_no_news_days += 1
                if num_no_news_days > 1:
                    break

        if num_no_news_days > 1:
            num_filtered_samples += 1
            continue

        label = int(y[stock_name][target_date])
        num_processed += 1

        if train_start_date <= target_date <= train_end_date:
            train_x.append(sample.flatten())
            train_y.append(label)
        elif dev_start_date <= target_date <= dev_end_date:
            dev_x.append(sample.flatten())
            dev_y.append(label)
        elif test_start_date <= target_date <= test_end_date:
            test_x.append(sample.flatten())
            test_y.append(label)

train_x = np.array(train_x) if len(train_x) > 0 else np.array([]).reshape(0, -1)
train_y = np.array(train_y) if len(train_y) > 0 else np.array([])
dev_x = np.array(dev_x) if len(dev_x) > 0 else np.array([]).reshape(0, -1)
dev_y = np.array(dev_y) if len(dev_y) > 0 else np.array([])
test_x = np.array(test_x) if len(test_x) > 0 else np.array([]).reshape(0, -1)
test_y = np.array(test_y) if len(test_y) > 0 else np.array([])

print(f"\nDataset Statistics:")
print(f"  Total samples processed: {num_processed}")
print(f"  Filtered (missing news): {num_filtered_samples}")
print(f"\n  Train: {len(train_y)} samples")
if len(train_y) > 0:
    print(f"    Up: {np.sum(train_y == 1)}, Down: {np.sum(train_y == 0)}")
print(f"  Dev: {len(dev_y)} samples")
if len(dev_y) > 0:
    print(f"    Up: {np.sum(dev_y == 1)}, Down: {np.sum(dev_y == 0)}")
print(f"  Test: {len(test_y)} samples")
if len(test_y) > 0:
    print(f"    Up: {np.sum(test_y == 1)}, Down: {np.sum(test_y == 0)}")

save_path = args.dataset_save_dir
Path(save_path).mkdir(parents=True, exist_ok=True)

print(f"\nSaving datasets to {save_path}...")
np.savetxt(save_path + 'train_x.csv', train_x, delimiter=',', fmt='%d')
np.savetxt(save_path + 'train_y.csv', train_y, delimiter=',', fmt='%d')
np.savetxt(save_path + 'dev_x.csv', dev_x, delimiter=',', fmt='%d')
np.savetxt(save_path + 'dev_y.csv', dev_y, delimiter=',', fmt='%d')
np.savetxt(save_path + 'test_x.csv', test_x, delimiter=',', fmt='%d')
np.savetxt(save_path + 'test_y.csv', test_y, delimiter=',', fmt='%d')

metadata = {
    'num_stocks': len(stock_names),
    'stocks': stock_names,
    'days': args.days,
    'max_tweets': args.max_num_tweets_len,
    'max_tokens': args.max_num_tokens_len,
    'train_samples': len(train_y),
    'dev_samples': len(dev_y),
    'test_samples': len(test_y),
}

import pickle

with open(save_path + 'metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE!")
print("=" * 70)
print(f"\nFiles saved in {save_path}")
print("  - train_x.csv, train_y.csv")
print("  - dev_x.csv, dev_y.csv")
print("  - test_x.csv, test_y.csv")
print("  - metadata.pkl")
print(f"\nReady for training with FinBERT-HAN!")
print(f"Run: python simple_train.py")