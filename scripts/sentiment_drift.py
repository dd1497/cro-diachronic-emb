import json, argparse
import numpy as np

from copy import deepcopy

from util import load_vectors, load_jsonl
from align import smart_procrustes_align_gensim

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

DATASET_NAME_TO_PATH = {
  'pauza': '../data/pauza_clean_tokens.jsonl',
}

def make_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data", default='pauza', type=str)
  parser.add_argument("--log_dir", default='../output/semantic_drift.txt', type=str)
  parser.add_argument("--test_size", default=0.2, type=float)
  parser.add_argument("--split", default=5, type=int)
  parser.add_argument("--seed", default=42, type=int)
  return parser.parse_args()

def do_eval(regressor, X_test, y_test):
    # Predict
    y_pred = regressor.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    aps = np.mean(y_pred) # Average predicted sentiment

    print(f"Test Mean Squared Error: {mse:.4f}")
    print(f"Test average score: {np.mean(aps):.2f}")

    return mse, aps

def compute_drift(args, split, base, target):
  args = make_parser()
  dataset = load_jsonl(DATASET_NAME_TO_PATH[args.data])

  vectors_root = f"/shared/lovorka/internal/retriever/data/{split}ysplits_models"
  base_vecs = load_vectors(f"{vectors_root}/diachronic_wave_{base}_processed_model.bin")
  target_vecs = load_vectors(f"{vectors_root}/diachronic_wave_{target}_processed_model.bin")


  # Copy base because alignment overwrites some attributes
  target_align_base = smart_procrustes_align_gensim(deepcopy(base_vecs), target_vecs)

  # Filter out instances where we don't find any tokens
  clean_dataset = [instance for instance in dataset if instance.aggregate_embeddings(base_vecs) is not None]
  clean_dataset = [instance for instance in clean_dataset if instance.aggregate_embeddings(target_vecs) is not None]

  # Average word vectors
  X = clean_dataset
  y = np.array([float(instance.rating) for instance in clean_dataset])

  # Train/test split
  X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)

  X_train_base = [instance.aggregate_embeddings(base_vecs) for instance in X_train_raw]

  X_test_base = [instance.aggregate_embeddings(base_vecs) for instance in X_test_raw]

  # Create and train the regressor
  regressor = LinearRegression()
  regressor.fit(X_train_base, y_train)

  # Predict
  mse_base, aps_base = do_eval(regressor, X_test_base, y_test)

  # Predict on aligned split
  X_test_target = [instance.aggregate_embeddings(target_align_base) for instance in X_test_raw]
  mse_target, aps_target = do_eval(regressor, X_test_target, y_test)
  N_train = len(X_train_raw)
  N_test = len(X_test_raw)

  return mse_base, aps_base, mse_target, aps_target, N_train, N_test

def main():
  args = make_parser()
  split = args.split # 5y period

  if split == 2:
    lo = 0
    hi = 12
  else:
    lo = 1
    hi = 5

  log_dir = args.log_dir

  with open(log_dir, 'w') as outfile:
    outfile.write(f"dataset,seed,split,base,target,test_size,mse_base,aps_base,mse_target,aps_target,N_train,N_test\n")

    for base in range(lo,hi+1):
      for target in range(lo, hi+1):
        if base == target: continue
        print(f"Running for {split}, {target} => {base}")

        mse_base, aps_base, mse_target, aps_target, N_train, N_test = compute_drift(args, split, base, target)
        log_data = [args.data, args.seed, split, base, target, args.test_size, mse_base, aps_base, mse_target, aps_target, N_train, N_test]
        log_data = [str(e) for e in log_data]
        outfile.write(",".join(log_data) + "\n")

if __name__ == '__main__':
  main()