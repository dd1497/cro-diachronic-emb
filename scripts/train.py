import os
import time

import pandas as pd
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

from tqdm import tqdm


import argparse
import logging


class LossLogger(CallbackAny2Vec):
    """Callback to log loss periodically during training."""
    def __init__(self, report_delay=10.0):
        self.epoch = 0
        self.batch = 0
        self.losses = []
        self.last_log_time = time.time()
        self.report_delay = report_delay # Log roughly every N seconds
        self.total_batches = 0 # Approximate total batches seen

    def on_epoch_begin(self, model):
        logging.info(f"Epoch {self.epoch + 1} start.")
        self.batch = 0
        self.last_log_time = time.time() # Reset timer at epoch start

    def on_batch_end(self, model):
        self.batch += 1
        self.total_batches += 1
        current_time = time.time()
        # Log loss if enough time has passed since the last log
        if current_time - self.last_log_time >= self.report_delay:
            # model.running_training_loss holds loss since last internal report
            current_loss = model.running_training_loss
            logging.info(
                f" Epoch {self.epoch + 1}, Batch ~{self.total_batches}: "
                f"Running loss cumulative since last report: {current_loss:.4f}"
            )
            self.last_log_time = current_time

    def on_epoch_end(self, model):
        # Gensim's internal logging usually reports final epoch loss here too
        # if compute_loss=True and INFO level is set.
        # We can explicitly fetch it if needed:
        final_epoch_loss = model.get_latest_training_loss()
        logging.info(f"Epoch {self.epoch + 1} end. Final loss for epoch: {final_epoch_loss:.4f}")
        self.epoch += 1




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--embedding-size", default=300, type=int)
    parser.add_argument("--window-size", default=4, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--min-count", default=1, type=int)
    parser.add_argument("--sg", default=1, type=int)
    parser.add_argument("--negative-samples", default=5, type=int)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--workers", default=24, type=str)
    parser.add_argument("--sample", default=1e-5, type=float)
    parser.add_argument('--alpha', default=0.02, type=float)
    parser.add_argument("--min-alpha", default=0, type=int)
    args = parser.parse_args()

    log_file = args.input.split('/')[-1] + '.log'

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO, filename=log_file)

    logging.info("Loading data...")


    if not os.path.isdir(args.input):
        input_filenames = [args.input]
    else:
        input_filenames = [os.path.join(args.input, f) for f in os.listdir(args.input)]

    logging.info(f"There are {len(input_filenames)} files to process.")

    for input_filename in sorted(input_filenames):
        logging.info(f"Loading {input_filename}...")

        df = pd.read_parquet(input_filename, engine="pyarrow")
        sentences = [sentence for article in tqdm(df["lemma_pos"]) for sentence in article]
        del df
        sentences = [[i.lower() for i in sentence if '_punct' not in i] for sentence in tqdm(sentences)]


        model = Word2Vec(
            vector_size=args.embedding_size,
            window=args.window_size,
            min_count=args.min_count,
            workers=args.workers,
            sg=args.sg,
            negative=args.negative_samples,
            sample=args.sample,
            alpha=args.alpha,
            epochs=args.epochs,
            compute_loss=True,
            min_alpha=args.min_alpha
        )

        model.build_vocab(sentences)

        for epoch in range(1, args.epochs + 1):
            # Calculate learning rate for this epoch (optional, Gensim does this internally too)
            # alpha_update = start_alpha - (start_alpha - end_alpha) * (epoch - 1) / EPOCHS
            # model.alpha = alpha_update # Update alpha manually if needed, but train() handles it

            logging.info(f"Starting Epoch {epoch}/{args.epochs}")
            # Train for one epoch
            # compute_loss=True calculates loss for this epoch's training pass
            model.train(
                sentences,
                total_examples=model.corpus_count,
                epochs=1,  # Train exactly one epoch in this call
                compute_loss=True,
                start_alpha=model.alpha,  # Pass current alpha if needed
                end_alpha=model.min_alpha  # Pass final alpha if needed
            )



            # Get the loss computed during the last call to train()
            current_loss = model.get_latest_training_loss()
            logging.info(f"Epoch {epoch} finished. Loss: {current_loss:.4f}")


        if not os.path.exists(args.output):
            os.makedirs(args.output)

        input_filename = input_filename.split('/')[-1]

        input_filename = input_filename.split('.')[0]
        model.save(os.path.join(args.output, f"{input_filename}_model.bin"))
