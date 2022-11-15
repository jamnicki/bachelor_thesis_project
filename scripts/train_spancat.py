from spacy.lang.pl import Polish
from spacy.util import minibatch, fix_random_seed
from spacy.training import Corpus
from spacy import displacy

from pathlib import Path
from jsonlines import jsonlines
from collections import defaultdict
from random import randint
from tqdm import tqdm
from time import time as etime
from datetime import datetime as dt

import logging
logging.basicConfig(level=logging.WARNING)


def query_random(records, exclude, n_instances):
    """Random query strategy"""
    n_queried = 0
    max_idx = len(records) - 1
    while n_queried < n_instances:
        idx = randint(0, max_idx)
        if idx not in exclude:
            exclude.add(idx)
            n_queried += 1
            yield idx, records[idx]


def log_results(results, out):
    """Log results to a file"""
    with jsonlines.open(out, mode="a") as writer:
        logging.info(f"Writing results to {out}")
        writer.write(results)


def init_nlp(labels):
    logging.info("Initializing spaCy model...")
    nlp = Polish()
    spancat = nlp.add_pipe("spancat")
    for label in labels:
        spancat.add_label(label)
    return nlp


def load_data(nlp, train_docbin_path, test_docbin_path):
    train_corpus = Corpus(train_docbin_path)
    test_corpus = Corpus(test_docbin_path)
    train_data = tuple(train_corpus(nlp))
    test_data = tuple(test_corpus(nlp))
    return train_data, test_data


def render_spans(prediction, spans_key):
    colors = {"nam_liv_person": "RGB(87, 151, 255)",
              "nam_loc_gpe_city": "RGB(64, 219, 64)",
              "nam_loc_gpe_country": "RGB(235, 159, 45)"}
    displacy.render(prediction,
                    style="span",
                    options={"colors": colors,
                             "spans_key": spans_key})


def data_exhausted(queried_idxs, set_len):
    return len(queried_idxs) >= set_len


def main():
    NAME = "random_lg_full"

    _start_etime_str = str(etime()).replace(".", "f")
    DATA_DIR = Path("data")
    TRAIN_DB = DATA_DIR / Path("inzynierka-kpwr-train-3.spacy")
    TEST_DB = DATA_DIR / Path("inzynierka-kpwr-test-3.spacy")
    LOGS_DIR = Path("logs")
    MODELS_DIR = Path("models")
    MODELS_DIR.mkdir(exist_ok=True)
    MODEL_OUT = MODELS_DIR / Path(f"{NAME}__{_start_etime_str}.spacy")
    METRICS_OUT = LOGS_DIR / Path(f"{NAME}__{_start_etime_str}.metrics.jsonl")

    SEED = 42
    MAX_ITER = 0
    N_INSTANCES = 50
    TRAIN_BATCH_SIZE = int(N_INSTANCES // 5)  # default 1000?
    TEST_BATCH_SIZE = int(N_INSTANCES // 5)  # default 1000?
    LABELS = ["nam_liv_person", "nam_loc_gpe_city", "nam_loc_gpe_country"]
    COMPONENT = "spancat"
    SPANS_KEY = "sc"

    fix_random_seed(SEED)
    assert not MODEL_OUT.exists()

    nlp = init_nlp(LABELS)

    train_data, test_data = load_data(nlp, TRAIN_DB, TEST_DB)
    train_len = len(train_data)

    # Training loop
    iteration = 1
    spans_queried = 0
    spans_num_history = []
    queried = set()
    labels_queried = defaultdict(int)
    _extended_train_data = []
    pbar = tqdm(total=MAX_ITER)  # TODO: wrong! data could be exhausted before
    while True:
        # Stopping criteria
        if iteration > MAX_ITER > 0:
            logging.warning("Stopped by max iterations")
            break
        if data_exhausted(queried, train_len):
            logging.warning("Stopped by data exhaustion")
            break

        datetime_str = dt.now().strftime("%d-%m-%Y %H:%M:%S")

        # Query the data
        logging.debug(f"Querying {N_INSTANCES} instances...")
        q_indexes = set()
        q_data = []
        for q_idx, q_doc in query_random(train_data, queried, N_INSTANCES):
            q_doc_annotation = q_doc.to_dict()["doc_annotation"]
            component_spans = q_doc_annotation["spans"][SPANS_KEY]
            for span in component_spans:
                # !! empty 'kd_id' field
                span_label = span[2]
                labels_queried[span_label] += 1
            q_indexes.add(q_idx)
            q_data.append(q_doc)
            spans_queried += len(component_spans)
        queried.update(q_indexes)
        spans_num_history.append(spans_queried)

        # Extend the training dataset
        _extended_train_data.extend(q_data)

        # Update the model with queried data
        logging.debug("Updating the model with queried data...")
        losses = {}
        with nlp.select_pipes(enable=COMPONENT):
            optimizer = nlp.initialize()
            for batch in minibatch(_extended_train_data, TRAIN_BATCH_SIZE):
                losses = nlp.update(batch, losses=losses, sgd=optimizer)
        sc_loss = losses[COMPONENT]

        # Evaluate the model on the test set
        logging.debug("Evaluating the model on the test set...")
        with nlp.select_pipes(enable=COMPONENT):
            eval_metrics = nlp.evaluate(test_data, batch_size=TEST_BATCH_SIZE)

        results = {
            "_date": datetime_str,
            "_iteration": iteration,
            "_spans_count": spans_queried,
            "_labels_count": labels_queried,
            "_sc_loss": sc_loss
        }
        results.update(eval_metrics)

        log_results(results,
                    out=METRICS_OUT)
        iteration += 1
        pbar.update(1)
    pbar.close()

    # Render the model's predictions
    logging.info("Rendering the model's sample prediction...")
    with nlp.select_pipes(enable=COMPONENT):
        pred = nlp(test_data[0].text)
    render_spans(pred, SPANS_KEY)

    # Save the model to binary file
    logging.info(f"Saving model to {MODEL_OUT}...")
    nlp.to_disk(MODEL_OUT)


if __name__ == "__main__":
    main()
    logging.info("Success!")
