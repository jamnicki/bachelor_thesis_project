from spacy.lang.pl import Polish
from spacy.util import minibatch, fix_random_seed
from spacy.training import Corpus
from spacy import displacy

from pathlib import Path
from jsonlines import jsonlines
from collections import defaultdict
from tqdm import tqdm
import random
from time import time as etime
from datetime import datetime as dt

from _temp_query_strategies import query_random

import logging
logging.basicConfig(level=logging.DEBUG)


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


def _wait_for_annotations(timeout=300):
    """Wait for user to annotate the data"""
    msg = "Annotate the data then press [Enter] to continue or [q] to quit."
    try:
        _input = input(msg)
    except KeyboardInterrupt:
        if "q" in _input.lower():
            return None


def stop_criteria(iteration, max_iter, queried, set_len):
    """Return True if stop criteria is met, False otherwise. Log the reason."""
    if iteration > max_iter > 0:
        logging.warning("Stopped by max iterations")
        return True
    if data_exhausted(queried, set_len):
        logging.warning("Stopped by data exhaustion")
        return True
    return False


def _query(func, _labels_queried, _spans_queried, _n_instances, _spans_key,
           _queried, **kwargs_dict):
    """Query the data using given function and keyword arguments.
       Update variables in place
    """
    logging.debug(f"Querying {_n_instances} instances...")
    q_indexes = set()
    q_data = []
    for q_idx, q_example in func(**kwargs_dict):
        q_doc_annotation = q_example.to_dict()["doc_annotation"]
        component_spans = q_doc_annotation["spans"][_spans_key]
        for span in component_spans:
            # !! empty 'kd_id' field
            span_label = span[2]
            _labels_queried[span_label] += 1
        q_indexes.add(q_idx)
        q_data.append(q_example)
        _spans_queried["count"] += len(component_spans)
    _queried.update(q_indexes)
    return q_indexes, q_data


def main():
    """pseudocode
    Input: Unlabeled dataset Du, Base model Mb, Acquisition Batch Size B,
           Strategy S, Labeling budget L
    Output: Labeled dataset Dl, Trained model Mt

    Query oracle for initial seed dataset Ds from Du (not RANDOM!!)
    Let Dl = Ds
    Mt = Train(Mb, Dl)
    while |Dl| < L do
        Ds = SelectInformativeSamples(Du , Mt , S, B)
        D's = Pre-tag(Ds, Mt)
        D''s = Query oracle for labels to D's
        Move new labeled instances D''s from Du to Dl
        Mt = Train(Mb, Dl)
    return Dl, Mt
    """

    NAME = "test_active_learned_kpwr-short"

    _start_etime_str = str(etime()).replace(".", "f")
    DATA_DIR = Path("data")
    MODELS_DIR = Path("models") / Path("scripts")
    LOGS_DIR = Path("logs") / Path("scripts")
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    TRAIN_DB = DATA_DIR / Path("inzynierka-kpwr-train-3.spacy")
    TEST_DB = DATA_DIR / Path("inzynierka-kpwr-test-3.spacy")
    MODEL_OUT = MODELS_DIR / Path(f"{NAME}__{_start_etime_str}.spacy")
    METRICS_OUT = LOGS_DIR / Path(f"{NAME}__{_start_etime_str}.metrics.jsonl")

    SEED = 42
    MAX_ITER = 10
    N_INSTANCES = 10
    TRAIN_BATCH_SIZE = int(N_INSTANCES // 5)  # default 1000?
    TEST_BATCH_SIZE = int(N_INSTANCES // 5)  # default 1000?
    LABELS = ["nam_liv_person", "nam_loc_gpe_city", "nam_loc_gpe_country"]
    COMPONENT = "spancat"
    SPANS_KEY = "sc"

    random.seed(SEED)
    fix_random_seed(SEED)
    assert not MODEL_OUT.exists()

    nlp = init_nlp(LABELS)

    train_data, test_data = load_data(nlp, TRAIN_DB, TEST_DB)
    train_len = len(train_data)

    # Training loop
    iteration = 1
    spans_queried = {"count": 0}
    queried = set()
    labels_queried = defaultdict(int)
    enlarged_train_data = []
    pbar = tqdm(total=MAX_ITER)  # TODO: wrong! data could be exhausted before
    while True:
        it_t0 = etime()
        pbar.update(1)
        datetime_str = dt.now().strftime("%d-%m-%Y %H:%M:%S")

        if stop_criteria(iteration, MAX_ITER, queried, train_len):
            break

        q_func = query_random
        func_kwargs = {
            "records": train_data,
            "exclude": queried,
            "n_instances": N_INSTANCES
        }
        _, q_data = _query(
            func=q_func,
            _labels_queried=labels_queried,
            _spans_queried=spans_queried,
            _n_instances=N_INSTANCES,
            _spans_key=SPANS_KEY,
            _queried=queried,
            **func_kwargs
        )

        # Extend the training dataset
        enlarged_train_data.extend(q_data)

        # Update the model with queried data
        logging.debug("Updating the model with queried data...")
        losses = {}
        optimizer = nlp.initialize()
        with nlp.select_pipes(enable=COMPONENT):
            for batch in minibatch(enlarged_train_data, TRAIN_BATCH_SIZE):
                nlp.update(batch,
                           losses=losses,
                           sgd=optimizer)
        sc_loss = losses[COMPONENT]

        # Evaluate the model on the test set
        logging.debug("Evaluating the model on the test set...")
        with nlp.select_pipes(enable=COMPONENT):
            eval_metrics = nlp.evaluate(test_data, batch_size=TEST_BATCH_SIZE)

        iteration_time = etime() - it_t0

        results = {
            "_date": datetime_str,
            "_iteration": iteration,
            "_iteration_time": iteration_time,
            "_spans_count": spans_queried["count"],
            "_labels_count": labels_queried,
            "_sc_loss": sc_loss
        }
        results.update(eval_metrics)
        log_results(results,
                    out=METRICS_OUT)

        iteration += 1

    # Render the model's sample prediction
    rand_indx = random.randint(0, len(test_data))
    text = test_data[rand_indx].text
    pred = nlp(text)

    logging.info("Rendering the model's sample prediction...")
    if pred.spans[SPANS_KEY]:
        render_spans(pred, SPANS_KEY)
    else:
        logging.warning("Nothing to render! No spans found.")

    # Save the model to binary file
    logging.info(f"Saving model to {MODEL_OUT}...")
    nlp.to_disk(MODEL_OUT)


if __name__ == "__main__":
    main()