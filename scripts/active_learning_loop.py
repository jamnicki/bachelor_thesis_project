from spacy.lang.pl import Polish
from spacy.util import minibatch, fix_random_seed
from spacy.training import Corpus
from spacy import displacy

from pathlib import Path
from jsonlines import jsonlines
from collections import defaultdict
from collections.abc import Iterable
from tqdm import tqdm
import random
from time import time as etime
from datetime import datetime as dt

from _temp_query_strategies import query_random, query_least_confidence

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
    # FIXME: implement Argilla "done annotating" event listener, continue when
    #        the event is received
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


def _query(func, func_kwargs, _labels_queried, _queried, spans_key):
    """Query the data using given function and keyword arguments.
       Using inplace object mutation on variables that start with underscore!
    """
    q_indexes = set()
    q_data = []
    for q_idx, q_example in func(**func_kwargs):
        q_doc_annotation = q_example.to_dict()["doc_annotation"]
        component_spans = q_doc_annotation["spans"][spans_key]
        for span in component_spans:
            span_label = span[2]
            _labels_queried[span_label] += 1
            _labels_queried["_all"] += 1
        q_indexes.add(q_idx)
        q_data.append(q_example)
    _queried.update(q_indexes)
    return q_indexes, q_data


def _update_model(nlp, included_components, examples, batch_size):
    """Update the model with the given data"""
    logging.debug("Updating the model with queried data...")
    losses = {}
    optimizer = nlp.initialize()
    with nlp.select_pipes(enable=included_components):
        for batch in minibatch(examples, batch_size):
            nlp.update(batch, losses=losses, sgd=optimizer)

    if isinstance(included_components, str):
        return losses[included_components]
    if isinstance(included_components, Iterable):
        return {
            component: losses[component]
            for component in included_components
        }
    return None


def _evaluate_model(nlp, included_components, examples, batch_size):
    """Evaluate the model with the given data.
       Returns spacy's evaluation metrics"""
    logging.debug("Evaluating the model...")
    with nlp.select_pipes(enable=included_components):
        eval_metrics = nlp.evaluate(examples, batch_size=batch_size)
    return eval_metrics


def _render_sample_prediction(nlp, examples, spans_key):
    rand_indx = random.randint(0, len(examples))
    text = examples[rand_indx].text
    pred = nlp(text)

    logging.info("Rendering the model's sample prediction...")
    if pred.spans[spans_key]:
        render_spans(pred, spans_key)
    else:
        logging.warning("Nothing to render! No spans found.")


def _run_loop(nlp, train_len, train_data, eval_data,
              included_components, max_iter, n_instances,
              train_batch_size, eval_batch_size,
              results_out, spans_key):
    """Functional approach based Active Learning loop implementation.
       Using inplace objects mutation!"""
    iteration = 1
    queried = set()
    labels_queried = defaultdict(int)
    enlarged_train_data = []

    pbar = tqdm(total=max_iter)
    while True:
        it_t0 = etime()
        datetime_str = dt.now().strftime("%d-%m-%Y %H:%M:%S")

        if stop_criteria(iteration, max_iter, queried, train_len):
            break

        func_kwargs = {
            "examples": train_data,
            "exclude": queried,
            "nlp": nlp,
            "spans_key": spans_key,
            "n_instances": n_instances
        }
        if iteration != 1:
            q_func = query_least_confidence
        else:
            logging.info("Querying seed data...")
            func_kwargs.pop("spans_key")
            q_func = query_random
        _, q_data = _query(
            func=q_func,
            func_kwargs=func_kwargs,
            _labels_queried=labels_queried,
            _queried=queried,
            spans_key=spans_key
        )

        # TODO: Log queried examples into Argilla as records wth Default status

        # TODO: query_oracle(), returns record indexes, annotations
        #        > TODO: create dummy oracle based on our train data

        # TODO: insert oracle's annotations into train data

        # Extend the training dataset
        enlarged_train_data.extend(q_data)

        # Update the model with queried data
        sc_loss = _update_model(nlp,
                                included_components=included_components,
                                examples=enlarged_train_data,
                                batch_size=train_batch_size)

        # Evaluate the model on the test set
        eval_metrics = _evaluate_model(nlp,
                                       included_components=included_components,
                                       examples=eval_data,
                                       batch_size=eval_batch_size)

        iteration_time = etime() - it_t0

        # Collect and save the results
        results = {
            "_date": datetime_str,
            "_iteration": iteration,
            "_iteration_time": iteration_time,
            "_spans_count": labels_queried["_all"],
            "_labels_count": labels_queried,
            "_sc_loss": sc_loss
        }
        results.update(eval_metrics)
        log_results(results, out=results_out)

        iteration += 1
        pbar.update(1)
    pbar.close()


def main():
    """AL loop pseudocode
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
    # TODO: FIXME: Refator to object based approach, obviously

    _start_etime_str = str(etime()).replace(".", "f")

    NAME = "test_active_learned_kpwr-short"

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
    TRAIN_BATCH_SIZE = int(N_INSTANCES // 5)
    TEST_BATCH_SIZE = int(N_INSTANCES // 5)
    LABELS = ["nam_liv_person", "nam_loc_gpe_city", "nam_loc_gpe_country"]
    COMPONENT = "spancat"
    SPANS_KEY = "sc"

    random.seed(SEED)
    fix_random_seed(SEED)
    assert not MODEL_OUT.exists()

    nlp = init_nlp(LABELS)

    train_data, test_data = load_data(nlp, TRAIN_DB, TEST_DB)
    train_len = len(train_data)

    _run_loop(nlp, train_len, train_data, test_data,
              COMPONENT, MAX_ITER, N_INSTANCES,
              TRAIN_BATCH_SIZE, TEST_BATCH_SIZE,
              METRICS_OUT, SPANS_KEY)

    # Render the model's sample prediction
    _render_sample_prediction(nlp, test_data, SPANS_KEY)

    # Save the model to binary file
    logging.info(f"Saving model to {MODEL_OUT}...")
    nlp.to_disk(MODEL_OUT)


if __name__ == "__main__":
    main()
