from spacy.lang.pl import Polish
from spacy.util import minibatch, fix_random_seed
from spacy.training import Corpus
from spacy import displacy

import argilla as rg
from argilla.client.sdk.commons.errors import NotFoundApiError as rgDatasetNotFound  # noqa: E501

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


def create_nlp(labels):
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


def data_exhausted(queried_idxs, set_len, n_instaces):
    return len(queried_idxs) + n_instaces > set_len


def _wait_for_user():
    """Wait for user to annotate the data"""
    # TODO: implement Argilla "done annotating" event listener, continue when
    #       the event is received
    msg = "Annotate the data then press [Enter] to continue or [q] to quit."
    msg.join("\n")
    _input = input(msg)
    if "q" in _input.lower():
        return 1
    return 0


def stop_criteria(iteration, max_iter, queried, set_len, n_instaces):
    """Return True if stop criteria is met, False otherwise. Log the reason."""
    if iteration > max_iter > 0:
        logging.warning("Stopped by max iterations")
        return True
    if data_exhausted(queried, set_len, n_instaces):
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


def ann_spacy2rg(spacy_ann):
    """Convert spaCy's span annotation to Argilla's"""
    start, end, label, *_ = spacy_ann
    return label, start, end


def ann_rg2spacy(rg_ann):
    """Convert Argilla's span annotation to spaCy's"""
    label, start, end = rg_ann
    return start, end, label


def serve_query_data_for_annotation(
        examples, q_indexes, ds_name, suggester_agent, spans_key):
    """Log the records to Argilla from given examples. Overwrite existing."""
    logging.info("Serving the queried data to annotation...")
    records = []
    for q_idx, example in zip(q_indexes, examples):
        example_dict = example.to_dict()
        doc_annotation = example_dict["doc_annotation"]
        suggestions = doc_annotation["spans"][spans_key]
        orth_list = example_dict["token_annotation"]["ORTH"]
        ann_suggestions = [
            ann_spacy2rg(annotation)
            for annotation in suggestions
        ]
        rg_record = rg.TokenClassificationRecord(
            id=q_idx,
            text=example.text,
            tokens=orth_list,
            prediction=ann_suggestions,
            prediction_agent=suggester_agent,
            status="Default"
        )
        records.append(rg_record)
    rg.log(records, ds_name)


def query_oracle(q_indexes, ds_name):
    """Query oracle"""
    annotated_records = rg.load(ds_name, ids=q_indexes)
    for record in annotated_records:
        yield record.id, record.annotation


def dummy_query_oracle(train_data, q_indexes, spans_key):
    """Dummy query oracle for experimental purposes"""
    for q_idx, train_data_idx in enumerate(q_indexes):
        example = train_data[train_data_idx]
        doc_annotation = example.to_dict()["doc_annotation"]
        annotation = doc_annotation["spans"][spans_key]
        yield q_idx, annotation


def _insert_oracle_annotation(_q_data, q_idx, q_oracle_ann, spans_key):
    """In-place insertion of oracle annotation into the queried data"""
    # FIXME: does't work, Example object in not subscriptable
    _q_data[q_idx]["doc_annotation"]["spans"][spans_key] = q_oracle_ann


def _update_model(nlp, included_components, examples, batch_size):
    """Update the model with the given data"""
    logging.debug("Updating the model with queried data...")
    losses = {}
    optimizer = nlp.initialize()
    with nlp.select_pipes(enable=included_components):
        for batch in minibatch(examples, batch_size):
            losses = nlp.update(batch, losses=losses, sgd=optimizer)

    if isinstance(included_components, str):
        return losses[included_components]
    if isinstance(included_components, Iterable):
        return {
            component: losses[component]
            for component in included_components
        }
    logging.error(f"{_update_model.__name__} function with no return!")


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
              rg_ds_name, rg_suggester_agent,
              results_out, spans_key,
              dummy):
    """Functional approach based Active Learning loop implementation.
       Using inplace objects mutation!"""
    iteration = 1
    queried = set()
    labels_queried = defaultdict(int)
    _loop_train_data = []
    pbar = tqdm(total=max_iter)
    while True:
        it_t0 = etime()
        datetime_str = dt.now().strftime("%d-%m-%Y %H:%M:%S")

        if stop_criteria(iteration, max_iter, queried, train_len, n_instances):
            break

        func_kwargs = {
            "examples": train_data,
            "exclude": queried,
            "nlp": nlp,
            "included_components": included_components,
            "spans_key": spans_key,
            "n_instances": n_instances
        }
        if iteration != 1:
            q_func = query_least_confidence
        else:
            logging.info("Querying seed data...")
            del func_kwargs["nlp"]
            del func_kwargs["included_components"]
            del func_kwargs["spans_key"]
            q_func = query_random

        q_indexes, q_data = _query(
            func=q_func,
            func_kwargs=func_kwargs,
            _labels_queried=labels_queried,
            _queried=queried,
            spans_key=spans_key
        )

        if not dummy:
            serve_query_data_for_annotation(q_data, q_indexes,
                                            rg_ds_name, rg_suggester_agent,
                                            spans_key)
            quit = _wait_for_user()
            if quit:
                return None

        # IN CASE OF DUMMY SYSTEM WE DO NOT NEED INSERT ANNOTATION TO QUERY
        # DATA BECAUSE IT IS ALREADY THERE
        # IT WILL BE NOT ANNOTATED IN THE FUTURE

        # TODO: Insert annotations from Oracle into the enlarged training data

        # for q_idx, qo_ann in dummy_query_oracle(train_data, q_indexes,
        #                                         spans_key):
        #     _insert_oracle_annotation(q_data, q_idx, qo_ann, spans_key)

        # Extend the training dataset
        _loop_train_data.extend(q_data)

        # Update the model with queried data
        sc_loss = _update_model(nlp,
                                included_components=included_components,
                                examples=_loop_train_data,
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
    # TODO: Refactor to object based approach, obviously

    _start_etime_str = str(etime()).replace(".", "f")

    DUMMY = True

    NAME = "active_learned_kpwr-full"
    AGENT_NAME = __file__.split("/")[-1].split(".")[0]
    RG_DATASET_NAME = "active_learninig_temp_dataset"

    DATA_DIR = Path("data")
    MODELS_DIR = Path("models") / Path("scripts")
    LOGS_DIR = Path("logs") / Path("scripts")
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    TRAIN_DB = DATA_DIR / Path("inzynierka-kpwr-train-3-full.spacy")
    TEST_DB = DATA_DIR / Path("inzynierka-kpwr-test-3-full.spacy")
    MODEL_OUT = MODELS_DIR / Path(f"{NAME}__{_start_etime_str}.spacy")
    METRICS_OUT = LOGS_DIR / Path(f"{NAME}__{_start_etime_str}.metrics.jsonl")

    SEED = 42
    MAX_ITER = 10
    N_INSTANCES = 10
    TRAIN_BATCH_SIZE = 1000
    TEST_BATCH_SIZE = 1000
    assert 0 not in [TRAIN_BATCH_SIZE, TEST_BATCH_SIZE]

    LABELS = ["nam_liv_person", "nam_loc_gpe_city", "nam_loc_gpe_country"]
    COMPONENT = "spancat"
    SPANS_KEY = "sc"

    random.seed(SEED)
    fix_random_seed(SEED)
    assert not MODEL_OUT.exists()

    nlp = create_nlp(LABELS)
    if not DUMMY:
        rg.monitor(nlp, RG_DATASET_NAME, agent=AGENT_NAME)

    # Raise error if temporary dataset already exists
    try:
        if not DUMMY:
            rg.load(RG_DATASET_NAME)
    except rgDatasetNotFound:
        logging.info("Temporary dataset not found and will be created")
    else:
        if not DUMMY:
            # TODO: Resume annotation from previous session
            logging.warning(f"Deleting {RG_DATASET_NAME} dataset...")
            rg.delete(RG_DATASET_NAME)

    train_data, test_data = load_data(nlp, TRAIN_DB, TEST_DB)
    train_len = len(train_data)

    _run_loop(nlp, train_len, train_data, test_data,
              COMPONENT, MAX_ITER, N_INSTANCES,
              TRAIN_BATCH_SIZE, TEST_BATCH_SIZE,
              RG_DATASET_NAME, AGENT_NAME,
              METRICS_OUT, SPANS_KEY,
              DUMMY)

    # Render the model's sample prediction
    _render_sample_prediction(nlp, test_data, SPANS_KEY)

    # Save the model to binary file
    logging.info(f"Saving model to {MODEL_OUT}...")
    nlp.to_disk(MODEL_OUT)


if __name__ == "__main__":
    main()
