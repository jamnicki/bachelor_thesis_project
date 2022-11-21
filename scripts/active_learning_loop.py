from spacy.lang.pl import Polish
from spacy.util import fix_random_seed, minibatch, compounding
from spacy.training import Corpus
from thinc.api import Config

import argilla as rg
from argilla.client.sdk.commons.errors import NotFoundApiError

from pathlib import Path
from jsonlines import jsonlines
from collections import defaultdict
from tqdm import tqdm
import random
from time import time as etime
from datetime import datetime as dt

from _temp_query_strategies import query_random, query_least_confidence

import logging
logging.basicConfig(level=logging.DEBUG)

random.seed(42)
fix_random_seed(42)


def log_results(results, out):
    """Log results to a file"""
    with jsonlines.open(out, mode="a") as writer:
        logging.info(f"Writing results to {out}")
        writer.write(results)


def create_nlp(labels, lang, config):
    logging.info("Initializing spaCy model...")
    nlp = lang.from_config(config)
    spancat = nlp.get_pipe("spancat")
    for label in labels:
        spancat.add_label(label)
    return nlp


def load_data(nlp, train_docbin_path, test_docbin_path):
    train_corpus = Corpus(train_docbin_path)
    test_corpus = Corpus(test_docbin_path)
    train_data = tuple(train_corpus(nlp))
    test_data = tuple(test_corpus(nlp))
    return train_data, test_data


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
        # TODO: continue and traing the model on the remaining data < n
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
    """Query oracle for annotations.
       Reads the annotations from Argilla records with given indexes."""
    annotated_records = rg.load(ds_name, ids=q_indexes)
    for record in annotated_records:
        yield record.id, record.annotation


def dummy_query_oracle(train_data, q_indexes, spans_key):
    """Dummy query oracle for experimental purposes.
       Simple gets annotations from the training data."""
    for q_idx, train_data_idx in enumerate(q_indexes):
        example = train_data[train_data_idx]
        doc_annotation = example.to_dict()["doc_annotation"]
        annotation = doc_annotation["spans"][spans_key]
        yield q_idx, annotation


def _insert_oracle_annotation(_q_data, q_idx, q_oracle_ann, spans_key):
    """In-place insertion of oracle annotation into the queried data"""
    # FIXME: does't work, Example object in not subscriptable, maybe create new
    #        example object with the new annotation, then overwrite?
    _q_data[q_idx]["doc_annotation"]["spans"][spans_key] = q_oracle_ann


def update_model(nlp, optimizer, included_components, examples):
    """Update the model with the given data"""
    logging.debug("Updating the model with queried data...")
    # TODO: get partial loss, not updated one
    # TODO: tune compounding and dropout, outsource minibatch size
    losses = {}
    with nlp.select_pipes(enable=included_components):
        for batch in minibatch(examples, size=compounding(4.0, 32.0, 1.001)):
            losses = nlp.update(batch, losses=losses, sgd=optimizer)
    return losses


def evaluate_model(nlp, included_components, examples):
    """Evaluate the model with the given data.
       Returns spacy's evaluation metrics"""
    logging.debug("Evaluating the model...")
    with nlp.select_pipes(enable=included_components):
        eval_metrics = nlp.evaluate(examples)
    return eval_metrics


def _run_loop(nlp, sampling_strategy,
              train_len, train_data, eval_data,
              included_components, max_iter, n_instances,
              rg_ds_name, rg_suggester_agent,
              results_out, spans_key, models_dir,
              dummy):
    """Functional approach based Active Learning loop implementation.
       Using inplace objects mutation!"""
    iteration = 1
    queried = set()
    labels_queried = defaultdict(int)
    _loop_train_data = []
    pbar = tqdm(total=max_iter)
    while True:
        pbar.update(1)
        it_t0 = etime()
        datetime_str = dt.now().strftime("%d-%m-%Y %H:%M:%S")

        if stop_criteria(iteration, max_iter, queried, train_len, n_instances):
            break

        optimizer = nlp.initialize()

        # vector representations spacy thing, raises hash error
        # if tok2vec is not disabled for nlp.pipe()
        nlp_pipe_included = set(included_components) - set(["tok2vec"])

        func_kwargs = {
            "examples": train_data,
            "exclude": queried,
            "nlp": nlp,
            "included_components": nlp_pipe_included,
            "spans_key": spans_key,
            "n_instances": n_instances
        }
        random_exluded = ["nlp", "included_components", "spans_key"]
        if iteration != 1:
            if sampling_strategy == "least_confidence":
                q_func = query_least_confidence
            elif sampling_strategy == "random":
                for ex_kwarg in random_exluded:
                    del func_kwargs[ex_kwarg]
                q_func = query_random
            else:
                raise ValueError("Unknown sampling strategy")
        else:
            logging.info("Querying seed data...")
            for ex_kwarg in random_exluded:
                del func_kwargs[ex_kwarg]
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
        losses = update_model(nlp, optimizer,
                              included_components=included_components,
                              examples=_loop_train_data)

        # Evaluate the model on the test set
        eval_metrics = evaluate_model(nlp,
                                      included_components=included_components,
                                      examples=eval_data)

        iteration_time = etime() - it_t0

        # Collect and save the results
        results = {
            "_date": datetime_str,
            "_iteration": iteration,
            "_iteration_time": iteration_time,
            "_spans_count": labels_queried["_all"],
            "_labels_count": labels_queried,
            "_sc_loss": losses["spancat"]
        }
        results.update(eval_metrics)
        log_results(results, out=results_out)

        iteration += 1

        # Save partial models
        spans_count = results["_spans_count"]
        model_name = f"model_{iteration}it_{spans_count}spans.spacy"
        model_path = models_dir / Path(model_name)

        yield nlp, model_path, results

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
    # TODO: Outsource the constans to config.ini file
    # TODO: Wrapp script with typer

    _start_etime_str = str(etime()).replace(".", "f")

    DUMMY = True

    MAX_ITER = 50
    N_INSTANCES = 50
    STRATEGY = "least_confidence"

    NAME = f"{STRATEGY}_{MAX_ITER}i_{N_INSTANCES}n_kpwr-full"
    CONFIG_PATH = "./config/spacy/config_sm.cfg"

    AGENT_NAME = __file__.split("/")[-1].split(".")[0]
    RG_DATASET_NAME = "active_learninig_temp_dataset"

    RUN_DIR = Path("runs") / Path(f"{NAME}_{_start_etime_str}")
    MODELS_RUN_DIR = Path(RUN_DIR) / Path("models")
    MODELS_RUN_DIR.mkdir(parents=True, exist_ok=True)

    DATA_DIR = Path("data")
    LOGS_DIR = RUN_DIR / Path("logs")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    TRAIN_DB = DATA_DIR / Path("inzynierka-kpwr-train-3-full.spacy")
    TEST_DB = DATA_DIR / Path("inzynierka-kpwr-test-3-full.spacy")

    METRICS_OUT = LOGS_DIR / Path(f"{NAME}.metrics.jsonl")

    LABELS = ["nam_liv_person", "nam_loc_gpe_city", "nam_loc_gpe_country"]
    COMPONENTS = ["tok2vec", "spancat"]
    SPANS_KEY = "sc"

    nlp_config = Config().from_disk(CONFIG_PATH)
    nlp = create_nlp(LABELS, lang=Polish, config=nlp_config)
    if not DUMMY:
        rg.monitor(nlp, RG_DATASET_NAME, agent=AGENT_NAME)

    # DELETE AL TEMP DATASET IF EXISTS
    try:
        if not DUMMY:
            rg.load(RG_DATASET_NAME)
    except NotFoundApiError:
        logging.info("Temporary dataset not found and will be created")
    else:
        if not DUMMY:
            # TODO: Resume annotation from previous session
            logging.warning(f"Deleting {RG_DATASET_NAME} dataset...")
            rg.delete(RG_DATASET_NAME)

    train_data, test_data = load_data(nlp, TRAIN_DB, TEST_DB)
    train_len = len(train_data)

    best_model = None
    best_model_fscore = -1
    best_model_path = None
    for _nlp, _out, _res in _run_loop(nlp, STRATEGY,
                                      train_len, train_data, test_data,
                                      COMPONENTS, MAX_ITER, N_INSTANCES,
                                      RG_DATASET_NAME, AGENT_NAME,
                                      METRICS_OUT, SPANS_KEY, MODELS_RUN_DIR,
                                      DUMMY):
        _nlp_fscore = _res[f"spans_{SPANS_KEY}_f"]
        if _nlp_fscore > best_model_fscore:
            best_model = _nlp
            best_model_path = _out
        _nlp.to_disk(_out)

    best_path = str(best_model_path).replace(".spacy", ".best.spacy")
    best_model.to_disk(best_path)


if __name__ == "__main__":
    main()
