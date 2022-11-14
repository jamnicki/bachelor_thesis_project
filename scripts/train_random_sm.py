import spacy
from spacy.cli.train import train
from spacy.cli.evaluate import evaluate
from spacy.tokens import DocBin

from pathlib import Path
from jsonlines import jsonlines
from collections import defaultdict
import random
from time import time as etime
from datetime import datetime as dt


def query_random(records, exclude, n_instances):
    """Random query strategy"""
    n_queried = 0
    max_idx = len(records) - 1
    while n_queried < n_instances:
        idx = random.randint(0, max_idx)
        if idx not in exclude:
            exclude.add(idx)
            n_queried += 1
            yield idx, records[idx]


def log_results(results, out):
    """Log results to a file"""
    with jsonlines.open(out, mode="a") as writer:
        writer.write(results)


def _docs_train(docbin_path, lang="pl"):
    """Get docs from Docbin using blank nlp object's vocabulary."""
    nlp = spacy.blank(lang)
    docs_train = list(DocBin().from_disk(docbin_path).get_docs(nlp.vocab))
    return docs_train


def main():
    NAME = "random_sm10"

    _start_etime_str = str(etime()).replace(".", "f")
    DATA_DIR = Path("data")
    TRAIN_DB = DATA_DIR / Path("inzynierka-kpwr-train-3.spacy")
    TEST_DB = DATA_DIR / Path("inzynierka-kpwr-test-3.spacy")
    TEMP_DB = DATA_DIR / Path(".temp-train.spacy")
    LOGS_DIR = Path("logs")
    CONFIG_DIR = Path("config") / Path("spacy")
    CONFIG = CONFIG_DIR / Path("config_sm.cfg")
    MODELS_DIR = Path("models")
    MODELS_DIR.mkdir(exist_ok=True)
    MODEL_OUT = MODELS_DIR / Path(f"{NAME}__{_start_etime_str}.spacy")
    MODEL_BEST = MODEL_OUT / Path("model-best")
    METRICS_OUT = LOGS_DIR / Path(f"{NAME}__{_start_etime_str}.metrics.jsonl")

    SEED = 42
    SPANS_KEY = "sc"
    N_INSTANCES = 5
    MAX_EPOCHS = 20

    random.seed(SEED)
    assert not MODEL_OUT.exists()

    docs_train = _docs_train(TRAIN_DB)
    docs_train_len = len(docs_train)

    iteration = 1
    max_iters = 10
    spans_queried = 0
    spans_num_history = []
    db = DocBin()
    queried = set()
    labels_queried = defaultdict(int)
    while True:
        if iteration > max_iters or len(queried) >= docs_train_len:
            break
        datetime_str = dt.now().strftime("%d-%m-%Y %H:%M:%S")

        for q_idx, q_doc in query_random(docs_train, queried, N_INSTANCES):
            for span in q_doc.spans[SPANS_KEY]:
                labels_queried[span.label_] += 1
            queried.add(q_idx)
            db.add(q_doc)
            spans_queried += len(q_doc.spans[SPANS_KEY])
        spans_num_history.append(spans_queried)

        db.to_disk(TEMP_DB)

        train(CONFIG,
              output_path=MODEL_OUT,
              overrides={
                  "paths.train": str(TEMP_DB),
                  "paths.dev": str(TEST_DB),
                  "training.seed": SEED,
                  "training.max_epochs": MAX_EPOCHS,
              })

        eval_metrics = evaluate(MODEL_BEST, TEST_DB)

        results = {
            "date": datetime_str,
            "_iteration": iteration,
            "_spans_count": spans_queried,
            "_labels_count": labels_queried
        }
        results.update(eval_metrics)

        iteration += 1
        log_results(results,
                    out=METRICS_OUT)


if __name__ == "__main__":
    main()
