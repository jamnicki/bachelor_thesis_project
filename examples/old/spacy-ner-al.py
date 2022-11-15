try:
    import pl_core_news_lg
except ImportError:
    import spacy.cli
    spacy.cli.download("pl_core_news_lg")
    import pl_core_news_lg

from spacy.training import Example
import argilla as rg
from matplotlib import pyplot as plt
from jsonlines import jsonlines

import random
random.seed(0)


def init_ner_pipeline():
    """Initialize pipeline with only the NER component"""
    nlp = pl_core_news_lg.load()
    return nlp


def query_random(records, n_instances, exclude):
    """Random query strategy"""
    n_queried = 0
    while n_queried < n_instances:
        idx = random.randint(0, len(records) - 1)
        if idx not in exclude:
            exclude.add(idx)
            n_queried += 1
            yield idx, records[idx]


def spacy2rg(annotations):
    # (start, end, label) -> (label, start, end)
    return [(ann[2], ann[0], ann[1]) for ann in annotations]


def plot_scores(scores, xlabel, ylabel):
    plt.plot(scores)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def main():
    DATASET = "/home/jjamnicki/Data/INZYNIERKA/calls_texts_sample200.jsonl"
    DS_NAME = "inzynierka"
    N_INSTANCES = 5

    with jsonlines.open(DATASET) as reader:
        dataset = [obj for obj in reader]

    nlp = init_ner_pipeline()
    pred_agent = "spacy-ner"

    nlp = rg.monitor(nlp, dataset=DS_NAME, agent=pred_agent)

    ds_settings = rg.TokenClassificationSettings(
        label_schema=nlp.pipe_labels["ner"]
    )
    rg.configure_dataset(name=DS_NAME, settings=ds_settings)

    query_history = set()
    loss_history = []
    losses = {}
    iteration = 1
    while True:
        queried = []
        records = []
        # query new instances
        for idx, call_data in query_random(dataset,
                                           n_instances=N_INSTANCES,
                                           exclude=query_history):
            doc = nlp(call_data["text"])  # process text over the pipeline

            # create Argilla record
            record = rg.TokenClassificationRecord(
                # each record must have unique id for further reference before
                # training
                id=idx,
                text=call_data["text"],
                tokens=[t.text for t in doc],
                prediction=[
                    (ent.label_, ent.start_char, ent.end_char)
                    for ent in doc.ents
                ],
                prediction_agent=pred_agent
            )

            records.append(record)
            queried.append(idx)

        query_history.update(queried)  # track queried instances

        # push records with predictions from our agent as recommendations
        # for the annotators
        rg.log(records, name=DS_NAME)

        # simple work around to wait for annotations
        _input = input("\n\nAnnotate records then press [Enter] to continue or [q] to quit")  # noqa: E501
        if "q" in _input.lower():
            break

        examples = []
        annotated_records = rg.load(name=DS_NAME, ids=queried)
        for rec in annotated_records:
            if hasattr(rec, "annotations"):
                annotations = spacy2rg(rec.annotations)
            else:
                annotations = []

            tmp_doc = nlp.make_doc(rec.text)
            example = Example.from_dict(tmp_doc, {"entities": annotations})
            examples.append(example)

        nlp.update(examples, losses=losses)

        loss = losses["ner"]
        loss_history.append(loss)

        print(f"Iteration: {iteration} Loss: {loss:.4f}")
        iteration += 1

    plot_scores(
        scores=loss_history,
        xlabel="Iteration",
        ylabel="Loss"
    )


if __name__ == "__main__":
    main()
