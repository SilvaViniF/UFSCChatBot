"""
AMNESTY INTERNATIONAL REPORTS QA DATASETS 

template from: https://github.com/huggingface/datasets/blob/main/templates/new_dataset_script.py
"""


import json

import datasets

_DESCRIPTION = """\
AMNESTY INTERNATIONAL REPORTS QA DATASETS
"""

_HOMEPAGE = "https://www.amnesty.org/en/research/"

# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={huggingface, Inc.
},
year={2020}
}
"""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = "Apache"

# make URLs form file in local directory
defined_csvs = ["english", "malayalam", "hindi"]
_URLS = {subset: f"{subset}.json" for subset in defined_csvs}
eval_csvs = []


class AmenstyConfig(datasets.BuilderConfig):
    """BuilderConfig for SuperGLUE."""

    def __init__(self, name, version, description, **kwargs):
        """BuilderConfig for SuperGLUE.

        Args:
        features: *list[string]*, list of the features that will appear in the
            feature dict. Should not include "label".
        data_url: *string*, url to download the zip file from.
        citation: *string*, citation for the data set.
        url: *string*, url for information about the data set.
        label_classes: *list[string]*, the list of classes for the label if the
            label is present as a string. Non-string labels will be cast to either
            'False' or 'True'.
        **kwargs: keyword arguments forwarded to super.
        """
        # Version history:
        # 2.0.0: changed ground_truths to ground_truth
        # 1.0.0: Initial version
        super().__init__(version=datasets.Version("2.0.0"), **kwargs)
        self.name = name
        self.version = version
        self.description = description


class Amnesty(datasets.GeneratorBasedBuilder):
    """
    Amnesty QA for RAG experiments
    """

    BUILDER_CONFIG_CLASS = AmenstyConfig
    VERSION_V1 = datasets.Version("1.0.0")
    VERSION_V2 = datasets.Version("2.0.0")

    # different configuration.
    # you can call it like load_dataset(dataset_repo, config)
    BUILDER_CONFIGS = [
        AmenstyConfig(
            name="english",
            version=VERSION_V1,
            description="Amnesty QA in English",
        ),
        AmenstyConfig(
            name="malayalam",
            version=VERSION_V1,
            description="Amnesty QA in Malayalam",
        ),
        AmenstyConfig(
            name="hindi",
            version=VERSION_V1,
            description="Amnesty QA in Hindi",
        ),
        AmenstyConfig(
            name="english_v2",
            version=VERSION_V2,
            description="Amnesty QA in English",
        ),
        AmenstyConfig(
            name="malayalam_v2",
            version=VERSION_V2,
            description="Amnesty QA in Malayalam",
        ),
        AmenstyConfig(
            name="hindi_v2",
            version=VERSION_V2,
            description="Amnesty QA in Hindi",
        ),
    ]

    DEFAULT_CONFIG_NAME = "english"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        self.is_v2 = self.config.name.endswith("_v2")
        features_v1 = datasets.Features(
            {
                "question": datasets.Value(dtype="string"),
                "ground_truths": datasets.Sequence(
                    feature=datasets.Value(dtype="string"), length=-1
                ),
                "answer": datasets.Value(dtype="string"),
                "contexts": datasets.Sequence(
                    feature=datasets.Value(dtype="string"),
                    length=-1,
                ),
            }
        )
        features_v2 = datasets.Features(
            {
                "question": datasets.Value(dtype="string"),
                "ground_truth": datasets.Value(dtype="string"),
                "answer": datasets.Value(dtype="string"),
                "contexts": datasets.Sequence(
                    feature=datasets.Value(dtype="string"),
                    length=-1,
                ),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features_v2 if self.is_v2 else features_v1,
            homepage=_HOMEPAGE,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        file_name = self.config.name[:-3] if self.is_v2 else self.config.name
        urls_to_download = [_URLS[file_name]]
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(
                name="eval",
                gen_kwargs={"filepath": downloaded_files[0], "is_v2": self.is_v2},
            ),
        ]

    def _generate_examples(self, filepath, is_v2, split=None):
        """
        This method handles input defined in _split_generators to yield (key, example)
        tuples from the dataset. The `key` is for legacy reasons (tfds) and is not
        important in itself, but must be unique for each example.
        """
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for i in range(len(data["question"])):
                data_row = {
                    "question": data["question"][i],
                    "contexts": data["contexts"][i],
                    "answer": data["answer"][i],
                }
                if is_v2:
                    data_row["ground_truth"] = data["ground_truths"][i][0]
                else:
                    data_row["ground_truths"] = data["ground_truths"][i]
                yield (i, data_row)
