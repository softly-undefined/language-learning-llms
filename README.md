# Original Idea
I've been meaning to study the impact of language fluency levels on translation, as well as specifically LLMs ability to role-play different experience levels.

- Testing the metrics/translation - we use LLMs to perturb a base set of translations into varying levels of fluency, and translate using these various levels.
- Testing the models - Use existing classification models that are used for evaluating how fluent you are on the LLM text with these varying fluency prompts to see if they can accurately simulate language learners (as far as I'm aware this part of the idea hasn't been done)

(for datasets there's high numbers of into english of various skill levels in multiple datasets, also into german, czech, and italian through MERLIN)



# DATA

MERLIN corpus: https://huggingface.co/datasets/symeneses/merlin

CLASSIFIER

UniversalCEFR classifier: https://huggingface.co/UniversalCEFR/models

UniversalCEFR/xlm-roberta-base-cefr-all-classifier

# Short Description of each directory:

- data-analysis: Final visualization creation

- merlin-text-v1.2: base MERLIN Corpus data

- merlin-extracted: the subset of MERLIN I extracted

- wmt-data: data extracted from wmt25' (isolated long ones)

- translation-levels: where each WMT segment was translated into each proficiency level

- translationmodels: some old api calling scripts i used to use (updated to work with each other to make model hot-swappable)

- universal-cefr-classifier: where the classifier was called on both merlin and wmt data