# HateFinetune

This repository contains the code for the paper [Probing Critical Learning Dynamics of PLMs for Hate Speech Detection](https://aclanthology.org/2024.findings-eacl.55/) accepted in the Findings of the Association for Computational Linguistics: EACL 2024

The `Data` dir has all the information to obtain the process the datasets while the `Experiments` dir has subfolders corresponding to different experiments from the paper.

For any question raise an issue on Github

If you find our work helpful, please don't forget to cite us -

```
@inproceedings{masud-etal-2024-probing,
    title = "Probing Critical Learning Dynamics of {PLM}s for Hate Speech Detection",
    author = "Masud, Sarah  and
      Khan, Mohammad Aflah  and
      Goyal, Vikram  and
      Akhtar, Md Shad  and
      Chakraborty, Tanmoy",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2024",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-eacl.55",
    pages = "826--845",
    abstract = "Despite the widespread adoption, there is a lack of research into how various critical aspects of pretrained language models (PLMs) affect their performance in hate speech detection. Through five research questions, our findings and recommendations lay the groundwork for empirically investigating different aspects of PLMs{'} use in hate speech detection. We deep dive into comparing different pretrained models, evaluating their seed robustness, finetuning settings, and the impact of pretraining data collection time. Our analysis reveals early peaks for downstream tasks during pretraining, the limited benefit of employing a more recent pretraining corpus, and the significance of specific layers during finetuning. We further call into question the use of domain-specific models and highlight the need for dynamic datasets for benchmarking hate speech detection.",
}
```

