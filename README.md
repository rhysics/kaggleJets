# QCD & TT Jet tagging CoDaS-HEP

[Kaggle challenge repository for CODAS-HEP 2026](
https://www.kaggle.com/competitions/qcd-tt-jet-tagging-co-da-s-he/)

## Overview from the challenge description

In this challenge, you’ll use simulations of real-world collider physics data to classify jets as either:
• QCD jets (background)
• Top quark (tt̄) jets (signal)

Whether you’re just starting out with ML or you’re looking to sharpen your skills on a scientific dataset, this is a great opportunity to explore particle physics, experiment with models, and learn to handle structured and image-based data alike.

The data is from the 2011 CMS Open Simulation, and has been made using the data published by Javier and Harrison [1].

[1]Javier Duarte and Harrison Prosper, ‘Fermilab LHC Physics Center Machine Learning Hands-On Advanced Tutorial Session Datasets’. Zenodo, Jun. 20, 2020. doi: 10.5281/zenodo.3901869.

## Benchmarks

We use the example bdt and dnn models provided in the challenge repository as benchmarks.

## Project structure

The existing root-level benchmark scripts remain reference implementations.
New work should use the structure below so models, experiment definitions, pipeline code, and generated artifacts stay separate.

```text
configs/
  data/                 Shared data and preprocessing defaults
  profiling/            Shared profiling defaults
  training/             Shared optimizer and training defaults
experiments/            Contributor-owned experiment specifications
models/
  bdt/                  Tree-based model implementations
  common/               Reusable model components
  dnn/                  Dense and image-based neural networks
  ensembles/            Stacking, blending, and voting models
  gnn/                  Graph neural network implementations
pipelines/
  evaluation/           Common metrics and comparison logic
  profiling/            Runtime, memory, throughput, and size measurement
  training/             Reusable training orchestration
reports/
  comparisons/          Reviewed comparison reports and decision records
tests/
  models/               Model interface and behavior tests
  pipelines/            Training, evaluation, and profiling tests
output/                  Local generated runs, ignored by Git
```

The tracked directories contain source code, configurations, tests, and concise reports.
The ignored `output/` directory contains trained weights, logs, predictions, metrics, plots, and profiling artifacts.
Keeping generated files out of Git prevents large binaries and concurrent runs from creating merge conflicts.

## Model conventions

- Put each reusable implementation in `models/<family>/<model_slug>.py`.
- Use lowercase `snake_case` for model slugs, configuration names, and Python files.
- Keep data loading, training loops, metric calculation, and profiling outside model implementations.
- Give each model one stable construction interface so the shared pipelines can instantiate it from configuration.
- Put layers or components used by more than one model family in `models/common/`.
- Add a new file for a meaningfully different architecture instead of repeatedly rewriting a teammate's model.
- Store trained model files only inside the run's `output/` directory.

## Experiment conventions

Each contributor creates `experiments/<github_handle>/` and owns the files inside that directory.
An experiment specification should be named `<yyyy-mm-dd>_<model_slug>_<variant>.yaml`.
It should reference a model implementation and record all settings needed to reproduce the run, including the data configuration, preprocessing, seed, optimizer, learning-rate schedule, batch size, stopping rule, and profiling options.
Shared defaults belong in `configs/`, while experiment-specific overrides belong in the contributor's experiment file.
Once an experiment has been used in a comparison, create a new specification for the next variant instead of changing its history.

## Run output contract

Every run should write to `output/<github_handle>/<run_id>/`.
Use `<yyyymmddThhmmssZ>_<model_slug>_<short_commit>` for the run ID so parallel runs cannot overwrite one another.

```text
output/<github_handle>/<run_id>/
  config.yaml            Resolved configuration snapshot
  metrics.json           Validation or test metrics
  profile.json           Timing, memory, throughput, and model size
  checkpoint.<ext>       Best trained model state
  predictions.<ext>      Optional per-example predictions
  logs/                   Training logs
  plots/                  Diagnostic plots
```

New experiments should produce one `metrics.json` per run rather than update a shared JSON file.
Comparison code can discover those run files and combine them in memory.
This makes runs append-only, reproducible, and safe when several contributors train at the same time.

At minimum, `metrics.json` should identify the schema version, run ID, contributor, Git commit, model name, model family, dataset split, random seed, and metric values.
At minimum, `profile.json` should record parameter count, serialized model size, training time, inference latency, throughput, peak memory, hardware, batch size, software versions, and the number of warm-up and measured iterations.

## Fair comparison rules

- Compare models on the same immutable split and preprocessing definition.
- Record every random seed and report variation across repeated runs when practical.
- Keep threshold-independent scores such as ROC AUC alongside thresholded scores such as accuracy, recall, and F1.
- Select classification thresholds on validation data and never tune them against test labels.
- Profile with the same hardware, precision, batch size, warm-up procedure, and measurement method.
- Rank candidates using both predictive quality and resource cost before selecting a final model or ensemble.
- Treat the competition test set as final evaluation data and avoid using it for routine iteration.

## Team workflow

1. Pull the latest `main` and create a short-lived branch for one model or infrastructure change.
2. Add a new model file or make a small, focused change to shared pipeline code.
3. Add the experiment specification under your own `experiments/<github_handle>/` directory.
4. Run the affected tests and a small end-to-end smoke experiment before launching a full run.
5. Keep all generated artifacts under your isolated `output/<github_handle>/<run_id>/` directory.
6. Open a focused pull request that explains the hypothesis, configuration, validation result, and profiling result.
7. Add a separate file under `reports/comparisons/` only when a comparison or model-selection decision is ready for team review.

Prefer one model or pipeline concern per pull request.
Avoid shared mutable result files, contributor-wide notebooks, or catch-all model modules because they make four-person development harder to merge.
Changes to shared interfaces should be small, tested, and coordinated with open model branches.

## Adding a model

- Add the implementation under the appropriate `models/<family>/` directory.
- Add focused model tests under `tests/models/`.
- Reuse the shared training, evaluation, and profiling pipelines.
- Add a contributor-owned experiment specification.
- Verify a smoke run from data loading through metric and profile output.
- Share the experiment file, commit hash, metrics, and profile when reporting results.
