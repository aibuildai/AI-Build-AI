<h1 align="center"><small>AIBuildAI – An AI agent that automatically builds AI models</small></h1>

<h1 align="center"><sub>🏆 #1 on OpenAI <a href="https://github.com/openai/mle-bench">MLE-Bench</a></sub></h1>

<p align="center">
  <img src="assets/downloads.svg" alt="Downloads">
  <a href="https://arxiv.org/abs/2604.14455">
  <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2604.14455-b31b1b.svg">
  </a>
</p>

---

## News

- **[4/27/2026]** Excited to announce **AIBuildAI Agent 2.0**! It has once again achieved #1 on OpenAI's MLE-Bench, reaching a score of 70.7% and substantially outperforming the agents ranked 2nd through 5th. Compared to version 1.0, Agent 2.0 introduces several technical advancements, which we will detail in an upcoming technical report. The 2.0 version will be available to Pro users soon.
- **[4/24/2026]** In a [heart disease prediction competition](https://www.kaggle.com/competitions/playground-series-s6e2/overview) hosted by Kaggle, the model automatically developed by our AIBuildAI Agent ranked in the top 6.6%. Among 4,370 teams composed of human experts, this performance reaches the level of top-tier human AI experts. Model and code developed by the agent: [tasks/playground-series-s6e2](https://github.com/aibuildai/AI-Build-AI/tree/main/tasks/playground-series-s6e2).

---

https://github.com/user-attachments/assets/b6043d39-43df-464a-8e25-d24006ba99c8

---


## Introduction

AIBuildAI is an AI agent that automatically builds AI models. Given a task, it runs an agent loop that analyzes the problem, designs models, writes code to implement them, trains them, tunes hyperparameters, evaluates model performance, and iteratively improves the models. By automating the model development workflow, AIBuildAI reduces much of the manual effort required to build AI models.

<p align="center">
  <img src="assets/workflow.png" width="70%" alt="AIBuildAI Architecture">
</p>

---

## Current Results

On OpenAI [MLE-Bench](https://github.com/openai/mle-bench), AIBuildAI ranked #1, demonstrating strong performance on real-world AI model building tasks.

<p align="center">
  <img src="assets/results.png" width="50%" alt="MLE-Bench Results">
</p>

---

## Quick Start

### Installation

AIBuildAI requires a **Linux x86_64** machine.

```bash
curl -L -O https://github.com/aibuildai/AI-Build-AI/releases/latest/download/aibuildai-linux-x86_64-v0.1.1.tar.gz
tar -xzf aibuildai-linux-x86_64-v0.1.1.tar.gz
cd aibuildai-linux-x86_64-v0.1.1
./install.sh
```

### Set up credentials

```bash
export ANTHROPIC_API_KEY=your-api-key
```

### Run

**Example task:** Predict the enzyme class of a protein from its amino acid sequence ([Yu et al., *Science* 2023](https://www.science.org/doi/10.1126/science.adf2465)).

```bash
git clone https://github.com/aibuildai/AI-Build-AI.git && cd AI-Build-AI
aibuildai --task-name protein-ec-prediction \
  --data-dir data/protein-ec-prediction \
  --playground-dir /path/to/playground \
  --model claude-opus-4-6 \
  --max-agent-calls 8 \
  --run-budget-minutes 60 \
  --num-candidates 3 \
  --instruction "$(cat tasks/protein-ec-prediction.md)" \
  --pipeline-budget-minutes 90 \
  --no-form
```

AIBuildAI takes two key inputs: `--data-dir`, the path to the training data for the task, and `--instruction`, a natural-language description of the AI task to solve.


**Important:**

Run the command directly in your terminal. Do not wrap the command in a `.sh` or `.bash` script. Running it through a script may cause the TUI (Text User Interface) to crash.

### Results

#### Output directory

After a run completes, the output directory usually looks like (structure may slightly vary by task):

```
├── candidate_1/  candidate_2/  candidate_3/  # Auto-generated training scripts and model checkpoints
├── checkpoint.pth       # Best model checkpoint
├── inference.py         # Standalone inference script for the final model
├── submission.csv       # Test predictions (if test inputs are provided)
└── progress.pdf         # Visual progress report
```

The main outputs of an AIBuildAI run are the model checkpoints and the script `inference.py`, which runs predictions with the final model on any data.

#### Evaluation

In the example protein-ec-prediction task, we provide unlabeled test data in the data folder, so AIBuildAI also generates a predicted-label file `submission.csv`. To evaluate the predictions against ground-truth labels:

```bash
python scripts/eval_protein_ec.py \
  --labels data/labels/protein-ec-prediction.csv \
  --submission /path/to/playground/code/protein-ec-prediction/timestamp/submission.csv
```

### Other tasks

We provide additional task markdowns in the `tasks/` folder. You can also write your own task markdown and point `--data-dir` to your own dataset.

---

### Command line options

To see all available options, run:

```bash
aibuildai -h
```

### Interactive form mode 

Alternatively, you can run AIBuildAI using the interactive form interface by running without `--no-form`:

```bash
aibuildai
```

This will launch a TUI (Text User Interface) where you can fill in the required parameters interactively.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

```bibtex
@article{zhang2026aibuildai,
    title={AIBuildAI: An AI Agent for Automatically Building AI Models},
    author={Ruiyi Zhang and Peijia Qin and Qi Cao and Li Zhang and Pengtao Xie},
    year={2026},
    journal={arXiv},
    url={https://arxiv.org/abs/2604.14455}
}
```


