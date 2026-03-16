<h1 align="center"><small>AIBuildAI – An AI agent that automatically builds AI models</small></h1>

<h1 align="center"><sub>🏆 #1 on OpenAI <a href="https://github.com/openai/mle-bench/pull/126">MLE-Bench</a></sub></h1>


<p align="center">
  <img src="https://img.shields.io/github/downloads/aibuildai/AI-Build-AI/total?style=flat-square&label=downloads" alt="Downloads">
</p>

---

https://github.com/user-attachments/assets/b6043d39-43df-464a-8e25-d24006ba99c8

---


## Introduction

AIBuildAI is an AI agent that automatically builds AI models. Given a task, it runs an agent loop that analyzes the problem, designs a model, writes the code to implement it, trains the model, performs hyperparameter tuning, evaluates model performance, and iteratively refines the solution. By automating the model development workflow, AIBuildAI reduces much of the manual effort required to build AI models.

<p align="center">
  <img src="assets/workflow.png" width="70%" alt="AIBuildAI Architecture">
</p>

---

## Current Results

On OpenAI [MLE-Bench](https://github.com/openai/mle-bench/pull/126), AIBuildAI ranked #1 (excluding methods that used leaked test labels), demonstrating strong performance on real-world AI model building tasks.

<p align="center">
  <img src="assets/results.png" width="50%" alt="MLE-Bench Results">
</p>

---

## Quick Start

### Installation

AIBuildAI requires a **Linux x86_64** machine.

```bash
curl -L -O https://github.com/aibuildai/AI-Build-AI/releases/latest/download/aibuildai-linux-x86_64-v0.1.0.tar.gz
tar -xzf aibuildai-linux-x86_64-v0.1.0.tar.gz
cd aibuildai-linux-x86_64-v0.1.0
./install.sh
```

---

### Usage (Recommended)

The recommended way to run AIBuildAI is via the command line.

First, set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=your-api-key
```

Next, we show an example of using AIBuildAI to build an AI model that classifies whether an aerial image contains a columnar cactus.

#### Data Setup

Run the provided script to download and prepare the dataset. You will need a [Kaggle Legacy API Key](https://www.kaggle.com/settings) and must accept the [competition rules](https://www.kaggle.com/competitions/aerial-cactus-identification/rules) beforehand. To get your legacy key, go to Kaggle Settings → API → Legacy API Credentials.

```bash
pip install kaggle==1.6.14
python scripts/download_aerial_cactus.py \
  --kaggle-username your_username \
  --kaggle-key your_api_key \
  --data-dir /path/to/data/aerial-cactus-identification
```

#### Running

AIBuildAI takes two key inputs: `--data-dir`, the path to the training data for the task, and `--instruction`, a natural-language description of the AI task to solve.

Example command:

```bash
aibuildai --task-name aerial-cactus-identification \
  --data-dir /path/to/data/aerial-cactus-identification \
  --playground-dir /path/to/playground \
  --model claude-opus-4-6 \
  --max-agent-calls 8 \
  --run-budget-minutes 10 \
  --num-candidates 3 \
  --instruction "$(cat tasks/aerial-cactus-identification.md)" \
  --no-form
```

**Important:**

Run the command directly in your terminal.

Do not wrap the command in a `.sh` or `.bash` script. Running it through a script may cause the TUI (Text User Interface) to crash.

#### Output

After a run completes, the output directory usually looks like (structure may slightly vary by task):

```
├── candidate_1/  candidate_2/  candidate_3/  # Auto-generated training scripts and model checkpoints
├── checkpoint.pth       # Best model checkpoint
├── inference.py         # Standalone inference script for the final model
└── progress.pdf         # Visual progress report
```

Use `inference.py` to run predictions with the final model.

#### Other Tasks

We provide additional task markdowns you can use in the same way — download the corresponding dataset, prepare the data directory, and pass the markdown as `--instruction`:

- `tasks/learning-agency-lab-automated-essay-scoring-2.md`
- `tasks/spaceship-titanic.md`

You can also write your own markdown to describe your own AI task in a similar format and let AIBuildAI automatically build a model for it.

---

#### Command Line Options

To see all available options, run:

```bash
aibuildai -h
```

### Interactive Form Mode (Optional)

Alternatively, you can run AIBuildAI using the interactive form interface by running without `--no-form`:

```bash
aibuildai
```

This will launch a TUI (Text User Interface) where you can fill in the required parameters interactively.

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you find AIBuildAI useful, please cite our paper:

```bibtex
@article{zhang2026aibuildai,
  title={AIBuildAI: An AI Agent that Automatically Builds AI Models},
  author={Zhang, Ruiyi and Qin, Peijia and Cao, Qi and Zhang, Li and Xie, Pengtao},
  year={2026}
}
```


