<div align="center">

# AIBuildAI

<sub>Autonomous Machine Learning Engineering through Collaborative AI Agents</sub>

</div>

---


https://github.com/user-attachments/assets/5d51d49a-f4bc-4e65-af42-ccf144d35839

---

**AIBuildAI** is a framework for autonomous machine learning engineering powered by an iterative multi-agent loop. You only need to provide a machine learning task description and data — AIBuildAI will automatically build ML models for you.

AIBuildAI decomposes the entire ML pipeline into specialized roles — setup, management, design, coding, tuning, and aggregation — each handled by a dedicated AI agent built on top of Claude models.

<p align="center">
  <img src="assets/workflow.png" width="70%" alt="AIBuildAI Architecture">
</p>

---

## Quick Start

### Installation

AIBuildAI requires a **Linux x86_64** machine.

Download the package and extract it:

```bash
curl -L -O https://github.com/aibuildai/AI-Build-AI/releases/latest/download/aibuildai-linux-x86_64-v0.1.0.tar.gz
tar -xzf aibuildai-linux-x86_64-v0.1.0.tar.gz
cd aibuildai-linux-x86_64-v0.1.0
./install.sh
```

This installs the AIBuildAI executable.

---

### Usage (Recommended)

The recommended way to run AIBuildAI is via the command line.

First, set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=your-api-key
```

Next, we show an example of using AIBuildAI to build a machine learning model that classifies whether an aerial image contains a columnar cactus.

#### Data Setup

For the aerial cactus example, download the dataset from https://www.kaggle.com/competitions/aerial-cactus-identification/data (this is a Kaggle competition — you must agree to the competition rules before downloading). Unzip the downloaded data and place its contents in the folder you pass as `--data-dir`.

After setup, the data directory should contain the files described in `tasks/aerial-cactus-identification.md`:

- `train/` — 17,500 images
- `train.csv` — columns: `id` (filename), `has_cactus` (0 or 1)
- `test/` — 4,000 images
- `sample_submission.csv` — columns: `id`, `has_cactus`

#### Running

`--playground-dir` can be any folder — it is used to store intermediate artifacts such as generated code, model checkpoints, and training logs.

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

You can also write your own markdown to describe your own machine learning task in a similar format and let AIBuildAI automatically build a model for it.

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

## Results on MLE-Bench

AIBuildAI achieves top performance on [MLE-Bench](https://github.com/openai/mle-bench), a benchmark that evaluates AI agents on real-world Kaggle machine learning competitions.

![MLE-Bench Results](assets/results.png)
