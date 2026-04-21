# LLM Product Description Generator & Judge

This project implements a pipeline for generating high-quality product descriptions using Large Language Models (LLMs) and evaluating them using an automated "LLM-as-a-Judge" system. It was developed as part of HW1 for the Nebius Academy project.

## Overview

The core objective is to automate the creation of product descriptions from structured data (product name, attributes, material, etc.) while ensuring the output remains grounded, fluent, and professional. The project includes a comprehensive evaluation framework to score the generated content across multiple criteria.

## Key Features

- **Description Generation**: Automated pipeline to transform product attributes into human-like descriptions.
- **Automated Judging**: An "LLM-as-a-Judge" pipeline that scores descriptions based on:
  - **Fluency**: Smoothness and readability of the text.
  - **Grammar**: Correctness of syntax and spelling.
  - **Tone**: Professional and engaging style.
  - **Length**: Adherence to length constraints.
  - **Grounding**: Verification that all facts come strictly from the source data.
- **Experimentation Framework**: Support for running experiments with different models, temperatures, and system prompts to optimize performance.
- **Data-Driven Analysis**: Integration with Jupyter Notebooks for detailed analysis of latency, cost, and quality metrics.

## Project Structure

```text
├── src/
│   ├── main.py            # Main entry point for the generation pipeline
│   ├── judge_main.py      # Main entry point for the evaluation (judge) pipeline
│   ├── pipeline.py       # Core logic for data processing and generation
│   ├── llm_client.py     # Wrapper for LLM API calls
│   ├── prompts.py        # System and user prompt templates
│   ├── schemas.py        # Pydantic models for structured outputs
│   └── judge/             # Sub-package for judging logic
├── notebooks/             # Task-specific analysis and development notebooks
├── data/                  # Input datasets and experiment results
├── pyproject.toml         # Project dependencies and metadata
└── .env                   # Configuration for API keys
```

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended for dependency management)

### Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   uv sync
   ```
3. Set up your environment variables:
   Create a `.env` file in the root directory and add your API key:
   ```text
   NEBIUS_TOKEN_FACTORY_API_KEY=your_api_key_here
   ```

### Running the Pipeline

#### 1. Generate Product Descriptions
Run the main generation script:
```bash
python -m src.main --take-rows 10
```
Optional arguments:
- `--experiment <name>`: Run a specific prompt/config experiment.
- `--llm-model <model>`: Specify the model (e.g., `gpt-4o`).
- `--temperature <float>`: Set the sampling temperature.

#### 2. Run the Judge
Evaluate the generated descriptions:
```bash
python -m src.judge_main --experiment grounding_only
```

## Development and Analysis

The `notebooks/` directory contains several Jupyter notebooks used for incremental development and deep-dive analysis:
- `task1_rubric.ipynb`: Developing the evaluation criteria.
- `task2_generation.ipynb`: Testing the generation logic.
- `task4_experiments.ipynb`: Comparing different configurations.
- `task6_analysis.ipynb`: Visualizing results and performance metrics.
