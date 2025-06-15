<div align="center">
  <img src="./assets/logo.png" alt="PromptPressure Logo" width="200"/>
  
  # PromptPressure Eval Suite
  
  [![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
  [![GitHub Issues](https://img.shields.io/github/issues/StressTestor/PromptPressure-EvalSuite)](https://github.com/StressTestor/PromptPressure-EvalSuite/issues)
  [![GitHub Stars](https://img.shields.io/github/stars/StressTestor/PromptPressure-EvalSuite?style=social)](https://github.com/StressTestor/PromptPressure-EvalSuite/stargazers)

  > A comprehensive evaluation suite for LLM assessment with CI/CD integration, automated visualizations, and advanced analytics.
  > 
  > **Features:** Automated model evaluation pipelines • Interactive dashboards • Multi-provider support • Extensible architecture
</div>

## 🌟 Features

- **Multi-Model Support**
  - Seamless integration with OpenAI, Groq, LM Studio, and custom models
  - Dynamic adapter system for easy extension
  - Local and cloud deployment options

- **Comprehensive Evaluation**
  - Refusal mapping and instruction following
  - Tone consistency and psychological reasoning
  - Performance metrics and benchmarking

- **Developer Experience**
  - CI/CD ready with GitHub Actions
  - Automated testing and validation
  - Detailed logging and debugging
  - Type hints and documentation

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- [Poetry](https://python-poetry.org/) (recommended) or pip
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/StressTestor/PromptPressure-EvalSuite.git
   cd PromptPressure-EvalSuite
   ```

2. **Set up the environment**
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # For development
   pip install -r requirements-dev.txt
   ```

3. **Configure environment variables**
   Create a `.env` file in the project root:
   ```env
   # Required API Keys
   OPENAI_API_KEY=your-openai-key-here
   GROQ_API_KEY=your-groq-key-here
   
   # Optional: LM Studio configuration
   LMSTUDIO_ENDPOINT=http://localhost:1234/v1
   ```

### Basic Usage

```bash
# Run evaluation with default settings
python run_eval.py

# Run with specific model
python run_eval.py --model groq --output results.csv

# Generate visualizations
python visualization/generate_plots.py
```

### Development

```bash
# Run tests
pytest

# Format code
black .

# Check types
mypy .

# Lint code
flake8 .
```

## 📚 Documentation

For detailed documentation, please visit our [documentation site](https://stressTestor.github.io/PromptPressure-EvalSuite).

- [API Reference](https://stressTestor.github.io/PromptPressure-EvalSuite/api)
- [Examples](https://stressTestor.github.io/PromptPressure-EvalSuite/examples)
- [Contributing Guide](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)

---

#### 🚀 Key Features (v1.5.3+)

- **Automated CI/CD Pipeline**
  - GitHub Actions workflow for automated testing
  - Trigger evaluations on dataset/model changes
  - Automatic artifact generation and storage

- **Advanced Visualization**
  - Interactive model performance dashboards
  - Success rate tracking over time
  - Latency distribution analysis
  - Model comparison tools

- **Enhanced Adapter System**
  - Dynamic adapter selection based on model provider
  - Built-in support for Groq, OpenAI, LM Studio, and Mock
  - Easy extension for new model providers
  - Comprehensive error handling and retries

- **Evaluation Categories**
  - Refusal Sensitivity
  - Instruction Following
  - Psychological Reasoning
  - Tone & Role Consistency
  - Emergent Story Logic

## 📁 Project Structure

```
.
├── .github/
│   └── workflows/           # GitHub Actions workflows
│       └── evaluation.yml    # CI/CD pipeline definition
├── adapters/                # Model adapters
│   ├── __init__.py          # Adapter registry
│   ├── groq_adapter.py      # Groq API integration
│   ├── openai_adapter.py    # OpenAI API integration
│   ├── lmstudio_adapter.py  # Local LM Studio integration
│   └── mock_adapter.py      # Mock adapter for testing
├── visualization/           # Visualization scripts
│   ├── generate_plots.py    # Plot generation
│   └── dashboard.py         # Interactive dashboard (coming soon)
├── configs/                 # Configuration presets
│   ├── default.yaml         # Default configuration
│   └── lmstudio.yaml        # LM Studio specific settings
├── datasets/                # Evaluation datasets
│   └── evals_dataset.json   # Default evaluation dataset
├── outputs/                 # Evaluation results
│   └── results_*.csv        # Generated result files
├── run_eval.py              # Main evaluation script
├── deepseek_post_analysis.py # Analysis and reporting
├── requirements.txt          # Production dependencies
├── requirements-dev.txt      # Development dependencies
└── README.md                # This file
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Git
- (Optional) Docker for containerized execution

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/StressTestor/PromptPressure-EvalSuite.git
   cd PromptPressure-EvalSuite
   ```

2. **Set up a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   # Core dependencies
   pip install -r requirements.txt
   
   # For development and visualization
   pip install -r requirements-dev.txt
   ```

4. **Configure environment variables**
   Create a `.env` file in the project root:
   ```env
   # Required API Keys
   OPENAI_API_KEY=your-openai-key-here
   GROQ_API_KEY=your-groq-key-here
   
   # Optional: LM Studio configuration
   LMSTUDIO_ENDPOINT=http://localhost:1234/v1
   
   # Logging
   LOG_LEVEL=INFO
   ERROR_LOG=error_log.txt
   ```

### Running Evaluations

#### Basic Usage
```bash
# Run with default settings
python run_eval.py

# Specify model and output file
python run_eval.py --model groq --output outputs/results.csv

# Run post-analysis and generate visualizations
python deepseek_post_analysis.py
```

#### Advanced Usage
```bash
# Run with custom configuration
python run_eval.py --config configs/custom.yaml

# Generate visualizations only
python visualization/generate_plots.py --input outputs/ --output visualization/

# Run in simulation mode (no API calls)
python run_eval.py --simulation --model mock
```

## 📊 CI/CD Integration

The GitHub Actions workflow automatically runs on:
- Pushes to `main` or `release/*` branches
- Changes to `datasets/`, `models/`, or `adapters/`
- Manual workflow dispatch

### Manual Trigger
1. Go to GitHub Actions
2. Select "Run workflow"
3. Choose branch and optional parameters
4. View results in the Actions tab

## 📈 Visualizations

### Success Rate Over Time
![Success Rate](visualization/success_rate.png)

### Latency Distribution
![Latency](visualization/latency_distribution.png)

### Model Comparison
```bash
# Generate comparison report
python visualization/compare_models.py --input outputs/ --output reports/
```

## 🔌 Adapters

### Available Adapters
- **OpenAI Adapter** - For GPT-4, GPT-3.5, and other OpenAI models
- **Groq Adapter** - For ultra-fast inference with Groq API
- **LM Studio Adapter** - For local model inference
- **Mock Adapter** - For testing and development

### Adding a New Adapter
1. Create a new file in `adapters/` (e.g., `my_adapter.py`)
2. Implement the required interface:
   ```python
   def generate_response(
       prompt: str,
       model_name: str,
       config: Dict[str, Any]
   ) -> Union[str, List[str], Iterator[str]]:
       # Your implementation here
       pass
   ```
3. Register it in `adapters/__init__.py`:
   ```python
   from .my_adapter import generate_response as my_adapter_resp
   
   ADAPTER_REGISTRY = {
       # ... existing adapters ...
       'my_adapter': my_adapter_resp
   }
   ```

## 📊 Metrics & Monitoring

### Key Metrics
- **Success Rate**: Percentage of successful responses
- **Latency**: Response time in seconds
- **Token Usage**: Input/Output tokens per request
- **Error Rate**: Percentage of failed requests

### Monitoring Setup
1. **Local Monitoring**
   ```bash
   # Start monitoring dashboard
   python -m http.server 8000 -d visualization/
   ```
   Then open `http://localhost:8000` in your browser

2. **Cloud Monitoring**
   - Configure your preferred monitoring solution (Datadog, Prometheus, etc.)
   - Point it to the metrics endpoint (coming in v1.6)

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black .

# Check types
mypy .

# Lint code
flake8 .
```

## 📄 License

MIT License. See [LICENSE](LICENSE) for more information.

## 📚 Resources

- [Documentation](https://github.com/StressTestor/PromptPressure-EvalSuite/wiki)
- [Changelog](CHANGELOG.md)
- [Issue Tracker](https://github.com/StressTestor/PromptPressure-EvalSuite/issues)

## 🙏 Acknowledgments

- Project maintained by Joseph Grey
- Built with ❤️ for the AI community
- Inspired by OpenAI's API Research & Evals initiative

---

<div align="center">
  Made with Python and ❤️ | v1.5.3
</div>