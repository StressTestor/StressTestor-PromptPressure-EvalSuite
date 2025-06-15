# Contributing to PromptPressure Eval Suite

Thank you for your interest in contributing! We welcome all forms of contributions, including bug reports, feature requests, documentation improvements, and code contributions.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Reporting Issues](#reporting-issues)
- [Feature Requests](#feature-requests)
- [Pull Requests](#pull-requests)
- [License](#license)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
   ```bash
   git clone https://github.com/your-username/PromptPressure-EvalSuite.git
   cd PromptPressure-EvalSuite
   ```
3. **Set up the development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements-dev.txt
   ```
4. **Create a branch** for your changes
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

1. **Sync with main**
   ```bash
   git fetch upstream
   git merge upstream/main
   ```
2. **Make your changes**
3. **Run tests**
   ```bash
   pytest
   ```
4. **Format your code**
   ```bash
   black .
   isort .
   ```
5. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature"
   ```
6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Open a Pull Request**

## Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code
- Use type hints for all function signatures
- Keep lines under 88 characters
- Use docstrings for all public functions and classes
- Use Google-style docstrings

## Testing

- Write tests for all new features and bug fixes
- Run tests with: `pytest`
- Aim for at least 80% test coverage
- Test different Python versions if possible

## Documentation

- Update the README.md with any new features or changes
- Add docstrings to all new functions and classes
- Update the CHANGELOG.md with notable changes
- Keep the ROADMAP.md up to date

## Reporting Issues

When reporting issues, please include:

1. A clear title and description
2. Steps to reproduce the issue
3. Expected vs actual behavior
4. Environment details (OS, Python version, etc.)
5. Any relevant logs or screenshots

## Feature Requests

We welcome feature requests! Please:

1. Check if a similar feature already exists
2. Explain why this feature would be valuable
3. Provide examples of how it would be used

## Pull Requests

1. Keep PRs focused on a single feature or bug fix
2. Update documentation and tests
3. Ensure all tests pass
4. Follow the PR template
5. Reference any related issues

## License

By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
