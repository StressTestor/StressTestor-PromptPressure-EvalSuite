# Changelog

All notable changes to the PromptPressure Eval Suite will be documented in this file.

## [1.5.3] - 2025-06-14

### Added
- **CI/CD Pipeline**
  - GitHub Actions workflow for automated testing and evaluation
  - Automated artifact generation and storage
  - Support for scheduled and manual workflow triggers

- **Visualization**
  - Success rate tracking over time
  - Latency distribution analysis
  - Model comparison tools
  - Interactive dashboards

- **Adapters**
  - Dynamic adapter selection based on model provider
  - Improved error handling and retry logic
  - Support for LM Studio local inference

- **Documentation**
  - Comprehensive README with setup and usage instructions
  - API documentation for extending adapters
  - Contribution guidelines

### Changed
- **Code Structure**
  - Reorganized project layout for better maintainability
  - Improved type hints and docstrings
  - Enhanced error messages and logging

- **Configuration**
  - Simplified environment variable handling
  - Support for multiple configuration presets
  - Better defaults and validation

### Fixed
- **Bug Fixes**
  - Fixed race conditions in concurrent evaluations
  - Improved handling of API rate limits
  - Better error recovery and reporting

## [1.4.0] - 2025-05-15

### Added
- Initial release with support for Groq, OpenAI, and Mock adapters
- Basic evaluation framework for LLM assessment
- Core functionality for refusal mapping and instruction following

---

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
