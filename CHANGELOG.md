# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - 2026-05-09
### Added
- Model Explainability tab with SHAP local explanations.
- Defensive model loading checks (predict_proba validation).
- Executive summary JSON export.
- Input sanity hints for high balance/low salary.

### Changed
- Refactored What-If Simulator to automatically sync with Prediction tab inputs.
- Moved versioning to a single source of truth in `src/version.py`.
- Bumped application version to 2.0.0.

## [1.1.0] - 2026-05-09
### Added
- Version display in the application header.

## [1.0.0] - 2026-05-09
### Added
- Initial release of the Bank Churn Predictive System.
- Basic prediction tab and What-If simulator.
