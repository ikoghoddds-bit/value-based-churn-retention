# Validation record

November 6, 2026 · Python 3.12 · Pension synthetic v2

- Generated 14,400 synthetic member records using the versioned seed and process.
- Executed fitting, calibration, model selection and temporal evaluation with label-maturity checks between stages; excluded 3,600 embargo rows overall.
- Executed all eight notebook code cells sequentially and captured results and figures. The build environment blocks kernel sockets, so execution used in-process IPython; the file is a standard Python 3 notebook for normal Jupyter/VS Code use.
- Tests cover member-assets versus fee arithmetic, flag/capacity logic, zero-effect cases, temporal stage boundaries, target exclusion, scoring schema, invalid inputs, serialization consistency and queue eligibility.
- Streamlit AppTest checks initial rendering, zero save rate and zero transfer fraction. No UI exceptions; zero assumptions correctly remove the corresponding financial exposure/benefit.
- Batch scoring succeeded on the example input with no target/outcome columns.
- Evaluation chart visually inspected; README references local evidence and actual CSV exports.
- ZIP contents and notebook outputs validated before delivery.

GitHub Actions workflows are included but have not been executed remotely. Local tests do not establish real pension predictive performance, treatment effect, real savings, regulatory compliance or production readiness. No external repository or deployment was created.
