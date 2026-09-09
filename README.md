# GenPark Instrumental Variable Two-Stage Least Squares (2SLS) Estimator Skill

Instrumental Variable (IV) Two-Stage Least Squares (2SLS) causal estimator isolating unobserved confounding.

Read more at [GenPark](https://genpark.ai) and the [GenPark MCP Catalog](https://genpark.ai/mcp).

```mermaid
graph LR
    Z[Exogenous Instrument Z] -->|Relevance| X[Endogenous Treatment X]
    U((Unobserved Confounder U)) --> X
    U --> Y[Outcome Y]
    X -->|True Causal Effect beta| Y
    style Z fill:#c8e6c9
    style U fill:#ffcdd2
    style X fill:#fff9c4
    style Y fill:#bbdefb
```

## Features
- Pure Python 2SLS estimator with Stage 1 instrument strength F-test.
- Unbiased causal effect recovery under unobserved endogeneity.
- Zero external dependencies.
