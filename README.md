# Generalized Linear Models for Econometric Analysis of Public Policy

Published: draft
Medium: [https://medium.com/@kyle-t-jones/generalized-linear-models-for-econometric-analysis-of-public-policy-915736acfa29](https://medium.com/@kyle-t-jones/generalized-linear-models-for-econometric-analysis-of-public-policy-915736acfa29)

## Business context

Generalized Linear Models (GLMs) extend the flexibility of traditional linear regression by allowing the dependent variable to follow different probability distributions. In public policy analysis, outcomes often involve counts, proportions, or binary responses, which do not satisfy the normality assumption of ordinary least squares (OLS) regression. GLMs address this limitation by modeling the relationship between the mean of the dependent variable and a linear combination of explanatory variables through a link function. This makes GLMs indispensable for accurate public policy evaluation.

Public policy decisions rely on a wide variety of data types. For example, public health policies use count data to analyze disease incidence, crime policies use rate data to evaluate crime trends, and election policies use binary data to predict voter turnout. Traditional linear regression assumes a normal distribution and constant variance, which are inappropriate for these data types.

GLMs provide a more flexible framework by allowing the dependent variable to follow distributions such as:

- Binomial Distribution for binary outcomes (e.g., policy adoption: yes/no)
- Poisson Distribution for count data (e.g., crime incidents, healthcare visits)
- Gamma Distribution for continuous, positive data (e.g., healthcare costs)

## Companion code

Companion materials for the Medium article on Generalized Linear Models — see `article.md` for the full narrative, including a worked Poisson regression example modeling hospital visits as a function of public health interventions, age, income, and urban residence.

## Project Structure

```
.
├── README.md           # This file
├── main.py             # Main entry point
├── config.yaml         # Configuration file
├── pyproject.toml      # Python dependencies
├── src/
│   └── core.py         # Data simulation, Poisson regression, and diagnostic plots
├── tests/              # Unit tests
└── images/             # Generated plots and figures
```

## Usage

```bash
python main.py
```

This simulates the public health intervention / hospital visits case study
from the article and fits a Poisson regression of `hospital_visits` on
`intervention`, `age`, `income`, and `urban`, printing the model summary,
deviance, and degrees of freedom, and saving diagnostic plots.

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).