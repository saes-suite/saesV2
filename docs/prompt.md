I am developing SAES2, a modern Python library for statistical analysis.

The existing data input specification is described in:

`/Users/emiliorodrigocarreiravillalta/Desktop/SAES2/README.md`

**IMPORTANT:** Read that README first and treat its data format specification as the source of truth. Do not invent a different input format.

The goal is to turn SAES2 into a modern, production-quality Python statistical analysis library with both a programmatic API and a GUI.

I want you to approach this as a senior Python software architect AND as a statistician. Do not blindly implement my assumptions. If any part of my proposed statistical methodology is incorrect or statistically incomplete, explicitly explain why and propose the correct approach before implementing it.

## 1. Project goals

SAES2 should eventually support this general workflow:

```text
Input file
    ↓
File parsing
    ↓
Schema / structural validation
    ↓
Dataset validation
    ↓
Validated dataset
    ↓
Data exploration
    ├── Descriptive statistics
    ├── Boxplots
    └── Statistical assumption analysis
    ↓
Statistical test selection
    ↓
Statistical analysis
    ↓
Results
    ├── Test statistics
    ├── p-values
    ├── Effect sizes
    ├── Confidence intervals
    ├── Assumption diagnostics
    └── Human-readable interpretation
```

For the first version, support:

### Input

The input format defined in the existing README.

The architecture should make it straightforward to later support:

* CSV
* JSON
* Excel
* Parquet
* other common tabular formats

Do not implement all of these immediately unless necessary. Design the abstraction so adding them later does not require modifying the statistical core.

### Visualization

Initially support:

* Boxplots

However, visualization must be designed around an extensible architecture so that later we can add:

* histograms
* violin plots
* scatter plots
* Q-Q plots
* bar plots
* etc.

Do not tightly couple the statistical logic to a specific plotting implementation.

### Statistical analysis

Initially support:

* t-tests
* one-way ANOVA

But design the statistical-analysis layer so additional tests can be added without modifying existing tests or creating a giant conditional/switch statement.

Before implementing this part, carefully determine the correct statistical decision process.

For example, do NOT assume:

> "If the data is normally distributed, use t-test or ANOVA."

The correct test depends on factors such as:

* number of groups
* independent vs paired/repeated observations
* normality
* homogeneity of variances
* experimental design
* variable types
* sample size

For the initial implementation, determine and document the assumptions and decision rules required for:

* independent two-sample t-test
* paired t-test, if appropriate
* one-way ANOVA

Also determine whether Welch's t-test should be the default when equal variances cannot be assumed.

For ANOVA, determine how assumption violations should be handled and whether Welch ANOVA should be included in the initial version or explicitly deferred.

Do not implement non-parametric tests yet unless you consider them essential to make the initial statistical decision system mathematically correct. If they are deferred, architect the system so they can be added later.

The results of every statistical test should be represented by structured result objects rather than raw tuples or unstructured dictionaries.

Results should ideally expose information such as:

* test name
* statistic
* p-value
* degrees of freedom
* effect size
* confidence interval
* assumptions evaluated
* warnings
* interpretation

Do not overstate statistical conclusions. The library should distinguish clearly between statistical significance and practical significance.

## 2. Data validation

The file-validation system is extremely important.

When a user provides a file, the GUI should first determine whether the dataset is valid.

Validation should distinguish between:

* file-format errors
* schema errors
* missing required columns
* invalid data types
* malformed values
* missing values
* duplicated observations where relevant
* statistical problems discovered later during analysis

The validation system should return structured validation results rather than simply raising generic exceptions.

The GUI should be able to explain errors to the user in a clear human-readable way.

The API should expose the same validation functionality programmatically.

## 3. Architecture

Design the project using clean architecture / modular architecture principles.

The statistical core must NOT depend on the GUI.

The project should conceptually separate:

```text
Core/domain
Application/use cases
Infrastructure
Visualization
API
GUI
```

The exact architecture is your decision, but explain why you choose it.

The system should follow principles such as:

* SOLID
* separation of concerns
* dependency inversion
* explicit domain models
* type hints
* immutable result objects where appropriate
* small cohesive modules
* dependency injection where useful
* extensibility
* testability

Avoid overengineering. The architecture should be sophisticated but appropriate for a Python scientific library.

## 4. Python ecosystem

Use modern Python tooling.

The project should use:

* `pyproject.toml`
* `uv`
* modern Python typing
* Ruff
* pytest
* mypy or an appropriate static type checker
* pre-commit if justified
* CI through GitHub Actions
* proper package metadata
* automated tests
* coverage
* documentation

Use a modern supported Python version.

Choose dependencies carefully and justify them.

For numerical/statistical functionality, evaluate appropriate established libraries such as:

* NumPy
* pandas
* SciPy
* statsmodels

Do not unnecessarily reimplement established statistical algorithms.

SAES should provide a clean domain-level API on top of these libraries.

## 5. API

The library should be pleasant to use programmatically.

For example, the final API should conceptually allow something like:

```python
dataset = saes.load("data.csv")

validation = dataset.validate()

if validation.is_valid:
    analysis = saes.analyze(dataset)
```

and eventually something like:

```python
result = saes.statistics.ttest(...)
```

or another API you consider architecturally superior.

The public API should be stable and significantly simpler than the underlying implementation.

Internal implementation details should not leak into the public API.

## 6. GUI

SAES should also have a GUI.

The GUI should initially provide:

1. File selection
2. File validation
3. Clear validation feedback
4. Dataset overview
5. Boxplot generation
6. Statistical analysis selection
7. Results visualization
8. Export of results where appropriate

The GUI should communicate with the application layer rather than directly manipulating statistical implementation details.

Choose an appropriate Python GUI/web technology and justify the choice.

The GUI must not contain statistical business logic.

## 7. Error handling

Design a coherent exception hierarchy.

Differentiate between:

* input errors
* validation errors
* configuration errors
* statistical-analysis errors
* visualization errors
* infrastructure errors

Do not use broad `except Exception` blocks unless there is a very good reason.

Errors should preserve enough context to diagnose failures.

## 8. Testing

Testing is a first-class requirement.

Include:

* unit tests
* integration tests
* statistical correctness tests
* input validation tests
* API tests
* visualization tests where appropriate

For statistical tests, use known datasets / analytically verifiable results to ensure that SAES produces correct results.

The tests should verify numerical correctness, not merely that functions execute successfully.

Include edge cases such as:

* empty datasets
* missing values
* constant variables
* insufficient observations
* invalid grouping variables
* unequal group sizes
* unequal variances
* non-normal data
* malformed files

## 10. Development process

Do not immediately generate a huge amount of code.

First:

1. Read and understand the existing README.
2. Inspect the existing repository structure and code.
3. Identify what can be reused.
4. Identify architectural problems.
5. Propose the target architecture.
6. Explain the statistical methodology and decision tree.
7. Propose the package structure.
8. Propose the dependency/tooling strategy.
9. Propose an incremental implementation plan.

Only after that should implementation begin.

When implementing, work incrementally and keep the project runnable and tested after each major step.

Do not rewrite working code unnecessarily.

## 11. Important requirement

Think of SAES as a real open-source scientific Python library, not as a one-off application.

The architecture must allow us to add new statistical tests, data formats, visualizations, reporting formats and GUI functionality without rewriting the existing system.

At the same time, avoid premature abstraction and unnecessary design-pattern complexity.

Whenever there is a conflict between my initial assumptions and established statistical methodology or good software engineering practice, explicitly point it out and explain the reasoning before making the implementation decision.
