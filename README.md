# shap4j

JVM interface for the [SHAP (SHapley Additive exPlanations) library](https://github.com/slundberg/shap) for tree 
ensembles (`TreeExplainer`.), built using [`javacpp`](https://github.com/bytedeco/javacpp)

## Use cases
`shap4j` enables lean SHAP integration in JVM projects, without dependencies on third party tree ensemble runtime 
libraries, _e.g._ XGBoost and LightGBM.

## Data generation
To generate SHAP values for a specific tree ensemble model, the model must be provided in a `.shap4j` data file, which
can be generated from model dumps of XGBoost/LightGBM/CatBoost/sklearn using the Python library
[`shap4j-data-converter`](https://github.com/xydrolase/shap4j-data-converter).

## Example usage
```java

```
