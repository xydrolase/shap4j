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
    package examples;

    import java.nio.file.Files;
    import java.io.File;
    import shap4j.TreeExplainer;

    class ExampleApp {
        public static void main(String[] args) throws Exception {
            byte[] data = Files.readAllBytes(new File("boston.shap4j").toPath());
            TreeExplainer explainer = new TreeExplainer(data);
            double[] x = {
                    6.320e-03, 1.800e+01, 2.310e+00, 0.000e+00, 5.380e-01, 6.575e+00,
                    6.520e+01, 4.090e+00, 1.000e+00, 2.960e+02, 1.530e+01, 3.969e+02,
                    4.980e+00
            };
            double[] shapValues = explainer.shapValues(x, false);

            System.out.println("SHAP values: " + Arrays.toString(shapValues));
        }
    }
```
