# shap4j

![Build Status](https://api.travis-ci.org/xydrolase/shap4j.svg?branch=master)
[![javadoc](https://javadoc.io/badge2/io.github.xydrolase/shap4j/javadoc.svg)](https://javadoc.io/doc/io.github.xydrolase/shap4j) 

Java interface for the [SHAP (SHapley Additive exPlanations) library](https://github.com/slundberg/shap) for tree 
ensembles (`TreeExplainer`). Note that `shap4j` is not a pure Java port of SHAP. Rather, it utilizes
[`JavaCPP`](https://github.com/bytedeco/javacpp) to provide a Java-Native Interface (JNI) on top of the
[fast C++ implementation of `TreeExplainer`](https://github.com/slundberg/shap/blob/master/shap/tree_shap.h).
In this sense, `shap4j` leverages the same underlying native code that powers the
[Python version of `TreeExplainer`](https://github.com/slundberg/shap#tree-ensemble-example-with-treeexplainer-xgboostlightgbmcatboostscikit-learnpyspark-models),
to ensure validity and efficiency.

Current supported platforms:

 - `macosx-x86_64`
 - `macosx-arm64`
 - `linux-x86_64`

#### Use cases
`shap4j` enables lean SHAP integration in JVM projects, _i.e._ a project can import `shap4j` as the sole dependency,
without having to depend on heavier third-party tree ensemble libraries, 
_e.g._ [`xgboost4j`](https://github.com/dmlc/xgboost/tree/master/jvm-packages).

## Tree ensemble model conversion 
In order for `shap4j` to explain a tree ensemble model, that model must be provided in a `.shap4j` data file, which can
be generated from model dumps (pickle files) of XGBoost/LightGBM/CatBoost/sklearn using the companion Python library
[`shap4j-data-converter`](https://github.com/xydrolase/shap4j-data-converter).

## Usage

#### Maven
```xml
<dependency>
  <groupId>io.github.xydrolase</groupId>
  <artifactId>shap4j-platform</artifactId>
  <version>0.0.3</version>
</dependency>
```

#### SBT
```
libraryDependencies += "io.github.xydrolase" % "shap4j-platform" % "0.0.3"
```

#### Example usage
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
The data file `boston.shap4j` can be found [here](https://github.com/xydrolase/shap4j/blob/master/shap4j/src/test/resources/boston.shap4j).
Or, you can follow the examples in [`shap4j-data-converter`](https://github.com/xydrolase/shap4j-data-converter) to
generate it yourself.

#### API Docs
 - [`TreeExplainer`](https://javadoc.io/doc/io.github.xydrolase/shap4j/latest/shap4j/TreeExplainer.html)
