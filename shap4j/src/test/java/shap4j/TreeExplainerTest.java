package shap4j;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import org.apache.commons.io.IOUtils;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import shap4j.shap.ShapUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;

public class TreeExplainerTest {
    private byte[] rawData;
    private TreeExplainer explainer;

    private double[][] approximateExpected = {
            {
                    0.        ,  0.        ,  0.00894827,  0.        , -1.1371417 ,
                    -1.4974804 , -0.02694616, -0.43359974,  0.        ,  0.        ,
                    0.        ,  0.        ,  5.345535
            },
            {
                    -0.07767473,  0.        ,  0.        ,  0.        ,  0.05598033,
                    -1.6990675 ,  0.01063945, -0.28113094,  0.        , -0.00349796,
                    0.04095722,  0.0068185 ,  2.621229
            }
    };

    private double[][] expected = {
            {
                    2.1455373e-01,  1.3466644e-03,  1.8625204e-02,  0.0000000e+00,
                    -4.0719920e-01, -1.1618356e+00, -5.1294092e-02, -3.4201440e-01,
                    -4.0446203e-02,  5.4874150e-03,  5.0436568e-02,  2.7335223e-02,
                    3.9443200e+00
            },
            {
                    1.5532924e-01, -1.4507028e-04,  1.5379548e-02,  0.0000000e+00,
                    2.0268284e-01, -1.4210027e+00,  7.8387614e-03, -2.1820213e-01,
                    -1.1171691e-02,  6.8911865e-02,  6.7537002e-02,  3.0773308e-02,
                    1.7763212e+00
            }
    };

    private double[][] X = {
            {
                    6.320e-03, 1.800e+01, 2.310e+00, 0.000e+00, 5.380e-01, 6.575e+00,
                    6.520e+01, 4.090e+00, 1.000e+00, 2.960e+02, 1.530e+01, 3.969e+02,
                    4.980e+00
            },
            {
                    2.7310e-02, 0.0000e+00, 7.0700e+00, 0.0000e+00, 4.6900e-01,
                    6.4210e+00, 7.8900e+01, 4.9671e+00, 2.0000e+00, 2.4200e+02,
                    1.7800e+01, 3.9690e+02, 9.1400e+00
            }
    };

    @BeforeEach
    public void setUp() {
        rawData = ShapUtils.readResourceAsBytes("/boston.shap4j");
        explainer = new TreeExplainer(rawData);
    }

    @Test
    public void testShapValuesForVector() {
        double[] shapValues = explainer.shapValues(X[0], false);
        assertArrayEquals(expected[0], shapValues, 1e-6);
    }

    @Test
    public void testApproximateShapValuesForVector() {
        double[] shapValues = explainer.shapValues(X[0], true, false);
        assertArrayEquals(approximateExpected[0], shapValues, 1e-6);
    }

    @Test
    public void testShapValuesForMatrix() {
        double[][] shapValues = explainer.shapValues(X, false);
        assertArrayEquals(expected[0], shapValues[0], 1e-6);
        assertArrayEquals(expected[1], shapValues[1], 1e-6);
    }

    @Test
    public void testApproximateShapValuesForMatrix() {
        double[][] shapValues = explainer.shapValues(X, true, false);
        assertArrayEquals(approximateExpected[0], shapValues[0], 1e-6);
        assertArrayEquals(approximateExpected[1], shapValues[1], 1e-6);
    }

    @Test
    public void testFromResource() throws IOException {
        TreeExplainer resourceExplainer = TreeExplainer.fromResource("/boston.shap4j");

        double[] shapValues = resourceExplainer.shapValues(X[0], false);
        assertArrayEquals(expected[0], shapValues, 1e-6);
    }

    @Test
    public void testFromFile() throws IOException {
        // copy the resource file to a temporary file on disk
        File tmpFile = Files.createTempFile("boston", ".shap4j").toFile();
        tmpFile.deleteOnExit();

        try (OutputStream os = new FileOutputStream(tmpFile)) {
            IOUtils.copy(this.getClass().getResourceAsStream("/boston.shap4j"), os);
        }

        TreeExplainer fileExplainer = TreeExplainer.fromFile(tmpFile.getAbsolutePath());

        double[] shapValues = fileExplainer.shapValues(X[0], false);
        assertArrayEquals(expected[0], shapValues, 1e-6);
    }
}
