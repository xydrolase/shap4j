package shap4j.shap;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class ExplanationDatasetTest {
    @Test
    public void testNonEmptyAssertion() {
        assertThrows(
                AssertionError.class,
                () -> {
                    double [][] emptyMatrix = {};
                    ExplanationDataset.fromMatrix(emptyMatrix, false);
                }
        );
    }

    @Test
    public void testExplanationDatasetFromMatrix() {
        double[][] matrix = {{1.0, 2.0, 3.0, Double.NaN}};
        ExplanationDataset dset = ExplanationDataset.fromMatrix(matrix, false);

        assertEquals(2.0, dset.X().get(1));
        assertTrue(Double.isNaN(dset.X().get(3)));
        // all X_missing will be set to false, even if there are missing values
        for (int i = 0; i < dset.X_missing().capacity(); ++i) {
            assertFalse(dset.X_missing().get(i));
        }
    }

    @Test
    public void testExplanationDatasetFromMatrixWithMultiRows() {
        double[][] matrix = {{1.0, 2.0, 3.0, Double.NaN}, {4.0, 5.0, 6.0, 7.0}};
        ExplanationDataset dset = ExplanationDataset.fromMatrix(matrix, false);

        // data should be row-major
        assertEquals(2.0, dset.X().get(1));
        assertEquals(4.0, dset.X().get(4));

        // all X_missing will be set to false, even if there are missing values
        for (int i = 0; i < dset.X_missing().capacity(); ++i) {
            assertFalse(dset.X_missing().get(i));
        }
    }

    @Test
    public void testExplanationDatasetFromMatrixWithCheckMissing() {
        double[][] matrix = {{1.0, 2.0, 3.0, Double.NaN}, {4.0, 5.0, 6.0, 7.0}};
        ExplanationDataset dset = ExplanationDataset.fromMatrix(matrix, true);

        for (int i = 0; i < dset.X_missing().capacity(); ++i) {
            if (i != 3) {
                assertFalse(dset.X_missing().get(i));
            } else {
                assertTrue(dset.X_missing().get(i));
            }
        }
    }
}
