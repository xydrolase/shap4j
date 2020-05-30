package shap4j;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.DoublePointer;
import shap4j.shap.ExplanationDataset;
import shap4j.shap.TreeEnsemble;
import shap4j.shap.TreeShap;

public class TreeExplainer {
    private static final int TREE_PATH_DEPENDENT_FEATURE = 1;
    private static final int IDENTITY_TRANSFORM = 0;

    private TreeEnsemble treeEnsemble;

    private TreeExplainer(TreeEnsemble ensemble) {
        this.treeEnsemble = ensemble;
    }

    public TreeExplainer(byte[] rawData) {
        this.treeEnsemble = TreeEnsemble.fromBytes(rawData);
    }

    // TODO: add support for TreeExplainer.data (a background dataset to use for integrating out features.)
    public double[][] shapValues(double [][] matrix, boolean checkMissing) {
        assert matrix.length > 0;

        int nRows = matrix.length;
        int nCols = matrix[0].length + 1;

        if (treeEnsemble.num_outputs() != 1) {
            throw new IllegalArgumentException("Currently only supporting models with num_outputs == 1");
        }

        ExplanationDataset dataset = ExplanationDataset.fromMatrix(matrix, checkMissing);
        DoublePointer phi = new DoublePointer(nRows * nCols);
        // set initial values for the SHAP values to zero, as tree_shap_recursive adds to these values.
        BytePointer.memset(phi, 0, nRows  * nCols * 8);

        TreeShap.dense_tree_shap(treeEnsemble, dataset, phi,
                TREE_PATH_DEPENDENT_FEATURE, IDENTITY_TRANSFORM, false
        );

        double[][] values = new double[nRows][nCols - 1];
        int offset = 0;
        for (int i = 0; i < nRows; ++i, offset += nCols) {
            phi.position(offset).limit(offset + nCols).asBuffer().get(values[i], 0, nCols - 1);
        }

        dataset.close();
        phi.close();

        return values;
    }

    public double[] shapValues(double [] vector, boolean checkMissing) {
        double[][] matrix = {vector};

        return shapValues(matrix, checkMissing)[0];
    }
}
