package shap4j;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.DoublePointer;
import shap4j.shap.ExplanationDataset;
import shap4j.shap.TreeEnsemble;
import shap4j.shap.TreeShap;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;

/**
 * A SHAP explainer using Tree SHAP algorithms to explain the output of tree ensemble models.
 *
 * @see <a href="https://github.com/slundberg/shap/blob/master/shap/explainers/tree.py">Python interface for TreeExplainer</a>
 */
public class TreeExplainer {
    private static final int TREE_PATH_DEPENDENT_FEATURE = 1;
    private static final int IDENTITY_TRANSFORM = 0;

    private TreeEnsemble treeEnsemble;

    private TreeExplainer(TreeEnsemble ensemble) {
        this.treeEnsemble = ensemble;
    }

    /**
     * Create a instance of <code>TreeExplainer</code> from binary data: <code>rawData</code>.
     * @see <a href="https://github.com/xydrolase/shap4j-data-converter/">shap4j-data-converter for creating the data</a>
     * @param rawData The binary raw data representing a tree ensemble model, upon which a <code>TreeExplainer</code> is
     *                based.
     */
    public TreeExplainer(byte[] rawData) {
        this.treeEnsemble = TreeEnsemble.fromBytes(rawData);
    }

    /**
     * Compute the SHAP values for a given <code>ExplanationDataset</code>.
     *
     * @param dataset An instance of <code>ExplanationDataset</code>.
     * @param approximate Run the approximate Saabas algorithm which is fast but only considers a single feature
     *                    ordering. See https://github.com/slundberg/shap/edit/master/shap/explainers/tree.py for
     *                    more details.
     * @return The SHAP values in a 2d array, which is of the same shape as <code>dataset.X()</code>.
     */
    public double[][] shapValues(ExplanationDataset dataset, boolean approximate) {
        int nRows = dataset.getNumRows();
        // the SHAP values, or phi, has an extra column
        int nCols = dataset.getNumCols() + 1;

        DoublePointer phi = new DoublePointer(nRows * nCols);
        // set initial values for the SHAP values to zero, as tree_shap_recursive adds to these values.
        BytePointer.memset(phi, 0, nRows  * nCols * 8);

        if (approximate) {
            TreeShap.dense_tree_saabas(phi, treeEnsemble, dataset);
        } else {
            TreeShap.dense_tree_shap(treeEnsemble, dataset, phi,
                    TREE_PATH_DEPENDENT_FEATURE, IDENTITY_TRANSFORM, false
            );
        }

        double[][] values = new double[nRows][nCols - 1];
        int offset = 0;
        for (int i = 0; i < nRows; ++i, offset += nCols) {
            phi.position(offset).limit(offset + nCols).asBuffer().get(values[i], 0, nCols - 1);
        }

        dataset.close();
        phi.close();

        return values;
    }

    /**
     * Compute the SHAP values for a given <code>ExplanationDataset</code> using the exact algorithm.
     * @param dataset An instance of <code>ExplanationDataset</code>.
     * @return The SHAP values in a 2d array, which is of the same shape as <code>dataset.X()</code>.
     */
    public double[][] shapValues(ExplanationDataset dataset) {
        return shapValues(dataset, false);
    }

    /**
     * Compute the SHAP values for a given 2-dimensional matrix: <code>matrix</code>.
     * @param matrix The 2d matrix from which the SHAP values are computed. Each row in this matrix should correspond
     *               to a feature vector.
     * @param approximate Run the approximate Saabas algorithm which is fast but only considers a single feature
     *                    ordering. See https://github.com/slundberg/shap/edit/master/shap/explainers/tree.py for
     *                    more details.
     * @param checkMissing Whether to check missing values in <code>matrix</code>. If set to false, all values in
     *                     <code>matrix</code> are assumed to be non-missing (i.e. not <code>NaN</code>.)
     * @return A 2d matrix containing the SHAP values, which should be of the same shape as the input <code>matrix</code>.
     */
    public double[][] shapValues(double [][] matrix, boolean approximate, boolean checkMissing) {
        assert matrix.length > 0;

        if (treeEnsemble.num_outputs() != 1) {
            throw new IllegalArgumentException("Currently only supporting models with num_outputs == 1");
        }

        ExplanationDataset dataset = ExplanationDataset.fromMatrix(matrix, checkMissing);

        return shapValues(dataset, approximate);
    }

    /**
     * Compute the SHAP values for a given 2-dimensional matrix: <code>matrix</code>, using the exact algorithm.
     * @param matrix The 2d matrix from which the SHAP values are computed. Each row in this matrix should correspond
     *               to a feature vector.
     * @param checkMissing Whether to check missing values in <code>matrix</code>. If set to false, all values in
     *                     <code>matrix</code> are assumed to be non-missing (i.e. not <code>NaN</code>.)
     * @return A 2d matrix containing the SHAP values, which should be of the same shape as the input <code>matrix</code>.
     */
    public double[][] shapValues(double [][] matrix, boolean checkMissing) {
        return shapValues(matrix, false, checkMissing);
    }

    /**
     * Compute SHAP values for a given feature vector: <code>vector</code>.
     * @param vector A feature vector compatible with the tree ensemble model.
     * @param approximate Run the approximate Saabas algorithm which is fast but only considers a single feature
     *                    ordering. See https://github.com/slundberg/shap/edit/master/shap/explainers/tree.py for
     *                    more details.
     * @param checkMissing Whether to check missing values in the feature vector (<code>NaN</code>'s)
     * @return An array containing SHAP values, which is of the same length of the input <code>vector</code>.
     */
    public double[] shapValues(double [] vector, boolean approximate, boolean checkMissing) {
        double[][] matrix = {vector};

        return shapValues(matrix, approximate, checkMissing)[0];
    }

    /**
     * Compute SHAP values for a given feature vector: <code>vector</code>, using the exact algorithm.
     * @param vector A feature vector compatible with the tree ensemble model.
     * @param checkMissing Whether to check missing values in the feature vector (<code>NaN</code>'s)
     * @return An array containing SHAP values, which is of the same length of the input <code>vector</code>.
     */
    public double[] shapValues(double [] vector, boolean checkMissing) {
        double[][] matrix = {vector};

        return shapValues(matrix, false, checkMissing)[0];
    }

    /**
     * Create a <code>TreeExplainer</code> from a resource file of the <code>.shap4j</code> format.
     * @param resource The path to the resource file, e.g. <code>"/boston.shap4j"</code> if the file is located under
     *                 <code>src/java/main/resources</code>.
     * @return An <code>TreeExplainer</code> instance corresponding to the tree model contained by the resource file.
     * @throws IOException
     * @see <a href="https://github.com/xydrolase/shap4j-data-converter>shap4j-data-converter to create the .shap4j file</a>
     */
    public static TreeExplainer fromResource(String resource) throws IOException {
        try(InputStream is = TreeExplainer.class.getResourceAsStream(resource)) {
            byte[] data = new byte[is.available()];
            is.read(data);

            return new TreeExplainer(data);
        }
        catch (NullPointerException e) {
            return null;
        }
    }

    /**
     * Create a <code>TreeExplainer</code> from a local file of the <code>.shap4j</code> format.
     * @param filePath The file path to the <code>.shap4j</code> file.
     * @return An <code>TreeExplainer</code> instance corresponding to the tree model contained by the local file.
     * @throws IOException
     * @see <a href="https://github.com/xydrolase/shap4j-data-converter>shap4j-data-converter to create the .shap4j file</a>
     */
    public static TreeExplainer fromFile(String filePath) throws IOException {
        byte[] data = Files.readAllBytes(new File(filePath).toPath());

        return new TreeExplainer(data);
    }
}
