package shap4j.shap;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.ByRef;
import org.bytedeco.javacpp.annotation.Const;
import org.bytedeco.javacpp.annotation.Platform;

@Platform(include="shap4j/shap/tree_shap.h")
public class ExplanationDataset extends Pointer {
    static {
        Loader.load();
    }

    private ExplanationDataset() {
        allocate();
    }
    private native void allocate();

    private PointerScope scope = null;

    // TODO: check memory leak: do we need to deallocate the pointers manually?
    private ExplanationDataset(DoublePointer X, BoolPointer X_missing, DoublePointer y, DoublePointer R,
                               BoolPointer R_missing, int num_X, int M, int num_R) {
        scope = new PointerScope();

        // attach all non-null pointers to a local scope, so that when ExplanationDataset is closed, all attached
        // pointers are closed/released accordingly as well.
        if (X != null) scope.attach(X);
        if (X_missing != null) scope.attach(X_missing);
        if (y != null) scope.attach(y);
        if (R != null) scope.attach(R);
        if (R_missing != null) scope.attach(R_missing);

        allocate(X, X_missing, y, R, R_missing, num_X, M, num_R);
    }

    @Override
    public void close() {
        // release all pointers attached to the current scope;
        if (scope != null) scope.close();
        super.close();
    }

    private native void allocate(DoublePointer X, BoolPointer X_missing, DoublePointer y, DoublePointer R,
                                 BoolPointer R_missing, int num_X, int M, int num_R);

    public native void get_x_instance(@ByRef ExplanationDataset instance, @Const int i);

    public static ExplanationDataset fromMatrix(double[][] matrix, boolean checkMissing) {
        assert matrix.length > 0;

        int num_X = matrix.length;
        int M = matrix[0].length;
        int Xsize = num_X * M;

        DoublePointer XPtr = new DoublePointer(Xsize);
        BoolPointer XmissingPtr = new BoolPointer(Xsize);

        // if checkMissing is false, we just pretend that none of the X elements are missing (nan)
        if (!checkMissing) {
            BoolPointer.memset(XmissingPtr, 0, Xsize);
        }

        int offset = 0;
        for (int i = 0; i < num_X; ++i, offset += M) {
            XPtr.put(matrix[i], offset, M);

            if (checkMissing) {
                for (int j = 0; j < M; ++j) {
                    XmissingPtr.put(offset + j, Double.isNaN(matrix[i][j]));
                }
            }
        }

        return new ExplanationDataset(XPtr, XmissingPtr, null, null, null, num_X, M, 0);
    }
}
