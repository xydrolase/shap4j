package shap4j.shap;

import org.bytedeco.javacpp.BoolPointer;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.ByRef;
import org.bytedeco.javacpp.annotation.Const;
import org.bytedeco.javacpp.annotation.Platform;

@Platform(include="shap4j/shap/tree_shap.h")
public class ExplanationDataset extends Pointer {
    static {
        Loader.load();
    }

    public ExplanationDataset() {
        allocate();
    }
    private native void allocate();

    public ExplanationDataset(DoublePointer X, BoolPointer X_missing, DoublePointer y, DoublePointer R,
                              BoolPointer R_missing, int num_X, int M, int num_R) {
       allocate(X, X_missing, y, R, R_missing, num_X, M, num_R);
    }

    private native void allocate(DoublePointer X, BoolPointer X_missing, DoublePointer y, DoublePointer R,
                                 BoolPointer R_missing, int num_X, int M, int num_R);

    public native void get_x_instance(@ByRef ExplanationDataset instance, @Const int i);
}
