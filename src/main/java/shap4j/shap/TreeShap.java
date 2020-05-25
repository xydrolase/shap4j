package shap4j.shap;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.ByRef;
import org.bytedeco.javacpp.annotation.Const;
import org.bytedeco.javacpp.annotation.Platform;

@Platform(include="shap4j/shap/tree_shap.h")
public class TreeShap {
    static {
        Loader.load();
    }
    public static native void dense_tree_shap(@Const @ByRef TreeEnsemble trees,
                                              @Const @ByRef ExplanationDataset data,
                                              DoublePointer out_contribs,
                                              @Const int feature_dependence,
                                              int model_transform, boolean interactions);
}
