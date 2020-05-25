package shap4j.shap;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.ByRef;
import org.bytedeco.javacpp.annotation.Const;
import org.bytedeco.javacpp.annotation.Platform;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;

@Platform(include="shap4j/shap/tree_shap.h")
public class TreeEnsemble extends Pointer {
    static {
        Loader.load();
    }

    private TreeEnsemble() {
        allocate();
    }
    private native void allocate();

    private TreeEnsemble(
            IntPointer children_left, IntPointer children_right, IntPointer children_default, IntPointer features,
            DoublePointer thresholds, DoublePointer values, DoublePointer node_sample_weights,
            int max_depth, int tree_limit, DoublePointer base_offset, int max_nodes, int num_outputs
    ) {
        allocate(children_left, children_right, children_default, features, thresholds, values, node_sample_weights,
                 max_depth, tree_limit, base_offset, max_nodes, num_outputs);
    }
    private native void allocate(
            IntPointer children_left, IntPointer children_right, IntPointer children_default, IntPointer features,
            DoublePointer thresholds, DoublePointer values, DoublePointer node_sample_weights,
            int max_depth, int tree_limit, DoublePointer base_offset, int max_nodes, int num_outputs
    );

    public native void get_tree(@ByRef TreeEnsemble tree, @Const int i);

    // FIXME: figure out how to map this to the C++ allocate() function
    // public native void allocate(int tree_limit_in, int max_nodes_in, int num_outputs_in);

    // methods mapping to the struct fields of TreeEnsemble; all setters are private to prevent those fields from being
    // updated in JVM;
    public native int num_outputs(); private native void num_outputs(int setter);
    public native int tree_limit(); private native void tree_limit(int setter);
    public native int max_nodes(); private native void max_nodes(int setter);

    public native void free();

    private static IntPointer getIntPointer(BytePointer base, int position, int numElements) {
        IntPointer ptr = new IntPointer(base);
        return ptr.position(position).limit(position + numElements);
    }

    private static DoublePointer getDoublePointer(BytePointer base, int position, int numElements) {
        DoublePointer ptr = new DoublePointer(base);
        return ptr.position(position).limit(position + numElements);
    }

    public static TreeEnsemble fromBytes(byte[] rawData) {
        ByteBuffer buffer = ByteBuffer.wrap(rawData).order(ByteOrder.nativeOrder());

        byte[] magicBytes = new byte[4];
        buffer.get(magicBytes, 0, 4);
        int version = buffer.getInt();

        assert new String(magicBytes).equals("SHAP");
        assert version == 1;

        int numTrees = buffer.getInt();
        int maxDepth = buffer.getInt();
        int maxNodes = buffer.getInt();
        int numOutputs = buffer.getInt();
        int offsetIntArrays = buffer.getInt();
        int offsetDoubleArrays = buffer.getInt();
        double baseOffset = buffer.getDouble();
        DoublePointer ptrBaseOffset = new DoublePointer(1);
        ptrBaseOffset.put(baseOffset);

        int numElements = numTrees * maxNodes;

        // allocate a native memory block, and copy the java array to the native memory block
        BytePointer rawDataPtr = new BytePointer(rawData.length);
        rawDataPtr.put(rawData, 0, rawData.length);

        // create pointers pointing to different sections of the memory block (allocated through rawDataPtr)
        IntPointer childrenLeft = getIntPointer(rawDataPtr, offsetIntArrays >> 2, numElements);
        IntPointer childrenRight = getIntPointer(rawDataPtr, (int) childrenLeft.limit(), numElements);
        IntPointer childrenDefault = getIntPointer(rawDataPtr, (int) childrenRight.limit(), numElements);
        IntPointer features = getIntPointer(rawDataPtr, (int) childrenDefault.limit(), numElements);

        DoublePointer thresholds = getDoublePointer(rawDataPtr, offsetDoubleArrays >> 3, numElements);
        DoublePointer values = getDoublePointer(rawDataPtr, (int) thresholds.limit(), numElements * numOutputs);
        DoublePointer nodeSampleWeight = getDoublePointer(rawDataPtr, (int) values.limit(), numElements);

        return new TreeEnsemble(
                childrenLeft, childrenRight, childrenDefault, features, thresholds, values, nodeSampleWeight,
                maxDepth, numTrees, ptrBaseOffset, maxNodes, numOutputs
        );
    }
}
