package shap4j.shap;

import static org.junit.jupiter.api.Assertions.assertEquals;
import org.junit.jupiter.api.Test;

public class TreeEnsembleTest {
    @Test
    public void testTreeEnsembleFromBytes() {
        byte[] raw = ShapUtils.readResourceAsBytes("/boston.shap4j");
        TreeEnsemble ensemble = TreeEnsemble.fromBytes(raw);

        assertEquals(1, ensemble.num_outputs());
        assertEquals(49, ensemble.max_nodes());
        assertEquals(100, ensemble.tree_limit());

        assertEquals(1, ensemble.getChildrenLeft(0, 0));
        assertEquals(3, ensemble.getChildrenLeft(0, 1));
    }
}
