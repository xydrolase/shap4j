package shap4j.shap;

import java.io.IOException;
import java.io.InputStream;

public class ShapUtils {
    public static byte[] readResourceAsBytes(String path) {
        try {
            InputStream is = ShapUtils.class.getResourceAsStream(path);
            byte[] data = new byte[is.available()];

            is.read(data);

            return data;
        }
        catch (IOException e) {
            return null;
        }
    }
}
