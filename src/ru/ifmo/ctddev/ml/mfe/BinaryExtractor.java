package ru.ifmo.ctddev.ml.mfe;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class BinaryExtractor {

    public static void main(String[] args) throws IOException {
        // LogManager.getLogManager().reset();

        int len = JointDecMF.LENGTH;
        final int objects = Integer.parseInt(args[0]), features = Integer.parseInt(args[1]), classes = 2;
        byte[] buffer = new byte[4096];

        try (InputStream input = System.in) {
            try (OutputStream output = System.out) {
                while (true) {
                    double[][] data = new double[objects][features];
                    int[] labels = new int[objects];

                    int length = objects * features * Double.BYTES;
                    ByteBuffer inputBuffer = ByteBuffer.allocate(length);
                    inputBuffer.order(ByteOrder.LITTLE_ENDIAN);

                    for (int read, offset = 0; offset < length; offset += read) {
                        read = input.read(buffer);
                        if (read < 0) {
                            return;
                        }
                        inputBuffer.put(buffer, 0, read);
                    }
                    inputBuffer.flip();

                    for (int oid = 0; oid < objects; oid++) {
                        for (int fid = 0; fid < features; fid++) {
                            data[oid][fid] = inputBuffer.getDouble();
                        }
                        labels[oid] = (2 * oid < objects) ? 0 : 1;
                    }

                    double[] metaFeatures = JointDecMF.extract(objects, features, classes, data, labels);
                    assert len == metaFeatures.length;
                    ByteBuffer outputBuffer = ByteBuffer.allocate(metaFeatures.length * Double.BYTES);
                    outputBuffer.order(ByteOrder.LITTLE_ENDIAN);
                    for (double value : metaFeatures) {
                        outputBuffer.putDouble(value);
                    }

                    output.write(outputBuffer.array());
                    output.flush();
                }
            }
        }
    }
}
