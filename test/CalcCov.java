import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import ru.ifmo.ctddev.ml.mfe.Landmarks;
import ru.ifmo.ctddev.ml.mfe.MetaFeatures;
import ru.ifmo.ctddev.ml.mfe.Utils;
import weka.core.Instances;

public class CalcCov {
    public static void main(String[] args) throws InterruptedException, ExecutionException, FileNotFoundException {
        String path = "D:\\eclipse\\MetaFeaturesSpace\\csv\\";

        ExecutorService executor = Executors.newFixedThreadPool(6);
        final int numFeatures = 16;
        final int numObjectsPerClass = 64;

        String prefix = "land";

        String[] classNames = { "zero", "one" };

        final int numClasses = classNames.length;
        final int numObjects = numObjectsPerClass * numClasses;
        List<Future<double[]>> fvectors = new ArrayList<>();

        for (File datafolder : new File(path).listFiles()) {
            try {
                double[][] data = new double[numObjects][numFeatures];
                int[] labels = new int[numObjects];

                String[] header = new String[numFeatures];

                for (int f = 0; f < numFeatures; f++) {
                    header[f] = "f" + f;
                }

                for (int oid = 0, label = 0; label < numClasses; label++) {
                    try (CSVParser parser = new CSVParser(new FileReader(datafolder.getPath() + File.separator + classNames[label] + ".csv"), CSVFormat.DEFAULT.withHeader(header))) {
                        for (CSVRecord record : parser) {
                            for (int fid = 0; fid < numFeatures; fid++) {
                                data[oid][fid] = Double.parseDouble(record.get(fid));
                            }
                            labels[oid++] = label;
                        }
                    }
                }

                System.out.println(datafolder.getName());

                fvectors.add(executor.submit(new Callable<double[]>() {
                    @Override
                    public double[] call() throws Exception {
                        Utils.normalize(numObjects, numFeatures, data);
                        Instances instances = Utils.convert(numObjects, numFeatures, numClasses, data, labels);
                        // return MetaFeatures.extract(numObjects, numFeatures, numClasses, data, labels, instances);
                        return Landmarks.extract(instances);
                    }
                }));

            } catch (IOException exception) {
                exception.printStackTrace();
            }
        }

        final List<double[]> vectors = new ArrayList<>();
        for (Future<double[]> future : fvectors) {
            vectors.add(future.get());
        }

        // int len = MetaFeatures.LENGTH;
        int len = Landmarks.LENGTH;

        double[] mean = new double[len];

        for (double[] vector : vectors) {
            double[] mf = vector;
            for (int i = 0; i < len; i++) {
                mean[i] += mf[i];
            }
        }

        for (int i = 0; i < len; i++) {
            mean[i] /= vectors.size();
        }
        try (PrintWriter writer = new PrintWriter(prefix + "_mean.txt")) {
            ArrayUtils.print(writer, mean);
        }

        System.out.println(vectors.size());
        double[][] cov = new double[len][len];

        for (double[] vector : vectors) {
            double[] mf = vector;

            for (int i = 0; i < len; i++) {
                for (int j = 0; j < len; j++) {
                    cov[i][j] += (mf[i] - mean[i]) * (mf[j] - mean[j]);
                }
            }
        }

        try (PrintWriter writer = new PrintWriter(prefix + "_cov.txt")) {
            ArrayUtils.print(writer, cov);
        }

        for (int i = 0; i < len; i++) {
            for (int j = 0; j < len; j++) {
                cov[i][j] /= vectors.size();
            }
            cov[i][i] += 1e-3;
        }

        double[][] inv = MatrixUtils.inv(len, cov);

        try (PrintWriter writer = new PrintWriter(prefix + "_inv.txt")) {
            ArrayUtils.print(writer, inv);
        }

        for (int i = 0; i < len; i++) {
            inv[i][i] += 1e-3;
        }

        double[][] sqr = MatrixUtils.sqrt(inv);

        try (PrintWriter writer = new PrintWriter(prefix + "_sqr.txt")) {
            ArrayUtils.print(writer, sqr);
        }

        executor.shutdown();

        // for (int i = 0; i < num; i++) {
        // for (int j = 0; j < num; j++) {
        // mf[i] += (vector[j] - mean[j]) * sqr[j][i];
        // }
        // }

    }
}
