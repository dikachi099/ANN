import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class boldDriverMain {
    private static final String TRAINING_SET_PATH = "trainingSet1.csv";
    private static final String TEST_SET_PATH = "Test.csv";
    private static  final String VALIDATION_SET_PATH = "ValidationDataset.csv";

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        List<Double> mseAtIntervals = new ArrayList<>();
        // Prompt the user to enter the number of hidden nodes
        System.out.print("Enter the number of hidden nodes: ");
        int hiddenNodesNum = scanner.nextInt();

        // Prompt the user to enter the number of epochs
        System.out.print("Enter the number of epochs: ");
        int numEpochs = scanner.nextInt();

        //load training data
        List<double[]> trainingData = loadData(TRAINING_SET_PATH);
        List<double[]> testData = loadData(TEST_SET_PATH);
        List<double[]> ValidateData = loadData(VALIDATION_SET_PATH);
        boldDriver boldDriver = new boldDriver(hiddenNodesNum, 1, 8);

        double prevTotalMSE = 0; // Initialize previous total MSE to a high value
        double[][] prevWeights = boldDriver.getWeights();


        List<Double> mseBlocks = new ArrayList<>();
        int prevBlockIndex = -1;

        for (int epoch = 1; epoch <= numEpochs; epoch++) {
            double totalMSE = 0;
            for (double[] data : trainingData) {
                double[] inputs = new double[data.length - 1];
                System.arraycopy(data, 0, inputs, 0, data.length - 1);
                double target = data[data.length - 1];
                double output = boldDriver.forwardPass(inputs);
                totalMSE += Math.pow(target - output, 2);

                // Update weights and bias
                boldDriver.backwardPass(target);
                boldDriver.updateWeightsandBias(inputs);
            }
            if (epoch % 1000 == 0) {
                double avgTrainingMSE = totalMSE / trainingData.size();

                double totalValidationMSE = 0;
                // Iterate over validation data
                for (double[] data : ValidateData) {
                    double[] inputs = new double[data.length - 1];
                    System.arraycopy(data, 0, inputs, 0, data.length - 1);
                    double target = data[data.length - 1];
                    double output = boldDriver.forwardPass(inputs);
                    totalValidationMSE += Math.pow(target - output, 2);
                }
                double avgValidationMSE = totalValidationMSE / ValidateData.size();

//                System.out.println("Epoch: " + epoch + ", Training MSE: " + avgTrainingMSE + ", Validation MSE: " + avgValidationMSE);
                System.out.printf("%-10d  %-20s  %-20s%n", epoch       ,      +                 avgTrainingMSE            ,        +                     avgValidationMSE);

            }

            if (epoch % 1000 == 0) {
                double avgMSE = totalMSE / trainingData.size();
//                System.out.println("Epoch: " + epoch + ", Average MSE: " + avgMSE);
                mseBlocks.add(avgMSE);
//                System.out.println("MSE Blocks: " + mseBlocks);
                prevBlockIndex++;

                // Compare current block with the previous block
                if (prevBlockIndex > 0) {
                    double prevMSE = mseBlocks.get(prevBlockIndex - 1);
//                    System.out.println(prevMSE);
                    double currentMSE = mseBlocks.get(prevBlockIndex);
//                    System.out.println(currentMSE);
                    if (currentMSE > (prevMSE * (1 + 0.04))) {
//                        System.out.println("MSE increased compared to the previous 1000 epochs (Epochs " + ((prevBlockIndex - 1) * 1000 + 1) + " to " + (prevBlockIndex * 1000) + ")");
                        // Revert weights
                        boldDriver.setWeights(prevWeights);
                        // Reduce error function by 30%
                        boldDriver.reduceLearningParameter();

                    } else {
//                        System.out.println("MSE did not increase");
                        // Accept weights
                        prevWeights = boldDriver.getWeights();
                        // Increase learning parameter by 5%
                        boldDriver.increaseLearningParameter();
                    }
                } else {
//                    System.out.println("yes");
                }


            }
        }
        // Calculate predictions for test data
        ArrayList<Double> predictands = new ArrayList<>();
        for (double[] data : testData) {
            double[] inputs = new double[data.length - 1];
            System.arraycopy(data, 0, inputs, 0, data.length - 1);
            predictands.add(boldDriver.forwardPass(inputs));
        }

        // Print predictions
        for (Double predictand : predictands) {
            System.out.println(predictand);
        }

    }

    private static List<double[]> loadData(String filePath) {
        List<double[]> dataList = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.replace("\uFEFF", "");
                String[] values = line.split("\\|");
                double[] data = new double[values.length];
                for (int i = 0; i < values.length; i++) {
                    String trimmedValue = values[i].trim();
                    data[i] = Double.parseDouble(trimmedValue);
                }
                dataList.add(data);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return dataList;
    }
}
