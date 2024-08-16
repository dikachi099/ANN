import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Main {
    private static final String TRAINING_SET_PATH = "trainingSet1.csv";
    private static final String TEST_SET_PATH = "Test.csv";
    private static  final String VALIDATION_SET_PATH = "ValidationDataset.csv";

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        // Prompt the user to enter the number of hidden nodes
        System.out.print("Enter the number of hidden nodes: ");
        int hiddenNodesNum = scanner.nextInt();

        // Prompt the user to enter the number of epochs
        System.out.print("Enter the number of epochs: ");
        int numEpochs = scanner.nextInt();

        //load training data
        List<double[]> trainingData = loadData(TRAINING_SET_PATH);
        List<double[]> TestData = loadData(TEST_SET_PATH);
        List<double[]> ValidateData = loadData(VALIDATION_SET_PATH);

        Annealing Annealing = new Annealing(hiddenNodesNum,1,8);

        //TRAIN DATA WITHOUT MOMENTUM

        for (int epoch = 1; epoch <= numEpochs; epoch++) {
            Annealing.updateLearningRate(epoch, numEpochs); //update the learning rate
            double totalMSE = 0;
            for (double[] data : trainingData) {
                double[] inputs = new double[data.length - 1];  //inputs are the predictors
                System.arraycopy(data, 0, inputs, 0, data.length - 1);
                double target = data[data.length - 1]; //predictand
                double output = Annealing.forwardPass(inputs);
                totalMSE += Math.pow(target - output, 2);

                Annealing.forwardPass(inputs);
                Annealing.backwardPass(target);
                Annealing.updateWeightsandBias(inputs);

            }
            // Print MSE every 100 epochs
            if (epoch % 100 == 0) {
                double avgTrainingMSE = totalMSE / trainingData.size();

                double totalValidationMSE = 0;
                // Iterate over validation data
                for (double[] data : ValidateData) {
                    double[] inputs = new double[data.length - 1];
                    System.arraycopy(data, 0, inputs, 0, data.length - 1);
                    double target = data[data.length - 1];
                    double output = Annealing.forwardPass(inputs);
                    totalValidationMSE += Math.pow(target - output, 2);
                }
                double avgValidationMSE = totalValidationMSE / ValidateData.size();

//                System.out.println("Epoch: " + epoch + ", Training MSE: " + avgTrainingMSE + ", Validation MSE: " + avgValidationMSE);
                System.out.printf("%-10d  %-20s  %-20s%n", epoch       ,      +                 avgTrainingMSE            ,        +                     avgValidationMSE);

            }



        }
        ArrayList<Double> predictands=new ArrayList<>();

        for (double[] data : TestData) {
            double[] inputs = new double[data.length - 1];  //inputs are the predictors
            System.arraycopy(data, 0, inputs, 0, data.length - 1);
            double target = data[data.length - 1]; //predictand
            predictands.add(Annealing.forwardPass(inputs));
//                backProp.backwardPass(target);
//                backProp.updateWeightsandBias(inputs);

        }

        for (Double predictand : predictands) {
            System.out.println(predictand);

        }


    }

    private static List<double[]> loadData(String filePath) {
        List<double[]> dataList = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                // Remove BOM if present
                line = line.replace("\uFEFF", "");
                String[] values = line.split("\\|");
                double[] data = new double[values.length];
                for (int i = 0; i < values.length; i++) {
                    // Trim any leading or trailing whitespace
                    String trimmedValue = values[i].trim();
                    // Parse the value to double
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