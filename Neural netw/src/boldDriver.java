import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
public class boldDriver {
    private double learningRate = 0.1;
    private double[] weightedSum; //array to store weighted sum of layers, tell that the last index is the output node
    private int hiddenNodesNum;
    private final Random random = new Random();
    private final double[][] nodeWeight;
    private double[] hiddenLayerBias;
    private double[] prevHiddenLayerBias;
    private double[] hiddenLayerBiasChange;
    private double[] hiddenNodeWeight;
    private double outputNodeBias;
    private double prevOutputNodeBias;

    private double outputNodeBiasChange;
    private int inputNodeNum;
    private int outputNodeNum;
    private double[] hiddenNodeSig;
    private double outputLayerSig;
    private double outputNodeDelta;
    private double[] hiddenNodeDeltaArray;

    private double[][] weightChange;

    private double[][] prevWeight;

    private double[] prevHiddenWeight;

    private double[] hiddenWeightChange;

    private double previousMSE;

    public double[][] getWeights() {
        return nodeWeight;
    }

    public void setWeights(double[][] weights) {
        for (int i = 0; i < hiddenNodesNum; i++) {
            System.arraycopy(weights[i], 0, nodeWeight[i], 0, inputNodeNum);
        }
    }


    private ArrayList<Double> predictedValue = new ArrayList<Double>();


    public boldDriver(int hiddenNodesNum, int outputNodeNum, int inputNodeNum) {
        this.nodeWeight = new double[hiddenNodesNum][inputNodeNum];
        this.hiddenNodesNum = hiddenNodesNum;
        this.outputNodeNum = outputNodeNum;
        this.inputNodeNum = inputNodeNum;
        this.hiddenNodeWeight = new double[hiddenNodesNum];
        this.hiddenLayerBias = new double[hiddenNodesNum];
        this.hiddenNodeSig = new double[hiddenNodesNum];
        this.outputLayerSig = outputLayerSig;
        this.hiddenNodeDeltaArray = new double[hiddenNodesNum];
        this.weightChange = new double[hiddenNodesNum][inputNodeNum];
        this.prevWeight = new double[hiddenNodesNum][inputNodeNum]; // store weight before and after iteration
        this.prevHiddenWeight = new double[hiddenNodesNum];
        this.hiddenWeightChange = new double[hiddenNodesNum];
        this.prevHiddenLayerBias = new double[hiddenNodesNum];
        this.hiddenLayerBiasChange = new double[hiddenNodesNum];
        this.outputNodeBias = outputNodeNum;
//        this.prevWeightValue = new double []
        initialiseWeight();
        initialiseBiases();


    }

    public void updatePreviousMSE(double currentMSE) {
        this.previousMSE = currentMSE;
    }


    //Initial weight for input to hidden layer
    public void initialiseWeight() {
        double inputUpper = 2.0 / 8;
        double inputLower = -2.0 / 8;

        for (int i = 0; i < inputNodeNum; i++) {
            for (int j = 0; j < hiddenNodesNum; j++) {

                nodeWeight[j][i] = (Math.random() * (inputUpper - inputLower)) + inputLower;
//                System.out.println(nodeWeight[i][j]);
            }

        }

        //Initial weight for hidden to output layer

        double outputUpper = 2.0 / hiddenNodesNum;
        double outputLower = -2.0 / hiddenNodesNum;

        for (int i = 0; i < hiddenNodesNum; i++) {

            hiddenNodeWeight[i] = (Math.random() * (outputUpper - outputLower)) + outputLower;
        }


    }

    //Initial bias for hidden layer
    public void initialiseBiases() {
        double outputUpper = 2.0 / inputNodeNum;
        double outputLower = -2.0 / inputNodeNum;
//        for (int i = 0; i < inputNodeNum; i++) {
        for (int j = 0; j < hiddenNodesNum; j++) {
            hiddenLayerBias[j] = (Math.random() * (outputUpper - outputLower)) + outputLower;
        }


        // Initial bias for output node
        outputNodeBias = (Math.random() * (outputUpper - outputLower)) + outputLower;

//            System.out.println(outputNodeBias);

    }


    // FORWARD PASS
    public double forwardPass(double[] inputs) {
//

        double weightSumOutput = 0.0;
        for (int i = 0; i < hiddenNodesNum; i++) {
            double weightedSum = 0.0;

            for (int j = 0; j < inputNodeNum; j++) {
                weightedSum += inputs[j] * nodeWeight[i][j];
            }
            double weightSumandBias = weightedSum + hiddenLayerBias[i] * 1;
            double weightsumSigmoid = sigmoid(weightSumandBias);

            hiddenNodeSig[i] = weightsumSigmoid;
            weightSumOutput += weightsumSigmoid * hiddenNodeWeight[i];

        }
        weightSumOutput += outputNodeBias * 1;
        predictedValue.add(sigmoid(weightSumOutput));

        outputLayerSig = sigmoid(weightSumOutput);

        double target = inputs[inputs.length - 1]; // Actual target value
        double error = (target - outputLayerSig);
        double errorSquare = error * error;


        return outputLayerSig;

    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }


    // BACKWARD PASS
    public double backwardPass(double target) {
        double firstDiv = outputLayerSig * (1 - outputLayerSig);

        outputNodeDelta = (target - outputLayerSig) * firstDiv;

        for (int i = 0; i < hiddenNodesNum; i++) {
            double hiddenNodeDiv = hiddenNodeSig[i] * (1 - hiddenNodeSig[i]);
            double hiddenNodedelta = hiddenNodeDiv * hiddenNodeWeight[i] * outputNodeDelta;
            hiddenNodeDeltaArray[i] = hiddenNodedelta;

        }
        return 0.0;

    }

// UPDATE WEIGHTS AND BIASES

    public double updateWeightsandBias(double[] inputs) {
        // Update input weight and calculate weight change
//        double[][] trainingDataArray = trainingData.toArray(new double[0][]);


        for (int i = 0; i < hiddenNodesNum; i++) {
            for (int j = 0; j < inputNodeNum; j++) {
                double prevWeight = nodeWeight[i][j]; // Store weight before iteration
                double deltaWeight = learningRate * hiddenNodeDeltaArray[i] * inputs[j]; // Calculate weight change
                nodeWeight[i][j] += deltaWeight; // Update weight with momentum
                // Apply momentum term
                nodeWeight[i][j] += 0.9 * (nodeWeight[i][j] - prevWeight);
            }
        }

        // Update hidden node weight and bias
        for (int i = 0; i < hiddenNodesNum; i++) {
            double prevHiddenWeight = hiddenNodeWeight[i]; // Store weight before iteration
//            System.out.println(prevHiddenWeight);
            double deltaHiddenWeight = learningRate * outputNodeDelta * hiddenNodeSig[i]; // Calculate weight change
            hiddenNodeWeight[i] += deltaHiddenWeight; // Update weight with momentum
            // Apply momentum term
            hiddenNodeWeight[i] += 0.9 * (hiddenNodeWeight[i] - prevHiddenWeight);

            double prevHiddenBias = hiddenLayerBias[i]; // Store bias before iteration
            hiddenLayerBias[i] += learningRate * hiddenNodeDeltaArray[i] * 1; // Update bias
            // Apply momentum term
            hiddenLayerBias[i] += 0.9 * (hiddenLayerBias[i] - prevHiddenBias);
        }

        // Update output node bias
        double prevOutputNodeBias = outputNodeBias;
//        System.out.println(prevOutputNodeBias);
        outputNodeBias += learningRate * outputNodeDelta * 1; // Update bias
        // Apply momentum term
        outputNodeBias += 0.9 * (outputNodeBias - prevOutputNodeBias);

        return 0.0;


    }

    public void reduceLearningParameter() {
        learningRate *= 0.7; // Reduce learning rate by 30%
        double minimumLearningRate = 0.01;
        if (learningRate < minimumLearningRate) {
            learningRate = minimumLearningRate; // Ensure the learning rate doesn't fall below 0.01
        }
    }

    public void increaseLearningParameter() {
        learningRate *= 1.05; // Reduce learning rate by 30%
        double maximumLearningRate = 0.5;
        if (learningRate < maximumLearningRate) {
            learningRate = maximumLearningRate; // Ensure the learning rate doesn't increase passed 0.5
        }
    }

}



