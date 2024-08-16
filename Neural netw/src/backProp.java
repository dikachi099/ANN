import java.util.ArrayList;
import java.util.Random;
public class backProp {
    private double learningRate = 0.1;
    private double[] weightedSum; //array to store weighted sum of layers, tell that the last index is the output node
    private int hiddenNodesNum;
    private final Random random = new Random();
    private final double[][] nodeWeight;
    private double[] hiddenLayerBias;
    private double[] hiddenNodeWeight;
    private double outputNodeBias;
    private int inputNodeNum;
    private int outputNodeNum;
    private double[] hiddenNodeSig;
    private double outputLayerSig;
    private double outputNodeDelta;
    private double [] hiddenNodeDeltaArray;


    private ArrayList<Double> predictedValue = new ArrayList<Double>();


    public backProp(int hiddenNodesNum, int outputNodeNum, int inputNodeNum) {
        this.nodeWeight = new double[hiddenNodesNum][inputNodeNum];
        this.hiddenNodesNum = hiddenNodesNum;
        this.outputNodeNum = outputNodeNum;
        this.inputNodeNum = inputNodeNum;
        this.hiddenNodeWeight = new double[hiddenNodesNum];
        this.hiddenLayerBias = new double[hiddenNodesNum];
        this.hiddenNodeSig = new double[hiddenNodesNum];
        this.outputLayerSig = outputLayerSig;
        this.hiddenNodeDeltaArray = new double[hiddenNodesNum];
        initialiseWeight();
        initialiseBiases();


    }


    //Initial weight for input to hidden layer
    public void initialiseWeight() {
        double inputUpper = 2.0 / 8;
        double inputLower = -2.0 / 8;

        for (int i = 0; i < inputNodeNum; i++) {
            for (int j = 0; j < hiddenNodesNum; j++) {

                nodeWeight[j][i] = (Math.random() * (inputUpper - inputLower)) + inputLower; //Let it know that the last weight in the array is for the output node
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

            hiddenNodeSig[i] = weightsumSigmoid; // may not be needed, she will say why
            weightSumOutput += weightsumSigmoid * hiddenNodeWeight[i];

        }
        weightSumOutput += outputNodeBias * 1;
        predictedValue.add(sigmoid(weightSumOutput));

        outputLayerSig = sigmoid(weightSumOutput);

        return outputLayerSig;

    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }


    // BACKWARD PASS
    public double backwardPass(double target) {
        double firstDiv =  outputLayerSig*(1-outputLayerSig);

        outputNodeDelta = (target - outputLayerSig) * firstDiv;

        for(int i =0; i < hiddenNodesNum; i++){
            double hiddenNodeDiv = hiddenNodeSig[i]*(1-hiddenNodeSig[i]);
            double hiddenNodedelta = hiddenNodeDiv*hiddenNodeWeight[i]*outputNodeDelta;
            hiddenNodeDeltaArray[i] = hiddenNodedelta;

        }
        return 0.0;

    }

// UPDATE WEIGHTS AND BIASES

    public double updateWeightsandBias(double[] inputs){
        //updating input weight
        for(int i = 0; i< hiddenNodesNum; i++){
            for(int j = 0 ; j < inputNodeNum; j++ ){
                nodeWeight[i][j] = nodeWeight[i][j]+ (0.1*hiddenNodeDeltaArray[i]*inputs[j])  ;
            }
        }

        //updating hidden node weight and bias
        for(int i =0; i<hiddenNodesNum; i++){
            hiddenNodeWeight[i] = hiddenNodeWeight[i] +(0.1*outputNodeDelta*hiddenNodeSig[i]);

            hiddenLayerBias[i] = hiddenLayerBias[i]+(0.1*hiddenNodeDeltaArray[i]*1);
        }

        outputNodeBias = outputNodeBias+(0.1*outputNodeDelta*1);


        return 0.0;

    }


}

