package org.example;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.io.*;
import java.sql.SQLException;
import java.util.*;
public class ChatbotModel {
    private VocabularyProcessor vp;
    private MultiLayerNetwork model;
    private DataHandler dataHandler;
    private NLPProcessor nlpProcessor;
    private Map<String, Integer> wordToIndex = new HashMap<>();
    private Map<Integer, String> indexToWord = new HashMap<>();
    private int wordIndex = 0;
    private int vocabularySize = 0;
    private int sequenceLength = 2147483647;
    private int vectorSize = 2147483647;
    public int LSTMLayerSize = 256;
    private double learningRate = 0.001;
    public String modelSavePath = "src/main/resources/chatbot_model.zip";
    public String vocabSavePath = "src/main/resources/vocabulary.bin";
    public ChatbotModel(DataHandler dataHandler) {
        this.dataHandler=dataHandler;
        this.nlpProcessor=new NLPProcessor();
        this.vp=new VocabularyProcessor(nlpProcessor);
    }
    public void initializeVocabulary(List<String[]> chatData) {
        for (String[] record : chatData) {
            String[] words = record[0].split("\\s+");
            for (String word : words) {
                if (!wordToIndex.containsKey(word)) {
                    wordToIndex.put(word, wordIndex);
                    indexToWord.put(wordIndex, word);
                    wordIndex++;
                }
            }
        }
        vocabularySize = wordToIndex.size();  // Set vocabulary size after processing data
    }
    public void trainFromDatabase() throws SQLException {
        List<String[]> chatData = dataHandler.readChatData();
        initializeVocabulary(chatData);  // Initialize vocabulary before training
        List<DataSet> trainingData = convertChatDataToDataSet(chatData);
        if (!trainingData.isEmpty()) {
            train(trainingData);
        }
    }
    public void train(List<DataSet> dataSets) {
        DataSetIterator iterator = new ListDataSetIterator<>(dataSets, dataSets.size());
        model.fit(iterator, 1000);
    }
    public String predict(String input) {
        INDArray inputVector = encodeString(input);
        if (inputVector == null) {
            return "I don't understand.";
        }
        INDArray outputVector = model.output(inputVector, false);
        return decodeOutput(outputVector);
    }
    public List<DataSet> convertChatDataToDataSet(List<String[]> chatData) {
        List<DataSet> dataSets = new ArrayList<>();
        for (String[] record : chatData) {
            INDArray inputVector = encodeString(record[0]);
            INDArray outputVector = encodeString(record[1]);
            if (inputVector != null && outputVector != null) {
                dataSets.add(new DataSet(inputVector, outputVector));
            }
        }
        return dataSets;
    }
    private INDArray encodeString(String input) {
        String[] words = input.split("\\s+");
        INDArray vector = Nd4j.zeros(1, vocabularySize);
        for (String word : words) {
            if (!wordToIndex.containsKey(word)) {
                wordToIndex.put(word, wordIndex);
                indexToWord.put(wordIndex, word);
                wordIndex++;
                vocabularySize++;
                vector = Nd4j.zeros(1, vocabularySize); //resize vector to fit new word.
            }
            vector.putScalar(0, wordToIndex.get(word), 1);
        }
        return vector;
    }
    private String decodeOutput(INDArray outputVector) {
        int maxIndex = Nd4j.argMax(outputVector, 1).getInt(0);
        return indexToWord.get(maxIndex);
    }
    public void buildModel() {
        if (vocabularySize <= 0) {
            throw new IllegalStateException("Vocabulary size must be greater than zero.");
        }
        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Sgd(0.01));
        NeuralNetConfiguration.ListBuilder listBuilder = builder.list();
        listBuilder.layer(0, new DenseLayer.Builder()
                .nIn(vocabularySize)
                .nOut(vocabularySize)
                .activation(Activation.RELU)
                .build());
        listBuilder.layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .nIn(vocabularySize)
                .nOut(vocabularySize)
                .activation(Activation.IDENTITY)
                .build());
        listBuilder.setInputType(InputType.feedForward(vocabularySize));
        model = new MultiLayerNetwork(listBuilder.build());
        model.init();
        model.setListeners(new ScoreIterationListener(10));
    }
    public void initialize() {
        List<String[]> chatData = dataHandler.readChatData();
        initializeVocabulary(chatData);  // Initialize vocabulary from chat data
        buildModel();
        System.out.println("Model initialized.");
    }
    public void saveModel(String modelPath) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelPath))) {
            oos.writeObject(model);
            System.out.println("Model saved successfully.");
        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("Error saving model.");
        }
    }
    public void loadModel(String modelPath) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(modelPath))) {
            model = (MultiLayerNetwork) ois.readObject();
            System.out.println("Model loaded successfully.");
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
            System.out.println("Error loading model.");
        }
    }
    public MultiLayerNetwork getModel() {
        return model;
    }
    public VocabularyProcessor getVocabularyProcessor() {
        if (vp==null) {
            vp=new VocabularyProcessor(nlpProcessor);
        }
        return vp;
    }
}