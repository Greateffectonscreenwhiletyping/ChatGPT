package org.example;
import org.deeplearning4j.datasets.iterator.ExistingDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import java.util.List;
import java.sql.SQLException;
public class TrainingPipeline {
    private final ChatbotModel chatbotModel;
    private final DataHandler dataHandler;
    public TrainingPipeline(ChatbotModel chatbotModel, DataHandler dataHandler) {
        this.chatbotModel = chatbotModel;
        this.dataHandler = dataHandler;
    }
    public void runTraining() throws SQLException {
        // Read chat data
        List<String[]> chatData = dataHandler.readChatData();
        // Convert chat data to DataSet objects
        List<DataSet> trainingData = chatbotModel.convertChatDataToDataSet(chatData);
        // Create a List of feature names (this is just an example, adjust based on your dataset)
        List<String> featureNames = List.of("feature1", "feature2", "feature3"); // Replace with your actual feature names
        // Create iterator using ExistingDataSetIterator
        DataSetIterator iterator = new ExistingDataSetIterator(trainingData.iterator(), featureNames); // iterator for training data
        // Configure EarlyStopping
        EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(100)) // Max 100 epochs
                .scoreCalculator(new DataSetLossCalculator(iterator, true)) // Score calculation using loss
                .modelSaver(new LocalFileModelSaver("models/")) // Saving model to "models" folder
                .build();
        // Create EarlyStoppingTrainer and fit the model
        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, chatbotModel.getModel(), iterator);
        trainer.fit();
    }
}