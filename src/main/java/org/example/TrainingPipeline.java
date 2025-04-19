package org.example;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.dataset.api.iterator.ExistingDataSetIterator;
import java.sql.SQLException;
import java.util.List;
public class TrainingPipeline {
    private final ChatbotModel chatbotModel;
    private final DataHandler dataHandler;
    public TrainingPipeline(ChatbotModel chatbotModel, DataHandler dataHandler) {
        this.chatbotModel = chatbotModel;
        this.dataHandler = dataHandler;
    }
    public void runTraining() throws SQLException {
        List<String[]> chatData = dataHandler.readChatData();
        List<DataSet> trainingData = chatbotModel.convertChatDataToDataSet(chatData);
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData);
        for (DataSet dataSet : trainingData) {
            normalizer.transform(dataSet);
        }
        DataSetIterator iterator = new ExistingDataSetIterator(trainingData);
        EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(100))
                .scoreCalculator(new DataSetLossCalculator(iterator, true))
                .modelSaver(new LocalFileModelSaver("models/"))
                .build();
        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(
                esConf,
                chatbotModel.getModel(),
                iterator
        );
        trainer.setListeners(new ScoreIterationListener(10));
        trainer.fit();
    }
}