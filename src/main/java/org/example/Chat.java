package org.example;
import java.io.*;
import java.sql.*;
import java.util.Scanner;
public class Chat {
    private ChatbotModel chatbotModel;
    private DataHandler dataHandler;
    private NLPProcessor nlpProcessor;
    public Chat() throws SQLException, IOException, ClassNotFoundException {
        this.dataHandler = new DataHandler();
        this.chatbotModel = new ChatbotModel(dataHandler);
        this.nlpProcessor = new NLPProcessor();
        initializeModel();
    }
    private void initializeModel() throws IOException, SQLException, ClassNotFoundException {
        File modelFile=new File("src/main/resources/chatbot_model.zip");
        File vocabFile=new File("src/main/resources/vocabulary.bin");
        if (modelFile.exists() && vocabFile.exists()) {
            chatbotModel.loadModel(modelFile.getPath());
            chatbotModel.getVocabularyProcessor().loadVocabulary(vocabFile.getPath());
            System.out.println("Loaded pre-trained model and vocabulary.");
        } else {
            System.out.println("Initializing new model...");
            chatbotModel.initialize();
            chatbotModel.trainFromDatabase();
            chatbotModel.saveModel(modelFile.getPath());
            chatbotModel.getVocabularyProcessor().saveVocabulary(vocabFile.getPath());
            System.out.println("Model trained and saved.");
        }
    }
    public void startChat() {
        Scanner scanner = new Scanner(System.in);
        System.out.println("Chatbot ready. Type 'quit' to exit.");
        while (true) {
            System.out.print("You: ");
            String input = scanner.nextLine().trim();
            if (input.equalsIgnoreCase("quit")) {
                break;
            }
            if (input.isEmpty()) {
                continue;
            }
            String response = chatbotModel.predict(input);
            System.out.println("Bot: " + response);
            dataHandler.createChatData(input, response);
        }
        scanner.close();
    }
    public static void main(String[] args) throws SQLException, IOException, ClassNotFoundException {
        Chat chat = new Chat();
        chat.startChat();
    }
}