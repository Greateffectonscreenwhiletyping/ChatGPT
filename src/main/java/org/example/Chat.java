package org.example;
import java.io.*;
import java.sql.*;
import java.util.Scanner;
public class Chat {
    private ChatbotModel chatbotModel;
    private DataHandler dataHandler;
    public Chat() throws SQLException, IOException {
        this.dataHandler = new DataHandler();
        this.chatbotModel = new ChatbotModel(dataHandler);
        File modelFile = new File("chatbot_model.zip");
        if (modelFile.exists()) {
            chatbotModel.loadModel(modelFile.getPath());
            System.out.println("Loaded pre-trained model.");
        } else {
            System.out.println("Initializing new model...");
            chatbotModel.initialize();
            chatbotModel.trainFromDatabase();
            chatbotModel.saveModel(modelFile.getPath());
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
    public static void main(String[] args) throws SQLException, IOException {
        Chat chat = new Chat();
        chat.startChat();
    }
}