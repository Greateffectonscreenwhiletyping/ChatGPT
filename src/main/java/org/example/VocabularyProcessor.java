package org.example;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import java.io.*;
import java.lang.ClassNotFoundException;
import java.util.*;
public class VocabularyProcessor {
    private final NLPProcessor nlpProcessor;
    private Map<String, Integer> wordToIndex = new HashMap<>();
    private Map<Integer, String> indexToWord = new HashMap<>();
    private int vocabularySize = 0;
    private final String UNK_TOKEN = "<UNK>";
    private final String PAD_TOKEN = "<PAD>";
    private final String SOS_TOKEN = "<SOS>";  // Start of sentence
    private final String EOS_TOKEN = "<EOS>";  // End of sentence
    public VocabularyProcessor(NLPProcessor nlpProcessor) {
        this.nlpProcessor = nlpProcessor;
        addSpecialTokens();
    }
    private void addSpecialTokens() {
        addWord(PAD_TOKEN);
        addWord(UNK_TOKEN);
        addWord(SOS_TOKEN);
        addWord(EOS_TOKEN);
    }
    public void addWord(String word) {
        if (!wordToIndex.containsKey(word)) {
            wordToIndex.put(word, vocabularySize);
            indexToWord.put(vocabularySize, word);
            vocabularySize++;
        }
    }
    public void buildVocabulary(List<String[]> chatData) {
        Set<String> vocabulary = new HashSet<>();
        for (String[] record : chatData) {
            String processedInput = nlpProcessor.processText(record[0]);
            String processedOutput = nlpProcessor.processText(record[1]);
            vocabulary.addAll(Arrays.asList(processedInput.split("\\s+")));
            vocabulary.addAll(Arrays.asList(processedOutput.split("\\s+")));
        }
        vocabulary.forEach(this::addWord);
    }
    public INDArray encodeInput(String input, int maxLength) {
        String processed = nlpProcessor.processText(input);
        String[] words = processed.split("\\s+");
        int length = Math.min(words.length, maxLength);
        INDArray vector = Nd4j.zeros(1, vocabularySize, maxLength);
        vector.putScalar(new int[]{0, wordToIndex.get(SOS_TOKEN), 0}, 1.0);
        for (int i = 0; i < length; i++) {
            String word = words[i];
            int wordIndex = wordToIndex.getOrDefault(word, wordToIndex.get(UNK_TOKEN));
            vector.putScalar(new int[]{0, wordIndex, i+1}, 1.0); // +1 for SOS position
        }
        int eosPos = Math.min(length + 1, maxLength - 1);
        vector.putScalar(new int[]{0, wordToIndex.get(EOS_TOKEN), eosPos}, 1.0);
        int padIndex = wordToIndex.get(PAD_TOKEN);
        for (int i = eosPos + 1; i < maxLength; i++) {
            vector.putScalar(new int[]{0, padIndex, i}, 1.0);
        }
        return vector;
    }
    public String decodeOutput(INDArray outputVector) {
        StringBuilder sb = new StringBuilder();
        int seqLength = (int) outputVector.size(2);
        for (int i=0; i<seqLength; i++) {
            int maxIndex = Nd4j.argMax(outputVector.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(i)), 0).getInt(0);
            String word = indexToWord.get(maxIndex);
            if (word.equals(EOS_TOKEN)) {
                break;
            }
            if (!word.equals(PAD_TOKEN) && !word.equals(SOS_TOKEN)) {
                sb.append(word).append(" ");
            }
        }

        return reconstructGrammar(sb.toString().trim());
    }

    private String reconstructGrammar(String text) {
        // Use OpenNLP to reconstruct proper grammar
        String[] sentences = nlpProcessor.detectSentences(text);
        StringBuilder result = new StringBuilder();

        for (String sentence : sentences) {
            // Capitalize first letter
            if (!sentence.isEmpty()) {
                sentence = sentence.substring(0, 1).toUpperCase() + sentence.substring(1);

                // Ensure proper ending punctuation
                if (!sentence.matches(".*[.!?]$")) {
                    sentence += ".";
                }

                result.append(sentence).append(" ");
            }
        }

        return result.toString().trim();
    }

    // Serialization methods
    public void saveVocabulary(String path) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path))) {
            oos.writeObject(wordToIndex);
            oos.writeObject(indexToWord);
            oos.writeInt(vocabularySize);
        }
    }
    public void loadVocabulary(String path) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path))) {
            wordToIndex = (Map<String, Integer>) ois.readObject();
            indexToWord = (Map<Integer, String>) ois.readObject();
            vocabularySize = ois.readInt();
        }
    }
}