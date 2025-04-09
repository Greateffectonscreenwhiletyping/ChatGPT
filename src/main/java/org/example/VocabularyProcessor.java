package org.example;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.io.*;
import java.lang.ClassNotFoundException;
import java.util.*;
public class VocabularyProcessor {
    private Map<String, Integer> wordToIndex = new HashMap<>();
    private Map<Integer, String> indexToWord = new HashMap<>();
    private int vocabularySize = 0;
    private final String UNK_TOKEN = "<UNK>";
    private final String PAD_TOKEN = "<PAD>";
    private final String SOS_TOKEN = "<SOS>";
    private final String EOS_TOKEN = "<EOS>";
    public VocabularyProcessor() {
        addWord(PAD_TOKEN);
        addWord(UNK_TOKEN);
    }
    public void buildVocabulary(List<String[]> chatData) {
        Set<String> vocabulary = new HashSet<>();
        for (String[] record : chatData) {
            vocabulary.addAll(Arrays.asList(record[0].toLowerCase().split("\\s+")));
            vocabulary.addAll(Arrays.asList(record[1].toLowerCase().split("\\s+")));
        }
        for (String word : vocabulary) {
            addWord(word);
        }
    }
    private void addWord(String word) {
        if (!wordToIndex.containsKey(word)) {
            wordToIndex.put(word, vocabularySize);
            indexToWord.put(vocabularySize, word);
            vocabularySize++;
        }
    }
    public INDArray encodeInput(String input, int maxLength) {
        String[] words = input.toLowerCase().split("\\s+");
        int length = Math.min(words.length, maxLength);
        INDArray vector = Nd4j.zeros(1, vocabularySize, maxLength);
        for (int i = 0; i < length; i++) {
            String word = words[i];
            int wordIndex = wordToIndex.getOrDefault(word, wordToIndex.get(UNK_TOKEN));
            vector.putScalar(new int[]{0, wordIndex, i}, 1.0);
        }
        int padIndex = wordToIndex.get(PAD_TOKEN);
        for (int i = length; i < maxLength; i++) {
            vector.putScalar(new int[]{0, padIndex, i}, 1.0);
        }
        return vector;
    }
    public INDArray encodeOutput(String output, int maxLength) {
        return encodeInput(output, maxLength);
    }
    public String decodeOutput(INDArray outputVector) {
        StringBuilder sb = new StringBuilder();
        if (outputVector.rank() != 3) {
            throw new IllegalArgumentException("Output vector must be 3-dimensional.");
        }
        INDArray argMax = Nd4j.argMax(outputVector, 1);
        for (int i=0; i<argMax.size(1); i++) {
            int maxIndex = argMax.getInt(0, i);
            String word = indexToWord.getOrDefault(maxIndex, UNK_TOKEN);
            if (!word.equals(PAD_TOKEN) && !word.equals(UNK_TOKEN)) {
                sb.append(word).append(" ");
            }
        }
        return sb.toString().trim();
    }
    public int getVocabularySize() {
        return vocabularySize;
    }
    public void saveVocabulary(String path) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path))) {
            oos.writeObject(wordToIndex);
        }
    }
    public void loadVocabulary(String path) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path))) {
            wordToIndex = (Map<String, Integer>) ois.readObject();
            indexToWord = new HashMap<>();
            for (Map.Entry<String, Integer> entry : wordToIndex.entrySet()) {
                indexToWord.put(entry.getValue(), entry.getKey());
            }
            vocabularySize = wordToIndex.size();
        }
    }
    public boolean containsWord(String word) {
        return wordToIndex.containsKey(word.toLowerCase());
    }
}