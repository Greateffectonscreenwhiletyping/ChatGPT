package org.example;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.lemmatizer.LemmatizerME;
import opennlp.tools.lemmatizer.LemmatizerModel;
import java.io.InputStream;
import java.util.Arrays;
public class NLPProcessor {
    private TokenizerME tokenizer;
    private SentenceDetectorME sentenceDetector;
    private POSTaggerME posTagger;
    private LemmatizerME lemmatizer;
    public NLPProcessor() {
        try {
            tokenizer = new TokenizerME(new TokenizerModel(loadModel("models/en-token.bin")));
            sentenceDetector = new SentenceDetectorME(new SentenceModel(loadModel("models/en-sent.bin")));
            posTagger = new POSTaggerME(new POSModel(loadModel("models/en-pos-maxent.bin")));
            lemmatizer = new LemmatizerME(new LemmatizerModel(loadModel("models/en-lemmatizer.bin")));
        } catch (Exception e) {
            throw new RuntimeException("Failed to initialize NLP models.", e);
        }
    }

    private InputStream loadModel(String path) {
        return getClass().getResourceAsStream(path);
    }
    public String[] tokenize(String text) {
        return tokenizer.tokenize(text);
    }
    public String[] detectSentences(String text) {
        return sentenceDetector.sentDetect(text);
    }
    public String[] tagPOS(String[] tokens) {
        return posTagger.tag(tokens);
    }
    public String[] lemmatize(String[] tokens, String[] posTags) {
        return lemmatizer.lemmatize(tokens, posTags);
    }
    public String processText(String text) {
        String[] sentences=detectSentences(text);
        StringBuilder processed=new StringBuilder();
        for (String sentence : sentences) {
            String[] tokens=tokenize(sentence);
            String[] posTags=tagPOS(tokens);
            String[] lemmas=lemmatize(tokens, posTags);
            for (int i=0; i<tokens.length; i++) {
                if (!posTags[i].matches("\\W")) {
                    processed.append(lemmas[i].toLowerCase()).append(" ");
                }
            }
        }
        return processed.toString().trim();
    }
}