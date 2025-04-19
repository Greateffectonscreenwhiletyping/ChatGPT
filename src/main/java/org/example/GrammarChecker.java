package org.example;
import opennlp.tools.parser.Parse;
import opennlp.tools.parser.Parser;
import opennlp.tools.parser.ParserFactory;
import opennlp.tools.parser.ParserModel;
import java.io.InputStream;
public class GrammarChecker {
    private final Parser parser;
    public GrammarChecker() {
        try (InputStream modelIn=getClass().getResourceAsStream("/models/en-parser-chunking.bin")) {
            ParserModel model=new ParserModel(modelIn);
            this.parser=ParserFactory.create(model);
        } catch (Exception e) {
            throw new RuntimeException("Failed to load parser model.", e);
        }
    }
    public String correctGrammar(String sentence) {
        Parse parser=parser.parse(sentence);
        if (!isGrammatical(parse)) {
            return reconstructSentence(parse);
        }
        return sentence;
    }
    private boolean isGrammatical(Parse parse) {
        return checkSubjectVerbAgreement(parse) && checkVerbTenseconsistency(parse) && checkProperArticleUse(parse);
    }
    private String reconstructSentence(Parse parse) {
        return applyGrammarRules(parse);
    }
}
