import pandas as pd
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
import logging

'''

Used for dataset preprocessing in text column exclusively

'''


class TextPreprocessor:
    def __init__(self, keep_pronouns=True, keep_negations=True):
 
        self.logger = logging.getLogger(__name__)
 
        self._download_nltk_resources()
 
        self.lemmatizer = WordNetLemmatizer()
 
        self._configure_stopwords(keep_pronouns, keep_negations)
    
    def _download_nltk_resources(self):
 
        resources = ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"]
        for resource in resources:
            try:
                nltk.download(resource, quiet=True)
            except Exception as e:
                self.logger.warning(f"Failed to download {resource}: {e}")
    
    def _configure_stopwords(self, keep_pronouns, keep_negations):
 
        self.stop_words = set(stopwords.words("english"))
 
        if keep_pronouns:
            pronouns = {"i", "me", "my", "mine", "you", "your", "yours", 
                        "he", "him", "his", "she", "her", "hers", 
                        "it", "its", "we", "us", "our", "ours", 
                        "they", "them", "their", "theirs"}
            self.stop_words -= pronouns
 
        self.negation_words = {"not", "no", "never", "none", "nothing", "neither", 
                          "nobody", "nowhere", "cannot", "cant", "isnt", 
                          "arent", "wasnt", "werent", "doesnt", "dont", 
                          "wont", "hasnt", "havent", "hadnt", "shouldnt", 
                          "wouldnt", "couldnt"}
 
        if not keep_negations:
            self.negation_words = set()
    
    def _get_wordnet_pos(self, word):
 
        tag = pos_tag([word])[0][1][0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }
        return tag_dict.get(tag, wordnet.NOUN)   
    
    def clean_text(self, text):
 
        if not isinstance(text, str):
            return ""
 
        text = BeautifulSoup(text, "html.parser").get_text()
  
        text = re.sub(r"http\S+|www\S+", "", text)
 
        text = re.sub(r"[^a-zA-Z\s!?]", "", text)
 
        text = re.sub(r"\s+", " ", text).strip()
 
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)
 
        return text.lower() if text else text
    
    def lemmatize_text(self, text):
 
        if not isinstance(text, str):
            return ""
 
        tokens = word_tokenize(text)
 
        processed_tokens = []
        for word in tokens:
 
            if word not in self.stop_words or word in self.negation_words:
                lemma = self.lemmatizer.lemmatize(word, self._get_wordnet_pos(word))
                processed_tokens.append(lemma)
                
        return " ".join(processed_tokens)
    
    def preprocess_dataframe(self, df, text_column="text", output_path=None):
 
        if text_column not in df.columns:
            self.logger.error(f"Column {text_column} not found in dataset.")
            raise ValueError(f"Column {text_column} not found in dataset.")
 
        processed_df = df.copy()
 
        self.logger.info("Cleaning text...")
        processed_df[text_column] = processed_df[text_column].apply(self.clean_text)
  
        self.logger.info("Lemmatizing text...")
        processed_df[text_column] = processed_df[text_column].apply(self.lemmatize_text)

        null_count = processed_df[text_column].isnull().sum()
        if null_count > 0:
            self.logger.warning(f"Found {null_count} null values in text column. Removing...")
            processed_df = processed_df.dropna(subset=[text_column])
 
        if output_path:
            processed_df.to_csv(output_path, index=False)
            self.logger.info(f"Cleaned dataset saved as {output_path}")
        
        return processed_df

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    preprocessor = TextPreprocessor(keep_pronouns=True, keep_negations=True)
    
    try:
        df = pd.read_csv("Trainer & Dataset/Emotion/emodata_augmented.csv")

        cleaned_df = preprocessor.preprocess_dataframe(
            df, 
            text_column="text",
            output_path="Trainer & Dataset/Emotion/emoAG_cleaned.csv"
        )
        
        print("Text preprocessing complete. Cleaned dataset saved as emoAG_cleaned.csv.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
