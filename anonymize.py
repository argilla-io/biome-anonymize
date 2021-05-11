import os
import pandas as pd
import numpy as np
import spacy
import xlrd
import re
import unidecode
import unicodedata
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.pattern import Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import RecognizerResult, AnonymizerConfig

class Anonymizer:
    """
    Anonymizes a text using presidio https://microsoft.github.io/presidio/api/
    """

    def __init__(self):
        pass
    
    def get_default_entities(self, language="es"):
        """
        language
            Supported language ("en", "es", "nl", etc)
        """
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers(languages=[language])
        entities=[]
        for recognizer in registry.recognizers:
            for entity in recognizer.get_supported_entities():
                entities.append(entity)
        return entities
    
    def get_default_recognizers(self, language="es"):
        """
        language
            Supported language ("en", "es", "nl", etc)
        """
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers(languages=[language])
        return registry.recognizers
    
    def recognizer_regex(self, regex, name, language):
        """
        regex
            Regex to match in the text
        name
            Name assigned to the recognizer
        language
            Supported language ("en", "es", "nl", etc)
        """
        pattern = Pattern(name=name, regex=regex, score=1)
        recognizer = PatternRecognizer(supported_entity=name,
                                       patterns=[pattern], supported_language=language)
        return recognizer
    
    def recognizer_deny_list(self, deny_list, name, language):
        """
        deny_list
            List of tokens to match in the text
        name
            Name assigned to the recognizer
        language
            Supported language ("en", "es", "nl", etc)
        """
        regex = r"(?<= )(" + "|".join(deny_list) + r")(?=[ ,;.:!?)_-]+|$)" # Improve?
        pattern = Pattern(name=name, regex=regex, score=1)
        recognizer = PatternRecognizer(supported_entity=name,
                                       patterns=[pattern], supported_language=language)
        return recognizer
    
    def anonymize_dataset(cls, dataset, column="text", language="es", model_name="es_core_news_sm", recognizers=[], save_path=None,
        entities=[], preprocess=lambda x: x):
        """
        Parameters
        ----------
        dataset
            The dataset to anonymize.
        language
            Text language
        model_name
            Spacy model name
        column
            Name of the column to anonymize.
        recognizers
            List of custom recognizers added to the default ones
        entities
            Supported entities to anonymize. https://microsoft.github.io/presidio/supported_entities/
        preprocess
            Optional function with only one parameter (string) that returns another string.
            This function will NOT modify the output text.
            In addition, this function expects the output string to have the same length as the input string.
            This can be useful to deal with accents, dieresis or special characters.
        save_path
            Path to save the anonymized dataset as csv

        Returns
        -------
        An anonymized pandas DataFrame

        """
        dataset=pd.DataFrame(dataset)
        
        if column not in dataset.columns:
            raise KeyError("Column '{}' not in dataset".format(column))

        # Create configuration containing engine name and models
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": language, "model_name": model_name}]
        }

        # Create NLP engine based on configuration
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()

        #Prepare base entities
        registry = RecognizerRegistry()
        if recognizers == []:
            registry.load_predefined_recognizers(languages=[language])

        # Add custom recognizers
        for recognizer in recognizers:
            registry.add_recognizer(recognizer)
        registry.recognizers = list(set(registry.recognizers))

        # Prepare analyzer
        analyzer = AnalyzerEngine(registry=registry, nlp_engine=nlp_engine, supported_languages=[language])

        # Prepare anonymizer
        anonymizer = AnonymizerEngine()

        # Analyzer entities
        if entities == []:
            entities = cls.get_default_entities(language)
            if recognizers != []:
                for recognizer in recognizers:
                    for entity in recognizer.get_supported_entities():
                        entities.append(entity)
        entities = list(set(entities))

        # Anonymizer mapping values
        anonymizers_config={}
        for entity in entities:
            anonymizers_config[entity] = AnonymizerConfig("replace", {"new_value": entity})

        # Preprocess in case there are NaNs
        dataset.dropna(how="all", axis=0, inplace=True)
        dataset.dropna(how="all", axis=1, inplace=True)
        dataset.fillna(value='nan', axis='columns', inplace=True)

        # Anonymize dataset
        dataset_PII = dataset.copy()
        dataset_PII[column] = dataset_PII[column].astype("str").apply(
            lambda x: anonymizer.anonymize(
                text=x,
                analyzer_results=analyzer.analyze(preprocess(x),language=language,entities=entities),
                anonymizers_config=anonymizers_config
            )
        )

        # Whether or not the row was modified during anonymization
        dataset_PII["has_PII"]=dataset_PII[column].apply(lambda x: any([value in x for value in entities]))

        if save_path:
            dataset_PII.to_csv(save_path)

        return dataset_PII

    def anonymize_text(cls, text, language="es", model_name="es_core_news_sm", recognizers=[],
                       entities=[], preprocess=lambda x: x):
        """
        Parameters
        ----------
        text
            The text to anonymize.
        language
            Text language ("en", "es", "nl", etc)
        model_name
            Spacy model name
        recognizers
            List of custom recognizers added to the default ones
        entities
            Supported entities to anonymize. https://microsoft.github.io/presidio/supported_entities/
        preprocess
            Optional function with only one parameter (string) that returns another string.
            This function will NOT modify the output text.
            In addition, this function expects the output string to have the same length as the input string.
            This can be useful to deal with accents, dieresis or special characters.

        Returns
        -------
        The anonymized text string

        """
        df = pd.DataFrame(data=[text], columns=["text"])
        df = cls.anonymize_dataset(df, column="text", language=language, model_name=model_name, recognizers=recognizers,
                                   save_path=None, entities=[], preprocess=preprocess)
        return df.iloc[0]["text"]