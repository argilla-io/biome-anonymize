import pandas as pd
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.pattern import Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import RecognizerResult, AnonymizerConfig


class Anonymizer:
    """
    Anonymizes a text using presidio https://microsoft.github.io/presidio/api/
    """

    def __init__(self, model_name="es_core_news_sm", language=None, default_entities=None):
        """
        Parameters
        ----------
        model_name
            Spacy model name
        language
            Supported language ("en", "es", "nl", etc)
        default_entities
            A list with the name of the default supported entities. If None, all the available for "language" are used.
        """
        self.language = model_name[:model_name.index("_")] if not language else language
        self.model_name = model_name
        self.nlp_engine = None
        self.registry = None
        self.entities = []
        self.anonymizers_config = {}
        self.analyzer = None
        self. anonymizer = None
        
        # Create configuration containing engine name and models
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": self.language, "model_name": self.model_name}]
        }

        # Create NLP engine based on configuration
        provider = NlpEngineProvider(nlp_configuration=configuration)
        self.nlp_engine = provider.create_engine()

        # Prepare base entities
        self.registry = RecognizerRegistry()
        if not default_entities:
            self.registry.load_predefined_recognizers(languages=[self.language])
        
        # Anonymizers mapping values
        for entity in self.entities:
            self.anonymizers_config[entity] = AnonymizerConfig("replace", {"new_value": entity})
            
        # Prepare analyzer
        self.analyzer = AnalyzerEngine(registry=self.registry, nlp_engine=self.nlp_engine, supported_languages=[self.language])
        # Prepare anonymizer
        self.anonymizer = AnonymizerEngine()
    
    def get_entities(self):
        """
        Returns
        -------
        List of entities used
        """
        return self.entities
    
    def add_recognizer_regex(self, regex, name):
        """
        Parameters
        ----------
        regex
            Regex to match in the text
        name
            Name assigned to the recognizer
        """
        pattern = Pattern(name=name, regex=regex, score=1)
        recognizer = PatternRecognizer(supported_entity=name,
                                       patterns=[pattern], supported_language=self.language)
        self.registry.add_recognizer(recognizer)
        self.entities.append(name)
        
        return None
    
    def add_recognizer_deny_list(self, deny_list, name):
        
        """
        Parameters
        ----------
        deny_list
            List of tokens to match in the text
        name
            Name assigned to the recognizer
        """
        regex = r"(?<= )(" + "|".join(deny_list) + r")(?=[ ,;.:!?)_-]+|$)" # Improve?
        pattern = Pattern(name=name, regex=regex, score=1)
        recognizer = PatternRecognizer(supported_entity=name,
                                       patterns=[pattern], supported_language=self.language)
        self.registry.add_recognizer(recognizer)
        self.entities.append(name)
        
        return None
    
    def anonymize_dataset(self, dataset, column="text", save_path=None, preprocess=lambda x: x):
        """
        Parameters
        ----------
        dataset
            The dataset to anonymize.
        column
            Name of the column to anonymize.
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

        # Preprocess in case there are NaNs
        dataset.dropna(how="all", axis=0, inplace=True)
        dataset.dropna(how="all", axis=1, inplace=True)
        dataset.fillna(value='nan', axis='columns', inplace=True)

        # Anonymize dataset
        dataset_PII = dataset.copy()
        dataset_PII[column] = dataset_PII[column].astype("str").apply(
            lambda x: self.anonymizer.anonymize(
                text=x,
                analyzer_results=self.analyzer.analyze(preprocess(x),language=self.language, entities=self.entities),
                anonymizers_config=self.anonymizers_config
            )
        )

        # Whether or not the row was modified during anonymization
        dataset_PII["has_PII"]=dataset_PII[column].apply(lambda x: any([value in x for value in self.entities]))

        if save_path:
            dataset_PII.to_csv(save_path)

        return dataset_PII

    def anonymize_text(self, text, preprocess=lambda x: x):
        """
        Parameters
        ----------
        text
            The text to anonymize.
        preprocess
            Optional function with only one parameter (string) that returns another string.
            This function will NOT modify the output text.
            In addition, this function expects the output string to have the same length as the input string.
            This can be useful to deal with accents, dieresis or special characters.

        Returns
        -------
        The anonymized text string

        """
        anonymized_text = self.anonymizer.anonymize(
                text=text,
                analyzer_results=self.analyzer.analyze(preprocess(text),language=self.language, entities=self.entities),
                anonymizers_config=self.anonymizers_config
        )
        has_PII = any([value in anonymized_text for value in self.entities])
        
        return (anonymized_text, has_PII)