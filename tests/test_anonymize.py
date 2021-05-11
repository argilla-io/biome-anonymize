from anonymize import Anonymizer


def test_anonymize_workflow():
    anonymizer = Anonymizer(lang="en")

    anonymizer.regex_recognize(r"The regex pattern", entity="TEST")
    black_list = ["BLACK", "LIST"]
    anonymizer.black_list_recognizer(
        black_list, entity="BLACK_LIST", lang="en"
    )  # Overrides default language

    anonymizer.default_recognizers(
        ["ACCOUNT_BANK", "PRESIDIO_PHONE_NUMBER"]  # other default presidio entities
    )
    anonymized_text, has_personal_info = anonymizer.anonymize_text(
        "The black text data"  # I'm not sure if text preprocessing should be includes as part anonymizer
    )
