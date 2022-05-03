from typing import Dict, Iterable
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import Field, LabelField, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer

"""
    This file is a dataset reader class.
    It takes the file path as an input and takes the parental claass from Allennlp's DatasetReader.

    This reader takes the string data as an input and 
        i)   tokenize it
        ii)  index tokens 
    this is built for ULF dataset.
"""

@DatasetReader.register("ULF")
class ULFreader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, data : str, label : str) -> Instance:
        #1. tokenize the text
        data_field = TextField(
            self._tokenizer.tokenize(data), #tokens
            self._token_indexers                #indexing tokens
        )

        fields: Dict[str, Field] = {label: data_field}
        fields["ULF"] = LabelField(label)
        return Instance(fields)
                
    def _read(self, file_path : str) -> Iterable[Instance]:
        data = open(file_path)
        lines = data.readlines()
        for line in lines:
            yield self.text_to_instance(line.strip(), "text")
        data.close()