from typing import Dict, Iterable, List
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import Field, LabelField, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, SpacyTokenizer

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

    def text_to_instance(self, sentence : str, label : str) -> Instance:
        # ID, sentence, ULF, ULF_AMR = datum[0], datum[1], datum[2], datum[3]

        #1. tokenize sentence
        sentence_field = TextField(
            self._tokenizer.tokenize(sentence), #tokens
            self._token_indexers                #indexing tokens
        )

        fields: Dict[str, Field] = {
            "sentence": sentence_field
        }
        fields["ULF"] = LabelField(label)
        return Instance(fields)
                
    def _read(self, file_path : str) -> Iterable[Instance]:
        data = open(file_path)
        lines = data.readlines()
        for line in lines:
            yield self.text_to_instance(line.strip(), "sentence")
        data.close()