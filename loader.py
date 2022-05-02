from typing import Dict
from ULFreader import ULFreader as reader 
from allennlp.data.data_loaders import MultiProcessDataLoader as loader
from allennlp.data.vocabulary import Vocabulary
import json

"""
    This file is a function that loads a dataset reader and loader.

    It takes the following input
        i)  isStog :  there are two available : dataset original and STOG.
        ii) _batch_size : batch size for the loader

    It returns a dictionary including
        i) loaders = a triple of loaders for ...
            1) raw sentence data 
            2) ULF preprocessed data 
            3) AMR parsed data 
        ii) vocabularies = a triple of vocabularies for ...
            1) raw sentence data 
            2) ULF preprocessed 
            3) AMR parsed data
"""

def load_ULF(isStog = False, _batch_size = 32) -> Dict:
    data_folder_path =  "ulf1" if not isStog else "ulf1stog"

    """
        #1. fetch the ULF and AMR parsed dataset for ULF interpretation.
            - from the JSON file, it extracts ULF preprocessed and AMR processed data.
            - and write to the file. 
    """
    ALL = open( "{}/all.json".format(data_folder_path)) #JSON data
    data = json.load(ALL)

    raw_data_path = "{}/raw".format(data_folder_path) #fetching the raw data file
    ULF, AMR = "ulf_extracted", "amr_extracted"
    ULF_f, AMR_f = open(ULF, 'w'), open(AMR, 'w')
    gen_line = lambda i, n, val : "{}{}".format(val, '\n') if i < n - 1 else val
    n = len(data)

    ULF_lines, AMR_lines = [], []
    for i, datum in enumerate(data):
        ulf, amr = datum[2], datum[3]
        ULF_line, AMR_line = gen_line(i, n, ulf), gen_line(i, n, amr) 
        ULF_lines.append(ULF_line)
        AMR_lines.append(AMR_line)
    ULF_f.writelines(ULF_lines)
    AMR_f.writelines(AMR_line)

    ALL.close() 
    ULF_f.close()
    AMR_f.close()

    """
        #2. read the raw data and return the dataset loader.
            - initiate a dataset reader.
            - construct vocabularies for raw sentence, ulf preprocessed, and amr processed data
            - construct data loaders based on the same three data.
            - return loaders and vocabularies
    """
    data_reader = reader()  #data reader
    
    #building vocabularies                        
    sentence_vocab = Vocabulary.from_instances(data_reader.read(raw_data_path))
    ULF_vocab = Vocabulary.from_instances(data_reader.read("ulf_extracted"))
    AMR_vocab = Vocabulary.from_instances(data_reader.read("amr_extracted"))

    #building a sentence loader.
    ULF_loader = init_loader(ULF_vocab, data_reader, "ulf_extracted", _batch_size)
    AMR_loader = init_loader(AMR_vocab, data_reader, "amr_extracted",  _batch_size)
    sentence_loader = init_loader(sentence_vocab, data_reader, raw_data_path, _batch_size)

    return {
        "loaders" : (sentence_loader, ULF_loader, AMR_loader),
        "vocabularies" : (sentence_vocab, ULF_vocab, AMR_vocab)
    }
    
def init_loader(vocab, data_reader, file_path, batch_size):
    data_loader = loader(                                  #init the data loader.
        data_reader,
        file_path,
        batch_size = batch_size
    )
    data_loader.index_with(vocab)                 #index the loader data with the vocabularies.
    return data_loader
