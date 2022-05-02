from ULFreader import ULFreader as reader 
from allennlp.data.data_loaders import MultiProcessDataLoader as loader
from allennlp.data.vocabulary import Vocabulary

# Construct a dataloader directly for a dataset which contains allennlp
# Instances which have _already_ been indexed.
def load_ULF(
    isStog = False,
    _batch_size = 32
):
    data_folder_path =  "ulf1" if not isStog else "ulf1stog"
    raw_data_path = "{}/raw".format(data_folder_path)
    data_reader = reader()

    vocab = Vocabulary.from_instances(data_reader.read(raw_data_path))
    ULF_loader = loader(
        data_reader,
        raw_data_path,
        batch_size = _batch_size
    )
    ULF_loader.index_with(vocab)
    return ULF_loader
    
