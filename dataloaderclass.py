class Instance(Mapping[str, Field]):
    def __init__(self, fields: MutableMapping[str, Field]) -> None:


class Batch(Iterable):
    def __init__(self, instances: Iterable[Instance]) -> None:
        

class DataLoader(Registrable):