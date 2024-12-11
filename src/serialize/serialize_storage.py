class SerializeStorage:
    def __init__(
            self,
            storage_name: str,
    ):
        self.storage_name = storage_name
        self.generators = {}

    def register_generator(self, generator_name, generator_fn):
        self[generator_name] = generator_fn

    def register_generators(self, generators):
        self.generators.update(generators)

    def __getitem__(self, generator_name):
        return self.generators[generator_name]

    def __setitem__(self, generator_name, generator_fn):
        self.generators[generator_name] = generator_fn
