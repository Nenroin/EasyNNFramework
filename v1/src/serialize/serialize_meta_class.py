from v1.src.serialize.serialize_storage import SerializeStorage

class SerializeMetaClass(
    type,
):
    def __init__(cls,
                 name,
                 bases,
                 namespace,

                 id_field: str = 'name',
                 saved_fields: [] = None,

                 serialized_fields: [] = None,
                 ):
        super().__init__(name, bases, namespace)

        if not hasattr(cls, '_SerializeMetaClass__serialize_storage'):
            cls.__serialize_storage = SerializeStorage(
                storage_name=cls.__name__,
            )
            cls.__serialized_fields = []
            cls.__saved_fields = []

        serialized_fields = serialized_fields or []
        serialized_fields = [f"_{cls.__name__}{saved_field}" if saved_field.startswith('__')
                           else saved_field
                             for saved_field in serialized_fields]
        cls.__serialized_fields += serialized_fields

        saved_fields = saved_fields or []
        saved_fields = [f"_{cls.__name__}{replaced_field}" if replaced_field.startswith('__')
                           else replaced_field
                        for replaced_field in saved_fields]
        cls.__saved_fields += saved_fields



        def getstate(self):
            state = {
                id_field: getattr(self, id_field),
                **{
                    saved_field: getattr(self, saved_field) for saved_field in cls.__serialized_fields
                }
            }
            return state

        cls.__getstate__ = getstate

        def setstate(self, state):
            if id_field in state:
                if len(cls.__saved_fields) != 0 and state[id_field] in cls.__serialize_storage.generators:
                    generator = cls.__serialize_storage[state[id_field]]
                    generated_instance = generator()
                    for replaced_field in cls.__saved_fields:
                        setattr(self, replaced_field, getattr(generated_instance, replaced_field))
                if len(cls.__serialized_fields) != 0:
                    for saved_field in cls.__serialized_fields:
                        setattr(self, saved_field, state[saved_field])

        cls.__setstate__ = setstate

        def register_generators(generators: []):
            if isinstance(generators, list):
                cls.__serialize_storage.register_generators(
                    {
                        getattr(generator(), id_field): generator for generator in generators
                    }
                )

        cls.__register_generators__ = register_generators

        def register_generator(generator):
            cls.__serialize_storage.register_generator(
                generator_name=generator[id_field],
                generator_fn=generator
            )

        cls.__register_generator__ = register_generator


    def __new__(metacls, name, bases, namespace, **kwargs):
        return super().__new__(metacls, name, bases, namespace)
