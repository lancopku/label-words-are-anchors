import weakref


class SetterWithRestore:
    def __init__(self):
        self.attribute_history = {}

    def set(self, bound_method, new_value):
        obj = bound_method.__self__
        attr_name = bound_method.__name__

        key = (weakref.ref(obj), attr_name)
        if key not in self.attribute_history:
            self.attribute_history[key] = getattr(obj, attr_name)

        setattr(obj, attr_name, new_value)

    def restore(self, bound_method):
        obj = bound_method.__self__
        attr_name = bound_method.__name__

        key = (weakref.ref(obj), attr_name)
        if key in self.attribute_history:
            setattr(obj, attr_name, self.attribute_history[key])
            del self.attribute_history[key]
        else:
            print(f"{attr_name} has not been set by the setter.")

    def restore_all(self):
        for key, original_value in self.attribute_history.items():
            obj_ref, attr_name = key
            obj = obj_ref()
            if obj is not None:
                setattr(obj, attr_name, original_value)

        self.attribute_history.clear()
