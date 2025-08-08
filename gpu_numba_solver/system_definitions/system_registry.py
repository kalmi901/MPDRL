import importlib
import os

registered_systems = {}

def ode_model(name:str|None = None):
    def decorator(cls):
        model_name = name or cls.__name__
        registered_systems[model_name] = cls
        return cls
    return decorator

def discover_systems_in_package(package_path: str, base_package: str):
    print("Discover systems in package...")
    for fname in os.listdir(package_path):
        if fname.endswith(".py") and not fname.startswith("_"):
            module_name = fname[:-3]
            full_module_path = f"{base_package}.{module_name}"
            importlib.import_module(full_module_path)

    print(registered_systems)