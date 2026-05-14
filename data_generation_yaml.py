# Using YAML file to acquire parameters for data generation

import yaml

# Reading parameters from YAML file
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

print("Parameters read from 'parameters.yaml':")
print(parameters)