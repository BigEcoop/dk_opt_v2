import json
import pkg_resources

def load_scoring():
    path = pkg_resources.resource_filename('optimizer', '../data/scoring.json')
    with open(path) as f:
        return json.load(f)