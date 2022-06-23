import json
import os.path

from tqdm import tqdm

downloads_dir: str = "/home/konstantin/Downloads"
path: str = os.path.join(downloads_dir, "data")
# for line in open(path).readlines():
#     a = 0
dbpedia_path = os.path.join(downloads_dir, "infobox-properties_lang=de.ttl")
count: int = 0
entities: set[str] = set()
properties: set[str] = set()
for line in tqdm(open(dbpedia_path).readlines()):
    # count += 1
    line_parts: list[str] = line.split("<")
    entity_no_namespace: str = line_parts[1].split("/")[-1]
    entities.add(entity_no_namespace[:-2])
    property_id: str = line_parts[2].split(">")[0]
    properties.add(property_id.split("/")[-1])
    # print(line)
    # if count == 50000:
    #     break
json.dump(list(entities), open("wikidata/entities.json", "w+"))
json.dump(list(properties), open("wikidata/properties.json", "w+"))
# data: dict
# with open(path) as f:
#     data = json.load(f)
b = 0
