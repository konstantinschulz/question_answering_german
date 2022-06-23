# <eventkg_relation_2866053> rdf:object <entity_13686511> .
# <entity_13686511> <http://dbpedia.org/resource/Oslo>
# <eventkg_relation_2866053> rdf:subject <entity_9279725> .

# <eventkg_relation_2866069> rdf:subject <entity_12568885> .
# <entity_12568885> <http://dbpedia.org/resource/Maimonides>
# <eventkg_relation_2866069> rdf:object <entity_124487> .
# <entity_10124487> <http://dbpedia.org/resource/Winterberg_bobsleigh,_luge,_and_skeleton_track>
# <eventkg_relation_2866069> sem:roleType dbo:influenced eventKG-g:dbpedia_en .
# <eventkg_relation_2866069> sem:roleType yago:influences eventKG-g:yago .

# <eventkg_relation_2866114> rdf:type eventKG-s:Relation eventKG-g:dbpedia_en .
# <eventkg_relation_2866114> rdf:subject <entity_10241086> eventKG-g:dbpedia_en .
# <eventkg_relation_2866114> rdf:object <entity_12680178> eventKG-g:dbpedia_en .
# <entity_12680178> <http://dbpedia.org/resource/Rock_music>
# <eventkg_relation_2866114> sem:roleType dbo:genre eventKG-g:dbpedia_en .
import json
import os.path

from tqdm import tqdm

target: str = "10241086"
input_dir: str = os.path.abspath("input")
relations_entities_path: str = os.path.join(input_dir, "relations_entities_other_dbo.nq")
types_path: str = os.path.join(input_dir, "types_dbpedia.nq")
relations_path: str = os.path.join(input_dir, "relations_other_dbo.nq")
event_kg_dir: str = os.path.abspath("event_kg")
entity_map_path: str = os.path.join(event_kg_dir, "entities.json")
entities_path: str = os.path.join(input_dir, "entities_filter.nq")
relations_filtered_path: str = os.path.join(event_kg_dir, "relations_filtered.json")


def filter_relations():
    current_relation: str = ""
    current_subject: str = ""
    current_object: str = ""
    current_relation_kind_set: set[str] = set()
    subject_string: str = "subject"
    object_string: str = "object"
    entity_string: str = "entity"
    role_type_string: str = "sem:roleType"
    relations_filtered: list[str] = []
    for line in tqdm(open(relations_entities_path).readlines()):
        # if len(relations_filtered) == 10:
        #     break
        line_parts: list[str] = line.split(" ")
        if line_parts[0] != current_relation:
            if current_subject and current_object:
                relations_filtered.append(
                    " ".join([current_subject, "_".join(current_relation_kind_set), current_object]))
            current_subject = ""
            current_object = ""
            current_relation_kind_set = set()
        current_relation = line_parts[0]
        relation_kind: str = line_parts[1]
        if relation_kind == role_type_string:
            current_relation_kind_set.add(line_parts[2])
            continue
        entity: str = line_parts[2]
        if entity_string in entity:
            if subject_string in relation_kind:
                current_subject = entity
            elif object_string in relation_kind:
                current_object = entity
    json.dump(relations_filtered, open(relations_filtered_path, "w+"))


def get_relations(target: str):
    target_with_marker: str = f"_{target}>"
    lines: list[str] = []
    for line in tqdm(open(relations_entities_path).readlines()):
        if target_with_marker in line:
            lines.append(line)
    return lines


def make_cache(src_path: str, cache_path: str):
    mapping: dict[str, str] = dict()
    for line in tqdm(open(src_path).readlines()):
        line_parts: list[str] = line.split(" ")
        if line_parts[0] not in mapping:
            mapping[line_parts[0]] = line_parts[1][:-1]
    json.dump(mapping, open(cache_path, "w+"))


# make_cache(entities_path, entity_map_path)
filter_relations()
