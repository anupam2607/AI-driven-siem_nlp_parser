# src/query_generator.py
from typing import Dict

def generate_elastic_query(nlp_output: Dict) -> Dict:
    """
    Build Elastic DSL JSON from nlp_output.
    nlp_output example:
    {"intent":"fetch_logs","entities":{"attack_category":"Exploits","service":"dns"}}
    """
    entities = nlp_output.get("entities", {})
    must = []
    for k, v in entities.items():
        # Use match for text fields; you can tune to term / wildcard as needed
        must.append({"match": {k: v}})
    q = {"query": {"bool": {"must": must}}}
    return q

def generate_kql_string(nlp_output: Dict, table: str = "SecurityEvent") -> str:
    entities = nlp_output.get("entities", {})
    if not entities:
        return f"{table}\n| order by TimeGenerated desc"
    conds = []
    for k, v in entities.items():
        # simple equality style
        conds.append(f'{k} == "{v}"')
    return table + "\n| where " + "\n| where ".join(conds) + "\n| order by TimeGenerated desc"
