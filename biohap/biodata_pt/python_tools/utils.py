import pandas as pd
from rdflib import Graph


def jsonld_to_rdflib(jsonld_text, base=None):
    """
    Parse JSON-LD text into an rdflib.Graph and return the Graph.
    """
    g = Graph()
    # rdflib accepts a JSON-LD string as input; base is optional
    g.parse(data=jsonld_text, format="json-ld", publicID=base)
    return g


def sparql_json_to_df(sparql_json):
    """
    Convert a SPARQL SELECT query JSON result to a pandas DataFrame.
    
    Parameters
    ----------
    sparql_json : dict
        JSON returned by Fuseki / SPARQL endpoint with Accept: application/sparql-results+json
    
    Returns
    -------
    pd.DataFrame
    """
    vars_ = sparql_json.get("head", {}).get("vars", [])
    rows = []

    for binding in sparql_json.get("results", {}).get("bindings", []):
        row = {}
        for var in vars_:
            # Some results might not bind all variables
            if var in binding:
                row[var] = binding[var]["value"]
            else:
                row[var] = None
        rows.append(row)

    df = pd.DataFrame(rows, columns=vars_)
    return df