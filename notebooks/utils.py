import re
def tokenize(text):
    text = text.lower()
    tokens = re.split('(\n|,|\s|\.|"|!|\?|-|“|\'|:|’|”|;|‘)',text)
    return tokens
def tokenize_to_id(tokens,token_to_id_mapping):
    # tokens = tokenize(text)
    ids = [token_to_id_mapping[t] for t in tokens]
    return ids
def detokenize_to_text(ids,id_to_token_mapping):
    tokens = [id_to_token_mapping[t] for t in ids]
    return "".join(tokens)