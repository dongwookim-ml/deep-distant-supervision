IGNORE_NRE_TAG = ('O')


def extract_ners(tokens):
    """
    Extract consecutive ners from the result of CoreNLPNERTagger
    :param tokens: list of tuple with token and tag
    :return: list of tuple with list of tokens and corresponding tag
    """
    ners = list()

    candid_entity = list()
    keep = False
    prev_tag = 'O'

    for i, (token, tag) in enumerate(tokens):
        if keep:
            if tag not in IGNORE_NRE_TAG:
                if prev_tag == tag:
                    candid_entity.append(token)
                    keep = True
                else:
                    ners.append((candid_entity, prev_tag))
                    candid_entity = list()
                    candid_entity.append(token)
                    keep = True
            else:
                ners.append((candid_entity, prev_tag))
                keep = False
        else:
            if tag not in IGNORE_NRE_TAG:
                candid_entity = list()
                candid_entity.append(token)
                keep = True
        prev_tag = tag

    for ner in ners:
        print(' '.join(ner[0]), ner[1])

    return ners


def lookup_freebase(ner):
    """
    Return existence of ner from Freebase entity dictionary
    :param ner:
    :return:
    """
    pass


def load_freebase():
    """
    Load freebase entity dictionary from saved dict
    :return:
    """
    pass
