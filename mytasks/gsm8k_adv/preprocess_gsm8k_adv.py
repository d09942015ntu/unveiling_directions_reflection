import re


def cot_preprocess(cot_raw):
    cot_raw = re.sub(r'[ \t]+', ' ', cot_raw)
    cot_raw = cot_raw.replace(" \n","\n")
    cot_raw = cot_raw.replace(".\n","\n")
    cot_raw = re.sub(r'([+\-*/=]) ', r'\1', cot_raw)
    return cot_raw


def doc_to_text(doc, cot_messy, wait_token="Wait"):
        return ("Answer the question: \n\n"
                + doc['question']
                + "\nPlease always end your response with the final numerical answer.\n"
                + " Let’s solve this step by step … "
                + cot_messy[:cot_messy.rfind("\n####")]
                + f" {wait_token} "
                )

def doc_to_text_wait(doc):
    return cot_preprocess(doc_to_text(doc, doc['messy_cot'], wait_token="Wait"))

def doc_to_text_alternatively(doc):
    return cot_preprocess(doc_to_text(doc, doc['messy_cot'], wait_token="Alternatively"))

def doc_to_text_check(doc):
    return cot_preprocess(doc_to_text(doc, doc['messy_cot'], wait_token="Check"))

def doc_to_text_eos(doc):
    return cot_preprocess(doc_to_text(doc, doc['messy_cot'], wait_token="<|endoftext|>"))

def doc_to_text_sharp(doc):
    return cot_preprocess(doc_to_text(doc, doc['messy_cot'], wait_token="#"))

def doc_to_text_answer(doc):
    return cot_preprocess(doc_to_text(doc, doc['messy_cot'], wait_token="Answer"))

def doc_to_text_result(doc):
    return cot_preprocess(doc_to_text(doc, doc['messy_cot'], wait_token="Result"))

def doc_to_text_retry(doc):
    return cot_preprocess(doc_to_text(doc, doc['messy_cot'], wait_token="Retry"))


def doc_to_target(doc):

    return doc['answer'] + "</s>"
