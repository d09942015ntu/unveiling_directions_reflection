import re
import os

pwd = os.path.dirname(__file__)
wait_token="NONE"
with open(os.path.join(pwd, "wait_token_1.txt"),"r") as f:
    wait_token="".join(f.readlines()).strip()
print(f"wait_token_in_task:{wait_token}")


def cot_preprocess(cot_raw):
    cot_raw = re.sub(r'[ \t]+', ' ', cot_raw)
    cot_raw = cot_raw.replace(" \n","\n")
    cot_raw = cot_raw.replace(".\n","\n")
    cot_raw = re.sub(r'([+\-*/=]) ', r'\1', cot_raw)
    return cot_raw


def doc_to_text(doc, cot_messy, wait_token):
        return ("Answer the question: \n\n"
                + doc['question']
                + "\nPlease always end your response with the final numerical answer.\n"
                + " Let’s solve this step by step … "
                + cot_messy[:cot_messy.rfind("\n####")]
                + f" {wait_token} "
                )

def doc_to_text_any(doc):
    return cot_preprocess(doc_to_text(doc, doc['messy_cot'], wait_token=wait_token))


def doc_to_target(doc):

    return doc['answer'] + "</s>"
