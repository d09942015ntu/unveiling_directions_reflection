import os

pwd = os.path.dirname(__file__)
wait_token="NONE"
with open(os.path.join(pwd, "wait_token_3.txt"),"r") as f:
    wait_token="".join(f.readlines()).strip()
print(f"wait_token_in_task:{wait_token}")


def doc_to_text(doc, messy_cot, wait_token="Wait"):
    code, inp = doc["code"], doc["input"]
    #doc["messy_cot"]

    return f"""You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Execute the program step by step before arriving at an answer, and provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.
    
[PYTHON]
code and assert statement
[/PYTHON]
[THOUGHT]
your step by step thought
[/THOUGHT]
[ANSWER]
assert f(input) == output
[/ANSWER]

[PYTHON]
{code}
assert f({inp}) == ??
[/PYTHON]
[THOUGHT]
{messy_cot}

""" + f" {wait_token} "


def doc_to_text_any(doc):
    return doc_to_text(doc, doc['messy_cot'], wait_token=wait_token)
