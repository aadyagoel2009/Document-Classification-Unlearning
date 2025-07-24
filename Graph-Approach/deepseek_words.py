import os
from openai import OpenAI
from sklearn.datasets import fetch_20newsgroups

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),         
    base_url="https://api.deepseek.com/v1",      
)

dataset = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
docs10 = dataset.data[:10]
total = []

for doc in docs10:
    instruction = (
        f"Identify the {10} most distinctive, generalizable keywords or short phrases that capture the core themes of this document "
        "in a way that would help categorize *other* similar texts. "
        "Focus on topic or concept terms. Exclude overly document-specific proper names."
        "Output exactly one `term:score` per line, where `score` is a decimal between 0.0 and 1.0. No additional text or formatting."
    )
    messages = [
            {"role": "system", "content": "Output only lines of `term:score`."},
            {"role": "user",   "content": doc + "\n\n" + instruction}
        ]


    resp = client.chat.completions.create(
        model="deepseek-chat",  
        messages=messages,
        stream = False       
    )

    text = resp.choices[0].message.content.strip()
    lines = [l.strip() for l in text.splitlines() if ":" in l]
    terms = []
    for line in lines:
        term, scr = line.split(":", 1)
        try:
            score = float(scr.strip())
            terms.append((term.strip(), score))
        except ValueError:
            continue
    #time.sleep(1)
    #print(terms)  
    total.append(terms) 

print(total)
print(docs10)

