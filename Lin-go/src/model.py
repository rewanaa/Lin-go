import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
from tqdm import tqdm

def infere_model(model_path,input_data,device):
    WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    results = {}
    for id in tqdm(input_data.keys(),desc="inferring model"):
        input_ids = tokenizer(
            [WHITESPACE_HANDLER(input_data[id])],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )["input_ids"].to(device)

        output_ids = model.generate(
            input_ids=input_ids,
            max_length=84,
            no_repeat_ngram_size=2,
            num_beams=4
        )[0].to("cpu")

        summary = tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        results[id]=summary
        
    return results