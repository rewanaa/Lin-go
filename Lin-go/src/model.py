import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
from tqdm import tqdm

def infere_model(model_path,input_data,device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    results = {}
    for id in tqdm(input_data.keys(),desc="inferring model"):
        input_ids = tokenizer(
            input_data[id],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )["input_ids"].to(device)

        output_ids = model.generate(
            input_ids=input_ids, 
            max_length=int(len(example_input_text)*100/35)+1, 
            num_beams=3, 
            repetition_penalty=3.0
        )[0].to("cpu")

        summary = tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        results[id]=summary
        
    return results
