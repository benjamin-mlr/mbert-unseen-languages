
import json
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

subword_char = set([])
for i in range(250002):
	try:
		subword = tokenizer.convert_ids_to_tokens(i)
		subword = set(list(subword))
		subword_char|=subword

		#print(f"--> i {i}",tokenizer.convert_ids_to_tokens(i))
	except Exception as e:
		print(e)
		print("Ending loop index ",i)
		break
print(subword_char)
print(len(subword_char), " char set in xlm-r vocabulary")
saving="./char_set_xlmr.json"
with open(saving,"w", encoding="utf-8") as f:
	json.dump(list(subword_char),f, ensure_ascii=False, indent=4)
print(saving)