from transformers import AutoTokenizer, LlamaForCausalLM
import torch

tok_dir = "/home/arxiv_code/tokenizers_rust/tokenizer_out_wikitext103_32k_n9_full_maxchars48_punctnorm_nostage_lowtpc_crossmix_v047"
model_dir = "/home/arxiv_code/tokenizers_rust/model_lenmax_lenmax_punctnorm_nostage_crossmix_v047_vs_superbpe_pack4000_evalval_steps10000_v048/best_bpc"

tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
model = LlamaForCausalLM.from_pretrained(model_dir).to("cuda").eval()

prompt = "= Valkyria Chronicles III ="
inputs = tok(prompt, return_tensors="pt").to("cuda")

out = model.generate(
    **inputs,
    max_new_tokens=160,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.12,
    no_repeat_ngram_size=3,
)

print(tok.decode(out[0], skip_special_tokens=True))