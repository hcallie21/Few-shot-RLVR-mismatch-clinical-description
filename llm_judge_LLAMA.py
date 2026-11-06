import os
# os.environ["CUDA_VISIBLE_DEVICES"]='0'
from huggingface_hub import login
login()
import torch
from transformers import  AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig #Llama4ForConditionalGeneration AutoModelForCausalLM

import sys
import json
import pandas as pd
from datasets import Dataset
from datasets import Features, Image as HFImage, Json, List, Value
import re
import time
import torch
import traceback
#file for analysis
input_dir = rf"/path/to/input"

model_id = "meta-llama/Llama-3.1-8B-Instruct"
response_save = rf"/save/output/here"


processor = AutoTokenizer.from_pretrained(
    model_id
)


SYSTEM_PROMPT = { 
    "role": "system",
    "content": """You are an expert judge who specializes in genetics. You will be provided with a question and response and your task is to evaluate whether the response confirms the presence of the feature in question. You have three options: yes, the text describes that the feature is present in the image, and no the text does not describe that the feature is present.
    Here are examples of the two accepted answer choices (Yes, No):  
    Example 1: 
    Question: This image shows a patient with Hereditary spherocytosis. Please evaluate whether the following clinical feature is visible and accurately describes what you observe in the image -- Feature: sickle cells.
    Model Response: **My Thought Process on Evaluating the Image for Sickle Cells**

        As an expert in this domain, I understand the nuances of interpreting peripheral blood smears. My primary directive here is to assess whether the presence of 'sickle cells' is discernible within the provided image.

        1.  **Initial Image Analysis:** My first step is always a thorough examination of the visual data. I observe a peripheral blood smear, and my attention is immediately drawn to the red blood cells.

        2.  **Feature Identification - The Target:** I'm specifically looking for 'sickle cells,' which, as I know, are red blood cells that have taken on a distinct elongated, crescent, or sickle-like shape. This is a hallmark of 
        conditions like sickle cell disease.

        3.  **Systematic Scan:** I conduct a systematic scan of the image, both horizontally and vertically, to ensure no area is overlooked. I pay close attention to the morphology of each red blood cell I encounter.

        4.  **Observation:** As I scan, I quickly note the presence of numerous red blood cells exhibiting the characteristic sickle or crescent shape. These are unmistakably the 'sickle cells' I was tasked with identifying.

        5.  **Contextual Note:** I also observe other red blood cells that appear round or slightly oval, which is typical. However, the distinct sickle shapes are the most striking and relevant finding for this specific evaluation.

        6.  **Addressing the Prompt's Nuance:** I recall the prompt's specific instruction to evaluate if ""sickle cells"" are visible. My observation confirms this.

        7.  **My Initial Conclusion:** Based purely on the visual evidence, my initial conclusion is a clear "Yes".
    Your Response Should Be: 'Yes', specifically because of this conclusion: 'my initial conclusion is a clear "Yes".'
            
    Example 2: 
    Question: This image shows a patient with Waardenburg syndrome. Please evaluate whether the following clinical feature is visible and accurately describes what you observe in the image -- Feature: microphthalmia.
    
    Model Response: Microphthalmia refers to abnormally small eyes. In the provided image, the patient's eyes appear to be of normal size relative to their face and do not exhibit the characteristic reduction in size associated with microphthalmia.
    Your Response Should Be: 'No', specifically because of this conclusion: 'do not exhibit the characteristic reduction in size associated with microphthalmia.'
    """
}
offload_dir = "./temp_offload"
os.makedirs(offload_dir, exist_ok=True)


rank = int(os.environ.get('RANK', '0'))
world = int(os.environ.get('WORLD_SIZE', '1'))
local_rank = int(os.environ.get('LOCAL_RANK', '0'))
torch.cuda.set_device(local_rank)
device = torch.device('cuda', local_rank)

print("Starting model load", flush=True)

# QLoRA: 4-bit quantization config passed to the trainer (set to None for plain LoRA).
quantization_config = BitsAndBytesConfig(     
                                         load_in_4bit=True,             # Load the model in 4-bit precision to save memory 
                                         bnb_4bit_compute_dtype=torch.float16,     # Data type used for internal computations in quantization 
                                         bnb_4bit_use_double_quant=True,           # Use double quantization to improve accuracy  
                                         bnb_4bit_quant_type="nf4",                # Type of quantization. "nf4" is recommended for recent LLMs
                                         )     
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    low_cpu_mem_usage=True,
    device_map="auto",
    quantization_config=quantization_config,
    max_memory={
        0: "70GiB",
        2: "70GiB",
        1: "70GiB",
        # 3: "70GiB",
        "cpu": "50GiB",
    },
    offload_folder = offload_dir,
)
model.eval()

print(model.hf_device_map)
print("Model Loaded... ", flush=True)

df = pd.read_csv(input_dir)

df = Dataset.from_pandas(df)
#change from original where do not require valid jsonlist in prompt
USER_QUESTION = """This image shows a patient with DISEASE_NAME. Please evaluate whether the following clinical feature is visible and accurately describes what you observe in the image -- Feature: PHENOTYPE_NAME. 
"""
def build_prompt(patient_condition, feature, USER_QUESTION):
    user_question = re.sub( "DISEASE_NAME", patient_condition, USER_QUESTION )
    user_question = re.sub( "PHENOTYPE_NAME", feature, user_question )
    return user_question

def make_conversation(example):
    # prompt = "Summarize the response provided to the question above ONLY with 'Yes' or 'No'."
    prompt = "Evaluate the following passage and confirm the following feature is present. "
    q = build_prompt(example['patient_condition'], example['feature'], USER_QUESTION=USER_QUESTION)
    
    return {
            "prompt":[ SYSTEM_PROMPT,
                      {
                          "role":"user",
                          "content": f"{q}\n{prompt}\n{example['answer']}",
                      },
                ],
            "question_index": example['question_index'],
            "answer": example['answer'],
    }
            


output_features = Features({
    "prompt": List({
        "role": Value("string"),
        "content": Value("string"),
    }),
    "question_index": Value("string"),
    "answer": Value("string"),
})

eval = df.map(
    make_conversation
)


def generate_text_from_sample( model, processor, sample, max_new_tokens=1000, device="cuda:0"):
        
    messages = sample['prompt']
    print(messages)

    model_inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
        )

    new_ids = generated_ids[:, model_inputs["input_ids"].shape[1]:]
    answer = processor.batch_decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    print(sample)
    return answer


results = []

for sample in eval: 
    parsed = ""
    raw_output = None
    error = None
    try:
        raw_output = generate_text_from_sample(
            model, processor, sample
        )
        normalized = raw_output.strip().lower()
        parsed_ok = normalized in {"yes", "no"}
        parsed = normalized.capitalize() if parsed_ok else ""
    except torch.cuda.OutOfMemoryError:
        traceback.print_exc()
        raise
    except Exception:
        parsed_ok = False
        error = traceback.format_exc()
        print(error, file=sys.stderr)
        raise

    results.append({
        "question_index":sample.get("question_index", ""),
        "summary": parsed,
        "raw_output":raw_output,
        "parsed_ok": parsed_ok,
        "answer":sample.get("answer",""),
        "error": error,
        "ppt_slide_number":sample.get("ppt_slide_number",""),
        "prompt_number":sample.get("prompt_number"),
        "patient_condition":sample.get("patient_condition"),
        "feature":sample.get("feature"),
    })

results = pd.DataFrame(results)
results.to_csv(response_save)
print("saved to : ", response_save)
# exit(0)
