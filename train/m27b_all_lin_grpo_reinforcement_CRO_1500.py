import os
import sys
import re
import pandas as pd
import argparse
import torch
import torch.distributed as dist
from PIL import Image
from datasets import Dataset
from datasets import Features, Image as HFImage, Json, List, Value
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from sentence_transformers import SentenceTransformer, util
from transformers import AutoProcessor, BitsAndBytesConfig,  AutoModelForImageTextToText
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import get_repetition_penalty_reward, get_soft_overlong_punishment
from sklearn.model_selection import GroupShuffleSplit
import logging
import threading
from collections.abc import Callable
from pathlib import Path

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
torch.cuda.set_device(LOCAL_RANK)
DEVICE = torch.device("cuda", LOCAL_RANK)

ds_path = "data/gemini_edited_long_thinking_content.csv"
ROOT = os.getcwd() or ""
output_dir = "medgemma_27b_grpo_trained" 
model_id = "google/medgemma-27b-it" 

SYSTEM_PROMPT = { 
        "role": "system",
        "content": "You are an expert clinician who specializes in genetics. You must respond with a logical rigor and deep explanatory depth." 
}

USER_QUESTION = """This image shows a patient with DISEASE_NAME. Please evaluate whether the following clinical feature is visible and accurately describes what you observe in the image -- Feature: PHENOTYPE_NAME. 
    """


def arguments(): 
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--output_dir', type=Path, default=Path(ROOT, output_dir))
    p.add_argument('--model_id', type=str, default="google/medgemma-27b-it")
    p.add_argument('--max_tokens', type=int, default=1500)
    p.add_argument("--lora", type=bool, default=False)
    p.add_argument(
        "--lora", action=argparse.BooleanOptionalAction, default=False
    )
    p.add_argument(
        "--qlora", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument(
        "--reasoning_path",
        type=Path,
        default=ds_path,
        help="CSV path for edited long-thinking reasoning",
    )
    p.add_argument("--image_root", type=Path, default=None,
                   help="path to image folder eg. <root>/slide_images/slide_002.png")
    a = p.parse_args()
    return a

def main(a):

    processor = AutoProcessor.from_pretrained(a.model_id, padding_side="left")
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
        "device_map": {"": LOCAL_RANK},
    }

    if a.qlora:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForImageTextToText.from_pretrained(
        a.model_id,
        **load_kwargs,
    )

    if a.qlora:
        model = prepare_model_for_kbit_training(model)

    if a.qlora or a.lora:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules="all-linear",
            use_dora=not a.qlora,
            init_lora_weights="gaussian",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
   
   
    df = pd.read_csv(a.reasoning_path)
    df = df.rename(columns={"edited_long_thinking_content":"solution"})
    df = df.drop(columns=['prompt_number','model','short_answer_text','answer','phenotype_in_answer','reasoning','long_thinking_content','parsed_ok','ppt_slide_number'])
    df = Dataset.from_pandas(df)


    def build_prompt(patient_condition, feature, USER_QUESTION):
        user_question = re.sub( "DISEASE_NAME", patient_condition, USER_QUESTION )
        user_question = re.sub( "PHENOTYPE_NAME", feature, user_question )
        return user_question

    def make_conversation(example):
        img_path = example["img_path"]
        image = Image.open(str(a.image_root / img_path))
        try: 
            if image.mode != 'RGB':
                image = image.convert('RGB') 
        except Exception as e: 
            print("Image conversion error")    
            print(e)
        prompt = build_prompt(example['patient_condition'], example['feature'], USER_QUESTION=USER_QUESTION)
        return {
                "prompt":[ SYSTEM_PROMPT,
                        {
                            "role":"user",
                            "content":[
                                {"type":"image"},
                                {"type":"text","text": (prompt)},
                            ],
                        },
                    ], 
                "image":image,
                "solution": example['solution'],
        }
                
    output_features = Features({
        "prompt": List({
            "role": Value("string"),
            "content": Json(),
        }),
        "image": HFImage(),
        "solution": Value("string"),
    })

    train = df.map(
        make_conversation,
        remove_columns=df.column_names,
        features=output_features,
    )

    repetition_reward = get_repetition_penalty_reward(ngram_size=3, max_penalty=-1.0) #default settings
    overlong_reward = get_soft_overlong_punishment(max_completion_len=1000, soft_punish_cache=500) #L optimal considered to be between 500 and 1000 tokens
    model_sentence_similarity = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def simple_cosine_sim_reward(completions, solution, log_extra: Callable[[str, list], None] | None = None, **kwargs) -> float:
        """Compute length-based rewards to discourage overthinking and promote token efficiency.

        Taken from the Kimi 1.5 tech report: https://huggingface.co/papers/2501.12599

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        Returns:
            List of rewards where:
            - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
            - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len)) #! cannot have min 0 with this dataset
        """
                            
        contents = completions

        answer_parsed_strs = []
        
        # First check correctness of answers
        correctness = []
        for content, sol in zip(contents, solution):
            
            #https://huggingface.co/tasks/sentence-similarity
            answer_parsed = content[0]["content"]
            embedding_1 = model_sentence_similarity.encode(answer_parsed, convert_to_tensor=True)
            embedding_2 = model_sentence_similarity.encode(sol, convert_to_tensor=True)
            cosine_sim_score = util.pytorch_cos_sim(embedding_1, embedding_2).cpu().numpy()[0][0]

            correctness.append ( cosine_sim_score )
            answer_parsed_strs.append(answer_parsed)

        # # Calculate lengths deleted        
        if log_extra is not None:
            log_extra("solution", list(solution))

        return correctness


    training_args = GRPOConfig(
        chat_template_kwargs={"return_tensors": "pt"},
        # Training schedule / optimization
        learning_rate=2e-5,
        num_train_epochs=20,
        beta = 0.01,
        bf16=True,
        fp16=False,
        # Parameters that control GRPO training (you can adapt them)
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 1,
        max_completion_length=a.max_tokens, 
        num_generations=8, # default: 8    
        # Optimizations
        optim = "paged_adamw_8bit",                          
        use_liger_kernel=True,                             
        #^ Ollama settings 
        # repeat_last_n = 64, #? i don't think that ollama supports that directly
        # num_predict=-1, #? i don't think that ollama supports that directly
        # draft_num_predict=4, #? i don't think that ollama supports that directly
        repetition_penalty = 1.0,
        temperature = 0.8,
        seed=0,
        top_k=40,
        top_p=0.9,
        min_p=0.0,
        reward_weights = [0.8,0.1, 0.1],
        # Parameters related to reporting and saving
        output_dir=str(a.output_dir),                              
        logging_steps=10,                             
        log_completions=False,       
        remove_unused_columns=False 
    )


    trainer = GRPOTrainer(
        model=model,
        reward_funcs= [simple_cosine_sim_reward, repetition_reward, overlong_reward], 
        args=training_args,
        train_dataset=train,
        processing_class=processor,
    )

    trainer.train()
    trainer.save_model(str(a.output_dir))
    print("Model is saved to: ", str(a.output_dir))

if __name__ == '__main__':
    args = arguments() 
    main(args)