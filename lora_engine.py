# reference: https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing#scrollTo=QmUBVEnvCDJv

from unsloth import FastLanguageModel
import multiprocessing
import json
import time
from datasets import Dataset
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from json_repair import repair_json
import gc
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers


class LoRAEngine:

    def __init__(self, max_steps=10):
        self.model_name = "unsloth/Llama-3.2-1B-Instruct"
        self.max_seq_length = 2048
        self.max_steps = max_steps

        self.alpaca_prompt = """Answer the instruction question with a valid JSON object with the key 'prediction' (value should be one of: 'SUPPORTS', 'REFUTES', 'NOTENOUGHINFO') and 'response' (your detailed answer). 

        ### Instruction:
        {}

        ### Input:
        {}

        ### Response:
        {}"""

    def formatting_prompts_func(self, examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = self.alpaca_prompt.format(instruction, input, output) + self.EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }

    def train(self, failure_label, failure_shard):
        base_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
            load_in_4bit=True,
        )

        self.EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
        
        my_dict = {'instruction': [], 'input': [], 'output': []}
        for sample in failure_shard:
            claim, label, evidences = sample
            my_dict["instruction"].append(claim)
            response = " ".join(evidences)
            if len(response.strip()) == 0:
                response = "There is not enough information available to respond to this claim."
            output = str({'prediction': label, 'response': response})
            my_dict["output"].append(output)
            my_dict["input"].append(None)

        dataset = Dataset.from_dict(my_dict)
        dataset = dataset.map(self.formatting_prompts_func, batched = True)

        model = FastLanguageModel.get_peft_model(
            base_model,
            r = 4,
            target_modules = [
                "q_proj", "k_proj", 
                # "v_proj", "o_proj",
                # "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha = 16,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 42,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )

        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = dataset,
            dataset_text_field = "text",
            max_seq_length = self.max_seq_length,
            dataset_num_proc = 2,
            packing = True, # Can make training 5x faster for short sequences.
            args = TrainingArguments(
                per_device_train_batch_size = 4,
                gradient_accumulation_steps = 4,
                warmup_steps = 5,
                # num_train_epochs = 1, # Set this for 1 full training run.
                max_steps = self.max_steps,
                learning_rate = 2e-4,
                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 42,
                # output_dir = "outputs",
                report_to = "none", # Use this for WandB etc
            ),
        )
        trainer_stats = trainer.train()

        model.save_pretrained(f"out/lora_adapter_{failure_label}")

        del base_model
        del tokenizer
        del model
        del trainer
        gc.collect()
        torch.cuda.empty_cache() 

    def query(self, failure_label, query_str, lora_type):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_mem = torch.cuda.memory_allocated()

        tic = time.perf_counter()

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            load_in_4bit= True,
            dtype=torch.bfloat16,
            device_map=lora_type,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = PeftModel.from_pretrained(
            base_model, 
            f"out/lora_adapter_{failure_label}"
        )
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map=lora_type
        )

        output = pipeline(
            self.alpaca_prompt.format(
                query_str, # instruction
                "", # input
                "", # output - leave this blank for generation!
            ),
            do_sample=False,
            top_k=1,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=128,
            return_full_text=False
        )
        response = output[0]["generated_text"].split("### Response:\n")[-1].split("<|eot_id|>")[0]
        
        print(response)
        try:
            response = json.loads(repair_json(response))
        except json.decoder.JSONDecodeError:
            response = {'prediction': 'NOTENOUGHINFO', 'response': ''}

        latency = time.perf_counter() - tic # seconds
        torch.cuda.synchronize()
        peak_after = torch.cuda.max_memory_allocated()
        vram_usage = (peak_after - start_mem) / 1048576 # MB, 1024 * 1024

        response["question"] = query_str
        response["claim"] = query_str
        response["latency"] = latency
        response["vram_usage"] = vram_usage

        del base_model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache() 

        return response
    
    def safe_query(self, failure_label, query_str, lora_type, max_mem_mb, max_time_s):
        ctx = multiprocessing.get_context("spawn")
        result_queue = ctx.Queue()

        p = ctx.Process(
            target=lora_target_fn,
            args=(self, failure_label, query_str, lora_type, result_queue)
        )
        p.start()

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_mem = torch.cuda.memory_allocated() / 1048576 # MB, 1024 * 1024
        start_time = time.time()

        while p.is_alive():
            elapsed = time.time() - start_time
            torch.cuda.synchronize()
            peak_after = torch.cuda.max_memory_allocated() / 1048576 # MB, 1024 * 1024
            gpu_mem = peak_after - start_mem
            
            if elapsed > max_time_s:
                print(f"Stopping process: exceeded time limit ({elapsed:.2f}s)")
                p.terminate()
                return {"status": "fail", "latency": elapsed, "vram_usage": gpu_mem}

            if peak_after > max_mem_mb:
                print(f"Stopping process: exceeded GPU memory limit ({gpu_mem} MB)")
                p.terminate()
                return {"status": "fail", "latency": elapsed, "vram_usage": gpu_mem}
            
            time.sleep(0.2)

        p.join()

        result = result_queue.get()
        result.update({"latency": elapsed, "vram_usage": gpu_mem})
        query_result = result
        
        if result["status"] == "fail":
            print("ERROR IN LORA QUERY")
            
        return query_result
    

def lora_target_fn(cls, failure_label, query_str, lora_type, result_queue):
    try:
        pred = cls.query(failure_label, query_str, lora_type)
        pred["status"] = "success"
        result_queue.put(pred)
    except Exception as e:
        result_queue.put({"status": "fail"})