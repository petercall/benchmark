#Regular Imports
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import re
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from peft import PeftModel

# File Imports
from hyperparams import model_name, model_args, tokenizer_args, dataset_name, system_prompt, max_length, generation_args, checkpoint_type, checkpoint_location


# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_args)
model = AutoModelForCausalLM.from_pretrained(model_name, **model_args)
device = next(model.parameters()).device

#Load in checkpoint
print(f"Checkpoint Type: {checkpoint_type}")
if checkpoint_type == "sft":
    checkpoint = torch.load(checkpoint_location)
    model.load_state_dict(checkpoint["model_state"])
elif checkpoint_type == "lora":
    model = PeftModel.from_pretrained(model, checkpoint_location)

# Load AIM24 dataset and put the model into evaluation mode
dataset = load_dataset(dataset_name, split="train")
model.eval()

# Format prompt
def format_prompt(my_dict):
    input = [{"role" : "system", "content" : system_prompt}, {"role" : "user", "content" : my_dict["problem"]}]
    tokens = tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt = True)
    return {"message" : tokens}

# Prepare data
dataset = dataset.map(format_prompt)
problems = [item['message'] for item in dataset]  
labels = [item['answer'] for item in dataset]

# Extract integer answer from generated output
def extract_integer_answer(text):
    answer_block = re.search(r"<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>", text, re.DOTALL)
    if answer_block:
        answer_text = answer_block.group(1)
        match = re.search(r'\d+', answer_text)
        if match:
            return int(match.group())
    return None

# Batching function
def batch_inference(problems, batch_size=8):
    predictions = []

    #Calculate how many batch iterations to do
    itrs = len(problems)//batch_size
    if itrs*batch_size != len(problems):
        itrs += 1
    
    #Loop through
    for i in tqdm(range(0, itrs)):
        #Grab a batch of problems
        if i == (itrs - 1):
            batch_problems = problems[i*batch_size:]
        else:
            batch_problems = problems[i*batch_size:(i+1)*batch_size]
        
        #Tokenize the batch and put it onto the GPU
        inputs = tokenizer(batch_problems, return_tensors="pt", truncation = True, padding = True, max_length=max_length)
        inputs = {name : tens.to(device) for name, tens in inputs.items()}

        #Get the model's prediction for that batch
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_args, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        
        #Turn the token ids back to text
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        #Extract just the 3-integer prediction for each response
        for j, text in enumerate(decoded_outputs):
            text = text[len(problems[(i*batch_size) + j])-56:]   #This cuts off the input question as part of the response
            predicted_answer = extract_integer_answer(text)
            if predicted_answer is not None:
                predictions.append(predicted_answer)
            else:
                predictions.append(-1)  # fallback for invalid generations
    
    return predictions


# Run batch inference
predictions = batch_inference(problems, batch_size=8)

# Calculate accuracy
accuracy = accuracy_score(labels, predictions)
print(f"Accuracy: {accuracy:.4f}")


