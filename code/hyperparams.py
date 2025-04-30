import torch

#Model hyperparameters
model_name = "Qwen/Qwen2.5-0.5B"
model_args = {"torch_dtype" : torch.bfloat16, "device_map" : "auto", "use_sliding_window" : True, "sliding_window" : 1000}

#Tokenization hyperparameters
tokenizer_args = {"padding_side" : "left"}
max_length = 5000

#Checkpoint hyperaparemters
checkpoint_type = "sft"
checkpoint_location = "/work/10509/ptc487/vista/classes/gen_models/sft/checkpoints/sft.pt"  #The sft checkpoint
# checkpoint_location = "/work/10509/ptc487/vista/classes/gen_models/lora/checkpoints/lora"  #The lora checkpoint

#Dataset hyperparameters
dataset_name = "HuggingFaceH4/aime_2024"

#System prompt
system_prompt = "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\n\n'} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, present the final solution that you deem correct, in 3-digit integer format with nothing else returned but the integer: <|begin_of_solution|> {single 3-digit integer such as 045 or 923} <|end_of_solution|> Now, try to solve the following question through the above guidelines:"

#Generation_args
generation_args = {
    "max_new_tokens" : 35000
}