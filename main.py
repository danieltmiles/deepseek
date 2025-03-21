import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set up device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype="auto",
    # attn_implementation="eager",
)
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Create a system prompt that instructs the model to be direct
# system_prompt = "You are a helpful assistant. Provide direct and concise answers without showing your reasoning process."
system_prompt = "You are a helpful assistant. Provide direct and concise answers and show your reasoning process."

# Combine system prompt with user prompt
user_prompt = "Tell me a short story"
combined_prompt = f"{system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"

# Tokenize input
inputs = tokenizer(combined_prompt, return_tensors="pt").to(device)

# Generate with beam search
start = time.time()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=2000,
        num_beams=5,           # Enable beam search with 5 beams
        early_stopping=True,   # Now early_stopping makes sense
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    )
end = time.time()

full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
if "</think>" in full_response:
    final_answer = full_response.split("</think>")[-1].strip()
else:
    final_answer = full_response
print(final_answer)
print(f"generated in {end - start} seconds")
