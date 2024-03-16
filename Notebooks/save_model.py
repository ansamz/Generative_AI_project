# this script is to save the pretrained model in a directory so it can be easier to retrieve it and not download it each time

from transformers import LlamaTokenizer, LlamaForCausalLM

def save_model_and_tokenizer(model_name, save_directory):
    # Initialize the tokenizer and model from the Hugging Face Hub
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name)

    # Save the tokenizer and model to the specified directory
    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)
    print(f"Model and tokenizer have been saved to {save_directory}")

save_model_and_tokenizer('psmathur/orca_mini_3b', './model')
