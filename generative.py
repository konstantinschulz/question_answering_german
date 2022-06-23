from transformers import pipeline, set_seed

generator = pipeline("text-generation", model="malteos/gpt2-xl-wechsel-german")
set_seed(42)
print(generator("Welcher König erwarb den Buckingham Palace 1761?", max_length=30, num_return_sequences=1))
