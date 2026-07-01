import mlflow

# prompt_v3 = """
# You are a helpful assistant who answers the provided question concisely and precisely
# Question: {{ question }} and {{ user_name }}
# """

# mlflow.genai.register_prompt(
#   name="FAQ-BOT1",
#   template=prompt_v3,
#   commit_message="version 3 prompt"
# )

prompt = mlflow.genai.load_prompt("prompts:/FAQ-BOT1/2")
# print(prompt.format(question="How is the wether like ?", user_name="Samuel"))

