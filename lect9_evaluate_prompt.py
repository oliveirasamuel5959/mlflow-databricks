import os
import mlflow
from dotenv import load_dotenv
from openai import OpenAI

# prompt_v3 = """
# You are a helpful assistant who answers the provided question concisely and precisely
# Question: {{ question }} and {{ user_name }}
# """

# mlflow.genai.register_prompt(
#   name="FAQ-BOT1",
#   template=prompt_v3,
#   commit_message="version 3 prompt"
# )

prompt = mlflow.genai.load_prompt("prompts:/FAQ-BOT/2")
# print(prompt.format(question="How is the wether like ?", user_name="Samuel"))

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY)

# message = prompt.format(question="What exactly is the difference between sun and moon?")
# print(message)

# response = client.chat.completions.create(
#   model="gpt-4o-mini",
#   messages=[{"role": "user", "content": message}]
# )

# print(response.choices[0].message.content)

mlflow.set_experiment("Prompt Evaluation")

def predict_fn(question):
  prompt = mlflow.genai.load_prompt("prompts:/FAQ-BOT/2")
  message = prompt.format(question="What exactly is the difference between sun and moon?")
  
  response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": message}]
  )
  output = response.choices[0].message.content
  
  return output

data = [
  {
    "inputs": {"question": "Who invented telephone?"},
    "expectations": {"expected_response": "Alexander Graham Bell"},
  }
]

scorers = [
  mlflow.genai.scorers.Correctness()
]

mlflow.genai.evaluate(
  data=data,
  scorers=scorers,
  predict_fn=predict_fn
)
