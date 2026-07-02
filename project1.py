import os
import pypdf
import json
import re
import mlflow

from dotenv import load_dotenv

from mlflow.genai.scorers import Correctness, Guidelines
from mlflow.genai import scorer
from mlflow.entities import Feedback
from openai import OpenAI


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

reader = pypdf.PdfReader("Samuel_Oliveira_CientistaDeDados_15012026.pdf")

resume_text = ""
for page in reader.pages:
  resume_text += page.extract_text()
  
def clean_text(text):
  text = text.lower()
  text = re.sub(r'[^a-z\s]', " ", text)
  text = re.sub(r'\s+', " ", text)
  return text

cleaned_resume_text = clean_text(resume_text)
# print(cleaned_resume_text[:200])

mlflow.set_experiment("Resume Skills Extraction Prompt Evaluation")

def predict_fn(cleaned_resume_text) -> str:
  response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": cleaned_resume_text}]
  )
  
  return response.choices[0].message.content

@scorer
def minimum_five_skills(inputs, outputs, expectations):
  try:
    skills_json = json.loads(outputs)
    if skills_json.get("skills"):
      if len(skills_json.get("skills")) > 5:
        return True
      else:
        return False
    else:
      return False
  except:
    return False

@scorer
def is_json(outputs):
  try:
    json.loads(outputs)
    return True
  except:
    return False

scorers = [
  Correctness(),
  Guidelines(name="coverage", guidelines="Are all the skills capture?"),
  minimum_five_skills,
  is_json
]

versions = [1, 2]

for v in versions:
  prompt = mlflow.genai.load_prompt(f"prompts:/RESUME_SKILL_EXTRACTION_PROMPT/{v}")
  formatted_prompt = prompt.format(cleaned_resume_text=cleaned_resume_text)
  
  eval_dataset = [
    {
      "inputs": {"cleaned_resume_text": formatted_prompt},
      "expectations": {"expected_response": "python,ml,sklearn,deep learning,machine learning", "skills": []}
    }
  ]
  
  mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=predict_fn,
    scorers=scorers
  )