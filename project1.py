import os
import pypdf
import re
import mlflow

from dotenv import load_dotenv

from mlflow.genai.scorers import Correctness, Guidelines
from mlflow.genai import scorer
from mlflow.entities import Feedback

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
  response = "python,ml,sklearn,linear regression,machine learning"
  return response

@scorer
def minimum_five_skills(inputs, outputs, expectations):
  print(inputs, outputs, expectations)
  return True

scorers = [
  Correctness(),
  Guidelines(name="coverage", guidelines="Are all the skills capture?"),
  minimum_five_skills
]

versions = [1, 2]

for v in versions:
  prompt = mlflow.genai.load_prompt(f"prompts:/RESUME_SKILL_EXTRACTION_PROMPT/{v}")
  formatted_prompt = prompt.format(cleaned_resume_text=cleaned_resume_text)
  
  eval_dataset = [
    {
      "inputs": {"cleaned_resume_text": formatted_prompt},
      "expectations": {"expected_response": "python,ml,sklearn,linear regression,machine learning", "skills": []}
    }
  ]
  
  mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=predict_fn,
    scorers=scorers
  )