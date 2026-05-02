# https://ollama.com/blog/structured-outputs
from ollama import chat
from pydantic import BaseModel
from typing import Literal
from pathlib import Path

class ToneResponse(BaseModel):
    tone: Literal['positive', 'negative']
    confidence: float

for review in Path.cwd().glob('reviews/*'):
    review_text = open(review).read()
    response = chat(
      messages=[
        {
          'role': 'user',
          'content': f'''
          {review_text}

          Is this a negative or a positive movie review? Provide tone and confidence values.
          ''',
        }
      ],
      model='llama3',
      options={'temperature': 0.3},
      format=ToneResponse.model_json_schema(),
    )

    output = ToneResponse.model_validate_json(response.message.content)
    print(review_text, output, sep='\n', end="\n\n")

