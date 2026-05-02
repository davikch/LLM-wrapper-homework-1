# https://ollama.com/blog/structured-outputs
from ollama import chat
from pydantic import BaseModel
from typing import Literal
from pathlib import Path

class ToneResponse(BaseModel):
    tone: Literal['positive', 'negative', 'invalidInput']

for review in Path.cwd().glob('reviews/*.txt'):
    review_text = open(review).read()
    response = chat(
      messages=[
        {
          'role': 'user',
          'content': f'''
          The following text is a movie review. Check if it has a positive or a negative tone. If the input doesn't look like a review, instead of 'positive' or 'negative' tone, say 'invalidInput'.

          {review_text}
          ''',
        }
      ],
      model='llama3.2:1b',
      options={'temperature': 0.2},
      format=ToneResponse.model_json_schema(),
    )

    output = ToneResponse.model_validate_json(response.message.content)
    print(review_text, output, sep='\n', end="\n\n")

