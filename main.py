# https://ollama.com/blog/structured-outputs
from ollama import chat
from pydantic import BaseModel
from typing import Literal
from pathlib import Path
import csv

class ToneResponse(BaseModel):
    tone: Literal['positive', 'negative', 'invalidInput']

output_file = open('answer.csv', 'w')
csvwriter = csv.writer(output_file)
csvwriter.writerow(['File', 'Verdict'])

for review in sorted(Path.cwd().glob('reviews/*.txt')):
    review_text = open(review).read()
    response = chat(
      messages=[
        {
          'role': 'user',
          'content': f'''
          The following text is a movie review. Check if it has a positive or a negative tone. If the input doesn't even distantly look like a meaningful text or a review, instead of 'positive' or 'negative' tone, say 'invalidInput'.

          {review_text}
          ''',
        }
      ],
      model='llama3.2:1b',
      options={'temperature': 0},
      format=ToneResponse.model_json_schema(),
    )

    output = ToneResponse.model_validate_json(response.message.content)
    csvwriter.writerow([review.name, output])

output_file.close()
