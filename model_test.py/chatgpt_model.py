import openai
import json


api_key = "sk-xxxxxx"

result = {
    "class": "mel",
    "confidence": 0.984
}

prompt = f"""
The model detected {result['class']} with {result['confidence']*100:.2f}% confidence.
Explain what this diagnosis means in simple terms for a non-medical user.
"""

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful medical assistant."},
        {"role": "user", "content": prompt}
    ]
)

print(response['choices'][0]['message']['content'])
