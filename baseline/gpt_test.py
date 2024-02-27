from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are an expert in sports. Your job is to classify a given list of objects\
      as either sports-related or not sports-related. Respond with exactly one line consisting of one of the \
     following labels: \nsports\nnot sports"},
    {"role": "user", "content": "Example: backpack, bicycle, person\n Label:"}
  ]
)

print(completion.choices[0].message)