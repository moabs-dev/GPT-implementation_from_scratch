from g4f.client import Client
from g4f.Provider import You

client=Client()

ask =input(' > ')

response=client.chat.completions.create(
    model='gpt-3.5-turbo',
    provider=You,
    messages=[
        {
            'role':'user',
            'content':ask
        }
    ]
)

print(response.choices[0].message.content)