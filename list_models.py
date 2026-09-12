from dotenv import load_dotenv
load_dotenv()

from google import genai

client = genai.Client()
for m in client.models.list():
    if "embedContent" in m.supported_actions:
        print(m.name)

client = genai.Client()
for m in client.models.list():
    if "generateContent" in m.supported_actions:
        print(m.name)