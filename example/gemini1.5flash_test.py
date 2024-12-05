import google.generativeai as genai

genai.configure(api_key="AIzaSyC20PNJ8xfgJKHK7Ew3ARDVuHJdIf0qfKg")
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Explain how AI works")
print(response.text)