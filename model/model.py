import google.generativeai as genai
import time
import json
import re
import ast

class GeminiModel():
    def __init__(self, api_key, model_name, task):
        self.api_key=api_key
        genai.configure(api_key=api_key)
        self.model=genai.GenerativeModel(model_name)
        self.max_attempts=5
        self.task = task

    def do_inference(self, prompt):
        global current_tpm, current_rpd, start_time, responses_today
        attempts = 0
        while attempts < self.max_attempts:
            try:
                # print('='*10)
                # print(prompt)
                # print('='*10)
                
                response = self.model.generate_content(prompt)
                time.sleep(5)
                
                # print("Response")
                # print(response)
                
                content = response.text  # Usar .text o .choices[0].message.content según tu modelo
                
                # Buscar el primer objeto JSON en el contenido
                match = re.search(r'\{.*\}', content, re.DOTALL)           
                answer = ast.literal_eval(match.group())
                answer = {int(k): v for k, v in answer.items()}
                print('='*10)
                print(answer)
                print('='*10)
                return answer
            except json.JSONDecodeError:
                print("JSON decoding failed")
                print(content)
                attempts += 1
                
            except Exception as error:
                print(f"Error encountered (attempt {attempts + 1}):", error)
                # print(f"Error encountered (attempt {attempts + 1})")
                attempts += 1
                
                if "GenerateRequestsPerMinutePerProjectPerModel" in str(error):
                    print("Rate limit hit, waiting 60 seconds before retrying...")
                    time.sleep(60)
                elif "429" in str(error):
                    raise RuntimeError(f"API key limit reached (429) for key {self.api_key}")
                else:
                    time.sleep(10)
                
                
        print("Failed after", self.max_attempts, "attempts.")
        return {}
            