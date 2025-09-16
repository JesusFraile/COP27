import google.generativeai as genai
import time
import json
import re
import ast

class GeminiModel():
    def __init__(self, api_key, model_name, task):
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
                
            
            except Exception as error:
                print(f"Error encountered (attempt {attempts + 1}):", error)
                if "429" in str(error):
                    print("Error 429: Resource has been exhausted. Waiting 20 seconds before retrying")
                    time.sleep(20)
                else:
                    time.sleep(10)
                attempts += 1
                
        print("Failed after", self.max_attempts, "attempts.")
        return {}

    # def do_inference(self, prompt):
    #     global current_tpm, current_rpd, start_time, responses_today
    #     attempts = 0
    #     while attempts < self.max_attempts:
    #         try:
    #             #Completar con el control de límite
    #             print('='*10)
    #             print(prompt)
    #             print('='*10)
    #             response=self.model.generate_content(prompt)
    #             time.sleep(5)
    #             print("Response")
    #             print(response)
    #             content=response.text
    #             # content = response.choices[0].message.content.strip()
    #             match = re.search(r'\[.*?\]', content, re.DOTALL)
    #             answer=json.loads(match.group())
    #             print('='*10)
    #             print(answer)
    #             print('='*10)
    #             return answer
                
    #         except (json.JSONDecodeError, Exception) as error:
    #             print(f"Error encountered (attempt {attempts + 1}):", error)
    #             attempts+=1
    #             time.sleep(10) 
    #         except Exception as error:
    #             if "429" in str(error):  
    #                 print("Error 429: Resource has been exhausted. Waiting 20 seconds before trying again")
    #                 time.sleep(20)  
    #                 continue
    #             print(f"Error encountered (attempt {attempts + 1}):", error)
    #             attempts += 1
    #             time.sleep(10)
                
    #     print("Failed after", self.max_attempts, "attempts.")
    #     return response.text