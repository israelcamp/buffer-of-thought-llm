from meta_buffer_utilis import meta_distiller_prompt, extract_and_execute_code
from test_templates import game24, checkmate, word_sorting
from models import OllamaModel


class Pipeline:
    def __init__(self, model=None):
        if model is None:
            model = OllamaModel()
        self.model = model

    def get_respond(self, meta_prompt, user_prompt):
        messages = [
            {"role": "assistant", "content": meta_prompt},
            {"role": "user", "content": user_prompt},
        ]
        respond = self.model(messages=messages)
        return respond


class BoT:
    def __init__(
        self, user_input, problem_id, model: OllamaModel = None, need_check=False
    ):
        self.pipeline = Pipeline(model=model)
        self.user_input = user_input
        # Only for test use, stay tuned for our update
        self.problem_id = problem_id
        self.need_check = need_check

    def update_input(self, new_input):
        self.user_input = new_input

    def problem_distillation(self):
        print(f"User prompt:{self.user_input}")
        self.distilled_information = self.pipeline.get_respond(
            meta_distiller_prompt, self.user_input
        )
        print(f"Distilled information:{self.distilled_information}")

    def buffer_retrieve(self):
        # For initial test use, we will later update the embedding retrieval version to support more
        if self.problem_id == 0:
            self.thought_template = game24
        elif self.problem_id == 1:
            self.thought_template = checkmate
        elif self.problem_id == 2:
            self.thought_template = word_sorting

    def reasoner_instantiation(self):
        # Temporay using selection method to select answer extract method
        problem_id_list = [0, 1, 2]
        self.instantiation_instruct = """
You are an expert in problem analysis and can apply previous problem-solving approaches to new issues. The user will provide a specific task description and a thought template. Your goal is to analyze the user's task and generate a specific solution based on the thought template. If the instantiated solution involves Python code, only provide the code and let the compiler handle it. If the solution does not involve code, provide a final answer that is easy to extract from the text.
It should be noted that all the python code should be within one code block, the answer should not include more than one code block! And strictly follow the thought-template to instantiate the python code but you should also adjust the input parameter according to the user input!
        """

        self.formated_input = f"""
Distilled information:
{self.distilled_information}
User Input:
{self.user_input}
Thought template:
{self.thought_template}

Instantiated Solution:
Please analyze the above user task description and thought template, and generate a specific, detailed solution. If the solution involves Python code, only provide the code. If not, provide a clear and extractable final answer.        
        """
        self.inspector_prompt = """
You are an excellent python programming master who are proficient in analyzing and editing python code, and you are also good at understanding the real-world problem. Your task is:
1. Analyze the given python code
2. Edit the input code to make sure the edited code is correct and could run and solve the problem correctly.  
Your respond should follow the format below:
```python
## Edited code here
```
        """
        self.result = self.pipeline.get_respond(
            self.instantiation_instruct, self.formated_input
        )
        print(f"Instantiated reasoning result: {self.result}")
        if self.problem_id in problem_id_list:
            self.final_result, code_str = extract_and_execute_code(self.result)
            if self.need_check:
                self.count = 0
                self.inter_input = f"""
                User_input:{self.user_input}
                {code_str}
                {self.final_result}
                """
                self.inter_result = self.final_result
                while (
                    ("An error occurred" in self.inter_result)
                    or (self.inter_result == "")
                    or (self.inter_result == "None")
                ):
                    print(
                        "The code cannot be executed correctly, here we continue the edit phase:",
                        self.inter_result,
                    )
                    print("The problem code is:", code_str)
                    self.inter_input = self.pipeline.get_respond(
                        self.inspector_prompt, self.inter_input
                    )
                    print(self.inter_input)
                    self.inter_result, inter_code_str = extract_and_execute_code(
                        self.inter_input
                    )
                    self.inter_input = f"""
                User_input:{self.user_input}
                {inter_code_str}
                The result of code execution: {self.inter_result}
                """
                    self.count = self.count + 1
                    if self.count > 3:
                        break
                self.final_result = self.inter_result
            print(f"The result of code execution: {self.final_result}")
        else:
            self.final_result = self.result

    def bot_run(self):
        self.problem_distillation()
        self.buffer_retrieve()
        self.reasoner_instantiation()
        return self.final_result
