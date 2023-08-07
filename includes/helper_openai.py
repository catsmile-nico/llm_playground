def print_job_detail(res):
    print("Model: {}".format(res.model))
    print("Tokens used: Prompt({}) + Completion({}) = {}".format(*res.usage.values()))

def print_chatcompletion_output(res):
    print_job_detail(res)
    print(res.choices[0].message.content)

def print_completion_output(res):
    print_job_detail(res)
    print(res.choices[0].text)