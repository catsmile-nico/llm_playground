import os, json, time

from llama_cpp import Llama

llama = Llama(model_path="/home/catsmile/models/llama-2-7b-chat.Q5_K_M.gguf", verbose=False, n_ctx=100, n_threads=1, n_gpu_layers=-1)

def get_reply(prompt):
    start_time = time.time()
    response = llama.create_completion(
        f"""{prompt}""", 
        max_tokens=100, 
        stop=["</s>"], 
        echo=False
    )
    end_time = time.time()
    duration = end_time - start_time
        
    print(response["choices"][0]["text"])
    print("Time-taken: ", duration)

    # dump response
    response = {"prompt":prompt} | response
    response["duration"] = round(duration,1)
    with open("./log.log", "a") as file: 
        json.dump(response, file, indent=4)
        file.write("\n"+"="*50+"\n")

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def main():

    while True:
        cli_prompt = input("\nYou: ")

        if cli_prompt == "exit":
            break
        else:
            print("LLM:", end="")
            get_reply(cli_prompt) # or just call this line

main()