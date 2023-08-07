# llm_playground

for file in $(find . -type f -name "*.ipynb" -printf '%P\n'); do $NBC $file; done
for file in $(find . -type f -name "new_*.ipynb" -printf '%P\n'); do new_name="${file#new_}"; mv "$file" "$new_name"; done