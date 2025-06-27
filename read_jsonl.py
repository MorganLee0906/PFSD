import json

with open("fine_tune_all.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        messages = data.get("messages", [])
        print("="*40)
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            print(f"[{role}]\n{content}\n")
        wait = input("Press Enter to continue...")  