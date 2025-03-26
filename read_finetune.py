import json

data = []
with open('fine_tune.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]
    df = {}
    for d in data:
        messages = d.get('messages', [])
        user_msg = ""
        assistant_msg = ""
        for message in messages:
            if message['role'] == 'user':
                user_msg = str(message['content'])
            elif message['role'] == 'assistant':
                assistant_msg = message['content']
            if assistant_msg not in df:
                df[assistant_msg] = []
        df[assistant_msg].append(user_msg)
    for k, v in df.items():
        print(f"Assistant content: {k}")
        for i in v:
            print(f"User content: {i}")
        print("=====================================")
        wait = input('Add it to the file? (y/n)')
print(len(data))
# with open('fine_tune.jsonl', 'w') as w:
#    for d in data:
#        messages = d.get('messages', [])
#        print("=====================================")
#        for message in messages:
#            if message['role'] == 'user':
#                print(f"User content: {message['content']}")
#            elif message['role'] == 'assistant':
#                print(f"Assistant content: {message['content']}")
#        wait = input('Add it to the file? (y/n)')
#        if wait == 'n':
#            continue
#        w.write(json.dumps(d, ensure_ascii=False))
#        w.write('\n')
