# list_vocab_stream.py
import ijson

path = "token_table_5b.json"

with open(path, "r", encoding="utf-8") as f:
    parser = ijson.parse(f)
    # 定位到 vocab 对象
    in_vocab = False
    for prefix, event, value in parser:
        if prefix == "vocab" and event == "map_key":
            # map_key 的 value 就是 token
            token = value
            # 下一个事件读取 id
            prefix_id, event_id, value_id = next(parser)
            print(f"{value_id}\t{token}")
        elif prefix == "vocab" and event == "start_map":
            in_vocab = True
        elif in_vocab and prefix == "vocab" and event == "end_map":
            break