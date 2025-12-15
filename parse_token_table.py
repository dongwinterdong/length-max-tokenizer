import ijson
import sys
import time
import os

def main():
    # 默认路径，也支持命令行参数
    filepath = "/home/arxiv_code/tokenizers_rust/token_table_safe.json"
    if len(sys.argv) > 1:
        filepath = sys.argv[1]

    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return

    print(f"Analyzing {filepath} (Size: {os.path.getsize(filepath) / (1024*1024*1024):.2f} GB)...")
    start_total = time.time()

    # 第一遍：解析 Merges 列表
    # 结构: { "merges": [ ... ] }
    print("\n[1/2] Scanning Merges...")
    merges_count = 0
    start_merges = time.time()
    
    try:
        with open(filepath, 'rb') as f:
            # ijson.items(f, 'merges.item') 会生成一个生成器，每次 yield 列表里的一个对象
            items = ijson.items(f, 'merges.item')
            for item in items:
                if merges_count < 5:
                    print(f"  Merge #{merges_count}: {item}")
                merges_count += 1
                if merges_count % 10000 == 0:
                    sys.stdout.write(f"\r  Processed {merges_count} merges...")
                    sys.stdout.flush()
        print(f"\n  Total Merges: {merges_count} (Time: {time.time() - start_merges:.2f}s)")

    except Exception as e:
        print(f"\nError reading merges: {e}")

    # 第二遍：解析 Vocab 映射
    # 结构: { ..., "vocab": { "token": freq, ... } }
    print("\n[2/2] Scanning Vocab...")
    vocab_count = 0
    start_vocab = time.time()
    
    try:
        with open(filepath, 'rb') as f:
            # ijson.kvitems(f, 'vocab') 专门用于迭代 JSON 对象中的键值对
            kvitems = ijson.kvitems(f, 'vocab')
            for key, value in kvitems:
                if vocab_count < 5:
                    # 避免打印过长的 key
                    display_key = key if len(key) < 50 else key[:47] + "..."
                    print(f"  Token #{vocab_count}: {display_key!r} -> {value}")
                vocab_count += 1
                if vocab_count % 100000 == 0:
                    sys.stdout.write(f"\r  Processed {vocab_count} tokens...")
                    sys.stdout.flush()
        print(f"\n  Total Vocab Size: {vocab_count} (Time: {time.time() - start_vocab:.2f}s)")

    except Exception as e:
        print(f"\nError reading vocab: {e}")

    print(f"\nDone. Total elapsed time: {time.time() - start_total:.2f}s")

if __name__ == "__main__":
    main()


